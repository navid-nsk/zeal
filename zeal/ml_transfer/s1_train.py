"""s1_train.py -- generate the declared synthetic data, train the declared models, save weights and nominal outputs.
Models (mlt_common): 'mlp' = raw-lag MLP; 'mlp_tc' = the same architecture trained with a temporal-consistency penalty
lambda * mean (f(x_{t+1}) - f(x_t))^2 on consecutive training windows (the standard anti-flip-flop regulariser of operational
risk scores); lambda = the LARGEST value in TC_GRID whose validation AUC is within 0.02 of the unregularised model (selection on
validation AUC only, never on any verifier/ZEAL quantity).  'feat' (rolling-feature MLP) and the 7-day-horizon variants are
kept as options (--models) and are reported in findings.md as set-up history.
Usage: python s1_train.py [--models mlp,mlp_tc]   (writes exp3_ml/model_<name>.pt, nominal_<name>.npz, s1_train_<models>.json)"""
import os, sys, time, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlt_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--models", default="mlp,mlp_tc"); A_ = ap.parse_args()
TC_GRID = [10.0, 30.0, 100.0, 300.0]


def samples(t0, t1, z, lab):
    X, Y, J, Tt = [], [], [], []
    for j in range(N_ENT):
        for t in range(max(t0, W - 1), t1):
            if np.isfinite(lab[j, t]):
                X.append(z[j, t - W + 1:t + 1]); Y.append(lab[j, t]); J.append(j); Tt.append(t)
    J, Tt = np.array(J), np.array(Tt)
    nxt = np.full(len(J), -1); ok = (J[1:] == J[:-1]) & (Tt[1:] == Tt[:-1] + 1); nxt[:-1][ok] = np.arange(1, len(J))[ok]
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y), dtype=torch.float32), torch.tensor(nxt)


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 > 0 and n0 > 0 else float("nan")


def train(name, lam, Xtr, Ytr, Ntr, Xva, Yva):
    torch.manual_seed(1)
    m = build(name).to(DEV); opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=5e-4, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss(); best = (np.inf, None, -1); has = torch.nonzero(Ntr >= 0).ravel()
    for ep in range(40):
        m.train(); perm = torch.randperm(len(has))
        for i in range(0, len(perm), 256):
            idx = has[perm[i:i + 256]]; xb, yb, xn = Xtr[idx].to(DEV), Ytr[idx].to(DEV), Xtr[Ntr[idx]].to(DEV)
            lg = m.logit(xb).squeeze(1); loss = bce(lg, yb)
            if lam > 0:
                loss = loss + lam * ((torch.sigmoid(m.logit(xn).squeeze(1)) - torch.sigmoid(lg)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = -auc(m.logit(Xva.to(DEV)).squeeze(1).cpu().numpy(), Yva.numpy())      # early stopping on validation AUC
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep)
    m.load_state_dict(best[1]); m.eval()
    return m, -best[0], best[2]


rep = {}
for name in A_.models.split(","):
    hz = horizon(name); D = make_data(0, hz); z, lab = D["z"].astype(np.float32), D["lab"]
    Xtr, Ytr, Ntr = samples(0, TRAIN_END - hz, z, lab); Xva, Yva, _ = samples(TRAIN_END, VAL_END - hz, z, lab); Xte, Yte, _ = samples(EVAL_START, H_TOT - hz, z, lab)
    rep[f"data_h{hz}"] = dict(n_train=len(Ytr), n_val=len(Yva), n_eval=len(Yte), pos_rate_train=float(Ytr.mean()), pos_rate_val=float(Yva.mean()),
                              pos_rate_eval=float(Yte.mean()), feeders=D["meta"], cap=D["cap"])
    t0 = time.time()
    if name.endswith("_tc"):
        base = name[:-3]; _, auc0, _ = train(base, 0.0, Xtr, Ytr, Ntr, Xva, Yva); sel = []
        for lam in TC_GRID:
            m_, a_, e_ = train(base, lam, Xtr, Ytr, Ntr, Xva, Yva); sel.append(dict(lam=lam, val_auc=a_))
            if a_ >= auc0 - 0.02:
                m, lam_sel, ep_sel, auc_sel = m_, lam, e_, a_
        rep[f"{name}_selection"] = dict(val_auc_unregularised=auc0, grid=sel, selected_lambda=lam_sel)
    else:
        m, auc_sel, ep_sel = train(name, 0.0, Xtr, Ytr, Ntr, Xva, Yva); lam_sel = 0.0
    torch.save(m.state_dict(), os.path.join(OUT, f"model_{name}.pt"))
    with torch.no_grad():
        pte = m(Xte.to(DEV)).squeeze(1).cpu().numpy()
        y = eval_series(D); f = m(windows(y).reshape(-1, W).to(DEV)).reshape(N_ENT, T).cpu().numpy()
    rep[name] = dict(horizon=hz, tc_lambda=lam_sel, best_epoch=ep_sel, val_auc=auc_sel, eval_auc=auc(pte, Yte.numpy()),
                     eval_brier=float(np.mean((pte - Yte.numpy()) ** 2)), seconds=time.time() - t0,
                     f_eval_mean=float(f.mean()), f_eval_sd=float(f.std()), f_eval_min=float(f.min()), f_eval_max=float(f.max()),
                     mean_abs_step=float(np.abs(np.diff(f, axis=1)).mean()))
    np.savez(os.path.join(OUT, f"nominal_{name}.npz"), f=f, y=y, lab=lab[:, EVAL_START:H_TOT])
    print(name, rep.get(f"{name}_selection"), rep[name], flush=True)
jdump(rep, os.path.join(OUT, f"s1_train_{A_.models.replace(',', '_')}.json"))
