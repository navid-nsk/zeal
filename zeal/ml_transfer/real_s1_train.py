"""real_s1_train.py -- train the three declared model families on the REAL data with exactly the exp3_ml recipe
(s1_train.train: Adam lr 5e-4, weight decay 1e-3, 40 epochs, batch 256, torch seed 1, early stopping on validation AUC) and write,
for each declared evaluation window, the model weights and nominal outputs into exp3_ml_real/<window>/.
Usage: python real_s1_train.py   (writes exp3_ml_real/s1_train_real.json, data_report.json, <window>/model_<m>.pt, nominal_<m>.npz)"""
import os, sys, time, shutil, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_common import *
from mlt_common import build, windows, DEV

MODELS = ["mlp", "feat", "feat24"]


def samples(t0, t1, z, lab):                         # identical to s1_train.samples
    X, Y, J, Tt = [], [], [], []
    for j in range(N_ENT):
        for t in range(max(t0, W - 1), t1):
            if np.isfinite(lab[j, t]):
                X.append(z[j, t - W + 1:t + 1]); Y.append(lab[j, t]); J.append(j); Tt.append(t)
    J, Tt = np.array(J), np.array(Tt)
    nxt = np.full(len(J), -1); ok = (J[1:] == J[:-1]) & (Tt[1:] == Tt[:-1] + 1); nxt[:-1][ok] = np.arange(1, len(J))[ok]
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32), torch.tensor(nxt)


def auc(p, y):                                       # identical to s1_train.auc
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 > 0 and n0 > 0 else float("nan")


def train(name, Xtr, Ytr, Ntr, Xva, Yva):           # identical to s1_train.train with lam = 0
    torch.manual_seed(1)
    m = build(name).to(DEV); opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=5e-4, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss(); best = (np.inf, None, -1); has = torch.nonzero(Ntr >= 0).ravel()
    for ep in range(40):
        m.train(); perm = torch.randperm(len(has))
        for i in range(0, len(perm), 256):
            idx = has[perm[i:i + 256]]; xb, yb = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
            loss = bce(m.logit(xb).squeeze(1), yb); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = -auc(m.logit(Xva.to(DEV)).squeeze(1).cpu().numpy(), Yva.numpy())
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep)
    m.load_state_dict(best[1]); m.eval()
    return m, -best[0], best[2]


D = make_real_data(); z, lab = D["z"].astype(np.float32), D["lab"]
jdump(dict(D["report"], clients=D["meta"]), os.path.join(REAL_OUT, "data_report.json"))
Xtr, Ytr, Ntr = samples(TRAIN_START, TRAIN_END - 24, z, lab); Xva, Yva, _ = samples(TRAIN_END, VAL_END - 24, z, lab)
rep = dict(n_train=len(Ytr), n_val=len(Yva), pos_rate_train=float(Ytr.mean()), pos_rate_val=float(Yva.mean()))
EV = {w: samples(E, E + T, z, lab) for w, E in WINDOWS.items()}
for w, (Xe, Ye, _) in EV.items():
    rep[f"pos_rate_{w}"] = float(Ye.mean()); rep[f"pos_rate_{w}_per_client"] = [float(np.nanmean(lab[j, WINDOWS[w]:WINDOWS[w] + T])) for j in range(N_ENT)]
print(rep, flush=True)
for name in MODELS:
    t0 = time.time(); m, auc_sel, ep_sel = train(name, Xtr, Ytr, Ntr, Xva, Yva)
    torch.save(m.state_dict(), os.path.join(REAL_OUT, f"model_{name}.pt"))
    rep[name] = dict(horizon=24, best_epoch=ep_sel, val_auc=auc_sel, seconds=time.time() - t0)
    for w, (Xe, Ye, _) in EV.items():
        wd = window_dir(w); shutil.copyfile(os.path.join(REAL_OUT, f"model_{name}.pt"), os.path.join(wd, f"model_{name}.pt"))
        with torch.no_grad():
            pe = m(Xe.to(DEV)).squeeze(1).cpu().numpy()
            y = eval_series_real(D, w); f = m(windows(y).reshape(-1, W).to(DEV)).reshape(N_ENT, T).cpu().numpy()
        rep[name][w] = dict(eval_auc=auc(pe, Ye.numpy()), eval_brier=float(np.mean((pe - Ye.numpy()) ** 2)),
                            f_eval_mean=float(f.mean()), f_eval_sd=float(f.std()), f_eval_min=float(f.min()), f_eval_max=float(f.max()),
                            mean_abs_step=float(np.abs(np.diff(f, axis=1)).mean()))
        np.savez(os.path.join(wd, f"nominal_{name}.npz"), f=f, y=y, lab=lab[:, WINDOWS[w]:WINDOWS[w] + T])
    print(name, rep[name], flush=True)
jdump(rep, os.path.join(REAL_OUT, "s1_train_real.json"))
for w in WINDOWS:                                    # per-window copy so that s5_tables' glob finds the training record
    jdump({k: (dict(v, **v[w]) if isinstance(v, dict) and w in v else v) for k, v in rep.items() if not k.startswith("pos_rate_") or w in k},
          os.path.join(window_dir(w), "s1_train_real.json"))
