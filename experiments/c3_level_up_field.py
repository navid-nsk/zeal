"""c3_level_up_field.py -- level-up shrinkage (Supplementary Theorem S13 (iv), level-up construction; results exp2/C3): retrain the Manchester
field on MSOA means ONLY (no LSOA outcomes), evaluate its LSOA predictions against the held-out observed LSOA means, estimate the
within-MSOA detail shrinkage kappa_up = <r_up, s_up>/||s_up||^2 there (r_up = (I - P_MSOA) y_LSOA, s_up = (I - P_MSOA) e_LSOA), and
report the R^2 at LSOA of the MSOA-trained field vs painting the MSOA mean; then transfer kappa_up to the LSOA-trained field's OA
detail (assumption: level invariance of the detail alignment, declared) and evaluate against OA truth (labelled 'transfer').
Same architecture/optimizer as uk_field.py (Field m=256, sigma=20, tanh, 3000 iters, beta=0.5). GPU. Usage: python c3_level_up_field.py --out <dir>
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, pandas as pd, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--iters", type=int, default=3000); p.add_argument("--seed", type=int, default=0); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
_argv = sys.argv; sys.argv = _argv[:1]
import geo_m4_multistate as M; sys.argv = _argv
from zeal_m1 import Field, dev
D = paths.GM_DIR; u = pd.read_csv(D + "gm_units.csv"); R = np.load(D + "gm_raster.npz")
A_ls = u["LSOA21CD_i"].values; A_ms = u["MSOA21CD_i"].values; K_ls, K_ms = A_ls.max() + 1, A_ms.max() + 1; ms_of_ls = np.zeros(K_ls, int); ms_of_ls[A_ls] = A_ms
wm = lambda v, w, g, K: np.bincount(g, weights=w * v, minlength=K) / np.maximum(np.bincount(g, weights=w, minlength=K), 1e-12)
def r2(t, pr, w): tb = (w * t).sum() / w.sum(); return float(1 - (w * (t - pr) ** 2).sum() / (w * (t - tb) ** 2).sum())
OUT = {}
for oc, wc in (("q4", "w_q4"), ("bad", "w_bad")):
    t0 = time.time(); s = M.S(f"GM{oc}", N=1024); cfg = json.load(open(paths.field_file(f"uk_field_{oc}.json")))["config"]
    torch.manual_seed(a.seed); net = Field(m=cfg["m"], sigma=cfg["sigma"], w=128, depth=3, act="tanh").to(dev); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for it in range(a.iters):            # MSOA (county) means only
        opt.zero_grad(); pred = net(s.coords); rec = (s.w * (s.wmean(pred, s.inv_co, s.Kco) - s.tgt_co) ** 2).sum() / s.w.sum(); (rec + cfg["beta"] * s.smooth_full(net)).backward(); opt.step()
    with torch.no_grad():
        pred = net(s.coords).cpu().numpy().ravel()
    w_pix = s.w.cpu().numpy(); mi = R["mask"].reshape(-1); oa_lab = R["oa_idx"].reshape(-1)[mi]; uniq, inv = np.unique(oa_lab, return_inverse=True)
    num = np.bincount(inv, weights=w_pix * pred); den = np.bincount(inv, weights=w_pix); e_oa = np.full(len(u), np.nan); e_oa[uniq[uniq >= 0]] = (num / np.maximum(den, 1e-12))[uniq >= 0]
    w = u[wc].values.astype(float); t = np.where(w > 0, u[oc].values.astype(float), 0.0); w = np.where(w > 0, w, 0.0); e_oa = np.where(np.isfinite(e_oa), e_oa, 0.0)
    W_ls = np.bincount(A_ls, weights=w, minlength=K_ls); y_ls = wm(t, w, A_ls, K_ls); y_ms = wm(t, w, A_ms, K_ms); e_ls = wm(e_oa, w, A_ls, K_ls); e_ms = wm(e_oa, w, A_ms, K_ms)
    r_up = y_ls - y_ms[ms_of_ls]; s_up = e_ls - wm(e_ls, W_ls, ms_of_ls, K_ms)[ms_of_ls]
    kappa_up = float((W_ls * r_up * s_up).sum() / (W_ls * s_up * s_up).sum()); m_up = float(np.sqrt((W_ls * s_up ** 2).sum() / (W_ls * r_up ** 2).sum())); rho_up = float((W_ls * r_up * s_up).sum() / np.sqrt((W_ls * r_up ** 2).sum() * (W_ls * s_up ** 2).sum()))
    res = dict(fit_R2_msoa=r2(y_ms, e_ms, np.bincount(A_ms, weights=w, minlength=K_ms)), R2_lsoa_msoa_trained_field=r2(y_ls, e_ls, W_ls), R2_lsoa_painting_msoa=r2(y_ls, y_ms[ms_of_ls], W_ls),
               kappa_up=kappa_up, m_up=m_up, rho_up=rho_up, beats_painting_at_lsoa=bool(rho_up > m_up / 2), R2_lsoa_shrunk=r2(y_ls, y_ms[ms_of_ls] + float(np.clip(kappa_up, 0, 1)) * s_up, W_ls), seconds=time.time() - t0)
    # transfer to the LSOA-trained field's OA detail (stored network) -- labelled transfer, assumption declared
    net2 = Field(m=cfg["m"], sigma=cfg["sigma"], w=128, depth=3, act="tanh").to(dev); net2.load_state_dict(torch.load(paths.field_file(f"uk_field_{oc}_net.pt"), map_location=dev))
    with torch.no_grad():
        pred2 = net2(s.coords).cpu().numpy().ravel()
    num2 = np.bincount(inv, weights=w_pix * pred2); e2 = np.full(len(u), np.nan); e2[uniq[uniq >= 0]] = (num2 / np.maximum(den, 1e-12))[uniq >= 0]; e2 = np.where(np.isfinite(e2), e2, 0.0)
    paint = y_ls[A_ls]; s_oa = e2 - wm(e2, w, A_ls, K_ls)[A_ls]; r_oa = t - paint
    res.update(transfer=dict(assumption="detail alignment kappa is level-invariant (MSOA->LSOA estimate applied to the LSOA-trained field's OA detail)", R2_oa_painting=r2(t, paint, w), R2_oa_field=r2(t, e2, w),
                             R2_oa_shrunk_transfer=r2(t, paint + float(np.clip(kappa_up, 0, 1)) * s_oa, w), kappa_oracle_oa=float((w * r_oa * s_oa).sum() / (w * s_oa * s_oa).sum())))
    OUT[oc] = res; print(oc, json.dumps(res, indent=1, default=float), flush=True)
    json.dump(OUT, open(os.path.join(a.out, "c3_level_up.json"), "w"), indent=1, default=float)
