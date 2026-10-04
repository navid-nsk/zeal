"""Certificate transport LP: for each nested pair (fine cell a inside
coarse cell b) the largest and smallest difference of population-weighted means consistent with the certified
first-order tiles, i.e. max / min of d . phi over the polytope
    lo_p <= phi_p <= hi_p,   c^-_pq <= phi_q - phi_p <= c^+_pq  (4-neighbour pixel pairs inside b),
with d = alpha_a - alpha_b (population shares) and c^+/- the gradient-interval bounds on the increment between
neighbouring pixels. Reports, per pair, U (full LP), V (value intervals only, closed form), T (gradient only), the
realized difference, and at map level the certified movement bound against the observed movement, next to the
global Lipschitz bound L*d_up and the Bhatia-Davis bound for nested coarsenings.
Usage: python transport_lp.py --field georgia [--max_pairs 0]
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--max_pairs", type=int, default=0); a = p.parse_args()
SPEC = {"georgia": (paths.RASTER["georgia"], "tract_idx", "county_idx", "weight", 0.1002, 136.3),
        "gm_q4": (paths.RASTER["gm_q4"], "tract_idx", "county_idx", "weight", None, None),
        "gm_bad": (paths.RASTER["gm_bad"], "tract_idx", "county_idx", "weight", None, None),
        "mx_rwi": (paths.RASTER["mx_rwi"], "tract_idx", "county_idx", "weight", None, None)}
ras_path, fine_k, coarse_k, w_k, d_up, L_old = SPEC[a.field]
F = np.load(paths.enclosure(a.field)); lo, hi = F["lo"], F["hi"]; g1l, g1h, g2l, g2h = F["g1l"], F["g1h"], F["g2l"], F["g2h"]
ras = np.load(ras_path); mask = ras["mask"].astype(bool) & np.isfinite(lo); N = mask.shape[0]; h = 2.0 / (N - 1)
w = np.where(mask, ras[w_k], 0.0).astype(float); fine = np.where(mask, ras[fine_k], -1); coarse = np.where(mask, ras[coarse_k], -1)
lam = (lo + hi) / 2                                                            # field value enclosed by [lo, hi]
pix = np.full((N, N), -1, dtype=np.int64); ii, jj = np.nonzero(mask); pix[ii, jj] = np.arange(len(ii))
# pixel-level increment bounds for right (x, column) and down (y, row) neighbours
cR_hi = h * np.maximum(g1h, np.roll(g1h, -1, axis=1)); cR_lo = h * np.minimum(g1l, np.roll(g1l, -1, axis=1))
cD_hi = h * np.maximum(g2h, np.roll(g2h, -1, axis=0)); cD_lo = h * np.minimum(g2l, np.roll(g2l, -1, axis=0))

rows, t0 = [], time.time()
mu_tot = w.sum(); mov_obs2 = 0.0; U2 = 0.0; V2 = 0.0; T2 = 0.0; BD2 = 0.0; Q2 = 0.0
for b in np.unique(coarse[coarse >= 0]):
    sel = (coarse == b); idx = pix[sel]; n = len(idx); loc = -np.ones(N * N, dtype=np.int64); loc[idx] = np.arange(n)
    rr, cc = np.nonzero(sel)
    # edges inside b
    E = []
    right = sel & np.roll(sel, -1, axis=1); right[:, -1] = False; r0, c0 = np.nonzero(right)
    E.append((loc[pix[r0, c0]], loc[pix[r0, c0 + 1]], cR_lo[r0, c0], cR_hi[r0, c0]))
    down = sel & np.roll(sel, -1, axis=0); down[-1, :] = False; r0, c0 = np.nonzero(down)
    E.append((loc[pix[r0, c0]], loc[pix[r0 + 1, c0]], cD_lo[r0, c0], cD_hi[r0, c0]))
    ep = np.concatenate([e[0] for e in E]); eq = np.concatenate([e[1] for e in E]); clo = np.concatenate([e[2] for e in E]); chi = np.concatenate([e[3] for e in E])
    m_ = len(ep); rows_i = np.repeat(np.arange(2 * m_), 2)
    cols = np.concatenate([np.stack([eq, ep], 1).ravel(), np.stack([ep, eq], 1).ravel()])
    vals = np.concatenate([np.tile([1.0, -1.0], m_), np.tile([1.0, -1.0], m_)])
    A = sp.csr_matrix((vals, (rows_i, cols)), shape=(2 * m_, n)); bub = np.concatenate([chi, -clo])   # phi_q - phi_p <= chi ; phi_p - phi_q <= -clo
    wb = w[rr, cc]; mu_b = wb.sum(); alpha_b = wb / mu_b; lo_b, hi_b = lo[rr, cc], hi[rr, cc]; lam_b = lam[rr, cc]
    mean_b = float((alpha_b * lam_b).sum()); Mb, mb = float(hi_b.max()), float(lo_b.min())
    for a_ in np.unique(fine[sel]):
        if a_ < 0:
            continue
        ina = (fine[rr, cc] == a_); mu_a = wb[ina].sum()
        if mu_a <= 0:
            continue
        d = -alpha_b.copy(); d[ina] += wb[ina] / mu_a
        real = float((d * lam_b).sum())
        V_pos = float((np.maximum(d, 0) * hi_b).sum() + (np.minimum(d, 0) * lo_b).sum()); V_neg = float(-((np.maximum(d, 0) * lo_b).sum() + (np.minimum(d, 0) * hi_b).sum()))
        res = linprog(-d, A_ub=A, b_ub=bub, bounds=list(zip(lo_b, hi_b)), method="highs"); U_pos = -res.fun if res.status == 0 else V_pos
        res = linprog(d, A_ub=A, b_ub=bub, bounds=list(zip(lo_b, hi_b)), method="highs"); U_neg = -res.fun if res.status == 0 else V_neg
        resT = linprog(-d, A_ub=A, b_ub=bub, bounds=[(None, None)] * n, method="highs"); T_pos = -resT.fun if resT.status == 0 else np.inf
        U = max(U_pos, U_neg); V = max(V_pos, V_neg); T = T_pos
        rows.append(dict(coarse=int(b), fine=int(a_), mu_a=float(mu_a), real=real, U_pos=U_pos, U_neg=U_neg, V=V, T=T, U=U))
        mov_obs2 += mu_a * real ** 2; U2 += mu_a * U ** 2; V2 += mu_a * V ** 2; T2 += mu_a * min(T, V) ** 2
        Q2 += mu_a * (max(abs(V_pos), abs(V_neg))) ** 2
    BD2 += mu_b * max((Mb - mean_b) * (mean_b - mb), 0.0)                           # Bhatia-Davis: within-b variance bound
    if a.max_pairs and len(rows) >= a.max_pairs:
        break
    if len(rows) % 200 < 20:
        print(f"  {len(rows)} pairs, {time.time() - t0:.0f}s", flush=True)
mov_obs = np.sqrt(mov_obs2 / mu_tot); out = dict(field=a.field, pairs=len(rows), movement_observed=mov_obs,
    bound_lp=float(np.sqrt(U2 / mu_tot)), bound_value_only=float(np.sqrt(V2 / mu_tot)), bound_gradient_only=float(np.sqrt(T2 / mu_tot)), bound_bhatia_davis=float(np.sqrt(BD2 / mu_tot)),
    bound_global=(L_old * d_up if L_old else None), seconds=time.time() - t0,
    share_lp_below_value=float(np.mean([r["U"] < r["V"] - 1e-9 for r in rows])), median_ratio_lp=float(np.median([r["U"] / max(abs(r["real"]), 1e-9) for r in rows])))
for k in ("bound_lp", "bound_value_only", "bound_gradient_only", "bound_bhatia_davis", "bound_global"):
    if out[k]:
        out[k + "_x"] = out[k] / mov_obs
print(json.dumps(out, indent=1))
json.dump(dict(summary=out, pairs=rows), open(paths.transport_lp(a.field), "w", encoding="utf-8"), indent=1)
print("saved", f"data/transport_lp_{a.field}.json")
