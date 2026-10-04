"""a3_manchester_identification.py -- the Manchester identification experiment
(identification and reconstruction, Supplementary Theorems S11-S13); results exp1/A3.

Fine truth: 8,966 Output Areas (two outcomes: q4 = degree share, bad = bad-health share; population bases w).
Observations: LSOA means (exact), MSOA means. Everything population-weighted; OA truth used only for evaluation,
oracle rows (labelled) and the identified-set coverage check.

Part 1  Field decomposition (Theorem S13 (i)-(iv)): per-OA predictions of the stored neural field (seed 0 network), m and rho
        computed DIRECTLY from the predictions, the exact-fit identity with the imperfect-fit term delta_fit (Theorem S13 (ii)),
        oracle shrinkage kappa* (labelled) and the operational shrinkage kappa_up trained one level up.
Part 2  Estimator table, two tracks (Theorem S13 (vi)):
        coarse-only: painting; affine-per-MSOA GLS (feasible GLS, binomial Sigma); tensor cubic-spline GLS (knots 2 km;
        1 km reported as not identified if rank-deficient); graph-energy interpolation (exact LSOA means); energy +
        nugget tuned one level up; neural field; field detail shrunk by kappa_up;
        covariate-augmented: dasymetric t = P t + beta * (I - P) x with beta ecological / one-level-up / oracle.
        Exact-data R^2 against OA truth, and repeated-noise mean risks (y = A t + eps, eps ~ N(0, Sigma_LSOA), R reps)
        with Monte-Carlo s.e.; the R_GLS / R_paint formulas (Theorem S13 (vi)) beside the realized means.
Part 3  Identified sets under declared graph classes (Theorem S12, graph form; graph class, NOT the continuous class):
        Lipschitz on the OA adjacency graph with centroid distances (km), L in a grid, and graph-TV budget tau in a grid;
        L_min and tau_min (feasibility frontier, Theorem S12); per-OA identified intervals on a random sample of OAs:
        width, compatibility with the OA truth (deterministic, no 'coverage' claimed), decision breakdown for
        'OA above its LSOA mean'.
Usage: python a3_manchester_identification.py --out <dir> [--reps 200] [--n_ids 150] [--skip_field]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, pandas as pd, scipy.sparse as sp, scipy.sparse.linalg as spl
from scipy.optimize import linprog
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--reps", type=int, default=200); p.add_argument("--n_ids", type=int, default=150)
p.add_argument("--skip_field", action="store_true"); p.add_argument("--seed", type=int, default=41); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); t_start = time.time()
D = paths.GM_DIR
u = pd.read_csv(D + "gm_units.csv"); adj = pd.read_csv(D + "gm_adjacency.csv").values; n = len(u)
A_ls = u["LSOA21CD_i"].values; A_ms = u["MSOA21CD_i"].values; K_ls, K_ms = A_ls.max() + 1, A_ms.max() + 1
ms_of_ls = np.zeros(K_ls, int); ms_of_ls[A_ls] = A_ms
E = adj[adj[:, 0] != adj[:, 1]]; E = np.unique(np.sort(E, 1), axis=0)
import geopandas as gpd
g = gpd.read_file(D + "oa_boundaries.geojson").set_index("OA21CD").loc[u["OA21CD"].values]
cx, cy = g.geometry.centroid.x.values / 1000, g.geometry.centroid.y.values / 1000           # km (BNG)
dist = np.sqrt((cx[E[:, 0]] - cx[E[:, 1]]) ** 2 + (cy[E[:, 0]] - cy[E[:, 1]]) ** 2)
L_oa = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); L_oa = L_oa + L_oa.T; L_oa = sp.diags(np.asarray(L_oa.sum(1)).ravel()) - L_oa
el = np.unique(np.sort(np.c_[A_ls[E[:, 0]], A_ls[E[:, 1]]], 1), axis=0); el = el[el[:, 0] != el[:, 1]]
L_ls = sp.coo_matrix((np.ones(len(el)), (el[:, 0], el[:, 1])), shape=(K_ls, K_ls)); L_ls = L_ls + L_ls.T; L_ls = sp.diags(np.asarray(L_ls.sum(1)).ravel()) - L_ls
wm = lambda v, w, gidx, K: np.bincount(gidx, weights=w * v, minlength=K) / np.maximum(np.bincount(gidx, weights=w, minlength=K), 1e-12)
def r2(t, pr, w): tb = (w * t).sum() / w.sum(); return float(1 - (w * (t - pr) ** 2).sum() / (w * (t - tb) ** 2).sum())
def wnorm2(v, w): return float((w * v * v).sum() / w.sum())


def pycno(Lap, w, gidx, K, y, paint, lam):
    nn = Lap.shape[0]; ww = w / w.mean()
    C = sp.coo_matrix((w / np.maximum(np.bincount(gidx, weights=w, minlength=K), 1e-12)[gidx], (gidx, np.arange(nn))), shape=(K, nn)).tocsr()
    H = 2 * (Lap + lam * sp.diags(ww)); rhs = np.concatenate([2 * lam * ww * paint, y])
    KKT = sp.bmat([[H, C.T], [C, None]]).tocsc()
    sol = spl.spsolve(KKT + sp.diags(np.r_[np.zeros(nn), -1e-12 * np.ones(K)]), rhs); return sol[:nn]


def field_oa_predictions(oc):
    """per-OA population-weighted means of the stored seed-0 network (CPU)."""
    import torch; _argv = sys.argv; sys.argv = _argv[:1]
    import geo_m4_multistate as M; sys.argv = _argv
    from zeal_m1 import Field, dev
    s = M.S(f"GM{oc}", N=1024); cfg = json.load(open(paths.field_file(f"uk_field_{oc}.json")))["config"]
    net = Field(m=cfg["m"], sigma=cfg["sigma"], w=128, depth=3, act="tanh").to(dev)
    net.load_state_dict(torch.load(paths.field_file(f"uk_field_{oc}_net.pt"), map_location=dev))
    with torch.no_grad():
        pred = net(s.coords)
    R = np.load(D + "gm_raster.npz"); mi = R["mask"].reshape(-1); oa_lab = R["oa_idx"].reshape(-1)[mi]
    w_pix = s.w.cpu().numpy(); pr = pred.cpu().numpy().ravel()
    uniq, inv = np.unique(oa_lab, return_inverse=True)
    num = np.bincount(inv, weights=w_pix * pr); den = np.bincount(inv, weights=w_pix)
    e = np.full(n, np.nan); e[uniq[uniq >= 0]] = (num / np.maximum(den, 1e-12))[uniq >= 0]
    tr = R[f"truth_{oc}"].reshape(-1)[mi]; tnum = np.bincount(inv, weights=w_pix * np.nan_to_num(tr)); t_chk = np.full(n, np.nan); t_chk[uniq[uniq >= 0]] = (tnum / np.maximum(den, 1e-12))[uniq >= 0]
    return e, t_chk


OUT = {}
for oc, wc, nc, xc in [("q4", "w_q4", "n_q4", "bad"), ("bad", "w_bad", "n_bad", "q4")]:
    res = {}; t0 = time.time()
    w = u[wc].values.astype(float); t = u[oc].values.astype(float); ok = w > 0; w = np.where(ok, w, 0.0); t = np.where(ok, t, 0.0)
    W_ls = np.bincount(A_ls, weights=w, minlength=K_ls); y_ls = wm(t, w, A_ls, K_ls); y_ms = wm(t, w, A_ms, K_ms)
    paint = y_ls[A_ls]; r = t - paint; Wsh = wnorm2(r, w) / wnorm2(t - (w * t).sum() / w.sum(), w)
    res["R2_paint"] = r2(t, paint, w); res["within_share_W"] = Wsh
    # ---------------- Part 1: field decomposition ----------------
    if not a.skip_field:
        e, t_chk = field_oa_predictions(oc); good = np.isfinite(e) & ok
        res["field_oa_truth_check_maxabs"] = float(np.nanmax(np.abs(t_chk[good] - t[good])))
        e = np.where(good, e, paint); s_ = e - wm(e, w, A_ls, K_ls)[A_ls]
        m = np.sqrt(wnorm2(s_, w) / wnorm2(r, w)); rho = float((w * r * s_).sum() / np.sqrt((w * r * r).sum() * (w * s_ * s_).sum()))
        Pte = wm(t - e, w, A_ls, K_ls)[A_ls]; delta_fit = wnorm2(Pte, w) / wnorm2(t - (w * t).sum() / w.sum(), w)
        R2_field = r2(t, e, w); R2_pred = 1 - (delta_fit + Wsh * (1 + m * m - 2 * rho * m))
        kappa_star = rho / m; kappa_star_c = float(np.clip(kappa_star, 0, 1))
        # operational shrinkage one level up: LSOA predictions of the field vs LSOA data, within MSOA
        e_ls = wm(e, w, A_ls, K_ls); r_up = y_ls - y_ms[ms_of_ls]; s_up = e_ls - wm(e_ls, W_ls, ms_of_ls, K_ms)[ms_of_ls]
        kappa_up = float((W_ls * r_up * s_up).sum() / (W_ls * s_up * s_up).sum())
        res["field"] = dict(R2_oa=R2_field, R2_lsoa_fit=r2(y_ls, e_ls, W_ls), m=float(m), rho=rho, delta_fit=float(delta_fit), R2_identity=float(R2_pred), identity_residual=float(R2_field - R2_pred),
                            beats_paint_iff_rho_gt_m_over_2=bool(rho > m / 2), kappa_star_oracle=float(kappa_star), kappa_up=kappa_up,
                            R2_shrink_oracle=r2(t, paint + kappa_star_c * s_, w), R2_shrink_up=r2(t, paint + float(np.clip(kappa_up, 0, 1)) * s_, w))
        field_pred = e; field_shrunk_up = paint + float(np.clip(kappa_up, 0, 1)) * s_
    else:
        field_pred = field_shrunk_up = None
    # ---------------- Part 2: estimators ----------------
    ests = {"painting": lambda y: y[A_ls]}
    # affine per MSOA: H columns [1, cx, cy] per MSOA block
    Hcols = []; rowsH = []; colsH = []; valsH = []
    for k in range(K_ms):
        idx = np.where(A_ms == k)[0]
        for j, f in enumerate((np.ones(len(idx)), cx[idx] - cx[idx].mean(), cy[idx] - cy[idx].mean())):
            rowsH += idx.tolist(); colsH += [3 * k + j] * len(idx); valsH += f.tolist()
    H_aff = sp.csr_matrix((valsH, (rowsH, colsH)), shape=(n, 3 * K_ms))
    Amat = sp.csr_matrix((w / np.maximum(W_ls, 1e-12)[A_ls], (A_ls, np.arange(n))), shape=(K_ls, n))   # LSOA averaging
    p_hat = np.clip(y_ls, 1e-4, 1 - 1e-4); Sig = p_hat * (1 - p_hat) / np.maximum(W_ls, 1)                 # feasible binomial variance per LSOA
    def gls_factory(H, name):
        X = (Amat @ H).toarray() if sp.issparse(H) else Amat @ H
        keep = np.linalg.norm(X, axis=0) > 1e-9; Xk = X[:, keep]; rank = np.linalg.matrix_rank(Xk)
        if rank < Xk.shape[1]:
            res[f"{name}_rank"] = dict(columns=int(Xk.shape[1]), rank=int(rank), identified=False)
            G = np.linalg.pinv(Xk.T @ (Xk / Sig[:, None])) @ Xk.T / Sig[None, :]
        else:
            res[f"{name}_rank"] = dict(columns=int(Xk.shape[1]), rank=int(rank), identified=True)
            G = np.linalg.solve(Xk.T @ (Xk / Sig[:, None]), Xk.T / Sig[None, :])
        Hk = H[:, keep] if sp.issparse(H) else H[:, keep]; Hk = Hk.toarray() if sp.issparse(Hk) else Hk
        est = lambda y: np.clip(Hk @ (G @ y), 0.0, 1.0)      # shares: clipped to [0, 1] (declared post-processing; the unclipped GLS is reported in *_rank)
        Wd = w / w.sum(); R_gls_formula = float(np.trace((Hk.T * Wd) @ Hk @ np.linalg.pinv(Xk.T @ (Xk / Sig[:, None]))))
        return est, R_gls_formula, Hk, G
    est_aff, R_gls_aff, _, _ = gls_factory(H_aff, "gls_affine_msoa"); ests["gls_affine_msoa"] = est_aff
    # tensor cubic B-spline bases at 2 km and 1 km
    from scipy.interpolate import BSpline
    def spline_basis(knot_km):
        kx = np.arange(cx.min() - 3 * knot_km, cx.max() + 4 * knot_km, knot_km); ky = np.arange(cy.min() - 3 * knot_km, cy.max() + 4 * knot_km, knot_km)
        Bx = BSpline.design_matrix(cx, kx, 3).toarray(); By = BSpline.design_matrix(cy, ky, 3).toarray()
        H = np.einsum("ni,nj->nij", Bx, By).reshape(n, -1); return H[:, np.abs(H).sum(0) > 1e-9]
    for km in (2.0, 1.0):
        H_sp = spline_basis(km); est_sp, R_gls_sp, _, _ = gls_factory(H_sp, f"gls_spline_{km:g}km"); res[f"gls_spline_{km:g}km_R_formula"] = R_gls_sp
        if res[f"gls_spline_{km:g}km_rank"]["identified"]:
            ests[f"gls_spline_{km:g}km"] = est_sp
        else:
            res[f"gls_spline_{km:g}km_rank"]["note"] = "rank-deficient: not identified from LSOA means; excluded from the estimator table"
    res["gls_affine_msoa_R_formula"] = R_gls_aff
    # graph energy (Tobler) exact means; nugget tuned one level up
    lams = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
    paint_ls = y_ms[ms_of_ls]; cal = {lam: r2(y_ls, pycno(L_ls, np.maximum(W_ls, 1e-9), ms_of_ls, K_ms, y_ms, paint_ls, lam), W_ls) for lam in lams}
    lam_up = max(cal, key=cal.get); res["energy_lambda_up"] = lam_up
    ests["energy_exact"] = lambda y: pycno(L_oa, np.maximum(w, 1e-9), A_ls, K_ls, y, y[A_ls], 0.0)
    ests["energy_nugget_up"] = lambda y: pycno(L_oa, np.maximum(w, 1e-9), A_ls, K_ls, y, y[A_ls], lam_up)
    # covariate track
    x = np.nan_to_num(u[xc].values.astype(float)); x_ls = wm(x, w, A_ls, K_ls); xi = x - x_ls[A_ls]; x_ms = wm(x_ls, W_ls, ms_of_ls, K_ms)
    dyl, dxl = y_ls - y_ms[ms_of_ls], x_ls - x_ms[ms_of_ls]; b_up = float((W_ls * dxl * dyl).sum() / (W_ls * dxl * dxl).sum())
    Xl = np.c_[np.ones(K_ls), x_ls]; sl = np.sqrt(W_ls); b_eco = float(np.linalg.lstsq(Xl * sl[:, None], y_ls * sl, rcond=None)[0][1])
    b_w = float((w * xi * r).sum() / (w * xi * xi).sum())
    ests["dasy_ecological"] = lambda y: y[A_ls] + b_eco * xi; ests["dasy_one_level_up"] = lambda y: y[A_ls] + b_up * xi; ests["dasy_oracle_within(ORACLE)"] = lambda y: y[A_ls] + b_w * xi
    res["covariate"] = dict(x=xc, beta_eco=b_eco, beta_up=b_up, beta_within_oracle=b_w, advantage_identity_up=float(wnorm2(xi, w) * (b_up ** 2 - 2 * b_up * b_w)),
                            beats_paint_iff=bool(0 < b_up / b_w < 2), level_invariance_check=dict(beta_LSOA_within_MSOA=b_up))
    # level-invariance one more level up (MSOA within LAD) -- no fine truth needed
    A_lad = u["LAD23CD_i"].values; lad_of_ms = np.zeros(K_ms, int); lad_of_ms[A_ms] = A_lad; W_ms = np.bincount(A_ms, weights=w, minlength=K_ms)
    y_lad = wm(y_ms, W_ms, lad_of_ms, A_lad.max() + 1); x_lad = wm(x_ms, W_ms, lad_of_ms, A_lad.max() + 1)
    dym, dxm = y_ms - y_lad[lad_of_ms], x_ms - x_lad[lad_of_ms]; res["covariate"]["level_invariance_check"]["beta_MSOA_within_LAD"] = float((W_ms * dxm * dym).sum() / (W_ms * dxm * dxm).sum())
    # exact-data R^2 table
    table = {}
    for name, f in ests.items():
        pr = f(y_ls); table[name] = dict(R2_exact_data=r2(t, pr, w), mass_preserving=bool(np.abs(wm(pr, w, A_ls, K_ls) - y_ls).max() < 1e-6))
    if field_pred is not None:
        table["neural_field"] = dict(R2_exact_data=r2(t, field_pred, w), mass_preserving=False); table["neural_field_shrunk_up"] = dict(R2_exact_data=r2(t, field_shrunk_up, w), mass_preserving=True)
    # repeated-noise mean risks (Theorem S13 (vi)): y = A t + eps, eps ~ N(0, Sigma)
    Wd = w / w.sum(); risks = {name: [] for name in ests}
    for rep in range(a.reps):
        y_sim = y_ls + rng.normal(size=K_ls) * np.sqrt(Sig)
        for name, f in ests.items():
            if "ORACLE" in name:
                continue
            pr = f(y_sim); risks[name].append(float((Wd * (pr - t) ** 2).sum()))
    for name in ests:
        if risks[name]:
            table[name]["risk_noisy_mean"] = float(np.mean(risks[name])); table[name]["risk_noisy_se"] = float(np.std(risks[name]) / np.sqrt(len(risks[name])))
        table[name]["risk_exact_data"] = float((Wd * (ests[name](y_ls) - t) ** 2).sum())
    Bp = sp.csr_matrix((np.ones(n), (np.arange(n), A_ls)), shape=(n, K_ls))
    table["painting"]["R_paint_formula"] = float((Wd * (paint - t) ** 2).sum() + (Wd * np.asarray((Bp @ sp.diags(Sig) @ Bp.T).diagonal())).sum())
    res["estimators"] = table
    # ---------------- Part 3: identified sets (graph classes) ----------------
    ids = {}
    nE = len(E); Aeq = Amat; beq = y_ls
    # L_min: variables (t, L): min L s.t. t_v - t_w <= L d, t_w - t_v <= L d, A t = y, 0 <= t <= 1
    Dif = sp.csr_matrix((np.r_[np.ones(nE), -np.ones(nE)], (np.r_[np.arange(nE), np.arange(nE)], np.r_[E[:, 0], E[:, 1]])), shape=(nE, n))
    A_ub = sp.vstack([sp.hstack([Dif, -sp.csr_matrix(dist[:, None])]), sp.hstack([-Dif, -sp.csr_matrix(dist[:, None])])]).tocsr()
    A_eq = sp.hstack([Aeq, sp.csr_matrix((K_ls, 1))]).tocsr()
    rr = linprog(np.r_[np.zeros(n), 1.0], A_ub=A_ub, b_ub=np.zeros(2 * nE), A_eq=A_eq, b_eq=beq, bounds=[(0, 1)] * n + [(0, None)], method="highs")
    L_min = float(rr.x[-1]) if rr.status == 0 else np.nan; ids["L_min_per_km"] = L_min
    # tau_min: min graph TV = sum |t_v - t_w| (unweighted by distance: declared graph-TV class) s.t. A t = y
    A_ub2 = sp.vstack([sp.hstack([Dif, -sp.identity(nE)]), sp.hstack([-Dif, -sp.identity(nE)])]).tocsr()
    rr = linprog(np.r_[np.zeros(n), np.ones(nE)], A_ub=A_ub2, b_ub=np.zeros(2 * nE), A_eq=sp.hstack([Aeq, sp.csr_matrix((K_ls, nE))]).tocsr(), b_eq=beq, bounds=[(0, 1)] * n + [(0, None)] * nE, method="highs")
    tau_min = float(rr.fun) if rr.status == 0 else np.nan; ids["tau_min_graphTV"] = tau_min
    ids["truth_L_graph"] = float(np.max(np.abs(t[E[:, 0]] - t[E[:, 1]]) / dist)); ids["truth_graphTV"] = float(np.abs(t[E[:, 0]] - t[E[:, 1]]).sum())
    sample = rng.choice(np.where(ok)[0], size=min(a.n_ids, ok.sum()), replace=False)
    def interval_L(L, idx):
        b_ub = np.r_[L * dist, L * dist]; A_ubL = sp.vstack([Dif, -Dif]).tocsr(); c = np.zeros(n); c[idx] = 1
        lo = linprog(c, A_ub=A_ubL, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=[(0, 1)] * n, method="highs"); hi = linprog(-c, A_ub=A_ubL, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=[(0, 1)] * n, method="highs")
        return (lo.fun if lo.status == 0 else np.nan, -hi.fun if hi.status == 0 else np.nan)
    def interval_tau(tau, idx):
        c = np.r_[np.zeros(n), np.zeros(nE)]; c[idx] = 1
        A_ub3 = sp.vstack([A_ub2, sp.csr_matrix(np.r_[np.zeros(n), np.ones(nE)][None, :])]).tocsr(); b3 = np.r_[np.zeros(2 * nE), tau]
        Aeq3 = sp.hstack([Aeq, sp.csr_matrix((K_ls, nE))]).tocsr(); bnd = [(0, 1)] * n + [(0, None)] * nE
        lo = linprog(c, A_ub=A_ub3, b_ub=b3, A_eq=Aeq3, b_eq=beq, bounds=bnd, method="highs"); hi = linprog(-c, A_ub=A_ub3, b_ub=b3, A_eq=Aeq3, b_eq=beq, bounds=bnd, method="highs")
        return (lo.fun if lo.status == 0 else np.nan, -hi.fun if hi.status == 0 else np.nan)
    for cls, grid, fn in (("lipschitz_graph", [max(L_min, 1e-9) * f for f in (1.0, 1.5, 2.0, 4.0)], interval_L), ("graphTV", [tau_min * f for f in (1.0, 1.5, 2.0, 4.0)], interval_tau)):
        ids[cls] = {}
        for kappa in grid:
            iv = np.array([fn(kappa, i) for i in sample]); widths = iv[:, 1] - iv[:, 0]
            inside = (iv[:, 0] - 1e-9 <= t[sample]) & (t[sample] <= iv[:, 1] + 1e-9)
            above = (iv[:, 0] > paint[sample]) | (iv[:, 1] < paint[sample])          # decision 'OA differs from LSOA mean' certified
            ids[cls][f"{kappa:.5g}"] = dict(width_median=float(np.nanmedian(widths)), width_mean=float(np.nanmean(widths)), truth_compatible_share=float(np.mean(inside)),
                                            decision_certified_share=float(np.mean(above)), oa_sd=float(np.sqrt(wnorm2(r, w))), n=int(len(sample)))
            print(f"  {oc} {cls} kappa={kappa:.4g}: width med {np.nanmedian(widths):.4f} (OA within-sd {np.sqrt(wnorm2(r, w)):.4f}) compatible {np.mean(inside):.3f} decisions {np.mean(above):.3f} {time.time()-t0:.0f}s", flush=True)
    res["identified_sets"] = ids
    OUT[oc] = res
    print(oc, json.dumps({k: v for k, v in res.items() if k in ("R2_paint", "within_share_W", "field", "covariate")}, indent=1, default=float))
    print(oc, "estimators", json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in table.items()}, indent=1), flush=True)
    json.dump(OUT, open(os.path.join(a.out, "a3_manchester.json"), "w"), indent=1, default=float)
print(f"done {time.time()-t_start:.0f}s")
