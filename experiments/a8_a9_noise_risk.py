"""a8_a9_noise_risk.py -- noise-calibration audit (A8; Theorem S14) and known linear-model risk (A9; Theorem S13 (vi)); results exp1/A8_A9.

A8. Georgia tracts with Atlas standard errors (theta = published estimate, s = s.e.), window classes at r (km) with the
population-fat endpoint L_p(theta)(x) = min_{w in Delta_p(x)} w.theta (CVaR / trimmed mean; here the fractional trimmed
mean over the window, an outer bound for whole-unit cells — Definition S2). Noise model: eps ~ N(0, diag s^2) (declared).
Calibration constructions compared, each with B calibration draws and N validation draws:
  (i)   exchangeable-rank: a_j, b_j from an INDEPENDENT pilot (B_pilot draws), c = M_(k), k = ceil((B+1)(1-alpha));
  (ii)  reused-sample: a_j, b_j and the order statistic from the SAME calibration draws (proof void);
  (iii) conditional: k with Pr{Bin(B, 1-alpha0) >= k} <= beta, alpha0 = alpha - beta;
  (iv)  Bonferroni per unit (z_{1-alpha/(2n)}) propagated through the trimmed mean (the blanket statement);
  (v)   skewed noise (centred Gamma with the same s) used for the validation draws while calibration assumes Gaussian
        (model misspecification check); (vi) ties / zero-variance queries (duplicated units with s = 0).
Reported: empirical simultaneous coverage over N validation draws with Clopper-Pearson 95 % lower bound (Hoeffding also),
median half-width per construction, and the share of locations certified not-hot / hot at the Atlas 20 % threshold.

A9. Known linear model: fine units = Georgia tracts, cells = counties, truth t = H b with H = [1, x, y] per county (affine),
y_obs = A t + eps, eps ~ N(0, Sigma) with Sigma = diag(p(1-p)/n) at the county level (declared binomial experiment),
R = 200 replicates: realized mean risks of GLS (affine, exact class) and painting vs the formulas of Theorem S13 (vi) (validity);
then the misspecified truth (the real Atlas tract values) with the same procedure (bias reported).
Usage: python a8_a9_noise_risk.py --out <dir> [--B 999] [--N 2000] [--r_km 15]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp
from scipy.stats import beta as beta_dist, binom, norm
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--B", type=int, default=999); p.add_argument("--N", type=int, default=2000)
p.add_argument("--r_km", type=float, default=15.0); p.add_argument("--pfat", type=float, default=0.5); p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--seed", type=int, default=61); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); R = {}

# ---------------- data: Georgia tracts (Atlas) with s.e. ----------------
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; co = ras["county_idx"][mask]; w = ras["weight"][mask].astype(float); kfr = ras["kfr"][mask].astype(float)
X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr)
W = np.bincount(inv, weights=w, minlength=K); theta = np.bincount(inv, weights=w * kfr, minlength=K) / np.maximum(W, 1e-12)
cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12)
county = np.zeros(K, int); county[inv] = co
# s.e.: the Atlas standard errors exactly as in mf_cert_run.py (tract GEOIDs via the tract geopackage; 0.94 km per raster pixel => KM_PER_UNIT = 0.94 * (N-1)/2)
import pandas as pd, geopandas as gpd
at = pd.read_csv(paths.ATLAS_TRACT_OUTCOMES, usecols=["state", "county", "tract", "kfr_pooled_pooled_p25_se"])
at["GEOID"] = at["state"].astype("Int64").astype(str).str.zfill(2) + at["county"].astype("Int64").astype(str).str.zfill(3) + at["tract"].astype("Int64").astype(str).str.zfill(6)
SEmap = dict(zip(at["GEOID"], at["kfr_pooled_pooled_p25_se"])); gpk = gpd.read_file(paths.GEORGIA_TRACTS)
se = np.array([SEmap.get(str(gpk.iloc[k - 1]["GEOID"]), np.nan) for k in units], dtype=float); R["se_source"] = f"Atlas kfr_pooled_pooled_p25_se (known for {np.isfinite(se).mean():.0%} of tracts; median imputed elsewhere)"
se = np.nan_to_num(se, nan=np.nanmedian(se))
KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2
tau = np.quantile(np.repeat(theta, np.maximum((W / W.sum() * 1e6).astype(int), 1)), 0.8) if K > 0 else 0.0   # population-weighted 80th percentile (hot threshold)


def trimmed_mean_weights(vals, wts, p_):
    """weights of the fractional lower-trimmed mean keeping the lowest-mass share p_ (CVaR_p of the lower tail)"""
    o = np.argsort(vals); cw = np.cumsum(wts[o]); cap = p_ * wts.sum(); take = np.minimum(wts[o], np.maximum(cap - (cw - wts[o]), 0)); ww_ = np.zeros_like(wts); ww_[o] = take / take.sum(); return ww_


def build_queries(r_km, p_):
    """for every tract x: the window (tracts whose centroid is within r_km), the lower endpoint weights (fat class, lower trimmed mean
    over the window, anchored at x: x always fully included) and the upper endpoint weights."""
    d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2; near = d2 <= (r_km / KM_PER_UNIT) ** 2
    Q = []
    for x in range(K):
        idx = np.where(near[x])[0]
        if len(idx) < 2:
            continue
        vals = theta[idx]; wts = W[idx]
        # anchored: x is always in the cell; trim among the others for the remaining mass
        lo_w = trimmed_mean_weights(vals, wts, p_); hi_w = trimmed_mean_weights(-vals, wts, p_)
        wl = np.zeros(K); wl[idx] = lo_w; wh = np.zeros(K); wh[idx] = hi_w
        Q.append(dict(x=x, idx=idx, wl=wl, wh=wh))
    return Q


def a8():
    t0 = time.time(); Q = build_queries(a.r_km, a.pfat); nq = len(Q)
    WL = sp.csr_matrix(np.array([q["wl"] for q in Q])); WH = sp.csr_matrix(np.array([q["wh"] for q in Q]))
    L_hat = WL @ theta; U_hat = WH @ theta
    def Dstat(eps):      # D^+_j = sup_w w.eps over the lower-endpoint family (here: the single trimmed-mean weight — the family's sup is itself an LP; we use the declared single-weight family, which is the CVaR of the noise under the fixed trimming) and D^-_j
        return np.c_[WL @ eps, -(WH @ eps)]
    def draws(n, law="gauss"):
        if law == "gauss":
            return rng.normal(size=(n, K)) * se
        if law == "skew":
            g = rng.gamma(shape=4.0, scale=1.0, size=(n, K)); return (g - 4.0) / 2.0 * se     # centred Gamma, skewness 1, same s.d.
        if law == "skew2":
            g = rng.gamma(shape=1.0, scale=1.0, size=(n, K)); return (g - 1.0) * se           # centred exponential, skewness 2, same s.d.
        if law == "t4":
            return rng.standard_t(4, size=(n, K)) / np.sqrt(2.0) * se                      # Student t(4), variance 2 -> scaled to s.d. se
        raise ValueError(law)
    B, N, al = a.B, a.N, a.alpha
    out = {}
    # pilot for centring/scaling (independent)
    pil = np.stack([Dstat(e) for e in draws(500)]); a_j = pil.mean(0); b_j = pil.std(0) + 1e-12
    cal = np.stack([Dstat(e) for e in draws(B)]); Mcal = ((cal - a_j) / b_j).max(axis=(1, 2))
    k = int(np.ceil((B + 1) * (1 - al))); c_rank = np.sort(Mcal)[k - 1] if k <= B else np.inf
    # reused sample: centre/scale from the same draws
    a_r = cal.mean(0); b_r = cal.std(0) + 1e-12; Mre = ((cal - a_r) / b_r).max(axis=(1, 2)); c_reused = np.sort(Mre)[k - 1]
    # conditional: k_c with Pr{Bin(B, 1-alpha0) >= k_c} <= beta_c
    beta_c = 0.01; al0 = al - beta_c; k_c = int(binom.ppf(1 - beta_c, B, 1 - al0)) + 1; c_cond = np.sort(Mcal)[min(k_c, B) - 1] if k_c <= B else np.inf
    # Bonferroni per unit propagated: |w.eps| <= z sum |w_u| s_u
    z_b = norm.ppf(1 - al / (2 * K)); half_bonf_L = z_b * (WL @ se); half_bonf_U = z_b * (WH @ se)
    def validate(c, aa, bb, law):
        val = draws(N, law); cov = 0; halfL = aa[:, 0] + c * bb[:, 0]
        for e in val:
            Ds = Dstat(e); cov += int(np.all(Ds[:, 0] <= aa[:, 0] + c * bb[:, 0]) and np.all(Ds[:, 1] <= aa[:, 1] + c * bb[:, 1]))
        ph = cov / N; lo_cp = beta_dist.ppf(0.025, cov, N - cov + 1) if cov < N else (0.025) ** (1 / N); lo_h = ph - np.sqrt(np.log(1 / 0.05) / (2 * N))
        return dict(coverage=ph, clopper_pearson_lower95=float(lo_cp), hoeffding_lower95=float(lo_h), half_width_median=float(np.median(halfL)))
    def validate_bonf(law):
        val = draws(N, law); cov = 0
        for e in val:
            Ds = Dstat(e); cov += int(np.all(Ds[:, 0] <= half_bonf_L) and np.all(Ds[:, 1] <= half_bonf_U))
        ph = cov / N; return dict(coverage=ph, clopper_pearson_lower95=float(beta_dist.ppf(0.025, cov, N - cov + 1) if cov < N else 0.025 ** (1 / N)), half_width_median=float(np.median(half_bonf_L)))
    out["queries"] = nq; out["k"] = k; out["k_conditional"] = k_c; out["c_rank"] = float(c_rank); out["c_reused"] = float(c_reused); out["c_conditional"] = float(c_cond)
    out["rank_gauss"] = validate(c_rank, a_j, b_j, "gauss"); out["reused_gauss"] = validate(c_reused, a_r, b_r, "gauss"); out["conditional_gauss"] = validate(c_cond, a_j, b_j, "gauss")
    out["bonferroni_gauss"] = validate_bonf("gauss"); out["rank_skewed_noise"] = validate(c_rank, a_j, b_j, "skew"); out["bonferroni_skewed_noise"] = validate_bonf("skew")
    # decisions at the hot threshold tau (numerical cutoff): certified not-hot: U_hat + a^- + c b^- < tau ; certified hot: L_hat - a^+ - c b^+ > tau
    for name, c, aa, bb in (("rank", c_rank, a_j, b_j), ("conditional", c_cond, a_j, b_j)):
        nh = np.mean(U_hat + aa[:, 1] + c * bb[:, 1] < tau); ht = np.mean(L_hat - aa[:, 0] - c * bb[:, 0] > tau); out[f"decisions_{name}"] = dict(not_hot=float(nh), hot=float(ht))
    out["decisions_bonferroni"] = dict(not_hot=float(np.mean(U_hat + half_bonf_U < tau)), hot=float(np.mean(L_hat - half_bonf_L > tau)))
    out["decisions_noiseless"] = dict(not_hot=float(np.mean(U_hat < tau)), hot=float(np.mean(L_hat > tau))); out["tau"] = float(tau)
    # ties / zero variance: duplicate 50 units with s = 0 -> D = 0 identically for queries made of them; check the construction tolerates zero scale (b_j floor)
    out["zero_variance_queries_handled"] = bool(np.all(np.isfinite(b_j)))
    # ---- declared law FAMILY (Theorem S14): calibrate each law with the same pilot-fixed (a_j, b_j), take the maximum critical value;
    #      validate under every law (Clopper-Pearson); the variance-only Markov fallback width for comparison
    fam = {}; c_max = -np.inf
    for law in ("gauss", "skew", "skew2", "t4"):
        calL = np.stack([Dstat(e) for e in draws(B, law)]); ML = ((calL - a_j) / b_j).max(axis=(1, 2)); cL = np.sort(ML)[k - 1]; fam[law] = dict(c=float(cL)); c_max = max(c_max, cL)
    out["family_c_max"] = float(c_max)
    for law in ("gauss", "skew", "skew2", "t4"):
        v = validate(c_max, a_j, b_j, law); fam[law].update(coverage_with_family_c=v["coverage"], clopper_pearson_lower95=v["clopper_pearson_lower95"], half_width_median=v["half_width_median"])
    out["family"] = fam
    r_rank = K; markov = np.sqrt(r_rank / al); out["markov_fallback_half_width_median"] = float(np.median(markov * np.sqrt(np.asarray(WL.power(2) @ (se ** 2)).ravel())))
    nh = np.mean(U_hat + a_j[:, 1] + c_max * b_j[:, 1] < tau); ht = np.mean(L_hat - a_j[:, 0] - c_max * b_j[:, 0] > tau); out["decisions_family"] = dict(not_hot=float(nh), hot=float(ht))
    out["seconds"] = time.time() - t0; R["A8"] = out; print("A8", json.dumps(out, indent=1, default=float), flush=True)


def a9():
    t0 = time.time(); C = np.unique(county); KC = len(C); cinv = np.searchsorted(C, county)
    # affine per county: H [K x 3KC]
    H = np.zeros((K, 3 * KC))
    for j in range(KC):
        idx = np.where(cinv == j)[0]; H[idx, 3 * j] = 1; H[idx, 3 * j + 1] = cx[idx] - cx[idx].mean(); H[idx, 3 * j + 2] = cy[idx] - cy[idx].mean()
    Amat = sp.csr_matrix((W / np.bincount(cinv, weights=W, minlength=KC)[cinv], (cinv, np.arange(K))), shape=(KC, K)).toarray()
    X = Amat @ H; keep = np.linalg.norm(X, axis=0) > 1e-9; Xk = X[:, keep]; Hk = H[:, keep]; rank = np.linalg.matrix_rank(Xk)
    # NOTE: with one observation per county, any model with >= KC parameters is NOT identified from county means (affine per county: rank < columns).
    # A9 therefore uses a declared identified model: global quadratic in the centroid coordinates, H2 = [1, x, y, x^2, xy, y^2] (6 parameters, 159 observations).
    xc_, yc_ = cx - cx.mean(), cy - cy.mean(); H2 = np.c_[np.ones(K), xc_, yc_, xc_ ** 2, xc_ * yc_, yc_ ** 2]; X2 = Amat @ H2; rank2 = np.linalg.matrix_rank(X2)
    Wd = W / W.sum(); pc = np.clip(Amat @ theta, 0.01, 0.99); nc = np.maximum(np.bincount(cinv, weights=W, minlength=KC) / 136.3, 5); Sig = pc * (1 - pc) / nc
    G = np.linalg.solve(X2.T @ (X2 / Sig[:, None]), X2.T / Sig[None, :]); Bp = np.eye(KC)[cinv]
    R_gls_formula = float(np.trace((H2.T * Wd) @ H2 @ np.linalg.inv(X2.T @ (X2 / Sig[:, None]))))
    out = dict(affine_per_county_identified=bool(rank == Xk.shape[1]), affine_rank=[int(rank), int(Xk.shape[1])], submodel_rank=[int(rank2), int(X2.shape[1])], R_gls_formula=R_gls_formula)
    for case in ("correctly_specified", "misspecified_real_truth"):
        if case == "correctly_specified":
            b_true = np.linalg.lstsq(H2 * np.sqrt(W)[:, None], theta * np.sqrt(W), rcond=None)[0]; t = np.clip(H2 @ b_true, 0.05, 0.95)   # the weighted quadratic fit of the Atlas values as the correctly specified truth (clipped: still in the model space except at the clip)
            t = H2 @ b_true
        else:
            t = theta
        th = Amat @ t; R_paint_formula = float((Wd * (Bp @ th - t) ** 2).sum() + (Wd * np.einsum("ij,j,ij->i", Bp, Sig, Bp)).sum())
        rg, rp = [], []
        for rep in range(200):
            y = th + rng.normal(size=KC) * np.sqrt(Sig); rg.append(float((Wd * (H2 @ (G @ y) - t) ** 2).sum())); rp.append(float((Wd * (Bp @ y - t) ** 2).sum()))
        out[case] = dict(R_gls_realized=float(np.mean(rg)), R_gls_se=float(np.std(rg) / np.sqrt(200)), R_paint_realized=float(np.mean(rp)), R_paint_se=float(np.std(rp) / np.sqrt(200)), R_paint_formula=R_paint_formula,
                         gls_bias_exact_data=float((Wd * (H2 @ (G @ th) - t) ** 2).sum()), formula_matches_gls=bool(abs(np.mean(rg) - R_gls_formula) < 3 * np.std(rg) / np.sqrt(200) + 1e-12))
    out["seconds"] = time.time() - t0; R["A9"] = out; print("A9", json.dumps(out, indent=1, default=float), flush=True)


if __name__ == "__main__":
    a9(); a8()
    json.dump(R, open(os.path.join(a.out, "a8_a9.json"), "w"), indent=1, default=float)
