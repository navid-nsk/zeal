"""b7_conditional.py -- conditional-calibration audit (Supplementary Theorem S14; results exp2/B7): Georgia tracts, fat class
at r km (fractional trimmed endpoints), Gaussian declared law.
Two experiments, as the theorem distinguishes:
  (J) JOINT coverage: in each of T trials, draw a fresh calibration sample (B draws), compute c = M_(k) with the exchangeable-rank k, then
      draw ONE observation error and record whether all 2n endpoint queries are covered. The exchangeable-rank theorem guarantees
      Pr >= k/(B+1) >= 1 - alpha for this experiment.
  (C) CONDITIONAL coverage: fix one calibration sample, estimate Pr_eps{M > c} by N validation draws -- for the exchangeable-rank k and for
      the binomial-k construction (Pr{Bin(B, 1 - alpha0) >= k_c} <= beta, alpha0 = alpha - beta), which guarantees conditional coverage
      >= 1 - alpha0 with probability >= 1 - beta over the calibration sample; repeated over R calibration samples to estimate the share of
      calibration samples whose conditional coverage falls below 1 - alpha (rank) / 1 - alpha0 (binomial).
Also: law-family Bonferroni (member tail quantiles: Gaussian, Gamma-4, exponential, t(4) scaled) as the like-for-like robust comparator.
Usage: python b7_conditional.py --out <dir> [--r_km 15] [--B 999] [--T 2000] [--R 40] [--N 500]."""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp, pandas as pd, geopandas as gpd
from scipy.stats import binom, norm, gamma as gamma_dist, t as t_dist, beta as beta_dist
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--r_km", type=float, default=15.0); p.add_argument("--B", type=int, default=999); p.add_argument("--T", type=int, default=2000)
p.add_argument("--R", type=int, default=40); p.add_argument("--N", type=int, default=500); p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--beta_c", type=float, default=0.01); p.add_argument("--pfat", type=float, default=0.5); p.add_argument("--seed", type=int, default=83); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; w = ras["weight"][mask].astype(float); kfr = ras["kfr"][mask].astype(float); X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr)
W = np.bincount(inv, weights=w, minlength=K); theta = np.bincount(inv, weights=w * kfr, minlength=K) / np.maximum(W, 1e-12)
cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12)
at = pd.read_csv(paths.ATLAS_TRACT_OUTCOMES, usecols=["state", "county", "tract", "kfr_pooled_pooled_p25_se"])
at["GEOID"] = at["state"].astype("Int64").astype(str).str.zfill(2) + at["county"].astype("Int64").astype(str).str.zfill(3) + at["tract"].astype("Int64").astype(str).str.zfill(6)
SEmap = dict(zip(at["GEOID"], at["kfr_pooled_pooled_p25_se"])); gpk = gpd.read_file(paths.GEORGIA_TRACTS)
se = np.array([SEmap.get(str(gpk.iloc[k - 1]["GEOID"]), np.nan) for k in units], dtype=float); se = np.nan_to_num(se, nan=np.nanmedian(se))
KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2; d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2; near = d2 <= (a.r_km / KM_PER_UNIT) ** 2


def trimmed_weights(vals, wts, p_):
    o_ = np.argsort(vals); cw_ = np.cumsum(wts[o_]); cap = p_ * wts.sum(); take = np.minimum(wts[o_], np.maximum(cap - (cw_ - wts[o_]), 0)); ww = np.zeros_like(wts)
    if take.sum() <= 0:
        return None
    ww[o_] = take / take.sum(); return ww


WL = sp.lil_matrix((K, K)); WH = sp.lil_matrix((K, K)); ok = np.zeros(K, bool)
for x in range(K):
    idx = np.where(near[x])[0]
    if len(idx) < 2 or W[idx].sum() <= 0:
        continue
    lw = trimmed_weights(theta[idx], W[idx], a.pfat); hw = trimmed_weights(-theta[idx], W[idx], a.pfat)
    if lw is None or hw is None:
        continue
    WL[x, idx] = lw; WH[x, idx] = hw; ok[x] = True
WL = WL[np.where(ok)[0]].tocsr(); WH = WH[np.where(ok)[0]].tocsr(); nq = WL.shape[0]
Dfun = lambda eps: np.c_[WL @ eps, -(WH @ eps)]
gauss = lambda n: rng.normal(size=(n, K)) * se
pil = np.stack([Dfun(e) for e in gauss(500)]); a_j = pil.mean(0); b_j = pil.std(0) + 1e-12       # pilot-fixed centring/scaling (independent of everything below)
Mstat = lambda e: ((Dfun(e) - a_j) / b_j).max()
B, al = a.B, a.alpha; k_rank = int(np.ceil((B + 1) * (1 - al))); al0 = al - a.beta_c; k_cond = int(binom.ppf(1 - a.beta_c, B, 1 - al0)) + 1
t0 = time.time(); res = dict(queries=int(nq), B=B, k_rank=k_rank, k_conditional=k_cond, alpha=al, alpha0=al0, beta_c=a.beta_c)
# (J) joint coverage: fresh calibration per trial
cov_rank = 0; cov_cond = 0
for t in range(a.T):
    M = np.sort(np.array([Mstat(e) for e in gauss(B)])); c_r = M[k_rank - 1]; c_c = M[min(k_cond, B) - 1] if k_cond <= B else np.inf
    m_obs = Mstat(gauss(1)[0]); cov_rank += int(m_obs <= c_r); cov_cond += int(m_obs <= c_c)
    if t % 200 == 0:
        print(f"  joint trial {t}: rank {cov_rank/(t+1):.4f} cond {cov_cond/(t+1):.4f} {time.time()-t0:.0f}s", flush=True)
res["joint"] = dict(T=a.T, coverage_rank=cov_rank / a.T, cp_lower_rank=float(beta_dist.ppf(0.025, cov_rank, a.T - cov_rank + 1)) if cov_rank < a.T else None, guarantee_rank=k_rank / (B + 1),
                    coverage_conditional_k=cov_cond / a.T, cp_lower_conditional_k=float(beta_dist.ppf(0.025, cov_cond, a.T - cov_cond + 1)) if cov_cond < a.T else None)
# (C) conditional coverage across R calibration samples
below_rank = 0; below_cond = 0; conds_r = []; conds_c = []
for r in range(a.R):
    M = np.sort(np.array([Mstat(e) for e in gauss(B)])); c_r = M[k_rank - 1]; c_c = M[min(k_cond, B) - 1]
    val = np.array([Mstat(e) for e in gauss(a.N)]); pr = float(np.mean(val <= c_r)); pc = float(np.mean(val <= c_c)); conds_r.append(pr); conds_c.append(pc)
    below_rank += int(pr < 1 - al); below_cond += int(pc < 1 - al0)
res["conditional"] = dict(R=a.R, N=a.N, rank_cond_coverage_median=float(np.median(conds_r)), rank_cond_coverage_min=float(np.min(conds_r)), share_calibrations_below_1_minus_alpha_rank=below_rank / a.R,
                          binomial_cond_coverage_median=float(np.median(conds_c)), binomial_cond_coverage_min=float(np.min(conds_c)), share_calibrations_below_1_minus_alpha0_binomial=below_cond / a.R, guarantee_binomial=f"<= {a.beta_c} of calibration samples below {1 - al0}")
# law-family Bonferroni comparator: per-unit two-sided tail quantiles at alpha/(2K) for each member, propagated through the trimmed weights; take the max member
q = 1 - al / (2 * K)
zq = dict(gauss=norm.ppf(q), gamma4=max(abs((gamma_dist.ppf(q, 4) - 4) / 2), abs((gamma_dist.ppf(1 - q, 4) - 4) / 2)), expo=max(abs(gamma_dist.ppf(q, 1) - 1), abs(gamma_dist.ppf(1 - q, 1) - 1)), t4=t_dist.ppf(q, 4) / np.sqrt(2))
zfam = max(zq.values()); halfL_g = zq["gauss"] * (WL @ se); halfL_f = zfam * (WL @ se)
res["bonferroni"] = dict(z_gauss=float(zq["gauss"]), z_family=float(zfam), half_width_median_gauss=float(np.median(halfL_g)), half_width_median_family=float(np.median(halfL_f)),
                         half_width_median_tailored_rank=float(np.median(a_j[:, 0] + np.sort(np.array([Mstat(e) for e in gauss(B)]))[k_rank - 1] * b_j[:, 0])))
res["seconds"] = time.time() - t0; json.dump(res, open(os.path.join(a.out, "b7_conditional.json"), "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
