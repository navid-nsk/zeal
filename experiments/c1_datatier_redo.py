"""c1_datatier_redo.py -- data-tier certificates (Definition S2 fat class, Supplementary Theorem S14 with the declared law family; results exp2/C1):
model-free data-tier certificates on Georgia tracts (Opportunity Atlas, s.e. known for 93 % of tracts) at radii r km,
classes 'any' (all unions containing x inside the window; endpoints = min/max unit value) and 'fat' (focal cell mass >= p * window
mass, anchored at x; fractional trimmed-mean endpoints = outer bound for whole-unit cells), hot threshold tau = population-weighted
80th percentile of the estimates (numerical cutoff, chosen before noise), denominators (i) all tracts, (ii) class non-empty (>= 2 units
in the window). Noise: Theorem S14 exchangeable-rank calibration with an independent pilot, Gaussian-only and the declared family
{Gaussian, Gamma skew 1, exponential skew 2, t(4)} with the maximum critical value; Bonferroni per unit as the blanket comparison.
Reported per (r, class): shares certified not-hot / hot without noise, with Gaussian calibration, with the family, with Bonferroni;
all as population shares over both denominators. Usage: python c1_datatier_redo.py --out <dir> [--radii 3.8,7.5,15,30] [--B 999]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp, pandas as pd, geopandas as gpd
from scipy.stats import norm
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--radii", default="3.8,7.5,15,30"); p.add_argument("--B", type=int, default=999); p.add_argument("--pfat", type=float, default=0.5)
p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--seed", type=int, default=71); p.add_argument("--whole_units", action="store_true"); p.add_argument("--tag", default=""); a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; w = ras["weight"][mask].astype(float); kfr = ras["kfr"][mask].astype(float); X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr); cnt_all = np.bincount(inv, minlength=K)
W = np.bincount(inv, weights=w, minlength=K); theta = np.bincount(inv, weights=w * kfr, minlength=K) / np.maximum(W, 1e-12)
cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12)
at = pd.read_csv(paths.ATLAS_TRACT_OUTCOMES, usecols=["state", "county", "tract", "kfr_pooled_pooled_p25_se"])
at["GEOID"] = at["state"].astype("Int64").astype(str).str.zfill(2) + at["county"].astype("Int64").astype(str).str.zfill(3) + at["tract"].astype("Int64").astype(str).str.zfill(6)
SEmap = dict(zip(at["GEOID"], at["kfr_pooled_pooled_p25_se"])); gpk = gpd.read_file(paths.GEORGIA_TRACTS)
se = np.array([SEmap.get(str(gpk.iloc[k - 1]["GEOID"]), np.nan) for k in units], dtype=float); se_known = float(np.isfinite(se).mean()); se = np.nan_to_num(se, nan=np.nanmedian(se))
KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2
o = np.argsort(theta); cw = np.cumsum(W[o]) / W.sum(); tau = float(theta[o][np.searchsorted(cw, 0.8)]); hot_truth_est = theta >= tau
pop = W / W.sum()
d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
diam = np.sqrt(np.bincount(inv, minlength=K)) * 0.94       # tract diameter proxy (km) for the 'anchor fits' check


def trimmed_weights(vals, wts, p_):
    o_ = np.argsort(vals); cw_ = np.cumsum(wts[o_]); cap = p_ * wts.sum(); take = np.minimum(wts[o_], np.maximum(cap - (cw_ - wts[o_]), 0)); ww = np.zeros_like(wts)
    if take.sum() <= 0:
        return None
    ww[o_] = take / take.sum(); return ww


def draws(n, law):
    if law == "gauss":
        return rng.normal(size=(n, K)) * se
    if law == "skew":
        g = rng.gamma(4.0, 1.0, size=(n, K)); return (g - 4.0) / 2.0 * se
    if law == "skew2":
        g = rng.gamma(1.0, 1.0, size=(n, K)); return (g - 1.0) * se
    if law == "t4":
        return rng.standard_t(4, size=(n, K)) / np.sqrt(2.0) * se


OUT = dict(tau=tau, se_known_share=se_known, radii={})
for r_km in [float(x) for x in a.radii.split(",")]:
    t0 = time.time(); near = d2 <= (r_km / KM_PER_UNIT) ** 2; muW_pix = None
    if a.whole_units:      # audited class (Definition S2): admissible units = tracts WHOLLY inside the pixel-level window around the anchor's centroid; mu(W) = window population
        r_ = r_km / KM_PER_UNIT; near = np.zeros((K, K), bool); muW_pix = np.zeros(K); contained = np.zeros(K, bool)
        for x in range(K):
            inwin = (X - cx[x]) ** 2 + (Y - cy[x]) ** 2 <= r_ * r_; cnt_in = np.bincount(inv[inwin], weights=None, minlength=K)
            whole = (cnt_in == cnt_all) & (cnt_all > 0); near[x] = whole; muW_pix[x] = w[inwin].sum(); contained[x] = whole[x]
    for cls in ("any", "fat"):
        WL = sp.lil_matrix((K, K)); WH = sp.lil_matrix((K, K)); nonempty = np.zeros(K, bool); feasible = np.zeros(K, bool)
        for x in range(K):
            idx = np.where(near[x])[0]
            if len(idx) < 2 or W[idx].sum() <= 0:
                continue
            if a.whole_units:
                feasible[x] = contained[x] and (W[x] > 0) and (W[idx].sum() >= a.pfat * muW_pix[x])     # audited: anchor wholly contained; eligible (wholly contained) mass reaches p * mu(W)
            else:
                feasible[x] = (diam[x] <= 2 * r_km) and (W[x] > 0)           # provisional diameter shortcut (replaced by --whole_units)
            vals = theta[idx]; wts = W[idx]
            if cls == "any":        # endpoints: min / max unit value in the window (singletons admissible)
                WL[x, idx[np.argmin(vals)]] = 1.0; WH[x, idx[np.argmax(vals)]] = 1.0; nonempty[x] = True
            else:
                pw = a.pfat * muW_pix[x] / max(W[idx].sum(), 1e-12) if a.whole_units else a.pfat     # fat threshold relative to the window POPULATION, expressed as a share of the eligible mass
                if pw > 1.0:
                    continue
                lo_w = trimmed_weights(vals, wts, pw); hi_w = trimmed_weights(-vals, wts, pw)
                if lo_w is None or hi_w is None:
                    continue
                WL[x, idx] = lo_w; WH[x, idx] = hi_w; nonempty[x] = True
        WL = WL.tocsr(); WH = WH.tocsr(); L_hat = WL @ theta; U_hat = WH @ theta
        res = dict(nonempty_pop_share=float(pop[nonempty].sum()), class_feasible_pop_share=float(pop[nonempty & feasible].sum()))
        res["noiseless"] = dict(not_hot=float(pop[nonempty & (U_hat < tau)].sum()), hot=float(pop[nonempty & (L_hat > tau)].sum()))
        if cls == "any":     # for the 'any' class the noise statistic is the max unit noise in the window: D_x = max_{u in W} eps_u (sup over all unions containing x)
            Wmax = sp.csr_matrix(near.astype(float))
            Dfun = lambda eps: np.c_[np.asarray([eps[near[x]].max() if nonempty[x] else 0.0 for x in range(K)]), np.asarray([(-eps[near[x]]).max() if nonempty[x] else 0.0 for x in range(K)])]
        else:
            Dfun = lambda eps: np.c_[WL @ eps, -(WH @ eps)]
        pil = np.stack([Dfun(e) for e in draws(300, "gauss")]); a_j = pil.mean(0); b_j = pil.std(0) + 1e-12
        k = int(np.ceil((a.B + 1) * (1 - a.alpha))); cs = {}
        for law in ("gauss", "skew", "skew2", "t4"):
            cal = np.stack([Dfun(e) for e in draws(a.B, law)]); M = ((cal - a_j) / b_j).max(axis=(1, 2)); cs[law] = float(np.sort(M)[k - 1])
        c_g = cs["gauss"]; c_f = max(cs.values())
        for name, c in (("gaussian", c_g), ("family", c_f)):
            nh = nonempty & (U_hat + a_j[:, 1] + c * b_j[:, 1] < tau); ht = nonempty & (L_hat - a_j[:, 0] - c * b_j[:, 0] > tau)
            res[name] = dict(c=c, not_hot=float(pop[nh].sum()), hot=float(pop[ht].sum()))
        z_b = norm.ppf(1 - a.alpha / (2 * K))
        if cls == "any":
            halfL = np.asarray([z_b * se[near[x]].max() if nonempty[x] else 0.0 for x in range(K)]); halfU = halfL
        else:
            halfL = z_b * (WL @ se); halfU = z_b * (WH @ se)
        res["bonferroni"] = dict(not_hot=float(pop[nonempty & (U_hat + halfU < tau)].sum()), hot=float(pop[nonempty & (L_hat - halfL > tau)].sum()))
        for key in ("noiseless", "gaussian", "family", "bonferroni"):
            res[key]["not_hot_over_nonempty"] = res[key]["not_hot"] / max(res["nonempty_pop_share"], 1e-12); res[key]["hot_over_nonempty"] = res[key]["hot"] / max(res["nonempty_pop_share"], 1e-12)
            res[key]["not_hot_over_feasible"] = res[key]["not_hot"] / max(res["class_feasible_pop_share"], 1e-12)
        OUT["radii"].setdefault(str(r_km), {})[cls] = res
        print(f"r={r_km} km {cls}: non-empty {res['nonempty_pop_share']:.2f} feasible {res['class_feasible_pop_share']:.2f} | not-hot: noiseless {res['noiseless']['not_hot']:.3f} gauss {res['gaussian']['not_hot']:.3f} family {res['family']['not_hot']:.3f} bonf {res['bonferroni']['not_hot']:.3f} | hot: {res['noiseless']['hot']:.3f} {res['gaussian']['hot']:.3f} {res['family']['hot']:.3f} {res['bonferroni']['hot']:.3f} | c {c_g:.2f}/{c_f:.2f} {time.time()-t0:.0f}s", flush=True)
    json.dump(OUT, open(os.path.join(a.out, f"c1_georgia{a.tag}.json"), "w"), indent=1, default=float)
