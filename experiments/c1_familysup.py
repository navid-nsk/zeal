"""c1_familysup.py -- data-tier certificates with the FAMILY-SUP noise statistic (Supplementary Theorem S14, Lemma S14.1;
V4/V5); results exp2/C1.

Why the family sup: the single trimmed weight vector w_H(y) of the noiseless maximizer as the noise statistic, D_x = -w_H(y).eps (the
statistic of c1_datatier_redo.py; reported below as a comparator), certifies the true mean of ONE
data-selected cell, not  U_x(theta) = sup_{a admissible} <theta>_a.  Theorem S14 needs
D_x^-(eps) = sup_{w in W_x} (-w.eps) over the whole admissible weight family.  For the fractional fat family
W_x = {w >= 0 : sum w = 1, w_u <= W_u / cap}, cap = p * mu(W_x) (the relaxation that also gives the outer endpoints), the sup is the
upper trimmed mean of -eps at mass share cap: fill the largest values of -eps up to capacity W_u each until the taken mass is cap.
The 'any' class (all unions containing x) already used the correct sup (max unit noise in the window).
Everything else (audited whole-unit windows, population-based fat threshold, exchangeable-rank calibration with an independent
Gaussian pilot, the four-law family with the maximum critical value, hot cutoff tau) is as in c1_datatier_redo.py --whole_units.
Status of the numbers: certified decisions and population shares, conditional on the declared law family and the stated joint
guarantee k/(B+1); Monte Carlo critical values (per-draw statistics are exact closed forms, so V4's per-draw upper bound is the
statistic itself up to floating-point rounding).
Usage: python c1_familysup.py --out <dir> [--radii 3.8,7.5,15,30] [--B 999]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp, pandas as pd, geopandas as gpd
from scipy.stats import norm
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--radii", default="3.8,7.5,15,30"); p.add_argument("--B", type=int, default=999); p.add_argument("--pfat", type=float, default=0.5)
p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--seed", type=int, default=71); p.add_argument("--tag", default="_familysup4"); a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
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
o = np.argsort(theta); cw = np.cumsum(W[o]) / W.sum(); tau = float(theta[o][np.searchsorted(cw, 0.8)])
pop = W / W.sum()


def trimmed_weights(vals, wts, p_):
    o_ = np.argsort(vals); cw_ = np.cumsum(wts[o_]); cap = p_ * wts.sum(); take = np.minimum(wts[o_], np.maximum(cap - (cw_ - wts[o_]), 0)); ww = np.zeros_like(wts)
    if take.sum() <= 0:
        return None
    ww[o_] = take / take.sum(); return ww


EPS_FP = np.finfo(float).eps


def family_sup(E, wts, cap):
    """sup_{w in Delta} w.eps for every row eps of E (n_draws x n_units): upper trimmed mean at mass cap (fractional fat family).
    V4 outward arithmetic: the returned value is an UPPER bound on the exact support function -- the forward rounding error of the
    cumulative masses, the take weights, the products and the final sum is bounded by (3n + 6) eps (sum |take eps| + cap |eps|_max) / cap."""
    o_ = np.argsort(-E, axis=1); Es = np.take_along_axis(E, o_, 1); Ws = wts[o_]; cw_ = np.cumsum(Ws, 1)
    take = np.minimum(Ws, np.maximum(cap - (cw_ - Ws), 0.0)); val = (take * Es).sum(1) / cap
    k_ = 3 * E.shape[1] + 6; err = (k_ * EPS_FP / (1 - k_ * EPS_FP)) * ((take * np.abs(Es)).sum(1) + cap * np.abs(Es).max(1)) / cap   # gamma_k = k u/(1 - k u) times the absolute magnitudes (V4)
    return val + err


def draws(n, law):
    if law == "gauss":
        return rng.normal(size=(n, K)) * se
    if law == "skew":
        g = rng.gamma(4.0, 1.0, size=(n, K)); return (g - 4.0) / 2.0 * se
    if law == "skew2":
        g = rng.gamma(1.0, 1.0, size=(n, K)); return (g - 1.0) * se
    if law == "t4":
        return rng.standard_t(4, size=(n, K)) / np.sqrt(2.0) * se


OUT = dict(tau=tau, se_known_share=se_known, statistic="family sup (upper trimmed mean of the noise over the fractional fat family; 'any': max unit noise)", radii={})
for r_km in [float(x) for x in a.radii.split(",")]:
    t0 = time.time(); r_ = r_km / KM_PER_UNIT; near = np.zeros((K, K), bool); muW_pix = np.zeros(K); contained = np.zeros(K, bool)
    for x in range(K):
        inwin = (X - cx[x]) ** 2 + (Y - cy[x]) ** 2 <= r_ * r_; cnt_in = np.bincount(inv[inwin], minlength=K)
        whole = (cnt_in == cnt_all) & (cnt_all > 0); near[x] = whole; muW_pix[x] = w[inwin].sum(); contained[x] = whole[x]
    for cls in ("any", "fat"):
        WL = sp.lil_matrix((K, K)); WH = sp.lil_matrix((K, K)); nonempty = np.zeros(K, bool); feasible = np.zeros(K, bool); caps = np.zeros(K); idxs = {}; distinct2 = np.zeros(K, bool)
        for x in range(K):
            idx = np.where(near[x])[0]
            if len(idx) < 2 or W[idx].sum() <= 0:
                continue
            feasible[x] = contained[x] and (W[x] > 0) and (W[idx].sum() >= a.pfat * muW_pix[x]); vals = theta[idx]; wts = W[idx]
            others = idx[idx != x]; distinct2[x] = bool(feasible[x] and len(others) > 0 and ((W[idx].sum() - W[others]) >= a.pfat * muW_pix[x]).any())   # >= 2 distinct admissible focal cells (exact criterion)
            if cls == "any":
                WL[x, idx[np.argmin(vals)]] = 1.0; WH[x, idx[np.argmax(vals)]] = 1.0; nonempty[x] = True; idxs[x] = idx
            else:
                pw = a.pfat * muW_pix[x] / max(W[idx].sum(), 1e-12)
                if pw > 1.0:
                    continue
                lo_w = trimmed_weights(vals, wts, pw); hi_w = trimmed_weights(-vals, wts, pw)
                if lo_w is None or hi_w is None:
                    continue
                WL[x, idx] = lo_w; WH[x, idx] = hi_w; nonempty[x] = True; caps[x] = pw * wts.sum(); idxs[x] = idx
        WL = WL.tocsr(); WH = WH.tocsr(); L_hat = WL @ theta; U_hat = WH @ theta; act = np.where(nonempty)[0]
        res = dict(nonempty_pop_share=float(pop[nonempty].sum()), class_feasible_pop_share=float(pop[nonempty & feasible].sum()), two_distinct_cells_pop_share=float(pop[nonempty & distinct2].sum()))
        res["noiseless"] = dict(not_hot=float(pop[nonempty & (U_hat < tau)].sum()), hot=float(pop[nonempty & (L_hat > tau)].sum()), not_hot_two_distinct=float(pop[nonempty & distinct2 & (U_hat < tau)].sum()),
                                not_hot_feasible=float(pop[nonempty & feasible & (U_hat < tau)].sum()), hot_feasible=float(pop[nonempty & feasible & (L_hat > tau)].sum()))
        def Dfun(E):          # columns: D^+_x = sup_w w.eps (for the lower endpoint), D^-_x = sup_w (-w.eps) (for the upper endpoint); rows = draws
            Dp = np.zeros((E.shape[0], K)); Dm = np.zeros((E.shape[0], K))
            for x in act:
                idx = idxs[x]; Ex = E[:, idx]
                if cls == "any":
                    Dp[:, x] = Ex.max(1); Dm[:, x] = (-Ex).max(1)
                else:
                    Dp[:, x] = family_sup(Ex, W[idx], caps[x]); Dm[:, x] = family_sup(-Ex, W[idx], caps[x])
            return np.stack([Dp, Dm], -1)
        pil = Dfun(draws(300, "gauss")); a_j = pil.mean(0); b_j = pil.std(0) + 1e-12
        k = int(np.ceil((a.B + 1) * (1 - a.alpha))); cs = {}
        for law in ("gauss", "skew", "skew2", "t4"):
            cal = Dfun(draws(a.B, law)); M = ((cal - a_j) / b_j)[:, act, :].max(axis=(1, 2)); cs[law] = float(np.sort(M)[k - 1])
        c_g = cs["gauss"]; c_f = max(cs.values())
        for name, c in (("gaussian", c_g), ("family", c_f)):
            nh = nonempty & (U_hat + a_j[:, 1] + c * b_j[:, 1] < tau); ht = nonempty & (L_hat - a_j[:, 0] - c * b_j[:, 0] > tau)
            res[name] = dict(c=c, not_hot=float(pop[nh].sum()), hot=float(pop[ht].sum()), not_hot_feasible=float(pop[nh & feasible].sum()), hot_feasible=float(pop[ht & feasible].sum()), not_hot_two_distinct=float(pop[nh & distinct2].sum()), hot_two_distinct=float(pop[ht & distinct2].sum()),
                             half_width_median=float(np.median((a_j[act, 1] + c * b_j[act, 1]))))
        # single-weight comparator (not valid for the class), same draws budget, for the record
        if cls == "fat":
            D1 = lambda E: np.stack([np.asarray(WL @ E.T).T, -np.asarray(WH @ E.T).T], -1)
            pil1 = D1(draws(300, "gauss")); a1 = pil1.mean(0); b1 = pil1.std(0) + 1e-12; cs1 = {}
            for law in ("gauss", "skew", "skew2", "t4"):
                cal1 = D1(draws(a.B, law)); M1 = ((cal1 - a1) / b1)[:, act, :].max(axis=(1, 2)); cs1[law] = float(np.sort(M1)[k - 1])
            c1f = max(cs1.values()); nh1 = nonempty & (U_hat + a1[:, 1] + c1f * b1[:, 1] < tau)
            res["single_weight_family_comparator"] = dict(c=c1f, not_hot=float(pop[nh1].sum()), half_width_median=float(np.median(a1[act, 1] + c1f * b1[act, 1])),
                                                          note="single data-chosen weight vector: certifies one selected cell, not the class; record only")
        z_b = norm.ppf(1 - a.alpha / (2 * K))
        if cls == "any":
            halfU = np.asarray([z_b * se[near[x]].max() if nonempty[x] else 0.0 for x in range(K)])
        else:                 # Bonferroni per unit propagated through the family sup: sup_w w.(z s) = upper trimmed mean of z*se
            halfU = np.zeros(K)
            for x in act:
                halfU[x] = family_sup((z_b * se[idxs[x]])[None, :], W[idxs[x]], caps[x])[0]
        nh = nonempty & (U_hat + halfU < tau); res["bonferroni"] = dict(not_hot=float(pop[nh].sum()), not_hot_feasible=float(pop[nh & feasible].sum()))
        res["c"] = cs; OUT["radii"].setdefault(str(r_km), {})[cls] = res
        print(f"r={r_km} {cls}: feasible {res['class_feasible_pop_share']:.3f} | not-hot noiseless {res['noiseless']['not_hot']:.3f} gauss {res['gaussian']['not_hot']:.3f} "
              f"family {res['family']['not_hot']:.3f} bonf {res['bonferroni']['not_hot']:.3f} | hot family {res['family']['hot']:.4f} gauss {res['gaussian']['hot']:.4f} | "
              f"c gauss/family {c_g:.2f}/{c_f:.2f} half-width family {res['family']['half_width_median']:.4f}"
              + (f" | single-weight comparator not-hot {res['single_weight_family_comparator']['not_hot']:.3f} hw {res['single_weight_family_comparator']['half_width_median']:.4f}" if cls == "fat" else "")
              + f" {time.time()-t0:.0f}s", flush=True)
    json.dump(OUT, open(os.path.join(a.out, f"c1_georgia{a.tag}.json"), "w"), indent=1, default=float)
