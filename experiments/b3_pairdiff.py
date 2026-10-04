"""b3_pairdiff.py -- direct pair-difference inference (Supplementary Theorem S16) with calibration (Theorem S14); results
exp2/B3: Georgia tracts, fat focal class C_{r,p} (p = 1/2, anchored at the named location, fractional trimmed endpoints
= outer family V^_jk ⊇ V_jk).

(1) FEASIBILITY DENOMINATORS per location: the anchor unit fits (unit inside the window), the admissible mass reaches
    p·mu(W) (a whole-unit contiguous... here: any union -- mass condition), the focal cell extends to a full zoning (singleton completion:
    always, Definition S2), and more than one focal choice exists (non-trivial).  Shares reported over (i) all, (ii) class-feasible.
(2) DIRECT PAIR-DIFFERENCE STATISTIC: for each pair (j,k) the difference query v = w_L(j) - w_U(k) (lower endpoint of j minus upper endpoint
    of k, common-zoning outer family); l_jk(y) = v.y; noise D_jk(eps) = v.eps (fixed weight -> variance v'Σv; the sup over the family is
    bounded by the two endpoint envelopes but we calibrate the DIFFERENCE statistic directly); M = max_{jk} (D_jk - a_jk)/b_jk calibrated
    by the exchangeable-rank rule (independent pilot), Gaussian and the four-law family; all pairs enter the calibration (no selection).
    Rank sets and certified-order shares as in b3_ranks.py; comparison with the separable-envelope treatment of b3_ranks.py.
(3) WITNESS CLASS MEMBERSHIP: for the county zoning, at each named location check whether its county cell lies inside the
    window and carries >= p·mu(W) -- i.e. belongs to C_{r,p} at that location; rank-membership flips are then split into in-class and
    separate-class references.
Usage: python b3_pairdiff.py --out <dir> [--radii 7.5,15] [--B 999] [--max_named 600].
Pairs: all pairs among a random subset of named locations (max_named) to keep the pair family O(10^5); the subset is prespecified (seed).
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp, pandas as pd, geopandas as gpd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--radii", default="7.5,15"); p.add_argument("--B", type=int, default=999); p.add_argument("--pfat", type=float, default=0.5)
p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--seed", type=int, default=79); p.add_argument("--max_named", type=int, default=600); p.add_argument("--whole_units", action="store_true"); p.add_argument("--tag", default=""); a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; co = ras["county_idx"][mask]; w = ras["weight"][mask].astype(float); kfr = ras["kfr"][mask].astype(float); X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr); cnt_all = np.bincount(inv, minlength=K)
W = np.bincount(inv, weights=w, minlength=K); theta = np.bincount(inv, weights=w * kfr, minlength=K) / np.maximum(W, 1e-12)
cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12); county = np.zeros(K, int); county[inv] = co
at = pd.read_csv(paths.ATLAS_TRACT_OUTCOMES, usecols=["state", "county", "tract", "kfr_pooled_pooled_p25_se"])
at["GEOID"] = at["state"].astype("Int64").astype(str).str.zfill(2) + at["county"].astype("Int64").astype(str).str.zfill(3) + at["tract"].astype("Int64").astype(str).str.zfill(6)
SEmap = dict(zip(at["GEOID"], at["kfr_pooled_pooled_p25_se"])); gpk = gpd.read_file(paths.GEORGIA_TRACTS)
se_raw = np.array([SEmap.get(str(gpk.iloc[k - 1]["GEOID"]), np.nan) for k in units], dtype=float); se = np.nan_to_num(se_raw, nan=np.nanmedian(se_raw))
KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2; pop = W / W.sum(); d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
o = np.argsort(-theta); cw = np.cumsum(pop[o]); top20_est = np.zeros(K, bool); top20_est[o[cw <= 0.2]] = True
# tract diameters (for 'anchor fits'): sqrt(area) from the raster pixel count
diam = np.sqrt(np.bincount(inv, minlength=K)) * 0.94


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
    return rng.standard_t(4, size=(n, K)) / np.sqrt(2.0) * se


OUT = dict(radii={}, ranking_universe='the prespecified named subset (feasible locations, seed-sampled up to max_named); rank sets describe ranks among those locations; witnesses labelled by class membership at the named locations')
for r_km in [float(x) for x in a.radii.split(",")]:
    t0 = time.time(); near = d2 <= (r_km / KM_PER_UNIT) ** 2; muW_pix = np.zeros(K); contained = np.ones(K, bool)
    if a.whole_units:      # audited class (Definition S2): admissible units wholly inside the Euclidean disc around the anchor's population centroid; mu(W) = disc population
        r_ = r_km / KM_PER_UNIT; near = np.zeros((K, K), bool)
        for x in range(K):
            inwin = (X - cx[x]) ** 2 + (Y - cy[x]) ** 2 <= r_ * r_; cnt_in = np.bincount(inv[inwin], minlength=K); whole = (cnt_in == cnt_all) & (cnt_all > 0)
            near[x] = whole; muW_pix[x] = w[inwin].sum(); contained[x] = whole[x]
    WL = sp.lil_matrix((K, K)); WH = sp.lil_matrix((K, K)); feas = dict(all=np.ones(K, bool), anchor_fits=np.zeros(K, bool), mass_ok=np.zeros(K, bool), nontrivial=np.zeros(K, bool), class_feasible=np.zeros(K, bool))
    for x in range(K):
        idx = np.where(near[x])[0]
        feas["anchor_fits"][x] = contained[x] if a.whole_units else (diam[x] <= 2 * r_km); feas["nontrivial"][x] = len(idx) >= 2
        if len(idx) < 2 or W[idx].sum() <= 0:
            continue
        feas["mass_ok"][x] = (W[idx].sum() >= a.pfat * muW_pix[x]) if a.whole_units else (W[x] > 0)
        pw = a.pfat * muW_pix[x] / max(W[idx].sum(), 1e-12) if a.whole_units else a.pfat
        if pw > 1.0:
            continue
        lw = trimmed_weights(theta[idx], W[idx], pw); hw = trimmed_weights(-theta[idx], W[idx], pw)
        if lw is None or hw is None:
            continue
        WL[x, idx] = lw; WH[x, idx] = hw; feas["class_feasible"][x] = feas["anchor_fits"][x] and feas["mass_ok"][x] and feas["nontrivial"][x]
    WL = WL.tocsr(); WH = WH.tocsr(); ok = feas["class_feasible"]
    den = {k: float(pop[v].sum()) for k, v in feas.items()}
    # named subset (prespecified)
    named = np.where(ok)[0]
    if len(named) > a.max_named:
        named = np.sort(rng.choice(named, a.max_named, replace=False))
    n = len(named); L0 = (WL @ theta)[named]; U0 = (WH @ theta)[named]
    # pair difference vectors: v_jk = w_L(j) - w_U(k)  (lower endpoint of j minus upper endpoint of k); l_jk(y) = L_j - U_k
    WLn = WL[named]; WHn = WH[named]
    # (A) separable treatment (as b3_ranks): envelopes on endpoints
    Dsep = lambda eps: np.c_[WLn @ eps, -(WHn @ eps)]
    # (B) direct pair-difference statistic: D_jk(eps) = (w_L(j) - w_U(k)).eps  for all ordered pairs j != k  -> matrix (n x n) = (WLn eps)[:,None] - (WHn eps)[None,:]
    def Dpair(eps):
        a_ = WLn @ eps; b_ = WHn @ eps; M = a_[:, None] - b_[None, :]; np.fill_diagonal(M, -np.inf); return M
    pilS = np.stack([Dsep(e) for e in draws(300, "gauss")]); aS = pilS.mean(0); bS = pilS.std(0) + 1e-12
    pilP = np.stack([Dpair(e) for e in draws(300, "gauss")]); pilP[~np.isfinite(pilP)] = 0.0; aP = pilP.mean(0); bP = pilP.std(0) + 1e-12
    k_ = int(np.ceil((a.B + 1) * (1 - a.alpha))); cS = {}; cP = {}
    for law in ("gauss", "skew", "skew2", "t4"):
        E = draws(a.B, law); MS = np.array([((Dsep(e) - aS) / bS).max() for e in E]); cS[law] = float(np.sort(MS)[k_ - 1])
        MP = np.array([np.nanmax(np.where(np.isfinite(Dpair(e)), (Dpair(e) - aP) / bP, -np.inf)) for e in E]); cP[law] = float(np.sort(MP)[k_ - 1])
    res = dict(n_named=int(n), denominators_pop_share=den, c_sep=cS, c_pair=cP, se_known_share=float(np.isfinite(se_raw).mean()))
    def rank_stats(Lmat):
        """Lmat[j,k] = certified lower bound on theta_j - theta_k over common zonings; order certified if > 0"""
        cert = Lmat > 0; np.fill_diagonal(cert, False); pairs = cert.sum() / (n * (n - 1))
        popn = pop[named]; above = np.array([popn[cert[:, j]].sum() for j in range(n)]); below = np.array([popn[cert[j, :]].sum() for j in range(n)])
        lo_r = above; hi_r = 1.0 - below; width = hi_r - lo_r; top = top20_est[named]
        return dict(ordered_pairs_share=float(pairs), rank_width_median=float(np.median(width)), top20_certified_top20_pop_share=float(popn[top & (hi_r <= 0.2)].sum() / max(popn[top].sum(), 1e-12)),
                    certified_not_top20_pop_share=float(popn[lo_r > 0.2].sum() / popn.sum()))
    for name, law in (("noiseless", None), ("gaussian", "gauss"), ("family", "max")):
        if law is None:
            Lsep = L0[:, None] - U0[None, :]; Lpair = Lsep.copy()
        else:
            cs = cS[law] if law != "max" else max(cS.values()); cp = cP[law] if law != "max" else max(cP.values())
            Lsep = (L0 - aS[:, 0] - cs * bS[:, 0])[:, None] - (U0 + aS[:, 1] + cs * bS[:, 1])[None, :]
            Lpair = (L0[:, None] - U0[None, :]) - (aP + cp * bP)
        res[name] = dict(separable=rank_stats(Lsep), direct_pair=rank_stats(Lpair))
    # (C) witness class membership: county zoning at each named location
    inclass = np.zeros(n, bool)
    for i, x in enumerate(named):
        cell = county == county[x]; idx = np.where(near[x])[0]
        inclass[i] = cell[idx].sum() == cell.sum() and W[cell].sum() >= a.pfat * W[idx].sum()      # county cell inside the window and fat enough
    Kc = county.max() + 1; m = np.bincount(county, weights=W * theta, minlength=Kc) / np.maximum(np.bincount(county, weights=W, minlength=Kc), 1e-12); v = m[county]
    oo = np.argsort(-v); cc = np.cumsum(pop[oo]); t20 = np.zeros(K, bool); t20[oo[cc <= 0.2]] = True; flip = (t20 != top20_est)[named]
    res["witness_county"] = dict(in_class_share_of_named=float(inclass.mean()), flips_in_class_pop_share=float(pop[named][flip & inclass].sum() / pop[named].sum()), flips_separate_class_pop_share=float(pop[named][flip & ~inclass].sum() / pop[named].sum()))
    OUT["radii"][str(r_km)] = res
    print(f"r={r_km}: named {n} | denominators (pop share) {dict((k_, round(v_, 3)) for k_, v_ in den.items())} | c sep gauss/family {cS['gauss']:.2f}/{max(cS.values()):.2f} pair {cP['gauss']:.2f}/{max(cP.values()):.2f} | ordered pairs: noiseless {res['noiseless']['separable']['ordered_pairs_share']:.3f} | gauss sep {res['gaussian']['separable']['ordered_pairs_share']:.3f} pair {res['gaussian']['direct_pair']['ordered_pairs_share']:.3f} | family sep {res['family']['separable']['ordered_pairs_share']:.3f} pair {res['family']['direct_pair']['ordered_pairs_share']:.3f} | not-top20 family sep/pair {res['family']['separable']['certified_not_top20_pop_share']:.3f}/{res['family']['direct_pair']['certified_not_top20_pop_share']:.3f} | county witness in-class {res['witness_county']['in_class_share_of_named']:.2f} flips in/sep {res['witness_county']['flips_in_class_pop_share']:.3f}/{res['witness_county']['flips_separate_class_pop_share']:.3f} {time.time()-t0:.0f}s", flush=True)
    json.dump(OUT, open(os.path.join(a.out, f"b3_pairdiff_georgia{a.tag}.json"), "w"), indent=1, default=float)
