"""b3_ranks.py -- data tier (Supplementary Theorems S14, S16; results exp2/B3): rank certificates on Georgia tracts.
Named locations = all tracts with a non-empty class; class = C_{r,p} (population-fat focal cell, p = 1/2, fractional trimmed endpoints =
outer for whole-unit cells; singleton completion elsewhere, Definition S2) at radius r. Pair differences over common zonings are
bounded by the separable bracket l_jk = L_j - U_k, u_jk = U_j - L_k, which is exact for disjoint windows under Theorem S16 and an
outer bound for overlapping windows (the zoning optimization is bounded by the separable sup). Rank sets (Theorem S16):
rank_j in [1 + |N^-_j|, n - |N^+_j|] with N^-_j = {k : l_kj > 0}, N^+_j = {k : l_jk > 0}; population-share version uses W_k.
Noise: Theorem S14 with the independent-pilot exchangeable-rank calibration, Gaussian and the declared four-law family (max c),
applied to the endpoint queries so that all l_jk are simultaneously valid. Reported: share of pairs with certified order; share of
the top-20 % (by estimate) certified in the top 20 % (rank upper bound <= 0.2 of population), and certified NOT in the top 20 %
(rank lower bound > 0.2); median rank-set width (population share); witnesses: the real (county) zoning's ranks and 200 random
balanced contiguous coarsenings (Definition S2 C_{K,bal,cont} over tracts, K = #counties) as demonstrated reversals.
Usage: python b3_ranks.py --out <dir> [--radii 3.8,7.5,15] [--B 999].
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, collections, numpy as np, scipy.sparse as sp, pandas as pd, geopandas as gpd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--radii", default="3.8,7.5,15"); p.add_argument("--B", type=int, default=999); p.add_argument("--pfat", type=float, default=0.5)
p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--seed", type=int, default=73); p.add_argument("--n_part", type=int, default=200); a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; co = ras["county_idx"][mask]; w = ras["weight"][mask].astype(float); kfr = ras["kfr"][mask].astype(float); X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr)
W = np.bincount(inv, weights=w, minlength=K); theta = np.bincount(inv, weights=w * kfr, minlength=K) / np.maximum(W, 1e-12)
cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12); county = np.zeros(K, int); county[inv] = co
at = pd.read_csv(paths.ATLAS_TRACT_OUTCOMES, usecols=["state", "county", "tract", "kfr_pooled_pooled_p25_se"])
at["GEOID"] = at["state"].astype("Int64").astype(str).str.zfill(2) + at["county"].astype("Int64").astype(str).str.zfill(3) + at["tract"].astype("Int64").astype(str).str.zfill(6)
SEmap = dict(zip(at["GEOID"], at["kfr_pooled_pooled_p25_se"])); gpk = gpd.read_file(paths.GEORGIA_TRACTS)
se = np.nan_to_num(np.array([SEmap.get(str(gpk.iloc[k - 1]["GEOID"]), np.nan) for k in units], dtype=float), nan=np.nanmedian([SEmap.get(str(gpk.iloc[k - 1]["GEOID"]), np.nan) for k in units]))
KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2; pop = W / W.sum(); d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
o = np.argsort(-theta); cw = np.cumsum(pop[o]); top20_est = np.zeros(K, bool); top20_est[o[cw <= 0.2]] = True


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


def rank_sets(L, U, ok):
    """population-share rank set from separable pair bounds: above_j = sum of pop of k with L_k > U_j (surely above), below_j = pop of k with U_k < L_j"""
    idx = np.where(ok)[0]; Lo, Uo, po = L[idx], U[idx], pop[idx]
    above = np.array([po[Lo > Uo[i]].sum() for i in range(len(idx))]); below = np.array([po[Uo < Lo[i]].sum() for i in range(len(idx))])
    pairs_cert = 0; tot = len(idx) * (len(idx) - 1) / 2
    srt = np.sort(Lo); pairs_cert = int(sum(np.searchsorted(srt, Uo[i], side="right") < len(idx) and (Lo > Uo[i]).sum() for i in range(len(idx))))   # ordered pairs with l_kj > 0
    lo_rank = above; hi_rank = 1.0 - below        # rank as population share from the top: in [above, 1 - below]
    return idx, lo_rank, hi_rank, pairs_cert / tot


OUT = dict(radii={}); adjacency = None
for r_km in [float(x) for x in a.radii.split(",")]:
    t0 = time.time(); near = d2 <= (r_km / KM_PER_UNIT) ** 2; WL = sp.lil_matrix((K, K)); WH = sp.lil_matrix((K, K)); ok = np.zeros(K, bool)
    for x in range(K):
        idx = np.where(near[x])[0]
        if len(idx) < 2 or W[idx].sum() <= 0:
            continue
        lw = trimmed_weights(theta[idx], W[idx], a.pfat); hw = trimmed_weights(-theta[idx], W[idx], a.pfat)
        if lw is None or hw is None:
            continue
        WL[x, idx] = lw; WH[x, idx] = hw; ok[x] = True
    WL = WL.tocsr(); WH = WH.tocsr(); L0 = WL @ theta; U0 = WH @ theta
    Dfun = lambda eps: np.c_[WL @ eps, -(WH @ eps)]
    pil = np.stack([Dfun(e) for e in draws(300, "gauss")]); a_j = pil.mean(0); b_j = pil.std(0) + 1e-12; k = int(np.ceil((a.B + 1) * (1 - a.alpha))); cs = {}
    for law in ("gauss", "skew", "skew2", "t4"):
        cal = np.stack([Dfun(e) for e in draws(a.B, law)]); M = ((cal - a_j) / b_j).max(axis=(1, 2)); cs[law] = float(np.sort(M)[k - 1])
    res = dict(n_named=int(ok.sum()), named_pop_share=float(pop[ok].sum()), c_gauss=cs["gauss"], c_family=max(cs.values()))
    for name, c in (("noiseless", 0.0), ("gaussian", cs["gauss"]), ("family", max(cs.values()))):
        L = L0 - (a_j[:, 0] + c * b_j[:, 0]) if c > 0 else L0; U = U0 + (a_j[:, 1] + c * b_j[:, 1]) if c > 0 else U0
        idx, lo_r, hi_r, pairs = rank_sets(L, U, ok); width = hi_r - lo_r
        top = top20_est[idx]; cert_top = (hi_r <= 0.2); cert_not_top = (lo_r > 0.2)
        res[name] = dict(pairs_certified_share=float(pairs), rank_width_median=float(np.median(width)), rank_width_q10=float(np.quantile(width, 0.1)),
                         top20_certified_top20_pop_share=float(pop[idx][top & cert_top].sum() / max(pop[idx][top].sum(), 1e-12)), certified_not_top20_pop_share=float(pop[idx][cert_not_top].sum() / pop[idx].sum()),
                         any_certified_top20_pop_share=float(pop[idx][cert_top].sum() / pop[idx].sum()))
    # witnesses: real county zoning and random balanced contiguous coarsenings of tracts (K = #counties): demonstrated rank reversals of top-20 % membership
    if adjacency is None:
        g = gpk.set_index("GEOID") if "GEOID" in gpk.columns else gpk; geoms = [gpk.iloc[k_ - 1].geometry for k_ in units]
        import shapely; adjacency = collections.defaultdict(set); sidx = gpd.GeoSeries(geoms).sindex
        for i_, gm in enumerate(geoms):
            for j_ in sidx.query(gm, predicate="touches"):
                if j_ != i_:
                    adjacency[i_].add(int(j_)); adjacency[int(j_)].add(i_)
    KC = len(np.unique(county)); flips = []
    def coarse_ranks(lab):
        Kz = lab.max() + 1; m = np.bincount(lab, weights=W * theta, minlength=Kz) / np.maximum(np.bincount(lab, weights=W, minlength=Kz), 1e-12); v = m[lab]
        oo = np.argsort(-v); cc = np.cumsum(pop[oo]); t20 = np.zeros(K, bool); t20[oo[cc <= 0.2]] = True; return t20
    t20_county = coarse_ranks(county); res["witness_county_zoning"] = dict(top20_membership_changed_pop_share=float(pop[t20_county != top20_est].sum()))
    for rep in range(a.n_part):
        lab = -np.ones(K, int); seeds = rng.choice(K, KC, replace=False); lab[seeds] = np.arange(KC); mass = W[seeds].copy(); tgt = W.sum() / KC; frontier = {int(s_): set(adjacency[int(s_)]) for s_ in seeds}
        active = set(range(KC)); un = K - KC
        while un > 0 and active:
            r_ = min(active, key=lambda q: mass[q]); cand = [v for v in frontier[int(seeds[r_])] if lab[v] < 0]
            if not cand:
                active.discard(r_); continue
            v = cand[int(rng.integers(len(cand)))]; lab[v] = r_; mass[r_] += W[v]; un -= 1; frontier[int(seeds[r_])] |= set(adjacency[v])
            if mass[r_] > 1.5 * tgt:
                active.discard(r_)
        for v in np.where(lab < 0)[0]:
            nb = [lab[z] for z in adjacency[v] if lab[z] >= 0]; lab[v] = nb[0] if nb else lab.max() + 1
        flips.append(float(pop[coarse_ranks(lab) != top20_est].sum()))
    res["witness_balanced_partitions"] = dict(n=a.n_part, top20_membership_changed_pop_share_median=float(np.median(flips)), q95=float(np.quantile(flips, 0.95)))
    OUT["radii"][str(r_km)] = res
    print(f"r={r_km}: named {res['n_named']} ({res['named_pop_share']:.2f}) | pairs certified: noiseless {res['noiseless']['pairs_certified_share']:.3f} gauss {res['gaussian']['pairs_certified_share']:.3f} family {res['family']['pairs_certified_share']:.3f} | top20 certified top20: {res['noiseless']['top20_certified_top20_pop_share']:.3f} / {res['gaussian']['top20_certified_top20_pop_share']:.3f} / {res['family']['top20_certified_top20_pop_share']:.3f} | certified not-top20: {res['noiseless']['certified_not_top20_pop_share']:.3f} / {res['gaussian']['certified_not_top20_pop_share']:.3f} / {res['family']['certified_not_top20_pop_share']:.3f} | rank width med {res['noiseless']['rank_width_median']:.3f} / {res['family']['rank_width_median']:.3f} | witnesses: county flips {res['witness_county_zoning']['top20_membership_changed_pop_share']:.3f}, balanced flips med {res['witness_balanced_partitions']['top20_membership_changed_pop_share_median']:.3f} {time.time()-t0:.0f}s", flush=True)
    json.dump(OUT, open(os.path.join(a.out, "b3_georgia.json"), "w"), indent=1, default=float)
