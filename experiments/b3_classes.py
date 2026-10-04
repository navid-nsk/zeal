"""b3_classes.py -- common-class audit (Supplementary Theorem S16, V5; results exp2/B3): which zoning class do the rank
statements refer to, is it non-empty, and what do the certificates say under each construction — with the FAMILY-SUP noise
statistic (Lemma S14.1) and exact whole-unit endpoints (anchor forced).

Georgia tracts (Opportunity Atlas kfr p25; s.e. known for 93 %), audited whole-unit windows (Definition S2: admissible units wholly inside the
Euclidean disc of radius r around the anchor's population centroid; fat threshold p * mu(W), mu(W) = disc population), p = 1/2,
600 prespecified named locations (seed-sampled among the class-feasible ones), ranking universe = the named set, ties unresolved
(tie-compatible rank interval).  Constructions:
  (C) pair-specific classes C_jk (fat at j and at k): l_jk = L_j - U_k is an outer bound on inf_{C_jk}(theta_j - theta_k); C_jk is
      non-empty iff admissible cells a at j, b at k exist with a = b or a, b disjoint (checked exactly: common cell test; disjoint-cell
      test by a fractional necessary condition, a greedy sufficient split, and a small MILP for the undecided pairs).
      Pair-order shares are reported over all named ordered pairs AND over the joint-class-feasible pairs.
  (A) the all-named class C_N (fat at every named location simultaneously): empty whenever some C_jk is empty (proof); otherwise a
      greedy witness partition is sought; the largest p on a grid at which a C_N witness is found is reported.
  (B) focal-specific classes C_j (fat at j only; competitors under the declared background rule 'focal window class without fatness':
      the cell containing k is any union of wholly contained units of W_k(r) that contains k's unit; endpoints and noise sups by the
      exact prefix rule).  Rank sets of j are valid over C_j and calibrated jointly across j.
Noise: y = theta + eps under the declared four-law family; D^+_x = sup over the (fractional) fat family of <eps>_a, D^-_x likewise for
-eps (upper trimmed means at mass cap_x) -- the family sup required by Theorem S14, not the single data-chosen weight vector;
background sups = max over anchored prefixes.  Exchangeable-rank calibration (B = 999 per law, independent Gaussian pilot centring,
max critical value over the family); statistics: separable (max over endpoint statistics) and direct pair (D^+_j + D^-_k).
Endpoints: exact whole-unit fat endpoints with the anchor forced (Dinkelbach + HiGHS MILP per location) and the fractional relaxation
(outer) -- both reported; certificates use the exact endpoints with the fractional-family noise sup (valid:
the fractional family contains the whole-unit family).
Usage: python b3_classes.py --out <dir> [--radii 7.5,15] [--B 999] [--max_named 600]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, pandas as pd, geopandas as gpd
from scipy.optimize import milp, LinearConstraint, Bounds
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--radii", default="7.5,15"); p.add_argument("--B", type=int, default=999); p.add_argument("--pfat", type=float, default=0.5)
p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--seed", type=int, default=79); p.add_argument("--max_named", type=int, default=600); p.add_argument("--tag", default="_v5")
p.add_argument("--pgrid", default="0.5,0.4,0.3,0.25,0.2,0.15,0.1"); a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; co = ras["county_idx"][mask]; w = ras["weight"][mask].astype(float); kfr = ras["kfr"][mask].astype(float); X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr); cnt_all = np.bincount(inv, minlength=K)
W = np.bincount(inv, weights=w, minlength=K); theta = np.bincount(inv, weights=w * kfr, minlength=K) / np.maximum(W, 1e-12)
cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12); county = np.zeros(K, int); county[inv] = co
at = pd.read_csv(paths.ATLAS_TRACT_OUTCOMES, usecols=["state", "county", "tract", "kfr_pooled_pooled_p25_se"])
at["GEOID"] = at["state"].astype("Int64").astype(str).str.zfill(2) + at["county"].astype("Int64").astype(str).str.zfill(3) + at["tract"].astype("Int64").astype(str).str.zfill(6)
SEmap = dict(zip(at["GEOID"], at["kfr_pooled_pooled_p25_se"])); gpk = gpd.read_file(paths.GEORGIA_TRACTS)
se_raw = np.array([SEmap.get(str(gpk.iloc[k - 1]["GEOID"]), np.nan) for k in units], dtype=float); se = np.nan_to_num(se_raw, nan=np.nanmedian(se_raw))
KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2; pop = W / W.sum()
o = np.argsort(-theta); cw = np.cumsum(pop[o]); top20_est = np.zeros(K, bool); top20_est[o[cw <= 0.2]] = True
TOL = 1e-9


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


def prefix_sup(E, wts, ia):
    """max over unions S (anchor ia forced in) of the weighted mean of eps over S, for every row of E: anchored prefix rule (exact)"""
    n_ = E.shape[1]; others = np.array([i for i in range(n_) if i != ia]); Eo = E[:, others]; Wo = wts[others]
    o_ = np.argsort(-Eo, axis=1); Es = np.take_along_axis(Eo, o_, 1); Ws = Wo[o_]
    num = np.cumsum(Ws * Es, 1) + wts[ia] * E[:, ia][:, None]; den = np.cumsum(Ws, 1) + wts[ia]
    means = np.concatenate([E[:, ia][:, None], num / den], 1); val = means.max(1)
    k_ = 3 * n_ + 6; return val + (k_ * EPS_FP / (1 - k_ * EPS_FP)) * np.abs(E).max(1)          # V4 outward inflation, gamma_k form (prefix means are convex combinations: |mean| <= max |eps|)


def exact_endpoint(vals, wts, cap, ia, sign):
    """exact min (sign=+1) / max (sign=-1) of the weighted mean over whole-unit sets S containing ia with mass >= cap (Dinkelbach + MILP)"""
    v = sign * vals; n_ = len(vals); lam = float(v[ia]); best = None
    lb = np.zeros(n_); ub = np.ones(n_); lb[ia] = 1.0
    for _ in range(30):
        res = milp(wts * (v - lam), integrality=np.ones(n_), bounds=Bounds(lb, ub), constraints=[LinearConstraint(wts[None, :], cap, np.inf)], options=dict(disp=False))
        if res.x is None:
            return None, None
        S = res.x > 0.5; val = float((wts[S] * v[S]).sum() / wts[S].sum())
        if best is None or val < best[0] - 1e-13:
            best = (val, np.where(S)[0])
        if res.fun >= -1e-12:
            break
        lam = val
    return sign * best[0], best[1]


def draws(n, law):
    if law == "gauss":
        return rng.normal(size=(n, K)) * se
    if law == "skew":
        g = rng.gamma(4.0, 1.0, size=(n, K)); return (g - 4.0) / 2.0 * se
    if law == "skew2":
        g = rng.gamma(1.0, 1.0, size=(n, K)); return (g - 1.0) * se
    return rng.standard_t(4, size=(n, K)) / np.sqrt(2.0) * se


def split_feasible(Ej, Ek, j, k, capj, capk):
    """do disjoint admissible cells a (in Ej, containing j, mass >= capj) and b (in Ek, containing k, mass >= capk) exist?
    Returns (feasible, proof): proof in {'anchor' (anchor outside the other's eligible set: exact), 'trivial' (exact), 'fractional' (exact
    arithmetic necessary condition with outward tolerance), 'milp' (HiGHS status, CS), 'brute' (exact subset enumeration, |both| <= 22)}"""
    Sj, Sk = set(Ej.tolist()), set(Ek.tolist())
    if j not in Sj or k not in Sk:
        return False, "anchor"
    both = sorted((Sj & Sk) - {j, k}); onlyj = W[list((Sj - Sk) - {j, k})].sum(); onlyk = W[list((Sk - Sj) - {j, k})].sum()
    needj = capj - onlyj - W[j]; needk = capk - onlyk - W[k]
    if needj <= TOL and needk <= TOL:
        return True, "trivial"
    if W[both].sum() < needj + needk - TOL or W[both].sum() < max(needj, needk) - TOL:
        return False, "fractional"                                           # exact arithmetic necessary condition (sums of <= 200 doubles; TOL >> rounding)
    if not both:
        return False, "fractional"
    wb = W[both]; n_ = len(both)                                             # x_u = 1 -> u to a; else to b
    lo_, hi_ = max(needj, 0.0), wb.sum() - max(needk, 0.0)
    if n_ <= 22:                                                             # exact: all subset sums
        sums = np.zeros(1)
        for v in wb:
            sums = np.concatenate([sums, sums + v])
        return bool(((sums >= lo_ - TOL) & (sums <= hi_ + TOL)).any()), "brute"
    res = milp(np.zeros(n_), integrality=np.ones(n_), bounds=Bounds(0, 1), constraints=[LinearConstraint(wb[None, :], lo_, hi_)], options=dict(disp=False))
    return res.x is not None, "milp"


OUT = dict(radii={}, ranking_universe="600 prespecified named class-feasible tracts; tie-compatible rank intervals; population-rank interval [share surely above, 1 - share surely below] (upper side includes j's own weight)",
           statistic="family sup over the fractional fat family (Theorem S14); background sup = anchored prefix max; exchangeable-rank calibration, Gaussian pilot, max over the four-law family")
for r_km in [float(x) for x in a.radii.split(",")]:
    t0 = time.time(); r_ = r_km / KM_PER_UNIT; near = np.zeros((K, K), bool); muW = np.zeros(K); contained = np.zeros(K, bool)
    for x in range(K):
        inwin = (X - cx[x]) ** 2 + (Y - cy[x]) ** 2 <= r_ * r_; cnt_in = np.bincount(inv[inwin], minlength=K)
        whole = (cnt_in == cnt_all) & (cnt_all > 0); near[x] = whole; muW[x] = w[inwin].sum(); contained[x] = whole[x]
    cap = a.pfat * muW; elig_mass = (near * W[None, :]).sum(1)
    feas = contained & (W > 0) & (elig_mass >= cap) & (near.sum(1) >= 2)
    named = np.where(feas)[0]
    if len(named) > a.max_named:
        named = np.sort(rng.choice(named, a.max_named, replace=False))
    n = len(named); popn = pop[named]
    # ---- endpoints: exact whole-unit (anchor forced) and fractional; background (prefix) endpoints
    Lex = np.zeros(n); Uex = np.zeros(n); Lfr = np.zeros(n); Ufr = np.zeros(n); Lbg = np.zeros(n); Ubg = np.zeros(n); idxs = {}; caps = np.zeros(n); n_cells_le2 = 0
    for i, x in enumerate(named):
        idx = np.where(near[x])[0]; idxs[i] = idx; ia = int(np.where(idx == x)[0][0]); vals = theta[idx]; wts = W[idx]; caps[i] = cap[x]
        Lex[i], _ = exact_endpoint(vals, wts, cap[x], ia, +1); Uex[i], _ = exact_endpoint(vals, wts, cap[x], ia, -1)
        pw = cap[x] / wts.sum(); Lfr[i] = trimmed_weights(vals, wts, pw) @ vals; Ufr[i] = trimmed_weights(-vals, wts, pw) @ vals
        Lbg[i] = -prefix_sup(-vals[None, :], wts, ia)[0]; Ubg[i] = prefix_sup(vals[None, :], wts, ia)[0]
    # ---- (C) pair-specific joint feasibility
    Eset = {i: set(idxs[i].tolist()) for i in range(n)}; joint = np.ones((n, n), bool); undecided = 0; t1 = time.time(); empty_proof = {}
    for i in range(n):
        for jj in range(n):
            if i == jj:
                continue
            x, y_ = named[i], named[jj]
            if not (Eset[i] & Eset[jj]):
                continue                                                     # disjoint eligible sets: compatible
            common = Eset[i] & Eset[jj]; ok_common = (x in common) and (y_ in common) and (W[list(common)].sum() >= max(cap[x], cap[y_]) - TOL)
            if ok_common:
                continue
            fz, proof = split_feasible(idxs[i], idxs[jj], x, y_, cap[x], cap[y_]); joint[i, jj] = fz
            if not fz:
                empty_proof[proof] = empty_proof.get(proof, 0) + 1
    np.fill_diagonal(joint, False); n_pairs = n * (n - 1); n_joint = int(joint.sum()); n_empty = n_pairs - n_joint
    # ---- (A) all-named class: pairwise necessary condition; greedy witness search over the p grid
    def greedy_witness(pf):
        capg = pf * muW; lab = -np.ones(K, int); order = named[np.argsort(-(capg[named] / np.maximum(elig_mass[named], 1e-12)))]; cell_id = 0
        d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
        for x in order:
            if lab[x] >= 0:
                continue
            members = [x]; allowed = set(np.where(near[x] & (lab < 0))[0].tolist()); need = {x}
            while True:
                mass = W[members].sum(); unmet = [y_ for y_ in members if y_ in set(named.tolist()) and mass < capg[y_] - TOL]
                if not unmet:
                    break
                cand = [u for u in allowed if u not in members]
                if not cand:
                    return None
                u = min(cand, key=lambda u_: d2[x, u_]); members.append(u)
                if u in set(named.tolist()):
                    allowed &= set(np.where(near[u])[0].tolist())
                    if any(m_ not in near_sets[u] for m_ in members):
                        return None
            for m_ in members:
                lab[m_] = cell_id
            cell_id += 1
        return lab
    near_sets = {u: set(np.where(near[u])[0].tolist()) for u in named}
    # ---- (A, witness-anchored universe of Theorem S16) all-named class: Z0 = Voronoi cells of seeds (named locations, greedy spacing s*r); N' = named locations fat-feasible in Z0 (geometry only, no outcomes)
    d2n = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
    def witness_zoning(spacing):
        seeds = []
        for x in named[np.argsort(-pop[named])]:
            if all(d2n[x, s_] >= spacing ** 2 for s_ in seeds):
                seeds.append(int(x))
        seeds = np.array(seeds)
        # a unit joins the nearest seed whose eligible set contains it (so every seed cell is inside the seed's window); the rest are singleton background cells
        D = d2n[:, seeds].copy(); D[~near[seeds].T] = np.inf; lab = np.where(np.isfinite(D.min(1)), np.argmin(D, 1), -1)
        free = np.where(lab < 0)[0]; lab[free] = len(seeds) + np.arange(len(free))
        okx = np.zeros(n, bool)
        for i, x in enumerate(named):
            cell = np.where(lab == lab[x])[0]; okx[i] = set(cell.tolist()) <= Eset[i] and W[cell].sum() >= cap[x] - TOL
        return lab, okx, len(seeds)
    bestA = None
    for sfac in (0.8, 1.0, 1.25, 1.5, 2.0):
        lab0, okx, nseed = witness_zoning(sfac * r_)
        if bestA is None or okx.sum() > bestA[1].sum():
            bestA = (lab0, okx, nseed, sfac)
    labA, okA, nseedA, sfacA = bestA; NA = np.where(okA)[0]
    pair_block = n_empty > 0; witness_p = None
    for pf in [float(s) for s in a.pgrid.split(",")]:
        if pf == a.pfat and pair_block:
            continue
        lab = greedy_witness(pf)
        if lab is not None:
            witness_p = pf; break
    # ---- noise statistics (family sups) and calibration
    act = np.arange(n)
    def Dfat(E):
        Dp = np.zeros((E.shape[0], n)); Dm = np.zeros((E.shape[0], n))
        for i in act:
            Ex = E[:, idxs[i]]; Dp[:, i] = family_sup(Ex, W[idxs[i]], caps[i]); Dm[:, i] = family_sup(-Ex, W[idxs[i]], caps[i])
        return Dp, Dm
    def Dbg(E):
        Dp = np.zeros((E.shape[0], n)); Dm = np.zeros((E.shape[0], n))
        for i in act:
            idx = idxs[i]; ia = int(np.where(idx == named[i])[0][0]); Ex = E[:, idx]; Dp[:, i] = prefix_sup(Ex, W[idx], ia); Dm[:, i] = prefix_sup(-Ex, W[idx], ia)
        return Dp, Dm
    k_ = int(np.ceil((a.B + 1) * (1 - a.alpha)))
    def calibrate(stat_fn, chunk=50):
        """streaming pilot moments (300 Gaussian draws) and per-law exchangeable-rank critical values (B draws), chunked over draws"""
        s1 = None; s2 = None; npil = 300
        for c0 in range(0, npil, chunk):
            S = stat_fn(draws(min(chunk, npil - c0), "gauss")); s1 = S.sum(0) if s1 is None else s1 + S.sum(0); s2 = (S ** 2).sum(0) if s2 is None else s2 + (S ** 2).sum(0)
        aa = s1 / npil; bb = np.sqrt(np.maximum(s2 / npil - aa ** 2, 0.0)) + 1e-12; cs = {}
        for law in ("gauss", "skew", "skew2", "t4"):
            Ms = []
            for c0 in range(0, a.B, chunk):
                S = stat_fn(draws(min(chunk, a.B - c0), law)); M = ((S - aa) / bb).reshape(S.shape[0], -1); Ms.append(np.nanmax(np.where(np.isfinite(M), M, -np.inf), 1))
            cs[law] = float(np.sort(np.concatenate(Ms))[k_ - 1])
        return aa, bb, cs
    # three query families (each construction calibrated on its own family): fat endpoints (C separable), fat + background (B), pair statistic (C direct)
    def fat_stat(E):
        Dp, Dm = Dfat(E); return np.stack([Dp, Dm], -1)
    def B_stat(E):
        Dp, Dm = Dfat(E); Bp, Bm = Dbg(E); return np.stack([Dp, Dm, Bp, Bm], -1)
    def pair_stat(E):
        Dp, Dm = Dfat(E); M = Dp[:, :, None] + Dm[:, None, :]; M[:, np.arange(n), np.arange(n)] = np.nan; return M
    aF, bF, cF = calibrate(fat_stat); aS, bS, cS = calibrate(B_stat); aP, bP, cP = calibrate(pair_stat)
    res = dict(n_named=int(n), feasible_pop_share=float(pop[feas].sum()), named_pop_share=float(popn.sum()), endpoints=dict(
        exact_vs_fractional_gap_median=float(np.median(np.r_[Lex - Lfr, Ufr - Uex])), exact_vs_fractional_gap_max=float(np.max(np.r_[Lex - Lfr, Ufr - Uex])),
        fat_width_median=float(np.median(Uex - Lex)), background_width_median=float(np.median(Ubg - Lbg))),
        pair_classes=dict(ordered_pairs=n_pairs, joint_feasible=n_joint, empty=n_empty, empty_share=n_empty / n_pairs, empty_proofs=empty_proof, secs=time.time() - t1),
        all_named_class=dict(empty_proven_by_pairs=bool(pair_block), greedy_witness_p=witness_p, note="C_N at p is empty whenever some pair class C_jk is empty; witness_p = largest grid p with a greedy witness partition"),
        witness_anchored_class=dict(universe_size=int(len(NA)), universe_pop_share_of_named=float(popn[NA].sum() / popn.sum()), seeds=int(nseedA), spacing_factor=sfacA,
                                    note="Z0 = Voronoi cells of named seeds at spacing s*r; N' = named locations whose Z0 cell is admissible for them; C_{N'} is non-empty (Z0 is a witness); rank sets over N' are over one common class"),
        c_fat=cF, c_B=cS, c_pair=cP)
    def rank_stats(Lmat, valid=None):
        cert = Lmat > 0; np.fill_diagonal(cert, False)
        if valid is not None:
            cert &= valid
        den_all = n * (n - 1); den_valid = int(valid.sum()) if valid is not None else den_all
        above = np.array([popn[cert[:, j]].sum() for j in range(n)]); below = np.array([popn[cert[j, :]].sum() for j in range(n)])
        lo_r = above; hi_r = 1.0 - below; top = top20_est[named]
        return dict(ordered_pairs_share_all=float(cert.sum() / den_all), ordered_pairs_share_jointfeasible=float(cert.sum() / max(den_valid, 1)),
                    rank_width_median=float(np.median(hi_r - lo_r)), top20_certified_top20_pop_share=float(popn[top & (hi_r <= 0.2)].sum() / max(popn[top].sum(), 1e-12)),
                    certified_not_top20_pop_share=float(popn[lo_r > 0.2].sum() / popn.sum()))
    for name, law in (("noiseless", None), ("gaussian", "gauss"), ("family", "max")):
        if law is None:
            dL = dU = dLB = dUB = dLb = dUb = np.zeros(n); dP = np.zeros((n, n))
        else:
            cf = cF[law] if law != "max" else max(cF.values()); cs = cS[law] if law != "max" else max(cS.values()); cp = cP[law] if law != "max" else max(cP.values())
            dL = aF[:, 0] + cf * bF[:, 0]; dU = aF[:, 1] + cf * bF[:, 1]; dP = aP + cp * bP                                   # construction C families
            dLB = aS[:, 0] + cs * bS[:, 0]; dUB = aS[:, 1] + cs * bS[:, 1]; dLb = aS[:, 2] + cs * bS[:, 2]; dUb = aS[:, 3] + cs * bS[:, 3]   # construction B family
        # (C) pair-specific: l_jk = (L_j - dL_j) - (U_k + dU_k)  [separable]  /  L_j - U_k - dP_jk [direct pair]; shares over all pairs and over joint-feasible pairs
        Lsep = (Lex - dL)[:, None] - (Uex + dU)[None, :]; Lpair = (Lex[:, None] - Uex[None, :]) - np.nan_to_num(dP, nan=np.inf)
        Lpair_fr = (Lfr[:, None] - Ufr[None, :]) - np.nan_to_num(dP, nan=np.inf)
        resC = dict(separable=rank_stats(Lsep, joint), direct_pair=rank_stats(Lpair, joint), separable_fractional_endpoints=rank_stats((Lfr - dL)[:, None] - (Ufr + dU)[None, :], joint),
                    direct_pair_fractional_endpoints=rank_stats(Lpair_fr, joint))
        # (B) focal-specific: for focal j, competitor k is under the background rule: k above j iff Lbg_k - dLb_k > Uex_j + dU_j ; k below j iff Lex_j - dL_j > Ubg_k + dUb_k
        above_B = (Lbg - dLb)[:, None] > (Uex + dUB)[None, :]         # [k, j]: k surely above j
        below_B = (Lex - dLB)[:, None] > (Ubg + dUb)[None, :]         # [j, k]: k surely below j
        np.fill_diagonal(above_B, False); np.fill_diagonal(below_B, False)
        ab = np.array([popn[above_B[:, j]].sum() for j in range(n)]); be = np.array([popn[below_B[j, :]].sum() for j in range(n)]); top = top20_est[named]
        resB = dict(ordered_pairs_share=float((above_B.sum() + below_B.sum()) / (2 * n * (n - 1))), rank_width_median=float(np.median((1 - be) - ab)),
                    top20_certified_top20_pop_share=float(popn[top & ((1 - be) <= 0.2)].sum() / max(popn[top].sum(), 1e-12)), certified_not_top20_pop_share=float(popn[ab > 0.2].sum() / popn.sum()))
        def rank_sub(Lmat, sub):
            Ls = Lmat[np.ix_(sub, sub)]; cert = Ls > 0; np.fill_diagonal(cert, False); ps = popn[sub]; m_ = len(sub)
            above = np.array([ps[cert[:, j]].sum() for j in range(m_)]); below = np.array([ps[cert[j, :]].sum() for j in range(m_)]); lo_r = above / ps.sum(); hi_r = 1.0 - below / ps.sum()
            top = top20_est[named][sub]
            return dict(universe=int(m_), ordered_pairs_share=float(cert.sum() / max(m_ * (m_ - 1), 1)), rank_width_median=float(np.median(hi_r - lo_r)) if m_ else None,
                        certified_not_top20_of_universe_pop_share=float(ps[lo_r > 0.2].sum() / ps.sum()) if m_ else None, top20_certified_top20_pop_share=float(ps[top & (hi_r <= 0.2)].sum() / max(ps[top].sum(), 1e-12)) if m_ else None)
        resA = dict(separable=rank_sub(Lsep, NA), direct_pair=rank_sub(Lpair, NA), direct_pair_fractional_endpoints=rank_sub(Lpair_fr, NA))
        res[name] = dict(C_pair_specific=resC, B_focal_specific=resB, A_witness_anchored=resA, A_all_named=("vacuous: C_N empty at p = %.2f" % a.pfat) if pair_block else "see greedy witness; rank sets = construction C assembled (valid only if C_N non-empty)")
    # ---- witness: county zoning membership in C_jk at named pairs and in C_j
    cell_ok = np.zeros(n, bool)
    for i, x in enumerate(named):
        cell = np.where(county == county[x])[0]; cell_ok[i] = set(cell.tolist()) <= Eset[i] and W[cell].sum() >= cap[x] - TOL
    Kc = county.max() + 1; m = np.bincount(county, weights=W * theta, minlength=Kc) / np.maximum(np.bincount(county, weights=W, minlength=Kc), 1e-12); v = m[county]
    oo = np.argsort(-v); cc = np.cumsum(pop[oo]); t20 = np.zeros(K, bool); t20[oo[cc <= 0.2]] = True; flip = (t20 != top20_est)[named]
    pair_in = cell_ok[:, None] & cell_ok[None, :]; np.fill_diagonal(pair_in, False)
    res["witness_county"] = dict(focal_in_class_share_of_named=float(cell_ok.mean()), pairs_in_joint_class=int(pair_in.sum()), flips_focal_in_class_pop_share=float(popn[flip & cell_ok].sum() / popn.sum()),
                                 flips_focal_separate_pop_share=float(popn[flip & ~cell_ok].sum() / popn.sum()))
    OUT["radii"][str(r_km)] = res
    f = res["family"]; g = res["gaussian"]; z = res["noiseless"]
    print(f"r={r_km}: named {n} feasible-pop {res['feasible_pop_share']:.3f} | exact-vs-frac endpoint gap median {res['endpoints']['exact_vs_fractional_gap_median']:.4f} max {res['endpoints']['exact_vs_fractional_gap_max']:.4f} | "
          f"pair classes: empty {n_empty}/{n_pairs} ({n_empty/n_pairs:.3f}) proofs {empty_proof} | C_N empty by pairs: {pair_block}, greedy witness p = {witness_p} | c fat gauss/family {cF['gauss']:.2f}/{max(cF.values()):.2f} B-family {cS['gauss']:.2f}/{max(cS.values()):.2f} pair {cP['gauss']:.2f}/{max(cP.values()):.2f}\n"
          f"   (C) ordered pairs over joint-feasible: noiseless {z['C_pair_specific']['direct_pair']['ordered_pairs_share_jointfeasible']:.3f} gauss {g['C_pair_specific']['direct_pair']['ordered_pairs_share_jointfeasible']:.3f} family {f['C_pair_specific']['direct_pair']['ordered_pairs_share_jointfeasible']:.3f} "
          f"(over all pairs: {z['C_pair_specific']['direct_pair']['ordered_pairs_share_all']:.3f}/{g['C_pair_specific']['direct_pair']['ordered_pairs_share_all']:.3f}/{f['C_pair_specific']['direct_pair']['ordered_pairs_share_all']:.3f}; fractional-endpoint family {f['C_pair_specific']['separable_fractional_endpoints']['ordered_pairs_share_jointfeasible']:.3f})\n"
          f"   (B) focal-specific ordered pairs: noiseless {z['B_focal_specific']['ordered_pairs_share']:.3f} gauss {g['B_focal_specific']['ordered_pairs_share']:.3f} family {f['B_focal_specific']['ordered_pairs_share']:.3f}; rank width median {f['B_focal_specific']['rank_width_median']:.3f}; not-top20 family {f['B_focal_specific']['certified_not_top20_pop_share']:.3f}\n"
          f"   (A, witness-anchored universe) N' = {len(NA)} named (pop share {popn[NA].sum()/popn.sum():.3f}; {nseedA} seeds, spacing {sfacA}r): ordered pairs noiseless {z['A_witness_anchored']['direct_pair']['ordered_pairs_share']:.3f} gauss {g['A_witness_anchored']['direct_pair']['ordered_pairs_share']:.3f} family {f['A_witness_anchored']['direct_pair']['ordered_pairs_share']:.3f}; rank width median family {f['A_witness_anchored']['direct_pair']['rank_width_median']}; not-top20-of-N' family {f['A_witness_anchored']['direct_pair']['certified_not_top20_of_universe_pop_share']}\n"
          f"   witness: county focal in-class {res['witness_county']['focal_in_class_share_of_named']:.3f}, pairs in joint class {res['witness_county']['pairs_in_joint_class']} {time.time()-t0:.0f}s", flush=True)
    json.dump(OUT, open(os.path.join(a.out, f"b3_classes_georgia{a.tag}.json"), "w"), indent=1, default=float)
