"""check_b3_incompat.py -- independent checker for the pair-class incompatibility proofs of b3_classes.py (Supplementary Theorem S16, V5): recomputes, in
EXACT rational arithmetic (Python Fractions of the float data), which ordered named pairs (j, k) admit no admissible focal cells a at j and
b at k that are equal or disjoint, under the declared class (audited whole-unit windows, p = 1/2, singleton-compatible background rule).
Independent of b3_classes.py except for the declared geometry rule (unit wholly inside the Euclidean disc around the anchor's population
centroid), which is re-implemented here from the raster.  A pair is proved incompatible iff (i) no common cell: not both anchors eligible
for each other, or mu(E_j ∩ E_k) < max(cap_j, cap_k); AND (ii) no disjoint split: either the exact fractional condition fails
(mu(both) < need_j + need_k or mu(both) < max(need_j, need_k)) or the exhaustive exact subset-sum test over the overlap finds no admissible
split (|overlap| <= 22; larger overlaps are reported as 'undecided by this checker', never counted).  No tolerance is used anywhere.
Output: the count of proved-incompatible ordered pairs, the first proved pair (the one-pair proof of C_N = ∅), the comparison with the
script's count, and the per-pair proof list.  Usage: python check_b3_incompat.py --out <dir> [--radii 7.5,15] [--max_named 600]
"""
try:
    from . import paths                     # imported as part of the package
except ImportError:
    import paths                            # flat import (this directory on sys.path)
import argparse, io, json, os, sys, time, numpy as np
from fractions import Fraction
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--radii", default="7.5,15"); p.add_argument("--pfat", type=float, default=0.5)
p.add_argument("--seed", type=int, default=79); p.add_argument("--max_named", type=int, default=600); p.add_argument("--compare", default=""); p.add_argument("--skip_pairs", default=""); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; w = ras["weight"][mask].astype(float); X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr); cnt_all = np.bincount(inv, minlength=K)
W = np.bincount(inv, weights=w, minlength=K); cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12)
KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2
WF = [Fraction(float(x)) for x in W]                      # exact rationals of the float unit masses (the declared data)
cmp = json.load(open(a.compare, encoding="utf-8"))["radii"] if a.compare else None
OUT = {}
for r_km in [float(x) for x in a.radii.split(",")]:
    t0 = time.time(); r_ = r_km / KM_PER_UNIT; near = np.zeros((K, K), bool); muW = np.zeros(K); contained = np.zeros(K, bool)
    for x in range(K):                                      # declared geometry rule (same float formula as the class definition)
        inwin = (X - cx[x]) ** 2 + (Y - cy[x]) ** 2 <= r_ * r_; cnt_in = np.bincount(inv[inwin], minlength=K)
        whole = (cnt_in == cnt_all) & (cnt_all > 0); near[x] = whole; muW[x] = w[inwin].sum(); contained[x] = whole[x]
    capF = [Fraction(a.pfat) * Fraction(float(muW[x])) for x in range(K)]
    elig_mass = (near * W[None, :]).sum(1); feas = contained & (W > 0) & (elig_mass >= a.pfat * muW) & (near.sum(1) >= 2)
    named = np.where(feas)[0]
    if len(named) > a.max_named:
        named = np.sort(rng.choice(named, a.max_named, replace=False))     # same prespecified sample (same seed and selection rule)
    n = len(named); E = {x: set(np.where(near[x])[0].tolist()) for x in named}
    def consume_rng():        # replicate b3_classes.py's noise-draw consumption per radius (3 calibrations x [300 Gaussian pilot + 999 per law]) so later radii see the same named sample
        for _ in range(3):
            rng.normal(size=(300, K)); rng.normal(size=(999, K)); rng.gamma(4.0, 1.0, size=(999, K)); rng.gamma(1.0, 1.0, size=(999, K)); rng.standard_t(4, size=(999, K))
    if str(r_km) in a.skip_pairs.split(","):
        consume_rng(); print(f"r={r_km}: rng consumed only (skip_pairs)", flush=True); continue
    proved = []; undecided = 0; by_type = {}
    for x in named:
        for y_ in named:
            if x == y_ or not (E[x] & E[y_]):
                continue
            common = E[x] & E[y_]
            if x in common and y_ in common and sum((WF[u] for u in common), Fraction(0)) >= max(capF[x], capF[y_]):
                continue                                                    # a common admissible cell exists: compatible
            both = sorted((E[x] & E[y_]) - {x, y_}); onlyx = sum((WF[u] for u in (E[x] - E[y_]) - {x, y_}), Fraction(0)); onlyy = sum((WF[u] for u in (E[y_] - E[x]) - {x, y_}), Fraction(0))
            needx = capF[x] - onlyx - WF[x]; needy = capF[y_] - onlyy - WF[y_]
            if needx <= 0 and needy <= 0:
                continue                                                    # trivially splittable
            mb = sum((WF[u] for u in both), Fraction(0))
            if mb < needx + needy or mb < max(needx, needy):
                proved.append((int(x), int(y_), "fractional")); by_type["fractional"] = by_type.get("fractional", 0) + 1; continue
            if len(both) > 22:
                undecided += 1; continue
            lo_, hi_ = max(needx, Fraction(0)), mb - max(needy, Fraction(0))
            sums = [Fraction(0)]
            for u in both:                                                  # exact subset sums (Python rationals; 2^|both| values)
                wu = WF[u]; sums = sums + [s_ + wu for s_ in sums]
            if any(lo_ <= s_ <= hi_ for s_ in sums):
                continue                                                    # an exact admissible split exists
            proved.append((int(x), int(y_), "subset_sums")); by_type["subset_sums"] = by_type.get("subset_sums", 0) + 1
    row = dict(n_named=int(n), ordered_pairs=int(n * (n - 1)), proved_incompatible=len(proved), by_proof_type=by_type, undecided_large_overlap=undecided,
               first_proved_pair=proved[0] if proved else None, C_N_empty_proved=bool(proved), seconds=time.time() - t0, pairs=proved)
    if cmp is not None:
        sc = cmp[str(r_km)]["pair_classes"]; row["script_count"] = sc["empty"]; row["script_proofs"] = sc.get("empty_proofs"); row["agree"] = (sc["empty"] == len(proved))
    OUT[str(r_km)] = row; consume_rng()
    print(f"r={r_km}: named {n}; proved incompatible ordered pairs {len(proved)} {by_type}; undecided {undecided}; first proved pair {row['first_proved_pair']}; "
          f"script {row.get('script_count')} {row.get('script_proofs')} agree={row.get('agree')} {time.time()-t0:.0f}s", flush=True)
json.dump(OUT, open(os.path.join(a.out, "check_b3_incompat" + ("_" + a.radii.replace(",", "_") if a.skip_pairs else "") + ".json"), "w"), indent=1)
