"""c1_focal_counts.py -- distinct admissible focal cells (results exp2/C1): per location x and radius r, the
audited whole-unit class C_{r,p} at x is (a) feasible (anchor wholly contained, eligible mass >= p mu(W_x)); (b) has >= 2 eligible units
(availability); (c) has >= 2 DISTINCT admissible focal cells -- exact criterion: the full eligible set E_x is admissible and some unit u != x
can be dropped with mu(E_x) - W_u >= cap_x (every admissible cell is a subset of E_x containing x; two distinct ones exist iff E_x minus its
lightest non-anchor unit is still fat); (d) the number of admissible focal cells is at least 2^m where m = #{u != x : mu(E_x) - W_u >= cap}
... (lower bound only; the exact count is a subset-sum count and is not needed).  Shares are population shares.
Usage: python c1_focal_counts.py --out <dir>
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--radii", default="3.8,7.5,15,30"); p.add_argument("--pfat", type=float, default=0.5); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; w = ras["weight"][mask].astype(float); X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr); cnt_all = np.bincount(inv, minlength=K)
W = np.bincount(inv, weights=w, minlength=K); cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12)
KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2; pop = W / W.sum(); OUT = {}
for r_km in [float(x) for x in a.radii.split(",")]:
    r_ = r_km / KM_PER_UNIT; feas = np.zeros(K, bool); avail = np.zeros(K, bool); distinct2 = np.zeros(K, bool); log2_lb = np.zeros(K)
    for x in range(K):
        inwin = (X - cx[x]) ** 2 + (Y - cy[x]) ** 2 <= r_ * r_; cnt_in = np.bincount(inv[inwin], minlength=K); whole = (cnt_in == cnt_all) & (cnt_all > 0)
        if not whole[x] or W[x] <= 0:
            continue
        idx = np.where(whole)[0]; cap = a.pfat * w[inwin].sum(); mE = W[idx].sum()
        feas[x] = mE >= cap; avail[x] = len(idx) >= 2
        if feas[x] and avail[x]:
            others = idx[idx != x]; droppable = (mE - W[others]) >= cap
            distinct2[x] = bool(droppable.any()); log2_lb[x] = float(droppable.sum())
    row = dict(feasible=float(pop[feas].sum()), feasible_and_available=float(pop[feas & avail].sum()), feasible_two_distinct_cells=float(pop[feas & distinct2].sum()),
               single_cell_only=float(pop[feas & avail & ~distinct2].sum()), log2_cells_lower_bound_median_over_feasible=float(np.median(log2_lb[feas & avail])))
    OUT[str(r_km)] = row; print(r_km, {k: round(v, 4) for k, v in row.items()}, flush=True)
json.dump(OUT, open(os.path.join(a.out, "c1_focal_counts.json"), "w"), indent=1)
