"""b14_conditioned.py -- conditioned classes (Supplementary Theorem S9; results exp2/B14): small-case validity tests of the exact
conditioned lifted outer (A C_m x = y), the exact reconstructed inner (A B_m x = y) and the approximate-data witness (|A B_m x - y| <= A beta).

Instances: synthetic sinusoidal fields f with sound boxes (synth.build); 'observed units' = random contiguous groups of pixels
(2-6 groups); y = A(Pf) computed exactly from the true pixel means; target T = mean of a random sub-group of pixels (cuts the units).
For m = 1, 2 we solve: outer_y = max/min d.C_m x s.t. x in X^(m), A C_m x = y   (general LP; verified duals);
inner_y = max/min d.B_m x s.t. x in X^(m), A B_m x = y (exact reconstructed inner; may be infeasible);
approx_y = same with |A B_m x - y| <= A beta^(m) (always feasible when outer_y is);
and the confidence-expanded outer with |A C_m x - y| <= eta (eta = 10 % of the unit s.d.).
Checks: truth target inside [outer_L, outer_U] (validity, 100 %); inner feasible => inner inside outer; approx feasible whenever outer is;
infeasible inner with feasible outer recorded (not a proof that the continuous identified set is empty); widths vs the unconditioned bracket.
Usage: python b14_conditioned.py --out <dir> [--n 30].
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import EPS
from outer_x import Mesh
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=30); p.add_argument("--seed", type=int, default=77); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); import synth; synth.set_rng(rng); from synth import build


def solve_eq(ms, obj, Aeq_rows, beq, ineq_rows=None, bineq=None):
    """max obj.x s.t. mesh system, extra equalities (rows over x) and extra inequalities; verified dual upper bound with equality multipliers."""
    A, b = ms.system(); lo_n, hi_n = ms.lo_n, ms.hi_n
    Aub = A if ineq_rows is None else sp.vstack([A, ineq_rows]).tocsr(); bub = b if ineq_rows is None else np.concatenate([b, bineq])
    res = linprog(-obj, A_ub=Aub, b_ub=bub, A_eq=Aeq_rows, b_eq=beq, bounds=list(zip(lo_n, hi_n)), method="highs")
    if res.status != 0:
        return np.nan, None, np.nan, res.status
    y = np.maximum(-res.ineqlin.marginals, 0.0); ye = -res.eqlin.marginals
    r = obj - Aub.T @ y - Aeq_rows.T @ ye; zp = np.maximum(r, 0); zm = np.maximum(-r, 0)
    if np.any((zp > 1e-12) & ~np.isfinite(hi_n)) or np.any((zm > 1e-12) & ~np.isfinite(lo_n)):
        return -res.fun, res.x, np.inf, 0
    hi_t = np.where(np.isfinite(hi_n), hi_n, 0.0); lo_t = np.where(np.isfinite(lo_n), lo_n, 0.0)
    terms = np.concatenate([bub * y, beq * ye, hi_t * zp, -lo_t * zm]); ub = float(np.sum(terms)) + (len(terms) + len(obj) + Aub.shape[0]) * EPS * float(np.sum(np.abs(terms)) + np.sum(np.abs(Aub.T @ y)) + np.sum(np.abs(Aeq_rows.T @ ye)) + np.sum(np.abs(obj)))
    return -res.fun, res.x, ub, 0


def grow_groups(mask, k):
    ii, jj = np.nonzero(mask); lab = -np.ones(mask.shape, int); seeds = rng.choice(len(ii), k, replace=False)
    for g, s_ in enumerate(seeds):
        lab[ii[s_], jj[s_]] = g
    changed = True
    while changed:
        changed = False
        for (i, j) in zip(*np.nonzero(mask & (lab < 0))):
            nb = [lab[x, y] for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)) if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and lab[x, y] >= 0]
            if nb:
                lab[i, j] = nb[int(rng.integers(len(nb)))]; changed = True
    return lab


rows = []
for t in range(a.n):
    n1, n2 = int(rng.integers(5, 9)), int(rng.integers(5, 9)); h = 0.25; mk = np.ones((n1, n2), bool); inst = build(n1, n2, h, float(rng.choice([0.0, 0.3])), mk)
    lo, hi, G1, G2, psi = inst["lo"], inst["hi"], inst["G1"], inst["G2"], inst["psi"]; w = rng.uniform(0.5, 1.5, (n1, n2))
    k = int(rng.integers(2, 7)); lab = grow_groups(mk, k); K = lab.max() + 1
    y = np.array([(w[lab == g] * psi[lab == g]).sum() / w[lab == g].sum() for g in range(K)])          # exact observed unit means of the TRUE field
    # target: a random contiguous sub-group cutting the units (a 2x2..3x3 block)
    bi, bj = int(rng.integers(0, n1 - 2)), int(rng.integers(0, n2 - 2)); bs = int(rng.integers(2, 4)); tgt = np.zeros((n1, n2), bool); tgt[bi:bi + bs, bj:bj + bs] = True
    d2 = np.where(tgt, w, 0.0); d2 = d2 / d2.sum(); truth = float((d2 * psi).sum())
    sd_unit = float(np.std(y)) if K > 1 else 0.05
    rec = dict(t=t, n_pix=int(n1 * n2), K=K, truth=truth, unit_sd=sd_unit)
    for m in (1, 2):
        ms = Mesh(lo, hi, G1, G2, h, mk, 2 * m); V = ms.nv; oc = ms.obj_centre(d2); oq = ms.obj_q1(d2)
        # rows for A C_m x (unit mean of pixel means = average of centre nodes) and A B_m x (unit mean of Q1 pixel means)
        rowsC = []; rowsB = []
        for g in range(K):
            wg = np.where(lab == g, w, 0.0); wg = wg / wg.sum(); rowsC.append(ms.obj_centre(wg)); rowsB.append(ms.obj_q1(wg))
        AC = sp.csr_matrix(np.array(rowsC)); AB = sp.csr_matrix(np.array(rowsB)); beta_rows = np.array([float(ms.deviation_bound(np.where(lab == g, w, 0.0) / np.where(lab == g, w, 0.0).sum(), G1, G2)) for g in range(K)])
        r = {}
        # unconditioned bracket for reference
        for sgn, nm in ((1, "U"), (-1, "L")):
            v, x, ub, st = solve_eq(ms, sgn * oc, sp.csr_matrix((0, V)), np.zeros(0)); r[f"{nm}_unc_outer"] = sgn * ub
            v, x, ub, st = solve_eq(ms, sgn * oc, AC, y); r[f"{nm}_cond_outer"] = sgn * ub; r["outer_status"] = st
            v, x, ub, st = solve_eq(ms, sgn * oq, AB, y); r[f"{nm}_cond_inner"] = sgn * v if st == 0 else float("nan"); r["inner_status"] = st
            ineq = sp.vstack([AB, -AB]).tocsr(); bi_ = np.concatenate([y + beta_rows, -(y - beta_rows)])
            v, x, ub, st = solve_eq(ms, sgn * oq, sp.csr_matrix((0, V)), np.zeros(0), ineq, bi_); r[f"{nm}_approx_inner"] = sgn * v if st == 0 else float("nan"); r["approx_status"] = st
            eta = 0.1 * sd_unit; ineq2 = sp.vstack([AC, -AC]).tocsr(); b2_ = np.concatenate([y + eta, -(y - eta)])
            v, x, ub, st = solve_eq(ms, sgn * oc, sp.csr_matrix((0, V)), np.zeros(0), ineq2, b2_); r[f"{nm}_expanded_outer"] = sgn * ub
        r["truth_inside_cond_outer"] = bool(r["L_cond_outer"] - 1e-9 <= truth <= r["U_cond_outer"] + 1e-9)
        r["inner_feasible"] = r["inner_status"] == 0; r["inner_inside_outer"] = bool(r["inner_feasible"] and r["L_cond_outer"] - 1e-9 <= r["L_cond_inner"] and r["U_cond_inner"] <= r["U_cond_outer"] + 1e-9)
        r["approx_feasible"] = r["approx_status"] == 0; r["width_unc"] = r["U_unc_outer"] - r["L_unc_outer"]; r["width_cond"] = r["U_cond_outer"] - r["L_cond_outer"]; r["width_expanded"] = r["U_expanded_outer"] - r["L_expanded_outer"]
        r["width_inner"] = (r["U_cond_inner"] - r["L_cond_inner"]) if r["inner_feasible"] else float("nan")
        rec[f"m{m}"] = r
    rows.append(rec); print(f"  t={t} K={K}: m=1 cond width {rec['m1']['width_cond']:.4f} (unc {rec['m1']['width_unc']:.4f}) inner {'feasible' if rec['m1']['inner_feasible'] else 'INFEASIBLE'} width {rec['m1']['width_inner']:.4f} truth inside {rec['m1']['truth_inside_cond_outer']} | m=2 cond {rec['m2']['width_cond']:.4f} inner {'feasible' if rec['m2']['inner_feasible'] else 'INFEASIBLE'} {rec['m2']['width_inner']:.4f}", flush=True)
S = {}
for m in (1, 2):
    R = [r[f"m{m}"] for r in rows]
    S[f"m{m}"] = dict(n=len(R), truth_inside_cond_outer=int(sum(r["truth_inside_cond_outer"] for r in R)), inner_feasible=int(sum(r["inner_feasible"] for r in R)), inner_inside_outer=int(sum(r["inner_inside_outer"] for r in R)),
                     approx_feasible=int(sum(r["approx_feasible"] for r in R)), width_cond_over_unc_median=float(np.median([r["width_cond"] / r["width_unc"] for r in R])),
                     width_inner_over_cond_median=float(np.nanmedian([r["width_inner"] / r["width_cond"] for r in R])), width_expanded_over_cond_median=float(np.median([r["width_expanded"] / r["width_cond"] for r in R])))
json.dump(dict(summary=S, rows=rows), open(os.path.join(a.out, "b14_conditioned.json"), "w"), indent=1, default=float); print(json.dumps(S, indent=1))
