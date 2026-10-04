"""b14_tent.py -- adversarial conditioned case (Supplementary Theorem S9; results exp2/B14): on [0,1] with |f'| <= 1,
f(0) = 0, f(1) = c (c = 0.43 so that the kink 0.5(1+c) = 0.715 is NOT a mesh node), condition the integral mean to its maximum
feasible value m* = int min(x, 1+c-x) dx. The continuous conditioned class is non-empty (the tent), but a fixed piecewise-linear
mesh without the kink node cannot attain m*: the exact reconstructed inner (A B_m x = y) is infeasible while the exact lifted outer
(A C_m x = y) is feasible. Implemented on a 1 x N pixel strip (G2 = {0}) with the lifted Mesh of outer_x.py, m = 1, 2, 4.
Usage: python b14_tent.py --out <dir>."""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from outer_x import Mesh
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--N", type=int, default=10); p.add_argument("--c", type=float, default=0.43); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
N, c = a.N, a.c; h = 1.0 / N; xk = 0.5 * (1 + c)
mstar = 0.5 * xk ** 2 + (1 + c) * (1 - xk) - 0.5 * (1 - xk ** 2)          # int_0^xk x dx + int_xk^1 (1+c-x) dx
lo = np.full((N, 1), -10.0); hi = np.full((N, 1), 10.0); G1 = np.zeros((N, 1, 2)); G1[..., 0] = -1; G1[..., 1] = 1; G2 = np.zeros((N, 1, 2)); mk = np.ones((N, 1), bool)
res = dict(N=N, c=c, kink=xk, m_star=mstar, mesh_nodes_contain_kink={}, rows=[])
for m in (1, 2, 4):
    ms = Mesh(lo, hi, G1, G2, h, mk, 2 * m); V = ms.nv; nodes_x = np.arange(0, N * 2 * m + 1) * h / (2 * m); res["mesh_nodes_contain_kink"][str(m)] = bool(np.any(np.abs(nodes_x - xk) < 1e-12))
    d = np.full((N, 1), 1.0 / N); oc = ms.obj_centre(d); oq = ms.obj_q1(d)
    # endpoint values f(0) = 0, f(1) = c: the corner nodes at x = 0 and x = 1 (left/right ends of the strip) -- the Mesh node ids: use nid(a, b) with a along axis 0
    nid = ms.nid; n_left = nid(0, 0); n_right = nid(N * 2 * m, 0)    # corner nodes at the ends (b = 0 row of corners)
    A, b = ms.system(); lo_n, hi_n = ms.lo_n.copy(), ms.hi_n.copy(); lo_n[n_left] = hi_n[n_left] = 0.0; lo_n[n_right] = hi_n[n_right] = c
    def solve(obj, eqrow, eqval):
        Aeq = sp.csr_matrix(eqrow[None, :]); r = linprog(-obj, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[eqval], bounds=list(zip(lo_n, hi_n)), method="highs"); return r.status, (-r.fun if r.status == 0 else np.nan)
    # (0) unconditioned maxima of the mean under the two objectives
    r_out = linprog(-oc, A_ub=A, b_ub=b, bounds=list(zip(lo_n, hi_n)), method="highs"); r_in = linprog(-oq, A_ub=A, b_ub=b, bounds=list(zip(lo_n, hi_n)), method="highs")
    max_outer, max_inner = -r_out.fun, -r_in.fun
    # (1) exact lifted outer conditioned on mean = m*: feasible?  (2) exact reconstructed inner conditioned: feasible?
    st_out, _ = solve(oc, oc, mstar); st_in, _ = solve(oq, oq, mstar)
    row = dict(m=m, nodes=int(V), kink_on_mesh=res["mesh_nodes_contain_kink"][str(m)], max_mean_outer=max_outer, max_mean_inner=max_inner, m_star=mstar,
               outer_conditioned_feasible=bool(st_out == 0), inner_conditioned_feasible=bool(st_in == 0), inner_deficit=float(mstar - max_inner))
    res["rows"].append(row); print(json.dumps(row))
json.dump(res, open(os.path.join(a.out, "b14_tent.json"), "w"), indent=1, default=float)
