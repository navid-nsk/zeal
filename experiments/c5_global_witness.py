"""c5_global_witness.py -- global-class map witness (Supplementary Theorems S7, S10; results exp2/C5): ONE globally feasible nodal field
of the lifted class on the whole raster, then feasibility-preserving local improvement.
Stage 1  global baseline: the lifted set X^(m0) (default m0 = 1, mesh h/2) on the full in-domain raster is built with the Mesh class
         (zeal/outer_x.py); a global LP (zero objective, or the linearized map objective when --linobj) gives a globally
         feasible nodal field x0 (verified against every constraint).
Stage 2  (optional) bilinear subdivision to the h/4 mesh is skipped here: improvement runs at the baseline mesh (m0) to keep one global object.
Stage 3  feasibility-preserving improvement: for each coarse cell b (region R = nodes adjacent to b's pixels), fix all other nodes at the
         current global field and maximize the linearization of the cell's map objective Q_b(x) = sum_{a in b} mu_a (d_a^T B x)^2 over the
         nodes of R with every incident constraint (ring pixels of neighbouring cells enter with their nodes fixed); accept iff the
         candidate is verified feasible and Q_b does not decrease (feasibility-preserving improvement, Theorem S10); 2 sweeps. The map statistic is evaluated on the one
         resulting global field: S_global = sqrt(sum_pairs mu_a (d_a^T B x)^2 / mu_tot) / movement.
Usage: python c5_global_witness.py --field georgia [--m0 1] [--sweeps 2].
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from outer_x import Mesh
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--m0", type=int, default=1); p.add_argument("--sweeps", type=int, default=2); p.add_argument("--out", default=None); p.add_argument("--max_cells", type=int, default=0); a = p.parse_args()
OUT = a.out or paths.results("exp2", "C5"); os.makedirs(OUT, exist_ok=True); M = 2 * a.m0
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float).T, F["hi"].astype(float).T
g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float).T for k in ("g1l", "g1h", "g2l", "g2h"))
ras = np.load(SPEC[a.field]); mask = ras["mask"].astype(bool).T & np.isfinite(lo0); N = mask.shape[0]; h = 2.0 / (N - 1)
w = np.where(mask, ras["weight"].T, 0.0).astype(float); fine = np.where(mask, ras["tract_idx"].T, -1); coarse = np.where(mask, ras["county_idx"].T, -1)
G1 = np.stack([g1l0, g1h0], -1); G2 = np.stack([g2l0, g2h0], -1); lam = (lo0 + hi0) / 2
t0 = time.time()
# ---------------- Stage 1: global lifted set and a feasible field
# crop to the bounding box of the mask to save nodes
ii, jj = np.nonzero(mask); I0, I1, J0, J1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1; sl = (slice(I0, I1), slice(J0, J1)); mk = mask[sl]
ms = Mesh(lo0[sl], hi0[sl], G1[sl], G2[sl], h, mk, M); A, b = ms.system(); V = ms.nv
print(f"global mesh M={M}: {V} nodes, {A.shape[0]} rows, built in {time.time()-t0:.0f}s", flush=True)
# pairs (fine a inside coarse b): objective rows d_a^T B on the global mesh (obj_q1 over the cropped arrays)
pairs = []; wmask = w[sl] * mk
for bb in np.unique(coarse[sl][mk]):
    selb = (coarse[sl] == bb) & mk; wb = wmask * selb
    for f_ in np.unique(fine[sl][selb]):
        if f_ < 0:
            continue
        wa = wb * (fine[sl] == f_)
        if wa.sum() <= 0 or wa.sum() >= wb.sum() - 1e-12:
            continue
        d2 = wa / wa.sum() - wb / wb.sum(); pairs.append((int(bb), int(f_), float(wa.sum()), d2))
print(f"{len(pairs)} pairs", flush=True)
Bq = sp.vstack([sp.csr_matrix(ms.obj_q1(d2)) for (_, _, _, d2) in pairs]).tocsr(); mus = np.array([m_ for (_, _, m_, _) in pairs]); reals = np.array([float(d2[mk] @ lam[sl][mk]) for (_, _, _, d2) in pairs])
cell_of_pair = np.array([bb for (bb, _, _, _) in pairs]); mu_tot = w.sum(); mov = np.sqrt((mus * reals ** 2).sum() / mu_tot)
def Qmap(x):
    return float(np.sum(mus * (Bq @ x) ** 2))
def feasible(x, tol=1e-7):
    return bool(np.all(A @ x <= b + tol) and np.all(x >= ms.lo_n - tol) and np.all(x <= ms.hi_n + tol))
t1 = time.time(); res = linprog(np.zeros(V), A_ub=A, b_ub=b, bounds=list(zip(ms.lo_n, ms.hi_n)), method="highs")
assert res.status == 0, res.message
x = res.x.copy(); print(f"Stage 1: global feasible field found in {time.time()-t1:.0f}s; feasible={feasible(x)}; S_global0 = {np.sqrt(Qmap(x)/mu_tot)/mov:.4f}x", flush=True)
# ---------------- Stage 3: feasibility-preserving local improvement, cell by cell
cells = sorted(set(cell_of_pair.tolist())); hist = []
for sweep in range(a.sweeps):
    for ci, bb in enumerate(cells):
        if a.max_cells and ci >= a.max_cells:
            break
        # region R: nodes adjacent to the cell's pixels; all other nodes fixed (bounds lo=hi=current value)
        pix = np.argwhere((coarse[sl] == bb) & mk); R = np.zeros(V, bool)
        for (i, j) in pix:
            for aa in range(i * M, i * M + M + 1):
                for bb_ in range(j * M, j * M + M + 1):
                    k = ms.col[ms.nid(aa, bb_)]
                    if k >= 0:
                        R[k] = True
        rows = np.where(cell_of_pair == bb)[0]; g = (mus[rows] * (Bq[rows] @ x)) @ Bq[rows]            # gradient of Q_b at x (restricted rows)
        if not np.any(g[R] != 0):
            continue
        lo_n = np.where(R, ms.lo_n, x); hi_n = np.where(R, ms.hi_n, x)
        # only constraints touching R matter; the rest hold at the fixed values -- pass the full system (HiGHS presolve removes fixed columns)
        r = linprog(-g, A_ub=A, b_ub=b, bounds=list(zip(lo_n, hi_n)), method="highs")
        if r.status != 0:
            continue
        xn = r.x; Qb_old = float(np.sum(mus[rows] * (Bq[rows] @ x) ** 2)); Qb_new = float(np.sum(mus[rows] * (Bq[rows] @ xn) ** 2))
        if feasible(xn) and Qb_new >= Qb_old - 1e-12:
            x = xn; hist.append(dict(sweep=sweep, cell=bb, Qb_old=Qb_old, Qb_new=Qb_new))
        if ci % 20 == 0:
            print(f"  sweep {sweep} cell {ci}/{len(cells)}: S_global = {np.sqrt(Qmap(x)/mu_tot)/mov:.4f}x  ({time.time()-t0:.0f}s)", flush=True)
S_global = np.sqrt(Qmap(x) / mu_tot) / mov
prod = json.load(open(os.path.join(OUT, f"c5_{a.field}.json")))["summary"] if os.path.exists(os.path.join(OUT, f"c5_{a.field}.json")) else {}
mw = json.load(open(os.path.join(OUT, f"c5_mapwitness_{a.field}.json")))["summary"] if os.path.exists(os.path.join(OUT, f"c5_mapwitness_{a.field}.json")) else {}
S = dict(field=a.field, mesh_M=M, nodes=int(V), rows=int(A.shape[0]), pairs=len(pairs), movement_mid=float(mov), S_global_witness_x=float(S_global), globally_feasible=feasible(x), accepted_improvements=len(hist),
         S_product_witness_x=mw.get("S_single_field_witness_x"), S_X_ceiling_x=prod.get("S_X_x"), S_separable_witness_x=prod.get("S_witness_x"), seconds=time.time() - t0,
         note="one globally feasible nodal field of the lifted class on the whole raster (Stage 1 feasibility LP + feasibility-preserving cellwise improvement); objectives = the same full-cell nested pairs as C5 (no cropped objectives)")
np.savez_compressed(os.path.join(OUT, f"c5_global_field_{a.field}_M{M}.npz"), x=x, I0=I0, J0=J0, M=M)
json.dump(dict(summary=S, history=hist), open(os.path.join(OUT, f"c5_global_witness_{a.field}.json"), "w"), indent=1, default=float); print(json.dumps(S, indent=1, default=float))
