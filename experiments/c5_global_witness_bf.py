"""c5_global_witness_bf.py -- global-class map witness (results exp2/C5), Stage 1 by the lattice top (Supplementary Theorem S3 (ii)): on the global lifted
difference-constraint system x_b - x_a <= c_ab (both directions) with bounds l <= x <= u, the vector u*_j = min_i (u_i + D(i,j)) (shortest
paths from the ground node with arc costs u_i on ground->i) is the TOP of the feasible lattice and is feasible whenever the system is.
Computed by vectorized Bellman-Ford (numpy np.minimum.at over all arcs until no change) -- no LP. Then Stage 3 as in c5_global_witness.py:
feasibility-preserving cellwise improvement with fixed outside nodes (local LPs), 2 sweeps; map statistic on the one global field.
Usage: python c5_global_witness_bf.py --field georgia [--m0 1] [--sweeps 2]."""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from outer_x import Mesh
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--m0", type=int, default=1); p.add_argument("--sweeps", type=int, default=2); p.add_argument("--out", default=None); a = p.parse_args()
OUT = a.out or paths.results("exp2", "C5"); os.makedirs(OUT, exist_ok=True); M = 2 * a.m0
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float).T, F["hi"].astype(float).T
g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float).T for k in ("g1l", "g1h", "g2l", "g2h"))
ras = np.load(SPEC[a.field]); mask = ras["mask"].astype(bool).T & np.isfinite(lo0); N = mask.shape[0]; h = 2.0 / (N - 1)
w = np.where(mask, ras["weight"].T, 0.0).astype(float); fine = np.where(mask, ras["tract_idx"].T, -1); coarse = np.where(mask, ras["county_idx"].T, -1)
G1 = np.stack([g1l0, g1h0], -1); G2 = np.stack([g2l0, g2h0], -1); lam = (lo0 + hi0) / 2; t0 = time.time()
ii, jj = np.nonzero(mask); I0, I1, J0, J1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1; sl = (slice(I0, I1), slice(J0, J1)); mk = mask[sl]
ms = Mesh(lo0[sl], hi0[sl], G1[sl], G2[sl], h, mk, M); A, b = ms.system(); V = ms.nv
# arcs of the difference system: row k of A is x_head - x_tail <= b_k  (A has +1 at head, -1 at tail)
Ac = A.tocoo(); heads = np.zeros(A.shape[0], np.int64); tails = np.zeros(A.shape[0], np.int64)
heads[Ac.row[Ac.data > 0]] = Ac.col[Ac.data > 0]; tails[Ac.row[Ac.data < 0]] = Ac.col[Ac.data < 0]; cost = b.copy()
print(f"global mesh M={M}: {V} nodes, {len(cost)} arcs, built in {time.time()-t0:.0f}s", flush=True)
# Stage 1: u* = shortest path from ground: dist = hi_n initially (ground->i costs hi_i), relax x_head <= x_tail + cost
dist = ms.hi_n.copy(); lo_n = ms.lo_n; it = 0; t1 = time.time()
while True:
    cand = dist[tails] + cost; new = dist.copy(); np.minimum.at(new, heads, cand)
    changed = np.count_nonzero(new < dist - 1e-15); dist = new; it += 1
    if changed == 0 or it > 20000:
        break
    if it % 100 == 0:
        print(f"  BF pass {it}: {changed} improved ({time.time()-t1:.0f}s)", flush=True)
x = dist; feas_bounds = bool(np.all(x >= lo_n - 1e-9)); feas_rows = bool(np.all(A @ x <= b + 1e-9))
print(f"Stage 1 (lattice top): {it} passes, {time.time()-t1:.0f}s; bounds feasible {feas_bounds}, rows feasible {feas_rows}; min slack to lo = {np.min(x - lo_n):.3e}", flush=True)
assert feas_bounds and feas_rows, "global system infeasible (l* > u*): the certified class is empty on this raster"
# pairs and map objective: objective rows built on county-local meshes (fast) and scattered to the global node indices
pairs = []; wmask = w[sl] * mk; rows_sp = []; mus = []; reals = []; cell_of_pair = []
N2g = (mk.shape[1]) * M + 1
for bb in np.unique(coarse[sl][mk]):
    selb = (coarse[sl] == bb) & mk; ib, jb = np.nonzero(selb); i0, i1, j0, j1 = ib.min(), ib.max() + 1, jb.min(), jb.max() + 1; slb = (slice(i0, i1), slice(j0, j1)); mkb = selb[slb]
    msb = Mesh(lo0[sl][slb], hi0[sl][slb], G1[sl][slb], G2[sl][slb], h, mkb, M)
    # local used node -> global used node
    loc_nodes = np.where(msb.col >= 0)[0]; a_loc, b_loc = loc_nodes // msb.N2, loc_nodes % msb.N2
    g_idx = ms.col[(a_loc + i0 * M) * N2g + (b_loc + j0 * M)]; assert np.all(g_idx >= 0)
    l2g = np.full(msb.nv, -1, np.int64); l2g[msb.col[loc_nodes]] = g_idx
    wb = wmask * selb
    for f_ in np.unique(fine[sl][selb]):
        if f_ < 0:
            continue
        wa = wb * (fine[sl] == f_)
        if wa.sum() <= 0 or wa.sum() >= wb.sum() - 1e-12:
            continue
        d2 = wa / wa.sum() - wb / wb.sum(); oq = msb.obj_q1(d2[slb]); nz = np.nonzero(oq)[0]
        rows_sp.append(sp.csr_matrix((oq[nz], (np.zeros(len(nz), np.int64), l2g[nz])), shape=(1, V))); mus.append(float(wa.sum())); reals.append(float(d2[mk] @ lam[sl][mk])); cell_of_pair.append(int(bb)); pairs.append((int(bb), int(f_)))
Bq = sp.vstack(rows_sp).tocsr(); mus = np.array(mus); reals = np.array(reals); cell_of_pair = np.array(cell_of_pair); mu_tot = w.sum(); mov = np.sqrt((mus * reals ** 2).sum() / mu_tot)
Acsc = A.tocsc()
Qmap = lambda x_: float(np.sum(mus * (Bq @ x_) ** 2)); feasible = lambda x_: bool(np.all(A @ x_ <= b + 1e-7) and np.all(x_ >= lo_n - 1e-7) and np.all(x_ <= ms.hi_n + 1e-7))
S0 = np.sqrt(Qmap(x) / mu_tot) / mov; print(f"{len(pairs)} pairs; S_global (lattice top) = {S0:.4f}x", flush=True)
# Stage 3: cellwise feasibility-preserving improvement (local LPs over the cell's nodes; every incident constraint via the full system with fixed bounds)
cells = sorted(set(cell_of_pair.tolist())); hist = []; AT = A.T.tocsr()
for sweep in range(a.sweeps):
    for ci, bb in enumerate(cells):
        pix = np.argwhere((coarse[sl] == bb) & mk); R = np.zeros(V, bool)
        for (i, j) in pix:
            for aa in range(i * M, i * M + M + 1):
                ks = ms.col[ms.nid(aa, j * M): ms.nid(aa, j * M) + M + 1]; R[ks[ks >= 0]] = True
        rows = np.where(cell_of_pair == bb)[0]; g = (mus[rows] * (Bq[rows] @ x)) @ Bq[rows]
        if not np.any(g[R] != 0):
            continue
        # restrict the LP to the rows touching R (others are fixed-feasible) and to R's columns; fixed neighbours enter through the rhs
        Rid = np.where(R)[0]; touch = np.unique(Acsc[:, Rid].tocoo().row); Asub = A[touch].tocsc(); frozen = np.where(~R)[0]
        rhs = b[touch] - Asub[:, frozen] @ x[frozen]; Ar = Asub[:, Rid].tocsr()
        r = linprog(-g[R], A_ub=Ar, b_ub=rhs, bounds=list(zip(lo_n[R], ms.hi_n[R])), method="highs")
        if r.status != 0:
            continue
        xn = x.copy(); xn[R] = r.x; Qb_old = float(np.sum(mus[rows] * (Bq[rows] @ x) ** 2)); Qb_new = float(np.sum(mus[rows] * (Bq[rows] @ xn) ** 2))
        ok = bool(np.all(A[touch] @ xn <= b[touch] + 1e-7))
        if ok and Qb_new >= Qb_old - 1e-12:
            x = xn; hist.append(dict(sweep=sweep, cell=bb, Qb_old=Qb_old, Qb_new=Qb_new))
        if ci % 20 == 0:
            print(f"  sweep {sweep} cell {ci}/{len(cells)}: S_global = {np.sqrt(Qmap(x)/mu_tot)/mov:.4f}x ({time.time()-t0:.0f}s)", flush=True)
assert feasible(x)
S_global = np.sqrt(Qmap(x) / mu_tot) / mov
prod = json.load(open(os.path.join(OUT, f"c5_{a.field}.json")))["summary"] if os.path.exists(os.path.join(OUT, f"c5_{a.field}.json")) else {}
mw = json.load(open(os.path.join(OUT, f"c5_mapwitness_{a.field}.json")))["summary"] if os.path.exists(os.path.join(OUT, f"c5_mapwitness_{a.field}.json")) else {}
S = dict(field=a.field, mesh_M=M, nodes=int(V), arcs=int(len(cost)), pairs=len(pairs), movement_mid=float(mov), S_global_lattice_top_x=float(S0), S_global_witness_x=float(S_global), globally_feasible=True, accepted_improvements=len(hist),
         S_product_witness_x=mw.get("S_single_field_witness_x"), S_X_ceiling_x=prod.get("S_X_x"), S_separable_witness_x=prod.get("S_witness_x"), seconds=time.time() - t0,
         note="ONE globally feasible nodal field of the lifted class on the whole raster: Stage 1 = lattice top u* by vectorized Bellman-Ford (Theorem S3 (ii)); Stage 3 = feasibility-preserving cellwise improvement (Theorem S10), all incident constraints re-verified; objectives = the full-cell nested pairs of C5")
np.savez_compressed(os.path.join(OUT, f"c5_global_field_{a.field}_M{M}.npz"), x=x, I0=I0, J0=J0, M=M)
json.dump(dict(summary=S, history=hist), open(os.path.join(OUT, f"c5_global_witness_{a.field}.json"), "w"), indent=1, default=float); print(json.dumps(S, indent=1, default=float))
