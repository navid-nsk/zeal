"""c5_mapwitness.py -- single-field map witness (Supplementary Theorems S7, S10; results exp2/C5): for each coarse cell b, ONE
field of the certified class (a Q_4 nodal vector on the lifted set X^(2) of the cell's patch) is found by alternating linearization
of the common-field objective Q(x) = sum_{a in b} mu_a (d_a^T B x)^2 (nonconvex; monotone ascent from the best single-pair optimizer;
each step maximizes the linear objective sum_a mu_a (d_a^T B x_prev)(d_a^T B x) over the same feasible set), and the map statistic
is evaluated on that field. The per-cell fields live on disjoint patches (the certified class in C5 is the product of per-patch box
classes), so their union is one admissible field for the class actually certified. Reported: S_single (RMS over all pairs of ONE
field's realized pair differences, x movement) beside the separable certified ceiling S_X and the separable witnessed reference
S_wit of c5_production.py. Cells larger than --max_pix are skipped (their pairs were cropped in C5; recorded).
Usage: python c5_mapwitness.py --field georgia [--iters 6]."""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import verified_solve_system
from outer_x import Mesh
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--max_pix", type=int, default=3000); p.add_argument("--iters", type=int, default=6); p.add_argument("--M", type=int, default=4); p.add_argument("--out", default=None); p.add_argument("--save", action="store_true"); a = p.parse_args()
OUT = a.out or paths.results("exp2", "C5"); os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(OUT, "mapwitness_fields"), exist_ok=True)
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
ras = np.load(SPEC[a.field]); mask = ras["mask"].astype(bool) & np.isfinite(lo0); N = mask.shape[0]; h = 2.0 / (N - 1)
w = np.where(mask, ras["weight"], 0.0).astype(float); fine = np.where(mask, ras["tract_idx"], -1); coarse = np.where(mask, ras["county_idx"], -1)
lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, fine, coarse = (A.T.copy() for A in (lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, fine, coarse))
G1 = np.stack([g1l0, g1h0], -1); G2 = np.stack([g2l0, g2h0], -1); lam = (lo0 + hi0) / 2
mu_tot = w.sum(); acc = dict(mov=0.0, single=0.0, sep_wit=0.0, X=0.0); rows = []; t0 = time.time(); skipped = 0
for b in np.unique(coarse[coarse >= 0]):
    b = int(b); sel = coarse == b
    if sel.sum() > a.max_pix:
        skipped += 1; continue
    fines = [f_ for f_ in np.unique(fine[sel]) if f_ >= 0 and w[sel & (fine == f_)].sum() > 0 and (fine[sel] == f_).sum() < sel.sum()]
    if not fines:
        continue
    i0, i1 = np.nonzero(sel.any(1))[0][[0, -1]]; j0, j1 = np.nonzero(sel.any(0))[0][[0, -1]]; sl = (slice(i0, i1 + 1), slice(j0, j1 + 1)); mk = sel[sl]
    ms = Mesh(lo0[sl], hi0[sl], G1[sl], G2[sl], h, mk, a.M); A_, b_ = ms.system(); wb = w[sl] * mk
    ds, mus, reals, ocs, oqs = [], [], [], [], []
    for f_ in fines:
        wa = wb * (fine[sl] == f_); d2 = wa / wa.sum() - wb / wb.sum(); ds.append(d2); mus.append(wa.sum()); reals.append(float(d2[mk] @ lam[sl][mk])); ocs.append(ms.obj_centre(d2)); oqs.append(ms.obj_q1(d2))
    mus = np.array(mus); Bq = np.array(oqs)                       # rows: d_a^T B (linear in nodal x)
    # separable pieces (for the record) and starting point: the single-pair outer optimizer with the largest mu_a * U_a^2
    best = None; sep = 0.0; Xc = 0.0
    for j in range(len(fines)):
        for sgn in (1, -1):
            v, x, ub = verified_solve_system(A_, b_, ms.lo_n, ms.hi_n, sgn * ocs[j]); val = (Bq[j] @ x) if x is not None else 0.0
            if x is not None and (best is None or mus[j] * val ** 2 > best[0]):
                best = (mus[j] * val ** 2, x)
        wit = max(abs(Bq[j] @ best[1]) if best is not None else 0.0, 0.0)
    # alternating linearization from the best start
    x = best[1]; Q_prev = float(np.sum(mus * (Bq @ x) ** 2))
    for it in range(a.iters):
        g = (mus * (Bq @ x)) @ Bq; vv, xn, _ = verified_solve_system(A_, b_, ms.lo_n, ms.hi_n, g)
        if xn is None:
            break
        Q_new = float(np.sum(mus * (Bq @ xn) ** 2))
        if Q_new <= Q_prev + 1e-12:
            break
        x, Q_prev = xn, Q_new
    pair_vals = Bq @ x                                                 # ONE field's realized pair differences
    if a.save:
        np.savez_compressed(os.path.join(OUT, "mapwitness_fields", f"{a.field}_cell{b}.npz"), x=x, i0=i0, j0=j0, n1=mk.shape[0], n2=mk.shape[1], mask=mk, M=a.M)
    acc["single"] += float(np.sum(mus * pair_vals ** 2)); acc["mov"] += float(np.sum(mus * np.array(reals) ** 2))
    rows.append(dict(coarse=b, n_fine=len(fines), n_pix=int(mk.sum()), Q_single=Q_prev, iters=it + 1));
    if len(rows) % 20 == 0:
        print(f"  {a.field} {len(rows)} cells ({time.time()-t0:.0f}s): S_single so far {np.sqrt(acc['single']/acc['mov']):.3f}x", flush=True)
mov = np.sqrt(acc["mov"] / mu_tot); S_single = np.sqrt(acc["single"] / mu_tot) / mov
prod = json.load(open(os.path.join(OUT, f"c5_{a.field}.json")))["summary"] if os.path.exists(os.path.join(OUT, f"c5_{a.field}.json")) else {}
S = dict(field=a.field, cells=len(rows), skipped_large_cells=skipped, movement_mid=float(mov), S_single_field_witness_x=float(S_single), S_X_ceiling_x=prod.get("S_X_x"), S_separable_witness_x=prod.get("S_witness_x"),
         note="single-field witness: one Q_4 field per coarse-cell patch (the class certified in C5 is the product of per-patch box classes); cells > max_pix skipped (their C5 pairs were cropped)", seconds=time.time() - t0)
json.dump(dict(summary=S, cells=rows), open(os.path.join(OUT, f"c5_mapwitness_{a.field}.json"), "w"), indent=1, default=float); print(json.dumps(S, indent=1))
