"""b11_tile_budget.py -- tile-budget ablation: when does the increment (gradient) information matter? Results exp1/B11.
Tile-budget ablation on the four ladders: the stored per-pixel first-order certificates are coarsened by k x k blocks
(value box = block min/max of lo/hi, gradient boxes = block min/max), which emulates a certifier run with k^2 fewer
tiles (conservatively: a real coarse tile is looser still). The pixel raster, the population allocation (uniform within
pixel) and the LP size are unchanged; only the boxes widen. For every nested pair: value-only V, propagated V*, the
programme U (tent increments + Lemma S6.1, verified dual), the gradient-only T, the Gamma criterion; map level: separable
bound S (RMS over pairs, population-weighted) for V and U vs the observed movement; U/V per pair by k.
Usage: python b11_tile_budget.py --field georgia [--ks 1,2,4,8,16] [--max_pairs 0] [--out <dir>]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import solve_max, lemma12, grid_poly, closure, mcshane, value_face
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--ks", default="1,2,4,8,16"); p.add_argument("--max_pairs", type=int, default=0)
p.add_argument("--out", default=None); p.add_argument("--seed", type=int, default=3); p.add_argument("--pairs_kvkg", default=""); a = p.parse_args()
OUT = a.out or paths.results("exp1", "B11"); os.makedirs(OUT, exist_ok=True)
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
ras = np.load(SPEC[a.field]); mask = ras["mask"].astype(bool) & np.isfinite(lo0); N = mask.shape[0]; h = 2.0 / (N - 1)
w = np.where(mask, ras["weight"], 0.0).astype(float); fine = np.where(mask, ras["tract_idx"], -1); coarse = np.where(mask, ras["county_idx"], -1)
lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, fine, coarse = (A.T.copy() for A in (lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, fine, coarse))   # axis 0 = x
lam = (lo0 + hi0) / 2
old = json.load(open(paths.transport_lp(a.field)))["summary"]


def block_boxes(k, kg=None):
    """coarsen boxes by k x k blocks (min of lower bounds, max of upper bounds over in-mask pixels of the block); kg: separate block size for the gradient boxes"""
    kv = k; kg = k if kg is None else kg
    def pool(A, fn, fill, k=kv):
        B = np.where(mask, A, fill); n1, n2 = B.shape; m1, m2 = -(-n1 // k), -(-n2 // k)
        P = np.full((m1 * k, m2 * k), fill); P[:n1, :n2] = B; P = P.reshape(m1, k, m2, k)
        R = fn(fn(P, axis=3), axis=1); return np.repeat(np.repeat(R, k, axis=0), k, axis=1)[:n1, :n2]
    return (pool(lo0, np.min, np.inf), pool(hi0, np.max, -np.inf), np.stack([pool(g1l0, np.min, np.inf, kg), pool(g1h0, np.max, -np.inf, kg)], -1), np.stack([pool(g2l0, np.min, np.inf, kg), pool(g2h0, np.max, -np.inf, kg)], -1))


rng = np.random.default_rng(a.seed); pairs = []
for b in np.unique(coarse[coarse >= 0]):
    for f_ in np.unique(fine[coarse == b]):
        if f_ >= 0 and w[(coarse == b) & (fine == f_)].sum() > 0:
            pairs.append((int(b), int(f_)))
if a.max_pairs and len(pairs) > a.max_pairs:
    pairs = [pairs[i] for i in rng.choice(len(pairs), a.max_pairs, replace=False)]
mu_tot = w.sum(); results = {}
grid = [(int(x), None) for x in a.ks.split(",")] if not a.pairs_kvkg else [tuple(int(v) for v in pr.split("x")) for pr in a.pairs_kvkg.split(",")]
for (k, kg) in grid:
    t0 = time.time(); lo, hi, G1, G2 = block_boxes(k, kg); lo_c, hi_c, _ = lemma12(lo, hi, G1, G2, h); key = f"{k}" if kg is None else f"v{k}g{kg}"
    cache = {}; rows = []; mov2 = U2 = V2 = Vs2 = 0.0
    for (b, f_) in pairs:
        if b not in cache:
            sel = coarse == b; i0, i1 = np.nonzero(sel.any(1))[0][[0, -1]]; j0, j1 = np.nonzero(sel.any(0))[0][[0, -1]]; sl = (slice(i0, i1 + 1), slice(j0, j1 + 1)); mk = sel[sl]
            poly, pix = grid_poly(lo_c[sl], hi_c[sl], G1[sl], G2[sl], h, mk); cache = {b: (sl, mk, poly)}
        sl, mk, poly = cache[b]
        wb = w[sl] * mk; wa = wb * (fine[sl] == f_); d2 = wa / wa.sum() - wb / wb.sum(); d = d2[mk]
        rp = solve_max(poly, d); rm = solve_max(poly, -d); U = max(rp["value"], rm["value"]); Uv = max(rp["verified"], rm["verified"])
        dp, dm = np.maximum(d, 0), np.maximum(-d, 0); V = max(dp @ poly.hi - dm @ poly.lo, dp @ (-poly.lo) - dm @ (-poly.hi))
        real = float(d @ lam[sl][mk]); mu_a = wa.sum()
        rows.append(dict(coarse=b, fine=f_, n_pix=int(mk.sum()), U=U, U_verified=Uv, V=V, real_mid=real, mu_a=float(mu_a)))
        mov2 += mu_a * real ** 2; U2 += mu_a * U ** 2; V2 += mu_a * V ** 2
    mov = np.sqrt(mov2 / mu_tot); ratio = np.array([r["U"] / r["V"] for r in rows if r["V"] > 1e-12])
    results[key] = dict(k=k, kg=kg, pairs=len(rows), tiles_equiv=int(mask.sum() / k / k), movement_mid=mov, S_U=float(np.sqrt(U2 / mu_tot)), S_V=float(np.sqrt(V2 / mu_tot)), S_U_x=float(np.sqrt(U2 / mu_tot) / mov), S_V_x=float(np.sqrt(V2 / mu_tot) / mov),
                      U_over_V_median=float(np.median(ratio)), U_over_V_q10=float(np.quantile(ratio, 0.1)), U_over_V_q90=float(np.quantile(ratio, 0.9)), share_U_below_V_5pct=float(np.mean(ratio < 0.95)),
                      median_U_over_real=float(np.median([r["U"] / max(abs(r["real_mid"]), 1e-9) for r in rows])), verified_ok=int(sum(r["U_verified"] >= r["U"] - 1e-12 for r in rows)), seconds=time.time() - t0)
    R_ = results[key]; print(f"{a.field} {key} (~{R_['tiles_equiv']} value tiles): S_U {R_['S_U_x']:.3f}x  S_V {R_['S_V_x']:.3f}x  U/V median {R_['U_over_V_median']:.4f} q10 {R_['U_over_V_q10']:.4f}  "
          f"share<0.95: {R_['share_U_below_V_5pct']:.2f}  U/|real| med {R_['median_U_over_real']:.2f}  {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(field=a.field, old_summary=old, by_k=results), open(os.path.join(OUT, f"b11_{a.field}" + ("_kvkg" if a.pairs_kvkg else "") + ".json"), "w"), indent=1, default=float)
