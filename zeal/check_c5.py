"""check_c5.py -- independent V1/V2 checker for the map-level transport certificates (Supplementary Theorems S7, S10;
requirements V1-V2 of Supplementary Note 8).  Inputs: the artifacts of c5_production_duals.py (per pair: window, M, the stored dual vector of each endpoint) and the
global witness field of c5_global_witness_bf.py.  Nothing from the producers' LP construction is reused: the lifted constraint system
X^(m) (Theorem S7 (X1)-(X2); code M = 2m) is RECONSTRUCTED here from the statement of Theorem S7 and the first-order data (value boxes lo/hi, derivative
boxes G1/G2, pixel side h, the cell mask and the declared row-ordering convention), and every inequality is evaluated with outward arithmetic.
Stage 1 (every endpoint, deterministic, no sampling): with y >= 0 the stored dual, r = c - A'y, the bound  y'b + sum_i max(r_i lo_i, r_i hi_i)
is an upper bound of max{c'x : Ax <= b, lo <= x <= hi} for ANY y >= 0 (weak duality); it is evaluated with the proved forward-error
inflation gamma_k = k u/(1 - k u) times the absolute magnitudes (dot products of length k), and r_i is enclosed in [r_i - e_i, r_i + e_i]
before the max.  PASS iff the artifact's reported endpoint >= the checker's upward-rounded bound (the artifact is then a valid bound);
otherwise the checker's own bound is the certified value.  Map ceiling: S_X = sqrt(sum_a mu_a B_a^2) / sqrt(sum_a mu_a mid_a^2) with
B_a = max(|U_a|, |L_a|) from the checker bounds (upward) and the reference movement enclosed from below.
Stage 2 (global witness, V2): the stored global nodal field (M = 2) is checked against EVERY constraint of the global system exactly
(float comparisons with an exact fallback via Fractions on the near-boundary cases); if violations exist, the recovery problem
'largest feasible vector below the stored one' is solved by downward-rounded Bellman-Ford relaxation and re-verified; the witness
statistic is then evaluated with downward rounding on the same pair family.  Output: per-pair rows + map summary with the three
categories (model reconstruction / proof coverage / arithmetic certificate).
Usage: python check_c5.py --field georgia --M 4 [--witness C5/c5_global_field_georgia_M2.npz] --out <dir>
"""
try:
    from . import paths                     # imported as part of the package
except ImportError:
    import paths                            # flat import (this directory on sys.path)
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp
from fractions import Fraction
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--M", type=int, default=4); p.add_argument("--artifacts", default=None)
p.add_argument("--witness", default=None); p.add_argument("--out", default=None); p.add_argument("--max_pairs", type=int, default=0); a = p.parse_args()
EXP = paths.results("exp2")
ART = a.artifacts or os.path.join(EXP, "C5_duals"); OUT = a.out or os.path.join(EXP, "checker"); os.makedirs(OUT, exist_ok=True)
U = np.finfo(float).eps / 2            # unit roundoff
def gam(k): return k * U / (1 - k * U)

# ---------------------------------------------------------------- data (first-order boxes; same declared inputs as the producer)
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
ras = np.load(SPEC[a.field]); mask0 = ras["mask"].astype(bool) & np.isfinite(lo0); N = mask0.shape[0]; h = 2.0 / (N - 1)
w0 = np.where(mask0, ras["weight"], 0.0).astype(float); fine0 = np.where(mask0, ras["tract_idx"], -1); coarse0 = np.where(mask0, ras["county_idx"], -1)
lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask0, w0, fine0, coarse0 = (A.T.copy() for A in (lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask0, w0, fine0, coarse0))
lam0 = (lo0 + hi0) / 2


def lifted_system(lo, hi, g1l, g1h, g2l, g2h, mask, M):
    """Independent reconstruction of X^(m) (M = 2m) on a pixel window: returns (A csr, b, lo_n, hi_n, node_map, sub-pixel-centre index lists).
    Nodes: (n1 M + 1) x (n2 M + 1) grid points; a node is used iff some masked pixel's closed square contains it; its value bounds are the
    intersection of the value boxes of all masked pixels containing it.  Edges: unit mesh segments between adjacent used nodes that lie in
    the closed square of at least one masked pixel; the segment's derivative box is the intersection over all masked pixels containing it;
    constraint hs*g_lo <= x_head - x_tail <= hs*g_hi with hs = h/M.  Row ordering (declared convention of the producer): for masked pixels in
    row-major order, the pixel's vertical segments (a -> a+1, b fixed) then horizontal (b -> b+1), first occurrence wins; rows 0..R-1 are the
    upper constraints, R..2R-1 the lower ones."""
    n1, n2 = mask.shape; N1, N2 = n1 * M + 1, n2 * M + 1; hs = h / M
    node_lo = np.full((N1, N2), -np.inf); node_hi = np.full((N1, N2), np.inf)
    pi_, pj_ = np.nonzero(mask)
    for (i, j) in zip(pi_, pj_):
        sl = (slice(i * M, i * M + M + 1), slice(j * M, j * M + M + 1))
        node_lo[sl] = np.maximum(node_lo[sl], lo[i, j]); node_hi[sl] = np.minimum(node_hi[sl], hi[i, j])
    used = np.isfinite(node_lo); nid = -np.ones((N1, N2), np.int64); nid[used] = np.arange(used.sum()); nv = int(used.sum())
    # segment boxes: vertical segment (a,b)->(a+1,b) belongs to pixel (i,j) iff i*M <= a < i*M+M and j*M <= b <= j*M+M
    vlo = np.full((N1 - 1, N2), -np.inf); vhi = np.full((N1 - 1, N2), np.inf); hlo = np.full((N1, N2 - 1), -np.inf); hhi = np.full((N1, N2 - 1), np.inf)
    vseen = np.zeros((N1 - 1, N2), bool); hseen = np.zeros((N1, N2 - 1), bool); order = []
    for (i, j) in zip(pi_, pj_):
        sv = (slice(i * M, i * M + M), slice(j * M, j * M + M + 1)); sh = (slice(i * M, i * M + M + 1), slice(j * M, j * M + M))
        vlo[sv] = np.maximum(vlo[sv], g1l[i, j]); vhi[sv] = np.minimum(vhi[sv], g1h[i, j]); hlo[sh] = np.maximum(hlo[sh], g2l[i, j]); hhi[sh] = np.minimum(hhi[sh], g2h[i, j])
        for b_ in range(j * M, j * M + M + 1):
            for a_ in range(i * M, i * M + M):
                if not vseen[a_, b_]:
                    vseen[a_, b_] = True; order.append((0, a_, b_))
        for a_ in range(i * M, i * M + M + 1):
            for b_ in range(j * M, j * M + M):
                if not hseen[a_, b_]:
                    hseen[a_, b_] = True; order.append((1, a_, b_))
    # NOTE: the producer inserts, per pixel, all vertical segments (a in range, b in range) in a-major order then horizontal; replicate exactly
    order = []; vseen[:] = False; hseen[:] = False
    for (i, j) in zip(pi_, pj_):
        for a_ in range(i * M, i * M + M):
            for b_ in range(j * M, j * M + M + 1):
                if not vseen[a_, b_]:
                    vseen[a_, b_] = True; order.append((0, a_, b_))
        for a_ in range(i * M, i * M + M + 1):
            for b_ in range(j * M, j * M + M):
                if not hseen[a_, b_]:
                    hseen[a_, b_] = True; order.append((1, a_, b_))
    R = len(order); tail = np.empty(R, np.int64); head = np.empty(R, np.int64); glo = np.empty(R); ghi = np.empty(R)
    for r_, (t, a_, b_) in enumerate(order):
        if t == 0:
            tail[r_] = nid[a_, b_]; head[r_] = nid[a_ + 1, b_]; glo[r_] = vlo[a_, b_]; ghi[r_] = vhi[a_, b_]
        else:
            tail[r_] = nid[a_, b_]; head[r_] = nid[a_, b_ + 1]; glo[r_] = hlo[a_, b_]; ghi[r_] = hhi[a_, b_]
    assert (tail >= 0).all() and (head >= 0).all()
    rows = np.concatenate([np.arange(R), np.arange(R), R + np.arange(R), R + np.arange(R)]); cols = np.concatenate([head, tail, tail, head])
    vals = np.concatenate([np.ones(R), -np.ones(R), np.ones(R), -np.ones(R)])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * R, nv)); b = np.concatenate([hs * ghi, -hs * glo])
    return A, b, node_lo[used], node_hi[used], nid, (tail, head, glo, ghi, hs)


def outer_objective(d, mask, nid, M):
    """c_i: for pixel p, each of its m x m sub-pixel centres (mesh index (iM + 2k + 1, jM + 2l + 1)) gets d_p / m^2"""
    m = M // 2; c = np.zeros(int((nid >= 0).sum())); pi_, pj_ = np.nonzero(mask)
    for (i, j) in zip(pi_, pj_):
        for k in range(m):
            for l in range(m):
                c[nid[i * M + 2 * k + 1, j * M + 2 * l + 1]] += d[i, j] / (m * m)
    return c


def dual_bound_outward(A, b, lo_n, hi_n, c, y):
    """upper bound on max c'x over {Ax <= b, lo <= x <= hi} from y >= 0, with proved forward-error inflation (upward)"""
    assert (y >= 0).all()
    Aty = A.T @ y; absAty = abs(A).T @ np.abs(y); deg = np.asarray((abs(A) != 0).sum(0)).ravel()
    e_r = gam(deg.max() + 2) * (absAty + np.abs(c))                       # enclosure of r_i = c_i - (A'y)_i
    r = c - Aty; rlo = r - e_r; rhi = r + e_r
    hi_t = np.where(np.isfinite(hi_n), hi_n, 0.0); lo_t = np.where(np.isfinite(lo_n), lo_n, 0.0)
    if np.any((rhi > 0) & ~np.isfinite(hi_n)) or np.any((rlo < 0) & ~np.isfinite(lo_n)):
        return np.inf
    term = np.maximum.reduce([rlo * lo_t, rhi * lo_t, rlo * hi_t, rhi * hi_t])   # sup over the enclosure of r_i of max(r lo, r hi)
    yb = y * b; S = float(yb.sum() + term.sum()); mag = float(np.abs(yb).sum() + np.abs(term).sum())
    k = len(yb) + len(term)
    return S + gam(k + 4) * mag + 2 * gam(4) * float(np.abs(term).sum())    # products rounded + summation rounded (upward)


# ---------------------------------------------------------------- stage 1: every endpoint
art = json.load(open(os.path.join(ART, f"c5_{a.field}.json"), encoding="utf-8")); rows = art["pairs"]; M = a.M
DU = os.path.join(ART, f"duals_{a.field}_M{M}"); t0 = time.time(); out_rows = []; acc = dict(X=0.0, mid_lo=0.0); n_pass = 0; n_recon = 0
cache = {}
for k_, r in enumerate(rows[: a.max_pairs or None]):
    i0, i1, j0, j1 = r["window"]; sl = (slice(i0, i1), slice(j0, j1)); mk = (coarse0 == r["coarse"])[sl]
    key = (r["coarse"], i0, i1, j0, j1)
    if key not in cache:
        cache = {key: lifted_system(lo0[sl], hi0[sl], g1l0[sl], g1h0[sl], g2l0[sl], g2h0[sl], mk, M)}
    A, b, lo_n, hi_n, nid, _ = cache[key]
    recon_ok = (A.shape[1] == r["nv"] and A.shape[0] == r["rows"]); n_recon += int(recon_ok)
    wb = w0[sl] * mk; wa = wb * (fine0[sl] == r["fine"]); d2 = wa / wa.sum() - wb / wb.sum(); c = outer_objective(d2, mk, nid, M)
    du = np.load(os.path.join(DU, r["dual_file"])); res = dict(coarse=r["coarse"], fine=r["fine"], reconstruction_ok=bool(recon_ok))
    for nm, sgn in (("U", 1), ("L", -1)):
        y = np.zeros(A.shape[0]); y[du[f"{nm}_idx"]] = du[f"{nm}_val"]
        ub = dual_bound_outward(A, b, lo_n, hi_n, sgn * c, y) if recon_ok else np.inf
        chk = sgn * ub; art_v = r[f"{nm}_X"]
        ok = (art_v >= chk) if sgn == 1 else (art_v <= chk)
        res[f"{nm}_checker"] = float(chk); res[f"{nm}_artifact"] = float(art_v); res[f"{nm}_ok"] = bool(ok)
    # reference movement (midpoints) enclosed from below; checker's certified endpoints = the tighter of (artifact if ok, checker)
    d = d2[mk]; mid = float(d @ lam0[sl][mk]); e_mid = gam(len(d) + 1) * float(np.abs(d * lam0[sl][mk]).sum()); mid_lo = max(abs(mid) - e_mid, 0.0)
    Ucert = min(res["U_artifact"], res["U_checker"]) if res["U_ok"] else res["U_checker"]; Lcert = max(res["L_artifact"], res["L_checker"]) if res["L_ok"] else res["L_checker"]
    B = max(abs(Ucert), abs(Lcert)); acc["X"] += r["mu_a"] * B * B * (1 + 4 * U); acc["mid_lo"] += r["mu_a"] * mid_lo * mid_lo * (1 - 4 * U)
    res.update(U_certified=float(Ucert), L_certified=float(Lcert), mid_lower=mid_lo); n_pass += int(res["U_ok"] and res["L_ok"] and recon_ok); out_rows.append(res)
    if (k_ + 1) % 100 == 0:
        print(f"  {a.field} {k_+1} pairs: reconstruction ok {n_recon}, endpoint pass {n_pass}, S_X(checker) {np.sqrt(acc['X']/acc['mid_lo']):.4f} ({time.time()-t0:.0f}s)", flush=True)
S_X_chk = float(np.sqrt(acc["X"] / acc["mid_lo"]) * (1 + 8 * U))
summary = dict(field=a.field, M=M, pairs=len(out_rows), model_reconstruction_pass=n_recon, endpoint_pass=n_pass, S_X_checker_upper=S_X_chk, seconds=time.time() - t0,
               note="S_X_checker_upper = outward map ceiling from the checker-certified endpoints over the reference movement enclosed from below; compare with the artifact's S_X_x")
print(json.dumps(summary, indent=1)); json.dump(dict(summary=summary, pairs=out_rows), open(os.path.join(OUT, f"check_c5_{a.field}_M{M}.json"), "w"), indent=1, default=float)
