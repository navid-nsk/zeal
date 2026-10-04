"""check_c5_witness.py -- stage 2 of the V1/V2 map checker (V2, Supplementary Note 8): the stored GLOBAL witness field (one nodal vector of
the lifted class on the whole raster, M = 2, from c5_global_witness_bf.py) is verified against EVERY constraint of the independently
reconstructed global system (lifted_system, same declared construction as check_c5.py); near-boundary comparisons are decided exactly with
Fractions.  If violations exist, the recovery problem 'largest feasible vector below the stored one' is solved by downward-rounded
Bellman-Ford relaxation (x_j <- min(x_j, x_i + c_ij), each update rounded down two ulps) and re-verified exactly; the repaired vector is the
witness.  The witness map statistic is then evaluated with downward rounding on the C5 pair family (bilinear pixel means of the Q_2 field:
each pixel's four sub-pixel corner nodes receive d_p/(4 M^2)), and the safe bracket [L_witness/mov_hi, U_ceiling/mov_lo] is reported
together with the stage-1 ceiling.  Usage: python check_c5_witness.py --field georgia [--ceiling checker/check_c5_georgia_M4.json]
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
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--ceiling", default=None); p.add_argument("--out", default=None); a = p.parse_args()
EXP = paths.results("exp2"); OUT = a.out or os.path.join(EXP, "checker"); os.makedirs(OUT, exist_ok=True)
U = np.finfo(float).eps / 2
def gam(k): return k * U / (1 - k * U)
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



# ---------------------------------------------------------------- global system and stored field
Z = np.load(os.path.join(EXP, "C5", f"c5_global_field_{a.field}_M2.npz")); x = Z["x"].astype(float); I0, J0, M = int(Z["I0"]), int(Z["J0"]), int(Z["M"])
ii, jj = np.nonzero(mask0); I1, J1 = ii.max() + 1, jj.max() + 1; assert I0 == ii.min() and J0 == jj.min()
sl = (slice(I0, I1), slice(J0, J1)); mk = mask0[sl]; t0 = time.time()
A, b, lo_n, hi_n, nid, (tail, head, glo, ghi, hs) = lifted_system(lo0[sl], hi0[sl], g1l0[sl], g1h0[sl], g2l0[sl], g2h0[sl], mk, M)
recon_ok = (A.shape[1] == len(x)); print(f"global system: nodes {A.shape[1]} (stored {len(x)}), rows {A.shape[0]}, reconstruction_ok={recon_ok} ({time.time()-t0:.0f}s)", flush=True)
R = len(tail); c_up = b[:R]; c_lo = -b[R:]            # x_head - x_tail <= c_up ;  x_head - x_tail >= c_lo


def exact_violations(xv):
    """rows with x_head - x_tail > c_up or < c_lo, and bound violations, decided exactly (float compare, Fraction fallback near the boundary)"""
    diff = xv[head] - xv[tail]; m_ = 4 * U * (np.abs(xv[head]) + np.abs(xv[tail]) + 1e-300)
    sure_up = diff > c_up + m_; sure_lo = diff < c_lo - m_; amb = (~sure_up) & (~sure_lo) & ((diff > c_up - m_) | (diff < c_lo + m_))
    viol_up = sure_up.copy(); viol_lo = sure_lo.copy()
    for k_ in np.nonzero(amb)[0]:
        dq = Fraction(float(xv[head[k_]])) - Fraction(float(xv[tail[k_]]))
        if dq > Fraction(float(c_up[k_])):
            viol_up[k_] = True
        if dq < Fraction(float(c_lo[k_])):
            viol_lo[k_] = True
    bviol = (xv < lo_n) | (xv > hi_n)
    return viol_up, viol_lo, bviol, float(np.max(np.r_[diff - c_up, c_lo - diff, lo_n - xv, xv - hi_n]))


vu, vl, bv, maxv = exact_violations(x); n_viol = int(vu.sum() + vl.sum() + bv.sum())
print(f"stored field: exact violations {n_viol} (rows up {int(vu.sum())}, rows lo {int(vl.sum())}, bounds {int(bv.sum())}), max violation {maxv:.3e}", flush=True)
repaired = False
if n_viol:
    # recovery (V2): a feasible point within t of x -- the lattice top of {phi <= min(x + t, hi), increments} (downward-rounded Bellman-Ford),
    # accepted iff it is exactly feasible (in particular >= lo); t is increased on a grid until acceptance
    feasible = False; x_w = x
    for t_ in (2e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        xr = np.minimum(x + t_, hi_n).copy(); it = 0
        while True:
            it += 1; new = xr.copy()
            cand = np.nextafter(np.nextafter(xr[tail] + c_up, -np.inf), -np.inf); np.minimum.at(new, head, cand)
            cand2 = np.nextafter(np.nextafter(xr[head] - c_lo, -np.inf), -np.inf); np.minimum.at(new, tail, cand2)
            if np.array_equal(new, xr) or it > 500:
                break
            xr = new
        vu2, vl2, bv2, maxv2 = exact_violations(xr); nv2 = int(vu2.sum()) + int(vl2.sum()) + int(bv2.sum())
        print(f"recovery t={t_:.0e}: {it} passes, exact violations after repair {nv2}, below-lo nodes {int((xr < lo_n).sum())}, max |x - x_repaired| = {np.max(np.abs(x - xr)):.3e}", flush=True)
        if nv2 == 0:
            feasible = True; x_w = xr; repaired = True; break
else:
    feasible = True; x_w = x
if not feasible:
    print('recovery failed: no exactly feasible point found on the t grid', flush=True)
# ---------------------------------------------------------------- witness statistic on the C5 pair family (bilinear pixel means), downward
num_lo = 0.0; mov_lo2 = 0.0; mov_hi2 = 0.0; n_pairs = 0; Mq = M
for bcell in np.unique(coarse0[coarse0 >= 0]):
    selc = (coarse0 == bcell)[sl]; wb = w0[sl] * selc
    fines = [f_ for f_ in np.unique(fine0[sl][selc]) if f_ >= 0]
    pi_, pj_ = np.nonzero(selc)
    for f_ in fines:
        wa = wb * (fine0[sl] == f_)
        if wa.sum() <= 0 or (fine0[sl][selc] == f_).sum() >= selc.sum():
            continue
        d2 = wa / wa.sum() - wb / wb.sum(); dv = d2[pi_, pj_]; nz = dv != 0
        val = 0.0; mag = 0.0
        for (i, j, dp) in zip(pi_[nz], pj_[nz], dv[nz]):
            s_ = 0.0
            for aa in range(i * Mq, i * Mq + Mq):
                for bb in range(j * Mq, j * Mq + Mq):
                    s_ += x_w[nid[aa, bb]] + x_w[nid[aa + 1, bb]] + x_w[nid[aa, bb + 1]] + x_w[nid[aa + 1, bb + 1]]
            val += dp * s_ / (4 * Mq * Mq); mag += abs(dp) * abs(s_) / (4 * Mq * Mq)
        e_v = gam(4 * Mq * Mq * int(nz.sum()) + 4) * mag; v_lo = max(abs(val) - e_v, 0.0)
        mid = float(dv @ lam0[sl][pi_, pj_]); e_m = gam(len(dv) + 1) * float(np.abs(dv * lam0[sl][pi_, pj_]).sum())
        mu_a = wa.sum(); num_lo += mu_a * v_lo * v_lo * (1 - 4 * U); mov_lo2 += mu_a * max(abs(mid) - e_m, 0) ** 2 * (1 - 4 * U); mov_hi2 += mu_a * (abs(mid) + e_m) ** 2 * (1 + 4 * U); n_pairs += 1
S_wit_lower = float(np.sqrt(num_lo / mov_hi2) * (1 - 8 * U))
res = dict(field=a.field, M=M, nodes=int(len(x)), reconstruction_ok=bool(recon_ok), stored_violations=n_viol, max_violation_stored=maxv, repaired=repaired, witness_feasible_exact=bool(feasible),
           pairs=n_pairs, S_witness_lower=S_wit_lower, seconds=time.time() - t0)
if a.ceiling and os.path.exists(a.ceiling):
    cj = json.load(open(a.ceiling))["summary"]; res["S_ceiling_upper"] = cj["S_X_checker_upper"]; res["bracket"] = [S_wit_lower, cj["S_X_checker_upper"]]
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(OUT, f"check_c5_witness_{a.field}.json"), "w"), indent=1, default=float)
