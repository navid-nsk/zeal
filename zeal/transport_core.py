"""transport_core.py -- the certificate transport programme (Supplementary Theorems S2-S8) as a library.

Objects
  Poly  : the polytope Phi = {lo <= phi <= hi, cm <= phi[eq] - phi[ep] <= cp}  (Theorem S2)
  solve_max(poly, d)       : U = max d.phi with a VERIFIED dual upper bound (V1)
  closure(poly)            : D(q,p) = shortest path avoiding the ground (signed quasi-metric; Theorem S3)
  mcshane(poly, D)         : hi*, lo*  (top and bottom of the lattice)
  closed_cost(D, hs, ls)   : Dbar = min{D, hi*_p - lo*_q}
  ot_cost(C, src, snk)     : optimal transport value (Theorem S3 (iii))
  value_face(...)          : Gamma, V*, V  (Theorem S4 (i))
  gradient_only(poly, d)   : T  (Theorem S4 (ii)), with the per-component zero-mass test
  prop_b(poly, d, psi)     : slack attribution (Theorem S5 (i))
  project_to_phi(poly, lam): nearest (L1) member of Phi to a candidate field
  lemma12(...)             : value-gradient coupling (Lemma S6.1), uniform-density form
  inner_q1(...)            : Q1 realizable inner LP (Theorem S6)
  outer_sub(...)           : sub-pixel outer LP with optional coupling
  r1_1d(...)               : 1-D gradient-only bracket and the h/8 bound (Theorem S8)

Conventions: arcs (ep -> eq) carry phi[eq] - phi[ep] in [cm, cp]; increments are tent-weighted,
cp = (h/2)(g+_p + g+_q) under uniform within-pixel density (Theorem S2).
"""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.csgraph import floyd_warshall, johnson, connected_components

EPS = np.finfo(float).eps


class Poly:
    def __init__(self, lo, hi, ep, eq, cm, cp):
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float)
        self.ep = np.asarray(ep, np.int64); self.eq = np.asarray(eq, np.int64)
        self.cm = np.asarray(cm, float); self.cp = np.asarray(cp, float)
        self.n = len(self.lo); self.E = len(self.ep)
        m = self.E; rows = np.repeat(np.arange(2 * m), 2)
        cols = np.concatenate([np.stack([self.eq, self.ep], 1).ravel(), np.stack([self.ep, self.eq], 1).ravel()])
        vals = np.concatenate([np.tile([1.0, -1.0], m), np.tile([1.0, -1.0], m)])
        self.A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * m, self.n))      # rows: phi_q - phi_p <= cp ; phi_p - phi_q <= -cm
        self.b = np.concatenate([self.cp, -self.cm])

    def feasible(self, phi, tol=1e-9):
        return bool(np.all(phi >= self.lo - tol) and np.all(phi <= self.hi + tol) and np.all(self.A @ phi <= self.b + tol))

    def shifted(self, psi):
        """the polytope of slacks around psi in Phi (Theorem S5 (i))"""
        inc = psi[self.eq] - psi[self.ep]
        return Poly(self.lo - psi, self.hi - psi, self.ep, self.eq, self.cm - inc, self.cp - inc)

    def components(self):
        G = sp.csr_matrix((np.ones(self.E), (self.ep, self.eq)), shape=(self.n, self.n))
        k, lab = connected_components(G, directed=False); return k, lab


def verified_upper(poly, d, y):
    """Dual feasible reconstruction: y >= 0 on the 2E inequality rows; residual r = d - A^T y is split into
    z+ (on hi) and z- (on lo).  Bound = b.y + hi.z+ - lo.z- is a valid upper bound on max d.phi for ANY y >= 0.
    Rounding: every sum of N terms is widened by N*eps*sum|terms| (outward rounding, V1)."""
    y = np.maximum(np.asarray(y, float), 0.0)
    r = d - poly.A.T @ y
    zp = np.maximum(r, 0.0); zm = np.maximum(-r, 0.0)
    terms = np.concatenate([poly.b * y, poly.hi * zp, -poly.lo * zm])
    val = float(np.sum(terms)); err = (len(terms) + poly.n + 2 * poly.E) * EPS * float(np.sum(np.abs(terms)) + np.sum(np.abs(poly.A.T @ y)) + np.sum(np.abs(d)))
    return val + err


def solve_max(poly, d, bounds=True):
    """max d.phi over Phi. Returns dict(value, phi, verified) where verified >= value is a dual-certified upper bound."""
    bnds = list(zip(poly.lo, poly.hi)) if bounds else [(None, None)] * poly.n
    res = linprog(-np.asarray(d, float), A_ub=poly.A, b_ub=poly.b, bounds=bnds, method="highs")
    if res.status == 3:       # unbounded
        return dict(value=np.inf, phi=None, verified=np.inf, status=res.status)
    if res.status != 0:
        return dict(value=np.nan, phi=None, verified=np.nan, status=res.status)
    y = -res.ineqlin.marginals          # HiGHS marginals of A_ub rows are <= 0 for a max problem posed as min(-d.phi)
    ver = verified_upper(poly, d, y) if bounds else np.nan
    return dict(value=-res.fun, phi=res.x, verified=ver, status=0)


def closure(poly, method="auto"):
    """D[s, p] = shortest-path distance s -> p (bounds phi_p - phi_s), avoiding the ground. inf if unreachable.
    Negative arcs allowed (signed certificates); raises if a negative cycle exists (Phi empty)."""
    n = poly.n
    rows = np.concatenate([poly.ep, poly.eq]); cols = np.concatenate([poly.eq, poly.ep])
    w = np.concatenate([poly.cp, -poly.cm])
    # parallel arcs: keep the minimum
    order = np.lexsort((w, cols, rows)); rows, cols, w = rows[order], cols[order], w[order]
    keep = np.ones(len(rows), bool); keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    rows, cols, w = rows[keep], cols[keep], w[keep]
    if method == "fw" or (method == "auto" and n <= 400):
        W = np.full((n, n), np.inf); np.fill_diagonal(W, 0.0); W[rows, cols] = np.minimum(W[rows, cols], w)
        try:
            D = floyd_warshall(W, directed=True)
        except Exception as e:   # NegativeCycleError
            raise ValueError(f"negative cycle: Phi is empty ({e})")
        if np.any(np.diag(D) < -1e-12):
            raise ValueError("negative cycle: Phi is empty")
        return D
    G = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    try:
        D = johnson(G, directed=True)
    except Exception as e:   # NegativeCycleError
        raise ValueError(f"negative cycle: Phi is empty ({e})")
    return D


def mcshane(poly, D):
    hs = np.min(poly.hi[:, None] + D, axis=0)        # hi*_p = min_s hi_s + D(s,p)
    ls = np.max(poly.lo[None, :] - D, axis=1)        # lo*_q = max_r lo_r - D(q,r)
    return hs, ls


def closed_cost(D, hs, ls):
    return np.minimum(D, hs[None, :] - ls[:, None])  # C[q, p]


def ot_cost(C, src, snk, tol=1e-15):
    """min sum pi(q,p) C[q,p] over couplings of src (on q) and snk (on p); masses are balanced by the caller."""
    S = np.where(src > tol)[0]; T = np.where(snk > tol)[0]
    if len(S) == 0 or len(T) == 0:
        return 0.0
    Csub = C[np.ix_(S, T)]
    if not np.all(np.isfinite(Csub)):
        return np.inf
    nS, nT = len(S), len(T)
    ridx = np.concatenate([np.repeat(np.arange(nS), nT), nS + np.tile(np.arange(nT), nS)])
    cidx = np.concatenate([np.arange(nS * nT), np.arange(nS * nT)])
    Aeq = sp.csr_matrix((np.ones(2 * nS * nT), (ridx, cidx)), shape=(nS + nT, nS * nT))
    beq = np.concatenate([src[S], snk[T]]); beq[:nS] *= beq[nS:].sum() / beq[:nS].sum()   # exact balance
    res = linprog(Csub.ravel(), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    if res.status != 0:
        return np.nan
    return float(res.fun)


def value_face(poly, D, hs, ls, d):
    dp, dm = np.maximum(d, 0), np.maximum(-d, 0)
    Vstar = float(dp @ hs - dm @ ls); V = float(dp @ poly.hi - dm @ poly.lo)
    Sm, Sp = np.where(dm > 0)[0], np.where(dp > 0)[0]
    Gam = float(np.max(hs[None, Sp] - ls[Sm, None] - D[np.ix_(Sm, Sp)])) if len(Sm) and len(Sp) else -np.inf
    return Gam, Vstar, V


def gradient_only(poly, d):
    """T = max d.phi with increments only. Finite iff d has zero mass on every connected component (Theorem S4 (ii))."""
    k, lab = poly.components()
    mass = np.array([d[lab == c].sum() for c in range(k)])
    zero_mass = bool(np.all(np.abs(mass) < 1e-12 * max(1.0, np.abs(d).sum())))
    r = solve_max(poly, d, bounds=False)
    return r["value"], zero_mass


def prop_b(poly, d, psi, D=None):
    """U = d.psi + OT(d-, d+; Dbar_psi) with Dbar_psi the closed cost of the slack polytope (Theorem S5 (i))."""
    sh = poly.shifted(psi)
    assert np.all(sh.lo <= 1e-9) and np.all(sh.hi >= -1e-9) and np.all(sh.cm <= 1e-9) and np.all(sh.cp >= -1e-9), "psi not in Phi"
    Ds = closure(sh); hs, ls = mcshane(sh, Ds); Cs = closed_cost(Ds, hs, ls)
    dp, dm = np.maximum(d, 0), np.maximum(-d, 0)
    slack = ot_cost(Cs, dm, dp)
    Sm, Sp = np.where(dm > 0)[0], np.where(dp > 0)[0]
    tv = dp.sum()
    bound_tv = tv * (np.max(hs[Sp]) + np.max(-ls[Sm])) if len(Sp) and len(Sm) else np.nan
    return dict(d_psi=float(d @ psi), slack_transport=slack, bound_tv=float(bound_tv), min_slack_cost=float(np.min(Cs[np.ix_(Sm, Sp)])) if len(Sm) and len(Sp) else np.nan)


def project_to_phi(poly, lam):
    """argmin ||psi - lam||_1 over Phi (LP with split variables). Returns psi in Phi."""
    n = poly.n
    # variables: psi (n), t (n) with t >= |psi - lam|
    c = np.concatenate([np.zeros(n), np.ones(n)])
    I = sp.identity(n, format="csr")
    A1 = sp.hstack([I, -I]); A2 = sp.hstack([-I, -I])
    A = sp.vstack([sp.hstack([poly.A, sp.csr_matrix((2 * poly.E, n))]), A1, A2]).tocsr()
    b = np.concatenate([poly.b, lam, -lam])
    bnds = list(zip(poly.lo, poly.hi)) + [(0, None)] * n
    res = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method="highs")
    assert res.status == 0, res.message
    return res.x[:n]


def lemma12(lo, hi, G1, G2, h, mask=None):
    """Lemma S6.1 (uniform density): mean in [lo + (h/2) delta, hi - (h/2) delta], delta = dist(0,G1)+dist(0,G2)."""
    d1 = np.maximum(G1[..., 0], 0) + np.maximum(-G1[..., 1], 0)
    d2 = np.maximum(G2[..., 0], 0) + np.maximum(-G2[..., 1], 0)
    sh = (h / 2) * (d1 + d2)
    lo2, hi2 = lo + sh, hi - sh
    bad = lo2 > hi2
    lo2 = np.where(bad, lo, lo2); hi2 = np.where(bad, hi, hi2)     # never cross (cannot happen for sound boxes)
    return lo2, hi2, sh


# ---------------------------------------------------------------- grid instances (synthetic and real patches)

def grid_poly(lo, hi, G1, G2, h, mask=None, neigh=4):
    """Polytope for an n1 x n2 pixel grid (arrays [n1, n2]; G1/G2 [n1, n2, 2]); axis 0 = x (columns i), axis 1 = y.
    neigh = 4: tent increments to right and down neighbours; neigh = 8: adds diagonal arcs with the exact
    diagonal tent weights (1/3, 1/3, 1/6, 1/6) over the 2x2 block (Theorem S4 (v), 8-neighbour option).
    Returns Poly over in-mask pixels and the index map."""
    n1, n2 = lo.shape
    if mask is None:
        mask = np.ones((n1, n2), bool)
    pix = np.full((n1, n2), -1, np.int64); ii, jj = np.nonzero(mask); pix[ii, jj] = np.arange(len(ii))
    ep, eq, cm, cp = [], [], [], []
    def add(p, q, lo_, hi_):
        ep.append(p); eq.append(q); cm.append(lo_); cp.append(hi_)
    for i, j in zip(ii, jj):
        p = pix[i, j]
        if i + 1 < n1 and mask[i + 1, j]:
            q = pix[i + 1, j]; add(p, q, h * (G1[i, j, 0] + G1[i + 1, j, 0]) / 2, h * (G1[i, j, 1] + G1[i + 1, j, 1]) / 2)
        if j + 1 < n2 and mask[i, j + 1]:
            q = pix[i, j + 1]; add(p, q, h * (G2[i, j, 0] + G2[i, j + 1, 0]) / 2, h * (G2[i, j, 1] + G2[i, j + 1, 1]) / 2)
        if neigh == 8:
            for (di, dj) in ((1, 1), (1, -1)):
                i2, j2 = i + di, j + dj
                if not (0 <= i2 < n1 and 0 <= j2 < n2 and mask[i2, j2] and mask[i2, j] and mask[i, j2]):
                    continue
                q = pix[i2, j2]
                wts = [((i, j), 1 / 3), ((i2, j2), 1 / 3), ((i2, j), 1 / 6), ((i, j2), 1 / 6)]
                # increment phi_q - phi_p = (1/h^2) int_p int_0^1 grad f(x + t v).v dt dx, v = (di h, dj h)
                up = h * sum(w_ * (G1[a, b, 1] if di > 0 else -G1[a, b, 0]) for (a, b), w_ in wts) + h * sum(w_ * (G2[a, b, 1] if dj > 0 else -G2[a, b, 0]) for (a, b), w_ in wts)
                lo_ = h * sum(w_ * (G1[a, b, 0] if di > 0 else -G1[a, b, 1]) for (a, b), w_ in wts) + h * sum(w_ * (G2[a, b, 0] if dj > 0 else -G2[a, b, 1]) for (a, b), w_ in wts)
                add(p, q, lo_, up)
    return Poly(lo[mask], hi[mask], ep, eq, cm, cp), pix


def inner_q1(lo, hi, G1, G2, h, d, m, mask=None, verified=False):
    """Theorem S6: max d.<f> over Q1 fields on the m-refined mesh with edge slopes in the pixel box and nodal values
    in every adjacent pixel's value box. Returns (value, nodal values)."""
    n1, n2 = lo.shape
    if mask is None:
        mask = np.ones((n1, n2), bool)
    N1, N2 = n1 * m + 1, n2 * m + 1; nid = lambda a, b: a * N2 + b; V = N1 * N2
    lo_n = np.full(V, -np.inf); hi_n = np.full(V, np.inf); obj = np.zeros(V)
    rp, rm, rhs = [], [], []
    hs = h / m
    for i in range(n1):
        for j in range(n2):
            if not mask[i, j]:
                continue
            for a in range(i * m, i * m + m + 1):
                for b in range(j * m, j * m + m + 1):
                    k = nid(a, b); lo_n[k] = max(lo_n[k], lo[i, j]); hi_n[k] = min(hi_n[k], hi[i, j])
            for a in range(i * m, i * m + m):
                for b in range(j * m, j * m + m):
                    for (k1, k2, G) in [(nid(a, b), nid(a + 1, b), G1[i, j]), (nid(a, b + 1), nid(a + 1, b + 1), G1[i, j]),
                                        (nid(a, b), nid(a, b + 1), G2[i, j]), (nid(a + 1, b), nid(a + 1, b + 1), G2[i, j])]:
                        rp.append(k2); rm.append(k1); rhs.append(hs * G[1])
                        rp.append(k1); rm.append(k2); rhs.append(-hs * G[0])
                    for k in (nid(a, b), nid(a + 1, b), nid(a, b + 1), nid(a + 1, b + 1)):
                        obj[k] += d[i, j] / (4 * m * m)
    R = len(rp); A = sp.csr_matrix((np.concatenate([np.ones(R), -np.ones(R)]), (np.concatenate([np.arange(R), np.arange(R)]), np.concatenate([rp, rm]))), shape=(R, V))
    lo_n[~np.isfinite(lo_n)] = -1e6; hi_n[~np.isfinite(hi_n)] = 1e6
    res = linprog(-obj, A_ub=A, b_ub=np.array(rhs), bounds=list(zip(lo_n, hi_n)), method="highs")
    assert res.status == 0, res.message
    if verified:
        # verified upper bound on the INNER LP optimum (dual reconstruction as in verified_upper); witness value = -res.fun
        y = np.maximum(-res.ineqlin.marginals, 0.0); r = obj - A.T @ y; zp = np.maximum(r, 0); zm = np.maximum(-r, 0)
        terms = np.concatenate([np.array(rhs) * y, hi_n * zp, -lo_n * zm]); ub = float(np.sum(terms)) + (len(terms) + V + R) * EPS * float(np.sum(np.abs(terms)) + np.sum(np.abs(A.T @ y)) + np.sum(np.abs(obj)))
        return -res.fun, res.x, ub
    return -res.fun, res.x


def eps_m(d, G1, G2, h, m, mask=None, weighted=False):
    """interpolation error of the Q1 hierarchy (Theorem S6): U_{Q_m} <= U^F <= U_{Q_m} + eps_m,
    eps_m = (h/(8m)) sum |d_p| (w1p + w2p) under uniform allocation; h/(4m) for arbitrary declared allocation."""
    w1 = G1[..., 1] - G1[..., 0]; w2 = G2[..., 1] - G2[..., 0]
    if mask is None:
        mask = np.ones(d.shape, bool)
    c = h / (4 * m) if weighted else h / (8 * m)
    return float(c * np.sum(np.abs(d[mask]) * (w1[mask] + w2[mask])))


def exact_1d(gm, gp, d, h):
    """Theorem S8: exact gradient-only 1-D support functions (uniform intervals, no value boxes, sum d = 0).
    b_0 = 0, b_i = -sum_{j<=i} d_j; s_G(z) = sup_{g in G} z g.
    U^F = h sum_i int_0^1 s_{G_i}((1-t) b_{i-1} + t b_i) dt ;  U^Phi = (h/2) sum_i [s_{G_i}(b_{i-1}) + s_{G_i}(b_i)] ;
    gap = (h/2) sum_{b_{i-1} b_i < 0} w_i |b_{i-1} b_i| / (|b_{i-1}| + |b_i|)."""
    N = len(d); b = np.concatenate([[0.0], -np.cumsum(d)]); assert abs(b[-1]) < 1e-9 * (1 + np.abs(d).sum()), "sum d must be 0"
    sG = lambda i, z: gp[i] * z if z >= 0 else gm[i] * z
    UF = 0.0; UPhi = 0.0; gap = 0.0
    for i in range(N):
        b0, b1 = b[i], b[i + 1]
        UPhi += h / 2 * (sG(i, b0) + sG(i, b1))
        if b0 * b1 < 0:
            t0 = b0 / (b0 - b1)           # zero crossing of the linear coefficient
            UF += h * (0.5 * t0 * sG(i, b0) + 0.5 * (1 - t0) * sG(i, b1))
            gap += h / 2 * (gp[i] - gm[i]) * abs(b0 * b1) / (abs(b0) + abs(b1))
        else:
            UF += h * 0.5 * (sG(i, b0) + sG(i, b1))
    return dict(UF=UF, UPhi=UPhi, gap_formula=gap, gap_direct=UPhi - UF)


def outer_sub(lo, hi, G1, G2, h, d, m, couple=True, mask=None, neigh=4):
    """sub-pixel outer LP (m x m sub-pixels per pixel, tent increments, sub-pixel means in the pixel value box
    tightened by Lemma S6.1 at the sub-pixel scale if couple)."""
    n1, n2 = lo.shape
    if mask is None:
        mask = np.ones((n1, n2), bool)
    rep = lambda A: np.repeat(np.repeat(A, m, axis=0), m, axis=1)
    lo_s, hi_s = rep(lo), rep(hi); G1s, G2s = rep(G1), rep(G2); ms = rep(mask)
    if couple:
        lo_s, hi_s, _ = lemma12(lo_s, hi_s, G1s, G2s, h / m)
    poly, pix = grid_poly(lo_s, hi_s, G1s, G2s, h / m, ms, neigh=neigh)
    ds = rep(d) / (m * m)
    r = solve_max(poly, ds[ms])
    return r["value"], r["verified"]


def r1_1d(gm, gp, d, h, m_inner=32):
    """Theorem S8 in 1-D: outer pixel LP (tent increments, no value boxes), inner P1 LP on m-refined mesh, and the
    h/8 bound at the sign changes of d.  Returns dict."""
    N = len(gm)
    def outer():
        ep = np.arange(N - 1); eq = ep + 1
        poly = Poly(np.full(N, -1e6), np.full(N, 1e6), ep, eq, h * (gm[:-1] + gm[1:]) / 2, h * (gp[:-1] + gp[1:]) / 2)
        r = solve_max(poly, d); return r["value"], r["phi"], poly
    def inner(m):
        M = N * m; hs = h / m
        ep = np.arange(M); eq = ep + 1; p = ep // m
        poly = Poly(np.full(M + 1, -1e6), np.full(M + 1, 1e6), ep, eq, hs * gm[p], hs * gp[p])
        obj = np.zeros(M + 1); np.add.at(obj, ep, d[p] / (2 * m)); np.add.at(obj, eq, d[p] / (2 * m))
        r = solve_max(poly, obj); return r["value"]
    U_out, phi, poly = outer(); U_in = inner(m_inner)
    w = gp - gm
    bound = h / 8 * np.sum(np.abs(d) * w)
    # kinks of the optimal outer potential: both adjacent increments tight at opposite bounds (peak or valley)
    inc = np.diff(phi); tol = 1e-7 * (1 + np.abs(poly.cp).max())
    tight_hi = inc >= poly.cp - tol; tight_lo = inc <= poly.cm + tol
    kinks = [p for p in range(1, N - 1) if (tight_hi[p - 1] and tight_lo[p]) or (tight_lo[p - 1] and tight_hi[p])]
    bound_kinks = h / 8 * sum(abs(d[p]) * w[p] for p in kinks)
    return dict(outer=U_out, inner=U_in, gap=U_out - U_in, bound=bound, ratio=(U_out - U_in) / bound if bound > 0 else np.nan,
                kinks=kinks, bound_kinks=bound_kinks, ratio_kinks=(U_out - U_in) / bound_kinks if bound_kinks > 0 else np.nan)


def saving_identity(poly, d, psi, D=None):
    """Exact transport-saving identity (Theorem S5 (ii)): with A_p = u*_p - psi_p, B_q = psi_q - l*_q and the increment-only slack
    distance D_psi(q,p) = D(q,p) - (psi_p - psi_q) >= 0,  V*(d) - U(d) = max_{pi in Pi(d-,d+)} sum pi_qp [A_p + B_q - D_psi(q,p)]_+ .
    Returns dict with V*, U, the saving, the saving-maximizing coupling's benefiting mass share and mean saving per benefiting unit mass."""
    if D is None:
        D = closure(poly)
    hs, ls = mcshane(poly, D); A = hs - psi; B = psi - ls
    dp, dm = np.maximum(d, 0), np.maximum(-d, 0); S = np.where(dm > 1e-15)[0]; T = np.where(dp > 1e-15)[0]
    Dpsi = D[np.ix_(S, T)] - (psi[T][None, :] - psi[S][:, None])
    gain = np.maximum(A[T][None, :] + B[S][:, None] - Dpsi, 0.0); gain[~np.isfinite(Dpsi)] = 0.0
    nS, nT = len(S), len(T)
    ridx = np.concatenate([np.repeat(np.arange(nS), nT), nS + np.tile(np.arange(nT), nS)]); cidx = np.concatenate([np.arange(nS * nT), np.arange(nS * nT)])
    Aeq = sp.csr_matrix((np.ones(2 * nS * nT), (ridx, cidx)), shape=(nS + nT, nS * nT)); beq = np.concatenate([dm[S], dp[T]]); beq[:nS] *= beq[nS:].sum() / beq[:nS].sum()
    res = linprog(-gain.ravel(), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    pi = res.x.reshape(nS, nT); saving = -res.fun
    Vstar = float(dp @ hs - dm @ ls); U = solve_max(poly, d)["value"]
    benefit = pi[gain > 1e-12].sum(); md = dp.sum()
    return dict(Vstar=Vstar, U=U, saving_identity=float(saving), saving_direct=float(Vstar - U), identity_residual=float(abs(Vstar - U - saving)),
                mass_benefiting_share=float(benefit / md), mean_saving_per_benefiting_mass=float(saving / benefit) if benefit > 0 else 0.0,
                value_slack_scale=float((Vstar - d @ psi) / md), normalized_saving=float((Vstar - U) / max(Vstar - d @ psi, 1e-15)))


def verified_solve_system(A, b, lo_n, hi_n, obj):
    """max obj.x s.t. A x <= b, lo_n <= x <= hi_n (bounds may be infinite) with a verified dual upper bound (V1).
    Returns (value, x, verified_ub); verified_ub = +inf if the dual residual falls on an unbounded coordinate."""
    res = linprog(-obj, A_ub=A, b_ub=b, bounds=list(zip(lo_n, hi_n)), method="highs")
    if res.status != 0:
        return np.nan, None, np.nan
    y = np.maximum(-res.ineqlin.marginals, 0.0); r = obj - A.T @ y; zp = np.maximum(r, 0); zm = np.maximum(-r, 0)
    if np.any((zp > 1e-12) & ~np.isfinite(hi_n)) or np.any((zm > 1e-12) & ~np.isfinite(lo_n)):
        return -res.fun, res.x, np.inf
    hi_t = np.where(np.isfinite(hi_n), hi_n, 0.0); lo_t = np.where(np.isfinite(lo_n), lo_n, 0.0)
    terms = np.concatenate([b * y, hi_t * zp, -lo_t * zm]); ub = float(np.sum(terms)) + (len(terms) + len(obj) + A.shape[0]) * EPS * float(np.sum(np.abs(terms)) + np.sum(np.abs(A.T @ y)) + np.sum(np.abs(obj)))
    return -res.fun, res.x, ub
