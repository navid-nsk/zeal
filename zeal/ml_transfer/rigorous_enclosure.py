"""rigorous_enclosure.py -- rigorous floating-point enclosures (IBP and CROWN) of the exp3_ml networks and of their shared-input
increment networks.  Independent of auto_LiRPA (torch is used only to READ the stored state dicts).

WHAT IS ENCLOSED
  Networks (declared data = the stored float32 weights, read as their exact float64 values):
    mlp    : x (168) -> W0 x + b0 -> ReLU -> W1 . + b1 -> ReLU -> W2 . + b2 = g(x) (logit);  f = sigmoid(g)
    feat*  : x (168) -> Wf x (fixed block-mean layer, no bias, no activation) -> the same three-layer ReLU MLP -> g -> sigmoid
  Input set of output t (one window):  X_t = { x : |x_i - y_{t,i}| <= eps }, y = the stored z-scored series (exact floats),
  eps = the DECLARED DECIMAL radius (0.01, 0.05, ...), enclosed by eps_up = nextafter(float(eps), +inf) >= eps.
  Increment network of lag k:  Incr_k(z) = f(z[k:k+W]) - f(z[:W]) on the joint window z of length W + k (shared values), i.e.
  f(t+k) - f(t) under ONE perturbed series.  Outputs, for every t: value box [lo_t, hi_t] and increment box [glo_tk, ghi_tk] with
      lo_t <= f_t(x) <= hi_t  and  glo_tk <= f_{t+k}(x) - f_t(x) <= ghi_tk   for EVERY admissible series x, in exact real arithmetic
  (the claim is about the real-valued network defined by the stored weights; the float forward pass is not what is enclosed).

ROUNDING ARGUMENT (IEEE-754 binary64, round-to-nearest-even, checked at import and at every call by `ieee_check`; gradual underflow
checked too).  u = 2^-53, REALMIN = 2^-1022, eta = 2^-1074.
  (R1) one operation.  For a single +, -, *, / of floats a, b with exact result v, fl(v) is the nearest float, so the exact v lies
       between nextafter(fl(v), -inf) and nextafter(fl(v), +inf) (a nearer float would otherwise exist).  dn(.)/up(.) below are these
       nextafter calls; every scalar/elementwise lower (upper) quantity is followed by dn (up), so it is a rigorous lower (upper)
       bound of the exact value of the expression of its (already rigorous) inputs.  Overflow is excluded by asserting finiteness.
       Products of an interval by a float use the min/max of the endpoint products, each rounded outward (`mul_iv`).
  (R2) sums of n products (matmul, einsum, sum), ANY summation order, with or without fused multiply-add:  every product a_i b_i
       passes through at most n roundings on its way to the result (one product or fma rounding, at most n - 1 additions on its
       path in the summation tree), hence  |fl(sum a_i b_i) - sum a_i b_i| <= gamma_n sum |a_i b_i| + n eta/2,  gamma_n = n u/(1 - n u)
       (Higham, Accuracy and Stability, Sec. 3.1; the eta term is the underflow error of products, additions being exact in gradual
       underflow).  The unknown exact sum |a||b| is bounded by the computed S = fl(sum |a_i||b_i|):  sum|a_i b_i| <= (S + n eta/2)/(1 - gamma_n).
       We use the radius   e = up( fl(2(n+2)u * S) + (n+2) REALMIN ),   which dominates gamma_n/(1-gamma_n) (S + n eta/2) + n eta/2
       for n u <= 1/4 (2(n+2)(1-u)(1-2nu) >= n), the rounding of the product 2(n+2)u*S (factor 1-u) and every eta term.  The enclosure
       of the exact bilinear form is then [dn(C - e), up(C + e)] with C the computed value (`bil`).  2(n+2)u = (n+2) 2^-52 is exact.
       The REALMIN term also covers flush-to-zero inside BLAS (each flushed product contributes < REALMIN); denormal-are-zero on INPUTS
       is excluded by refusing subnormal inputs (`_nosub` raises).
  (R3) transcendental functions.  exp is enclosed WITHOUT libm: x = k ln2 + r with k = rint(x/ln2) and ln2 in [LN2_LO, LN2_HI]
       (two consecutive floats, bracketing ln 2 verified at import against a 40-digit decimal), r enclosed by (R1), |r| <= 0.36
       asserted; exp(r) = Taylor polynomial of degree 20 evaluated at the endpoint floats of r by interval Horner (coefficients
       1/i! enclosed exactly via Fraction), plus the Lagrange remainder |r|^21 e^|r| / 21! <= REM = 1e-27 (verified at import);
       exp is increasing so exp([rl, rh]) = [exp(rl), exp(rh)]; scaling by 2^k is exact (ldexp), widened by one ulp for safety.
       sigmoid(z) = 1/(1 + exp(-z)) and sigmoid'(z) = sigmoid(z) sigmoid(-z) are enclosed from it by (R1).  log/sqrt are used ONLY to
       pick tangent points, which need not be exact (see the relaxations).
  (R4) relaxations: slopes are chosen in plain floating point (any float value is admissible), intercepts are COMPUTED rigorously
       as the exact maximum (minimum) of the gap over the pre-activation interval, so the soundness never depends on the slope:
       ReLU   : for a backward coefficient m on relu(z_j), z_j in [l, u] (rigorous):  stable active (l >= 0) -> m z, exact; stable
                inactive (u <= 0) -> 0, exact; unstable with m < 0 -> m alpha z with alpha in {0, 1} exact (relu(z) >= alpha z), alpha =
                1 iff u/(u - l) > 1/2 (auto_LiRPA's default 'adaptive' lower-slope rule); unstable with m >= 0 -> lam z + nu with lam =
                fl(m u/(u - l)) (the CROWN triangle upper slope, as computed in float) and nu = up(max over z in {l, 0, u} of
                m relu(z) - lam z)  (m relu(z) - lam z is convex and piecewise linear, so its max over [l, u] is at an endpoint).
       sigmoid (increment networks only): sigma(g) <= s g + mu_hi(s) and sigma(g) >= s g + mu_lo(s) on g in [l, u] with
                mu_hi/lo(s) = exact max/min of h(z) = sigma(z) - s z over [l, u]: h is convex on [l, min(u, 0)] (max at an endpoint,
                min >= tangent bound h(c) + h'(c)(z - c) at any c in the piece) and concave on [max(l, 0), u] (symmetric); c = the
                float critical point sigma'(c) = s clipped into the piece.  The slope pair (s_A, s_B) of the two windows is chosen as
                the best of 6 x 6 float candidates {chord, sigma'(l), sigma'(mid), sigma'(u), lower-hull edge, upper-hull edge} per
                window (every candidate is sound, so their minimum is sound; the set contains auto_LiRPA's vanilla choices for intervals
                not containing 0 and the convex-hull tangents that its d_lower/d_upper tables approximate for intervals containing 0),
                then improved by 2 rounds of golden-section search (20 steps) along the common-slope direction s_A = s_B and along
                each coordinate (the bound is jointly convex in (s_A, s_B); each evaluated pair is itself a rigorous bound).
       sigmoid (value boxes): sigma is increasing, so the value box is [sigma_lo(logit_lo), sigma_hi(logit_hi)] (no relaxation).
  (R5) back-substitution.  The CROWN state is "target <= M . a + c for every admissible input" with M a FLOAT matrix and c a rigorous
       upper constant.  Passing a linear layer a = W a' + b:  M . (W a' + b) = (M W) a' + M.b exactly; fl(M W) = M' with
       |M W - M'| <= E elementwise by (R2), so  M.(W a' + b) <= M'.a' + up(M.b) + up(E . amax')  where amax' >= |a'| is a rigorous
       bound of the previous layer's activations (relu: max(u, 0); identity layer: max(|l|, |u|); input: up(|y| + eps_up)).  At the
       input, sup over the box of M.x = M.y + eps sum|M| (exact identity), enclosed by (R2).  Intermediate (pre-activation) bounds of
       every hidden layer are themselves CROWN bounds (rows +-e_j), exactly as auto_LiRPA's 'backward' method computes them; the
       first layer's bounds are the exact interval image of the box.  Increment boxes:  sigma(g_B) - sigma(g_A) <=
       s_B (a+_B . x_B + c+_B) + s_A (a-_A . x_A + c-_A) + mu_hi_B(s_B) - mu_lo_A(s_A)  where (a+, c+) is the window's rigorous CROWN
       linear upper bound of g and (a-, c-) that of -g (the ReLU relaxations depend only on the coefficient SIGN, which s >= 0 does not
       change, so this equals CROWN on the joint increment network); the joint coefficient m = s_A a_A + s_B a_B (shared positions add:
       the cancellation of the shared perturbation) is formed in float with |m^ - m| <= 4u(|fl(s_A a_A)| + |fl(s_B a_B)|) + 4 REALMIN
       per entry, and the sup is s_A (a_A.y_A) + s_B (a_B.y_B) + eps (sum|m^| + sum err), every term rounded upward.
  IBP: midpoint-radius interval matrix products with (R1)/(R2); exact interval image for a point matrix and a box.

What is NOT claimed: correctness of numpy/BLAS beyond IEEE-754 binary64 round-to-nearest semantics of +, -, *, /, nextafter, ldexp
(and the absence of extended precision).  No libm function is trusted.
Usage: from rigorous_enclosure import load_net, enclose_model ;  see r1_rigorous_boxes.py.
"""
import os as _os, sys as _sys
try:
    from .. import paths                    # imported as part of the package
except ImportError:
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")); import paths
import os
import numpy as np
from fractions import Fraction
from math import factorial

U = 2.0 ** -53
REALMIN = 2.0 ** -1022
INF = np.inf


# ---------------------------------------------------------------- IEEE environment checks
def ieee_check():
    """raise unless binary64 arithmetic rounds to nearest-even with gradual underflow (numpy and Python floats)"""
    one = np.array([1.0, -1.0])
    t = 2.0 ** -53 + 2.0 ** -60
    ok = (bool(np.all(one + np.array([2.0 ** -53, -2.0 ** -53]) == one))                 # ties to even
          and float((one + t)[0]) == 1.0 + 2.0 ** -52 and float((one - t)[1]) == -(1.0 + 2.0 ** -52)     # nearest (not RZ/RD/RU)
          and float((one - t)[0]) == 1.0 - 2.0 ** -53                                          # nearest below 1 (RD would give 1 - 2^-52)
          and float((np.array([REALMIN]) / 2.0)[0]) == 2.0 ** -1023                           # gradual underflow (no FTZ)
          and float((np.array([2.0 ** -1074]) * 2.0)[0]) == 2.0 ** -1073                        # subnormal inputs not zeroed (no DAZ)
          and 1.0 + 2.0 ** -53 == 1.0 and 1.0 + t == 1.0 + 2.0 ** -52
          and float((np.array([[1.0, 2.0 ** -53]]) @ np.array([1.0, 1.0]))[0]) in (1.0, 1.0 + 2.0 ** -52))
    if not ok:
        raise RuntimeError("floating-point environment is not IEEE binary64 round-to-nearest with gradual underflow")


ieee_check()


def up(x):
    return np.nextafter(x, INF)


def dn(x):
    return np.nextafter(x, -INF)


def _nosub(*arrs):
    for a in arrs:
        a = np.asarray(a)
        if np.any((a != 0) & (np.abs(a) < REALMIN)):
            raise FloatingPointError("subnormal input to a rigorous sum")


def err_from_abs(S, n):
    """(R2) radius for a computed sum of n products whose computed absolute sum is S"""
    return up((n + 2) * 2.0 ** -52 * S + (n + 2) * REALMIN)


def bil(f, A, B, n):
    """(R2) enclosure [lo, hi] of the exact bilinear form f(A, B) (sums of n products per entry, any order)"""
    _nosub(A, B)
    C = f(A, B); S = f(np.abs(A), np.abs(B)); e = err_from_abs(S, n)
    return dn(C - e), up(C + e)


def bil_hi(f, A, B, n):
    lo, hi = bil(f, A, B, n); return hi


def sum_hi_nonneg(V, axis, n):
    """upper bound of the exact sum of nonnegative floats V along axis (n terms); subnormal entries are first rounded UP to
    REALMIN (sound for an upper bound, and keeps DAZ out of the argument)"""
    V = np.where((V > 0) & (V < REALMIN), REALMIN, V)
    C = V.sum(axis=axis); return up(C + err_from_abs(C, n))


def mul_iv(al, ah, bl, bh):
    """(R1) interval product"""
    p = (al * bl, al * bh, ah * bl, ah * bh)
    return dn(np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3]))), up(np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3])))


def mul_up(a, b):
    return up(a * b)


def mul_dn(a, b):
    return dn(a * b)


def _check_finite(*arrs):
    for a in arrs:
        if not np.all(np.isfinite(a)):
            raise FloatingPointError("non-finite bound (overflow): enclosure aborted")


# ---------------------------------------------------------------- (R3) exp, sigmoid
LN2_LO = float.fromhex("0x1.62e42fefa39efp-1")
LN2_HI = float(np.nextafter(LN2_LO, INF))
_LN2_DEC = Fraction("0.6931471805599453094172321214581765680755")     # ln 2 truncated to 40 decimals: ln 2 in (_LN2_DEC, _LN2_DEC + 1e-40)
assert Fraction(LN2_LO) < _LN2_DEC and Fraction(LN2_HI) > _LN2_DEC + Fraction(1, 10 ** 40), "ln2 bracket"
TAYLOR_N = 20
R_MAX = 0.36
REM = 1e-27
# Lagrange remainder bound |r|^(N+1) e^|r| / (N+1)! with e^0.36 < 3/2 (since e^0.36 < 1 + 0.36 + 0.36^2 (sum of a geometric tail) < 1.5)
assert Fraction(R_MAX) ** (TAYLOR_N + 1) * Fraction(3, 2) / factorial(TAYLOR_N + 1) < Fraction(REM), "Taylor remainder"


def _coef_bracket(q):
    f = float(q); F = Fraction(f)
    if F == q:
        return f, f
    return (f, float(np.nextafter(f, INF))) if F < q else (float(np.nextafter(f, -INF)), f)


_TC = [_coef_bracket(Fraction(1, factorial(i))) for i in range(TAYLOR_N + 1)]


def _exp_point_poly(r):
    """interval Horner enclosure of the degree-N Taylor polynomial of exp at the float points r"""
    pl = np.full(r.shape, _TC[TAYLOR_N][0]); ph = np.full(r.shape, _TC[TAYLOR_N][1])
    for i in range(TAYLOR_N - 1, -1, -1):
        a = pl * r; b = ph * r
        ql = dn(np.minimum(a, b)); qh = up(np.maximum(a, b))
        pl = dn(ql + _TC[i][0]); ph = up(qh + _TC[i][1])
    return pl, ph


def exp_enclose(x):
    """rigorous [lo, hi] with lo <= exp(x) <= hi for float array x"""
    ieee_check()
    x = np.asarray(x, dtype=np.float64)
    big = x > 709.0; small = x < -740.0
    xc = np.clip(x, -740.0, 709.0)
    k = np.rint(xc / LN2_LO)
    kl = np.where(k >= 0, dn(k * LN2_LO), dn(k * LN2_HI)); kh = np.where(k >= 0, up(k * LN2_HI), up(k * LN2_LO))
    rl = dn(xc - kh); rh = up(xc - kl)
    if not (np.all(rl >= -R_MAX) and np.all(rh <= R_MAX)):
        raise FloatingPointError("range reduction failed")
    pl, _ = _exp_point_poly(rl); _, ph = _exp_point_poly(rh)
    ki = k.astype(np.int32)
    lo = dn(np.ldexp(dn(pl - REM), ki)); hi = up(np.ldexp(up(ph + REM), ki))
    lo = np.maximum(lo, 0.0)
    lo = np.where(small, 0.0, lo); hi = np.where(big, INF, hi)
    return lo, hi


def sig_enclose(z):
    """rigorous enclosure of sigmoid(z) = 1/(1 + exp(-z))"""
    el, eh = exp_enclose(-np.asarray(z, dtype=np.float64))
    dl = dn(1.0 + el); dh = up(1.0 + eh)
    with np.errstate(divide="ignore"):
        sl = dn(1.0 / dh); sh = up(1.0 / dl)
    return np.maximum(sl, 0.0), np.minimum(sh, 1.0)


def dsig_enclose(z):
    """rigorous enclosure of sigmoid'(z) = sigmoid(z) sigmoid(-z)"""
    z = np.asarray(z, dtype=np.float64)
    al, ah = sig_enclose(z); bl, bh = sig_enclose(-z)
    return np.maximum(dn(al * bl), 0.0), up(ah * bh)


def _h(z, s):
    """enclosure of h(z) = sigmoid(z) - s z at float points"""
    sl, sh = sig_enclose(z)
    return dn(sl - up(s * z)), up(sh - dn(s * z))


def _dh(z, s):
    dl, dh = dsig_enclose(z)
    return dn(dl - s), up(dh - s)


def _crit(s):
    """float critical point zc <= 0 of h(z) = sigmoid(z) - s z on the convex side (sigmoid'(zc) = s; -zc on the concave side).
    Only used after clipping into a piece: any value is admissible."""
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.sqrt(np.maximum(1.0 - 4.0 * s, 0.0)); sc = (1.0 - q) / 2.0
        zc = np.log(sc) - np.log1p(-sc)
    return np.where(np.isnan(zc), 0.0, zc)


class SigPieces:
    """(R4) rigorous sigmoid relaxation intercepts on the intervals [l, u] (float arrays, l <= u), for any float slopes s:
    mu_lo(s) <= sigmoid(z) - s z <= mu_hi(s) for every z in [l, u].  h(z) = sigmoid(z) - s z is convex on the piece [l, min(u, 0)]
    (present iff l <= 0) and concave on [max(l, 0), u] (present iff u >= 0): on the convex piece the max is at an endpoint and the
    min is >= the tangent bound h(c) + h'(c)(z - c) at any c in the piece (evaluated at the endpoints, interval arithmetic); symmetric
    on the concave piece.  The sigmoid enclosures at the four piece endpoints are cached (they do not depend on s)."""

    def __init__(self, l, u):
        self.l = np.asarray(l, float); self.u = np.asarray(u, float)
        self.a1, self.b1 = self.l, np.minimum(self.u, 0.0); self.a2, self.b2 = np.maximum(self.l, 0.0), self.u
        self.cv = self.l <= 0; self.cc = self.u >= 0
        self.sig = {nm: sig_enclose(getattr(self, nm)) for nm in ("a1", "b1", "a2", "b2")}

    def _hend(self, nm, s):
        sl, sh = self.sig[nm]; z = getattr(self, nm)
        return dn(sl - up(s * z)), up(sh - dn(s * z))

    def _tangent(self, s, a, b, c):
        """interval of h(c) and of h'(c)(a - c), h'(c)(b - c)"""
        Hcl, Hch = _h(c, s); Dl, Dh = _dh(c, s)
        pa = mul_iv(Dl, Dh, dn(a - c), up(a - c)); pb = mul_iv(Dl, Dh, dn(b - c), up(b - c))
        return Hcl, Hch, pa, pb

    def mu_hi(self, s):
        s = np.asarray(s, float); out = np.full(s.shape, -INF)
        mx1 = np.maximum(self._hend("a1", s)[1], self._hend("b1", s)[1])                      # convex piece: endpoint max
        out = np.where(self.cv, np.maximum(out, mx1), out)
        c2 = np.clip(-_crit(s), self.a2, self.b2)
        _, Hch, pa, pb = self._tangent(s, self.a2, self.b2, c2)
        mx2 = up(Hch + np.maximum(pa[1], pb[1]))                                               # concave piece: tangent bound
        out = np.where(self.cc, np.maximum(out, mx2), out)
        _check_finite(out); return out

    def mu_lo(self, s):
        s = np.asarray(s, float); out = np.full(s.shape, INF)
        c1 = np.clip(_crit(s), self.a1, self.b1)
        Hcl, _, pa, pb = self._tangent(s, self.a1, self.b1, c1)
        mn1 = dn(Hcl + np.minimum(pa[0], pb[0]))                                               # convex piece: tangent bound
        out = np.where(self.cv, np.minimum(out, mn1), out)
        mn2 = np.minimum(self._hend("a2", s)[0], self._hend("b2", s)[0])                       # concave piece: endpoint min
        out = np.where(self.cc, np.minimum(out, mn2), out)
        _check_finite(out); return out


def sig_intercepts(s, l, u):
    """(R4) rigorous mu_lo <= sigmoid(z) - s z <= mu_hi for all z in [l, u]"""
    P = SigPieces(l, u); return P.mu_lo(s), P.mu_hi(s)


def sig_slope_candidates(l, u, iters=60):
    """float slope candidates per window (any float >= 0 is admissible; intercepts are verified separately):
    chord, sigma'(l), sigma'(mid), sigma'(u), and the two convex-hull edge slopes of sigmoid on [l, u] (the tangent at d <= 0 through
    (u, sigma(u)) and the tangent at d >= 0 through (l, sigma(l)), found by float bisection; chord when the tangent point would fall
    outside [l, u]).  The hull slopes play the role of auto_LiRPA's d_lower / d_upper tangent points for intervals containing 0."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        sg = lambda z: 1.0 / (1.0 + np.exp(-z))
        ds = lambda z: sg(z) * sg(-z)
        m = 0.5 * (l + u); w = u - l
        chord = np.where(w > 1e-12, (sg(u) - sg(l)) / np.where(w > 1e-12, w, 1.0), ds(m))
        mixed = (l < 0) & (u > 0)
        # lower hull edge: g(d) = sigma(d) + sigma'(d)(u - d) - sigma(u), root in [l, 0] iff g(l) >= 0 (g(0) >= 0 always)
        gl = lambda d: sg(d) + ds(d) * (u - d) - sg(u)
        a = np.minimum(l, 0.0); b = np.zeros_like(l); has = mixed & (gl(a) < 0)
        for _ in range(iters):
            c = 0.5 * (a + b); neg = gl(c) < 0; a = np.where(neg, c, a); b = np.where(neg, b, c)
        hull_lo = np.where(has, ds(b), chord)
        # upper hull edge: g(d) = sigma(d) + sigma'(d)(l - d) - sigma(l), root in [0, u] iff g(u) >= 0 (g(0) <= 0 always)
        gu = lambda d: sg(d) + ds(d) * (l - d) - sg(l)
        a = np.zeros_like(l); b = np.maximum(u, 0.0); has = mixed & (gu(b) > 0)
        for _ in range(iters):
            c = 0.5 * (a + b); pos = gu(c) > 0; b = np.where(pos, c, b); a = np.where(pos, a, c)
        hull_hi = np.where(has, ds(a), chord)
        S = np.stack([chord, ds(l), ds(m), ds(u), hull_lo, hull_hi], -1)
    return np.maximum(np.nan_to_num(S, nan=0.0), 0.0)


# ---------------------------------------------------------------- networks
def load_net(model, out_dir=None):
    """the stored model as a list of layers dict(W [out, in] float64, b [out] float64, act in {'relu', 'id', None})"""
    import torch
    if out_dir is None:
        out_dir = paths.EXP3_ML
    sd = torch.load(os.path.join(out_dir, f"model_{model}.pt"), map_location="cpu")
    g = lambda k: sd[k].detach().to(torch.float64).numpy().copy()
    net = []
    if "feat.weight" in sd:
        Wf = g("feat.weight"); net.append(dict(W=Wf, b=np.zeros(Wf.shape[0]), act="id"))
    net.append(dict(W=g("net.0.weight"), b=g("net.0.bias"), act="relu"))
    net.append(dict(W=g("net.2.weight"), b=g("net.2.bias"), act="relu"))
    net.append(dict(W=g("net.4.weight"), b=g("net.4.bias"), act=None))
    for L in net:
        _nosub(L["W"], L["b"])
    return net


def forward_float(net, X):
    """plain float64 forward pass (NOT rigorous; nominal values only): logits [N]"""
    a = X
    for L in net:
        a = a @ L["W"].T + L["b"]
        if L["act"] == "relu":
            a = np.maximum(a, 0.0)
    return a[:, 0]


# ---------------------------------------------------------------- IBP (rigorous)
def ibp_logit(net, Yw, eps):
    """rigorous IBP bounds of the logit for windows Yw [N, W]"""
    ieee_check()
    eps_up = up(float(eps))
    c = Yw; r = np.full_like(Yw, eps_up)
    mm = lambda A, B: A @ B
    for L in net:
        W, b = L["W"], L["b"]; n = W.shape[1]
        Pl, Ph = bil(mm, c, W.T, n); Q = bil_hi(mm, r, np.abs(W).T, n)
        zl = dn(dn(Pl - Q) + b); zh = up(up(Ph + Q) + b)
        if L["act"] == "relu":
            zl, zh = np.maximum(zl, 0.0), np.maximum(zh, 0.0)
        if L["act"] is None:
            _check_finite(zl, zh); return zl[:, 0], zh[:, 0]
        c = 0.5 * (zl + zh); c = np.where(np.abs(c) < REALMIN, 0.0, c)          # any float centre is admissible
        r = up(np.maximum(up(zh - c), up(c - zl))); r = np.maximum(r, REALMIN)    # [c - r, c + r] contains [zl, zh]; enlarging r is sound


# ---------------------------------------------------------------- CROWN (rigorous)
def _relax_relu(M, l, u):
    """(R4) ReLU relaxation for an upper bound of M . relu(z), z in [l, u]: returns (lam, nu_sum_hi)"""
    L = l[:, None, :]; Uu = u[:, None, :]
    pos = L >= 0; neg = Uu <= 0; unst = ~(pos | neg)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(unst, Uu / np.where(unst, Uu - L, 1.0), 0.0)
    alpha = (s > 0.5).astype(np.float64)
    mpos = unst & (M > 0); mneg = unst & (M <= 0)          # m = 0: lam = 0, nu = 0 exactly
    lam_p = M * s
    lam = np.where(pos, M, 0.0)
    lam = np.where(mneg, M * alpha, lam)
    lam = np.where(mpos, lam_p, lam)
    nu = np.maximum(np.maximum(up(lam_p * (-L)), up(up(M * Uu) - dn(lam_p * Uu))), 0.0)
    nu = np.where(mpos, nu, 0.0)
    return lam, sum_hi_nonneg(nu, -1, M.shape[-1])


def _backward(net, i, M, c, lb, ub, amax, xmax):
    """rows target <= M . a_{i-1} + c (M float [N, R, n_{i-1}], c rigorous upper [N, R]); back-substitute to the input"""
    for j in range(i - 1, -1, -1):
        Lj = net[j]
        if Lj["act"] == "relu":
            lam, nu = _relax_relu(M, lb[j], ub[j]); c = up(c + nu)
        else:
            lam = M
        N, R, k = lam.shape; W, b = Lj["W"], Lj["b"]
        cb = bil_hi(lambda A, B: A @ B, lam.reshape(N * R, k), b, k).reshape(N, R)
        lam2 = lam.reshape(N * R, k)
        _nosub(lam2)
        Mn = (lam2 @ W).reshape(N, R, -1); E = err_from_abs((np.abs(lam2) @ np.abs(W)).reshape(N, R, -1), k)
        am = amax[j - 1] if j >= 1 else xmax
        am = np.where((am > 0) & (am < REALMIN), REALMIN, am)                 # rounding a nonnegative bound up is sound
        sub = (Mn != 0) & (np.abs(Mn) < REALMIN)
        if np.any(sub):                                                        # flush subnormal coefficients: |dropped . a| <= REALMIN sum amax
            E = E + np.where(sub, REALMIN, 0.0); Mn = np.where(sub, 0.0, Mn)
        ce = up(np.einsum("nri,ni->nr", E, am)); ce = up(ce + err_from_abs(ce, E.shape[-1]))
        c = up(up(c + cb) + ce)
        M = Mn
    return M, c


def _sup_box(M, c, Yw, eps_up):
    """rigorous upper bound of sup_{|x - y| <= eps} M.x + c; also returns upper(M.y) and upper(sum|M|)"""
    n = M.shape[-1]
    dot_hi = bil_hi(lambda A, B: np.einsum("nrk,nk->nr", A, B), M, Yw, n)
    sabs = sum_hi_nonneg(np.abs(M), -1, n)
    return up(up(dot_hi + mul_up(eps_up, sabs)) + c), dot_hi, sabs


def crown_windows(net, Yw, eps, chunk=512):
    """rigorous CROWN bounds for windows Yw [N, W]: logit bounds and the logit's linear upper bounds (rows: +g, -g)"""
    ieee_check()
    eps_up = up(float(eps)); N = len(Yw); nL = len(net)
    out = dict(logit_lo=np.empty(N), logit_hi=np.empty(N), A=np.empty((N, 2, Yw.shape[1])), c=np.empty((N, 2)),
               dot_hi=np.empty((N, 2)), sabs=np.empty((N, 2)), n_unstable=np.zeros(nL, np.int64))
    for s0 in range(0, N, chunk):
        Y = Yw[s0:s0 + chunk]; n = len(Y)
        xmax = up(np.abs(Y) + eps_up)
        lb, ub, amax = [None] * nL, [None] * nL, [None] * nL
        for i in range(nL):
            W, b = net[i]["W"], net[i]["b"]; o = W.shape[0]
            M0 = np.broadcast_to(np.concatenate([W, -W], 0)[None], (n, 2 * o, W.shape[1])).copy()
            c0 = np.broadcast_to(np.concatenate([b, -b])[None], (n, 2 * o)).copy()
            M, c = _backward(net, i, M0, c0, lb, ub, amax, xmax)
            sup, dot_hi, sabs = _sup_box(M, c, Y, eps_up)
            _check_finite(sup)
            if net[i]["act"] is None:
                out["logit_hi"][s0:s0 + n] = sup[:, 0]; out["logit_lo"][s0:s0 + n] = -sup[:, 1]
                out["A"][s0:s0 + n] = M; out["c"][s0:s0 + n] = c; out["dot_hi"][s0:s0 + n] = dot_hi; out["sabs"][s0:s0 + n] = sabs
            else:
                ub[i] = sup[:, :o]; lb[i] = -sup[:, o:]
                if np.any(lb[i] > ub[i]):
                    raise FloatingPointError("crossed intermediate bounds")
                amax[i] = np.maximum(ub[i], 0.0) if net[i]["act"] == "relu" else np.maximum(np.abs(lb[i]), np.abs(ub[i]))
                if net[i]["act"] == "relu":
                    out["n_unstable"][i] += int(np.sum((lb[i] < 0) & (ub[i] > 0)))
    return out


def value_boxes(logit_lo, logit_hi):
    lo, _ = sig_enclose(logit_lo); _, hi = sig_enclose(logit_hi)
    return lo, hi


def increment_boxes(cw, Ywin_shape, k, W, refine_rounds=2, golden_iters=20):
    """rigorous CROWN increment boxes [glo, ghi] of f(t+k) - f(t) for every feeder j and t = 0..T-k-1.
    cw: crown_windows output (+ 'eps_up', 'slopes') for the windows indexed (j, t) -> j*T + t (Ywin_shape = (n_feeders, T)).
    Per pair and side, the sigmoid slope pair (s_A, s_B) is the best of the candidate grid, then improved by `refine_rounds` rounds of
    golden-section search along s_A = s_B, then s_A, then s_B (the bound is jointly convex in the slopes; the common-slope direction
    is needed because coordinate search stalls at kinks of the coupling term); EVERY evaluated pair yields a rigorous bound
    (intercepts are recomputed rigorously for each slope), so the minimum over evaluated pairs is rigorous whatever the search does."""
    nF, T = Ywin_shape
    assert 0 < k < W
    idxA = (np.arange(nF)[:, None] * T + np.arange(T - k)[None, :]).ravel(); idxB = idxA + k
    eps_up = cw["eps_up"]; S = cw["slopes"]; nS = S.shape[1]
    PA, PB = SigPieces(cw["logit_lo"][idxA], cw["logit_hi"][idxA]), SigPieces(cw["logit_lo"][idxB], cw["logit_hi"][idxB])
    MUL, MUH = cw["mu_lo"], cw["mu_hi"]                                       # [Nw, nS] grid intercepts (rigorous)
    res = {}
    for name, (rowA, rowB) in {"hi": (1, 0), "lo": (0, 1)}.items():
        # "hi": upper of sigma(g_B) - sigma(g_A): A uses the bound of -g (row 1) and mu_lo; B uses the bound of +g (row 0) and mu_hi.
        # "lo": upper of sigma(g_A) - sigma(g_B): A uses row 0 and mu_hi, B uses row 1 and mu_lo.
        aA = cw["A"][idxA, rowA]; aB = cw["A"][idxB, rowB]                    # [P, W]
        dA = cw["dot_hi"][idxA, rowA]; dB = cw["dot_hi"][idxB, rowB]
        cA = cw["c"][idxA, rowA]; cB = cw["c"][idxB, rowB]
        sabA = cw["sabs"][idxA, rowA]; sabB = cw["sabs"][idxB, rowB]
        aAo, aBo = aA[:, k:], aB[:, :W - k]
        nA_only = np.abs(aA[:, :k]).sum(1); nB_only = np.abs(aB[:, W - k:]).sum(1)
        muA_f = PA.mu_lo if name == "hi" else PA.mu_hi; muB_f = PB.mu_hi if name == "hi" else PB.mu_lo
        gA = (MUL if name == "hi" else MUH)[idxA]; gB = (MUH if name == "hi" else MUL)[idxB]

        def bound(sa, sb, mA, mB):
            """rigorous upper bound for slope vectors sa, sb >= 0 with their rigorous intercepts mA (window A), mB (window B)"""
            ov = np.abs(sa[:, None] * aAo + sb[:, None] * aBo).sum(1)
            tot = ov + sa * nA_only + sb * nB_only                             # computed sum |m^| (W + k terms, plus 2 scalings)
            tot_hi = up(tot + err_from_abs(tot, W + k + 4))
            errsum = up(8 * U * up(mul_up(sa, sabA) + mul_up(sb, sabB)) + 4 * (W + k) * REALMIN)
            T1 = up(mul_up(sb, dB) + mul_up(sa, dA))
            T3 = mul_up(eps_up, up(tot_hi + errsum))
            T4 = up(mul_up(sb, cB) + mul_up(sa, cA))
            T5 = up(mB - mA) if name == "hi" else up(mA - mB)                 # mu_hi_B - mu_lo_A  |  mu_hi_A - mu_lo_B
            return up(up(T1 + T3) + up(T4 + T5))

        best = np.full(len(idxA), INF); bA = np.zeros(len(idxA)); bB = np.zeros(len(idxA)); bestc = np.zeros(len(idxA), np.int64)
        for iA in range(nS):
            for iB in range(nS):
                sa, sb = S[idxA, iA], S[idxB, iB]; Ub = bound(sa, sb, gA[:, iA], gB[:, iB])
                bt = Ub < best; best = np.where(bt, Ub, best); bA = np.where(bt, sa, bA); bB = np.where(bt, sb, bB)
                bestc = np.where(bt, iA * nS + iB, bestc)
        grid_best = best.copy()
        loA, hiA = S[idxA].min(1), S[idxA].max(1); loB, hiB = S[idxB].min(1), S[idxB].max(1)
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        for _ in range(refine_rounds):
            for coord in ("D", "A", "B"):                  # D = common slope s_A = s_B (maximal cancellation of the shared input)
                if coord == "D":
                    a, b = np.minimum(loA, loB), np.maximum(hiA, hiB)
                else:
                    a, b = (loA.copy(), hiA.copy()) if coord == "A" else (loB.copy(), hiB.copy())
                fixA, fixB = bA.copy(), bB.copy()
                if coord == "D":
                    f = lambda s_: bound(s_, s_, muA_f(s_), muB_f(s_))
                elif coord == "A":
                    mfix = muB_f(fixB); f = lambda s_: bound(s_, fixB, muA_f(s_), mfix)
                else:
                    mfix = muA_f(fixA); f = lambda s_: bound(fixA, s_, mfix, muB_f(s_))

                def record(xv, fv):
                    nonlocal best, bA, bB
                    bt = fv < best; best = np.where(bt, fv, best)
                    if coord == "D":
                        bA = np.where(bt, xv, bA); bB = np.where(bt, xv, bB)
                    elif coord == "A":
                        bA = np.where(bt, xv, bA); bB = np.where(bt, fixB, bB)
                    else:
                        bB = np.where(bt, xv, bB); bA = np.where(bt, fixA, bA)
                x1 = b - gr * (b - a); x2 = a + gr * (b - a); f1 = f(x1); f2 = f(x2); record(x1, f1); record(x2, f2)
                for _it in range(golden_iters):
                    left = f1 < f2                                              # minimum of the convex section lies in [a, x2]
                    b = np.where(left, x2, b); a = np.where(left, a, x1)
                    xn = np.where(left, b - gr * (b - a), a + gr * (b - a)); fn = f(xn); record(xn, fn)
                    x1, f1, x2, f2 = (np.where(left, xn, x2), np.where(left, fn, f2), np.where(left, x1, xn), np.where(left, f1, fn))
        _check_finite(best)
        res[name] = best; res[name + "_combo"] = bestc; res[name + "_refine_gain"] = grid_best - best
    ghi = res["hi"].reshape(nF, T - k); glo = (-res["lo"]).reshape(nF, T - k)
    gain = float(max(np.max(res["hi_refine_gain"]), np.max(res["lo_refine_gain"])))
    return glo, ghi, res["hi_combo"].reshape(nF, T - k), res["lo_combo"].reshape(nF, T - k), gain


def enclose_model(net, Y, eps, lags, W=168, chunk=512, verbose=False):
    """rigorous IBP and CROWN value boxes and CROWN increment boxes for every feeder j and output t.
    Y [n_feeders, T + W - 1] stored series (exact floats).  Returns dict of arrays."""
    import time
    ieee_check()
    nF, Ltot = Y.shape; T = Ltot - W + 1
    Yw = np.lib.stride_tricks.sliding_window_view(Y, W, axis=1).reshape(nF * T, W).astype(np.float64).copy()
    t0 = time.time()
    il, ih = ibp_logit(net, Yw, eps)
    ibp_lo, ibp_hi = value_boxes(il, ih)
    t_ibp = time.time() - t0
    t0 = time.time()
    cw = crown_windows(net, Yw, eps, chunk=chunk)
    lo, hi = value_boxes(cw["logit_lo"], cw["logit_hi"])
    t_val = time.time() - t0
    t0 = time.time()
    cw["eps_up"] = up(float(eps))
    cw["slopes"] = sig_slope_candidates(cw["logit_lo"], cw["logit_hi"])                           # [Nw, 6]
    Pw = SigPieces(np.repeat(cw["logit_lo"][:, None], cw["slopes"].shape[1], 1), np.repeat(cw["logit_hi"][:, None], cw["slopes"].shape[1], 1))
    cw["mu_lo"], cw["mu_hi"] = Pw.mu_lo(cw["slopes"]), Pw.mu_hi(cw["slopes"])
    out = dict(lo=lo.reshape(nF, T), hi=hi.reshape(nF, T), ibp_lo=ibp_lo.reshape(nF, T), ibp_hi=ibp_hi.reshape(nF, T),
               logit_lo=cw["logit_lo"].reshape(nF, T), logit_hi=cw["logit_hi"].reshape(nF, T),
               ibp_logit_lo=il.reshape(nF, T), ibp_logit_hi=ih.reshape(nF, T), n_unstable=cw["n_unstable"])
    for k in lags:
        glo, ghi, chi, clo, rg = increment_boxes(cw, (nF, T), k, W)
        out[f"glo|{k}"] = glo; out[f"ghi|{k}"] = ghi; out[f"combo_hi|{k}"] = chi; out[f"combo_lo|{k}"] = clo
        out.setdefault("refine_gain_max", {})[k] = rg
        out[f"ibp_glo|{k}"] = dn(out["ibp_lo"][:, k:] - out["ibp_hi"][:, :-k]); out[f"ibp_ghi|{k}"] = up(out["ibp_hi"][:, k:] - out["ibp_lo"][:, :-k])
    t_inc = time.time() - t0
    out["seconds"] = dict(ibp=t_ibp, crown_values=t_val, crown_increments=t_inc)
    if verbose:
        print("  timings", out["seconds"], flush=True)
    return out


# ---------------------------------------------------------------- self-tests (not part of the enclosure)
def selftest(seed=0):
    """exp/sigmoid enclosures against mpmath (if present) and the sigmoid intercepts against dense sampling"""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-50, 50, 2000), rng.uniform(-1, 1, 2000), [0.0, -0.0, 1e-300, -745.5, 710.0, 700.0, -700.0]])
    lo, hi = exp_enclose(x)
    rep = dict()
    try:
        import mpmath as mp
        mp.mp.prec = 200
        bad = 0; rel = []
        for xi, a, b in zip(x, lo, hi):
            v = mp.e ** mp.mpf(float(xi))
            if not (mp.mpf(float(a)) <= v <= mp.mpf(float(b))):
                bad += 1
            if np.isfinite(b) and a > 0:
                rel.append(float((mp.mpf(float(b)) - mp.mpf(float(a))) / v))
        rep["exp_vs_mpmath_bad"] = bad; rep["exp_rel_width_max"] = float(np.max(rel))
        z = rng.uniform(-30, 30, 2000); sl, sh = sig_enclose(z); bad = 0
        for zi, a, b in zip(z, sl, sh):
            v = 1 / (1 + mp.e ** (-mp.mpf(float(zi))))
            bad += not (mp.mpf(float(a)) <= v <= mp.mpf(float(b)))
        rep["sigmoid_vs_mpmath_bad"] = bad
    except ImportError:
        rep["mpmath"] = "absent"
    # intercepts vs dense sampling
    l = rng.uniform(-6, 6, 3000); w = rng.exponential(0.5, 3000); u = l + w
    s = sig_slope_candidates(l, u)[np.arange(3000), rng.integers(0, 6, 3000)]
    mlo, mhi = sig_intercepts(s, l, u)
    zz = l[:, None] + w[:, None] * np.linspace(0, 1, 2001)[None]
    h = 1 / (1 + np.exp(-zz)) - s[:, None] * zz
    rep["intercept_sampled_violation"] = float(max(np.max(h.max(1) - mhi), np.max(mlo - h.min(1))))
    rep["intercept_slack_median"] = float(np.median((mhi - h.max(1)) + (h.min(1) - mlo)))
    return rep


if __name__ == "__main__":
    print(selftest())
