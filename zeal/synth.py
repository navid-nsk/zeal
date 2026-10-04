"""synth.py -- shared synthetic instance generator for exp1 (sinusoidal fields, sound boxes, masks, objectives).
Call set_rng(np.random.Generator) before use."""
try:
    from . import paths                     # imported as part of the package
except ImportError:
    import paths                            # flat import (this directory on sys.path)
import numpy as np
rng = np.random.default_rng(0)
def set_rng(r):
    global rng; rng = r

SPEC = {"georgia": paths.RASTER["georgia"], "gm_q4": paths.RASTER["gm_q4"],
        "gm_bad": paths.RASTER["gm_bad"], "mx_rwi": paths.RASTER["mx_rwi"]}

def field(K=6, scale=2.0):
    am = rng.normal(size=K); B = rng.normal(size=(K, 2)) * scale; c = rng.uniform(0, 2 * np.pi, K)
    f = lambda x, y: sum(am[k] * np.sin(B[k, 0] * x + B[k, 1] * y + c[k]) for k in range(K))
    g1 = lambda x, y: sum(am[k] * B[k, 0] * np.cos(B[k, 0] * x + B[k, 1] * y + c[k]) for k in range(K))
    g2 = lambda x, y: sum(am[k] * B[k, 1] * np.cos(B[k, 0] * x + B[k, 1] * y + c[k]) for k in range(K))
    M1 = np.sum(np.abs(am) * np.sqrt(np.sum(B ** 2, 1))); M2 = np.sum(np.abs(am) * np.sum(B ** 2, 1))
    return f, g1, g2, M1, M2


def build(n1, n2, h, widen, mask):
    f, g1, g2, M1, M2 = field()
    s = 21; t = (np.arange(s) + 0.5) / s
    gl, gw = np.polynomial.legendre.leggauss(8); gl = (gl + 1) / 2; gw = gw / 2
    psi = np.zeros((n1, n2)); lo = np.zeros((n1, n2)); hi = np.zeros((n1, n2)); G1 = np.zeros((n1, n2, 2)); G2 = np.zeros((n1, n2, 2))
    pad1 = M1 * h / s * 1.5; pad2 = M2 * h / s * 1.5
    for i in range(n1):
        for j in range(n2):
            X, Y = np.meshgrid((i + t) * h, (j + t) * h, indexing="ij"); XG, YG = np.meshgrid((i + gl) * h, (j + gl) * h, indexing="ij")
            psi[i, j] = np.sum(np.outer(gw, gw) * f(XG, YG))
            F, A_, B_ = f(X, Y), g1(X, Y), g2(X, Y)
            r = F.max() - F.min(); lo[i, j] = F.min() - pad1 - widen * r * rng.uniform(0.5, 1.5); hi[i, j] = F.max() + pad1 + widen * r * rng.uniform(0.5, 1.5)
            rA, rB = A_.max() - A_.min(), B_.max() - B_.min()
            G1[i, j] = [A_.min() - pad2 - widen * rA * rng.uniform(0.5, 1.5), A_.max() + pad2 + widen * rA * rng.uniform(0.5, 1.5)]
            G2[i, j] = [B_.min() - pad2 - widen * rB * rng.uniform(0.5, 1.5), B_.max() + pad2 + widen * rB * rng.uniform(0.5, 1.5)]
    return dict(psi=psi, lo=lo, hi=hi, G1=G1, G2=G2, h=h, mask=mask)


def random_mask(n1, n2, kind):
    if kind == "full":
        return np.ones((n1, n2), bool)
    m = np.zeros((n1, n2), bool)
    for _ in range(rng.integers(1, 4)):
        i0, j0 = rng.integers(0, n1 - 2), rng.integers(0, n2 - 2); i1, j1 = rng.integers(i0 + 2, n1 + 1), rng.integers(j0 + 2, n2 + 1)
        m[i0:i1, j0:j1] = True
    return m


def random_d(inst, kind):
    mask = inst["mask"]; n1, n2 = mask.shape
    w = rng.uniform(0.2, 1.0, (n1, n2)) * mask                          # pixel population
    if kind == "sparse":
        wa = w * (rng.random((n1, n2)) < 0.4); wb = w * (rng.random((n1, n2)) < 0.4)
        if wa.sum() == 0 or wb.sum() == 0:
            return None
        return wa / wa.sum() - wb / wb.sum()
    if kind == "nested":
        ii, jj = np.nonzero(mask); k = rng.integers(len(ii)); i0, j0 = ii[k], jj[k]
        si, sj = rng.integers(1, 4), rng.integers(1, 4)
        wa = w.copy(); wa[:, :] = 0; wa[i0:i0 + si, j0:j0 + sj] = w[i0:i0 + si, j0:j0 + sj]
        if wa.sum() == 0:
            return None
        return wa / wa.sum() - w / w.sum()
    if kind == "unbalanced":                                              # nonzero mass on a component (exercises the ground)
        wa = w * (rng.random((n1, n2)) < 0.4); wb = w * (rng.random((n1, n2)) < 0.4)
        if wa.sum() == 0 or wb.sum() == 0:
            return None
        return 1.3 * wa / wa.sum() - wb / wb.sum()


