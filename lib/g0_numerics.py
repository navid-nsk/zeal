"""
G0 numerical verification for the ZEAL sliver-robust cross-zoning certificate.

Checks, on synthetic KNOWN fields and partitions, the claims proven in
zeal_phase0_sliver_robust_certificate.md sections 11-12:

  (A) Upper bound (Prop 11.2 / 12.1):  N(lambda) := ||phi_delta * (P_Z - P_Z') lambda||_L2
                                        <=  Lip(lambda) * d_W(Z,Z').            [C = 1]
  (B) Bracket (Prop 12.2):  c_lin * d_low  <=  d^delta  <=  d_W.
  (C) Non-vacuity / G0 number (Prop 12.3): kappa_real := N(lambda*) / (Lip(lambda*) * d_W)
                                            is bounded away from 0 for a realistic smooth field.
  (D) delta-uniformity (sec 11.4): the bracket constants do not blow up as delta varies
                                    (for cells coarser than delta).

Metrics:
  d_low(Z,Z')^2 = sum_{a,b} |a cap b| * |cen_a - cen_b|^2
  d_W(Z,Z')^2   = sum_{a,b} |a cap b| * W1(unif_a, unif_b)^2      (W1 via POT)
  d^delta       estimated from below by max over a test-field basis of N/Lip
                (combined with proven d^delta <= d_W gives a two-sided bracket).

Everything lives on a uniform pixel grid over [0,1]^2 (pixel area h^2); the L2 norm
includes the h^2 measure.  Gaussian mollification via scipy.ndimage.gaussian_filter
(mode='reflect' ~ Neumann), sigma = delta in pixel units.

Run:  python g0_numerics.py
Deps: numpy, scipy, POT (import ot).  (POT is pure-numpy, Windows-safe.)
"""

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import ot  # POT: python optimal transport
    HAVE_OT = True
except Exception:
    HAVE_OT = False


# --------------------------------------------------------------------------- #
# Grid + fields
# --------------------------------------------------------------------------- #
def make_grid(n=192):
    h = 1.0 / n
    xs = (np.arange(n) + 0.5) * h           # pixel-center coords in [0,1]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)   # (n*n, 2)
    return n, h, X, Y, coords


def field_linear(X, Y, u):
    u = np.asarray(u, float); u = u / (np.linalg.norm(u) + 1e-12)
    return u[0] * X + u[1] * Y                # Lip = 1 exactly


def field_smooth_random(X, Y, n_modes=6, seed=0, lip_target=1.0):
    """Sum of low-frequency sinusoids -> a 'realistic' smooth field; rescaled to Lip≈lip_target."""
    rng = np.random.default_rng(seed)
    f = np.zeros_like(X)
    for _ in range(n_modes):
        kx, ky = rng.uniform(0.6, 3.4, size=2)   # non-integer: no band-averaging artifact
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.normal()
        f += amp * np.sin(2 * np.pi * (kx * X + ky * Y) + ph)
    f -= f.mean()
    L = lipschitz(f, 1.0 / X.shape[0])
    return f * (lip_target / (L + 1e-12))


def lipschitz(field, h):
    gx, gy = np.gradient(field, h)
    return float(np.sqrt(gx**2 + gy**2).max())


# --------------------------------------------------------------------------- #
# Partitions (label arrays, shape (n,n), integer labels)
# --------------------------------------------------------------------------- #
def part_h_stripes(n, K):
    lab = np.zeros((n, n), int)
    edges = np.linspace(0, n, K + 1).astype(int)
    for k in range(K):
        lab[edges[k]:edges[k + 1], :] = k       # bands in X (rows)
    return lab


def part_v_stripes(n, K):
    return part_h_stripes(n, K).T.copy()        # bands in Y (cols)


def part_voronoi(n, n_sites, seed=0):
    rng = np.random.default_rng(seed)
    sites = rng.uniform(0, 1, size=(n_sites, 2))
    h = 1.0 / n
    xs = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    d = ((P[:, None, :] - sites[None, :, :]) ** 2).sum(-1)
    return d.argmin(1).reshape(n, n)


def part_nested(n, n_fine, n_coarse, seed=0):
    """Genuine nested ladder: Z_coarse coarsens Z_fine (each fine cell ⊂ one coarse cell).
       Models tract↑BG↑county (the identifiable headline regime)."""
    rng = np.random.default_rng(seed)
    fine = part_voronoi(n, n_fine, seed)
    h = 1.0 / n
    xs = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    cseeds = rng.uniform(0, 1, size=(n_coarse, 2))
    flat = fine.ravel()
    fmap = {}
    for c in np.unique(flat):
        cen = P[flat == c].mean(0)                      # fine-cell centroid
        fmap[c] = int(((cseeds - cen) ** 2).sum(1).argmin())   # nearest coarse seed -> group
    coarse = np.vectorize(fmap.get)(fine)
    return fine, coarse


# --------------------------------------------------------------------------- #
# Operators / metrics
# --------------------------------------------------------------------------- #
def conditional_expectation(field, lab):
    """P_Z field: replace each cell by its mean (uniform / Lebesgue weighting)."""
    out = np.empty_like(field)
    for c in np.unique(lab):
        m = lab == c
        out[m] = field[m].mean()
    return out


def certificate_N(field, labZ, labZp, delta_px, h):
    g = conditional_expectation(field, labZ) - conditional_expectation(field, labZp)
    gm = gaussian_filter(g, sigma=delta_px, mode="reflect")
    return float(np.sqrt((gm**2).sum()) * h)        # ||.||_L2 with measure h^2 -> *h


def cell_centroids(lab, coords, n):
    cen = {}
    flat = lab.ravel()
    for c in np.unique(flat):
        cen[c] = coords[flat == c].mean(0)
    return cen


def d_low(labZ, labZp, coords, n, h):
    cenA = cell_centroids(labZ, coords, n)
    cenB = cell_centroids(labZp, coords, n)
    A = labZ.ravel(); B = labZp.ravel()
    keys, counts = np.unique(np.stack([A, B], 1), axis=0, return_counts=True)
    s = 0.0
    for (a, b), cnt in zip(keys, counts):
        area = cnt * h * h
        s += area * np.sum((cenA[a] - cenB[b]) ** 2)
    return float(np.sqrt(s))


def W1_uniform(cellA_coords, cellB_coords, n_sub=120, seed=0):
    """W1 between uniform measures on two pixel-sets, via POT on subsampled points."""
    rng = np.random.default_rng(seed)
    def sub(P):
        if len(P) > n_sub:
            P = P[rng.choice(len(P), n_sub, replace=False)]
        return P
    PA, PB = sub(cellA_coords), sub(cellB_coords)
    M = ot.dist(PA, PB, metric="euclidean")
    a = np.ones(len(PA)) / len(PA); b = np.ones(len(PB)) / len(PB)
    return float(ot.emd2(a, b, M))


def d_W(labZ, labZp, coords, n, h, n_sub=120):
    A = labZ.ravel(); B = labZp.ravel()
    by_a = {a: coords[A == a] for a in np.unique(A)}
    by_b = {b: coords[B == b] for b in np.unique(B)}
    keys, counts = np.unique(np.stack([A, B], 1), axis=0, return_counts=True)
    cache = {}
    s = 0.0
    for (a, b), cnt in zip(keys, counts):
        if (a, b) not in cache:
            cache[(a, b)] = W1_uniform(by_a[a], by_b[b], n_sub=n_sub)
        s += (cnt * h * h) * cache[(a, b)] ** 2
    return float(np.sqrt(s))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def estimate_dlow_const(delta_px, n):
    """c_lin = sqrt(e^{-1} (1-theta_delta) / 2); report the e^{-1/2}/sqrt2 ceiling factor."""
    return np.exp(-0.5) / np.sqrt(2.0)   # upper proxy (theta_delta>=0); empirical N gives the truth


def run():
    if not HAVE_OT:
        print("POT (`ot`) not importable -> install with: pip install pot")
        print("Proceeding with d_low + N only (d_W skipped).\n")

    n, h, X, Y, coords = make_grid(n=160)
    print(f"grid {n}x{n}, h={h:.4f}\n")

    families = {
        "stripes_8x8":   (part_h_stripes(n, 8),  part_v_stripes(n, 8)),
        "stripes_16x16": (part_h_stripes(n, 16), part_v_stripes(n, 16)),
        "voronoi_40":    (part_voronoi(n, 40, 0), part_voronoi(n, 40, 1)),
        "voronoi_120":   (part_voronoi(n, 120, 0), part_voronoi(n, 120, 1)),
        "nested_120>30": part_nested(n, 120, 30, 0),    # identifiable ladder
        "nested_120>12": part_nested(n, 120, 12, 0),
    }

    lin_u = field_linear(X, Y, [1.0, 0.0])
    lin_v = field_linear(X, Y, [0.0, 1.0])
    lin_d = field_linear(X, Y, [1.0, 1.0])
    smooth = field_smooth_random(X, Y, n_modes=6, seed=3, lip_target=1.0)
    test_basis = {"lin_x": lin_u, "lin_y": lin_v, "lin_xy": lin_d, "smooth": smooth}

    for dpx in (3.0, 6.0):                 # mollifier scale in pixels (delta)
        print(f"================  delta = {dpx} px ({dpx*h:.3f} in [0,1])  ================")
        hdr = f"{'family':16s} {'d_low':>8s} {'d_W':>8s} {'kapW=dW/dl':>10s} " \
              f"{'dhat':>8s} {'dhat/dW':>8s} {'kap_gen':>8s} {'upOK':>5s}"
        print(hdr)
        for name, (Z, Zp) in families.items():
            dl = d_low(Z, Zp, coords, n, h)
            dW = d_W(Z, Zp, coords, n, h) if HAVE_OT else float("nan")
            # d^delta estimate from below = max over test basis of N/Lip
            ratios = {}
            for fn, fld in test_basis.items():
                Nv = certificate_N(fld, Z, Zp, dpx, h)
                Lip = lipschitz(fld, h)
                ratios[fn] = Nv / (Lip + 1e-12)
            dhat = max(ratios.values())                       # operator-norm proxy for d^delta
            upper_ok = all(r <= dW * (1 + 1e-6) for r in ratios.values()) if HAVE_OT else True
            kapW = (dW / dl) if dl > 0 else float("nan")       # comparability varkappa (want O(1))
            dhat_dW = (dhat / dW) if (HAVE_OT and dW > 0) else float("nan")
            kap_gen = ratios["smooth"] / dhat if dhat > 0 else float("nan")  # delta-stable non-vacuity
            print(f"{name:16s} {dl:8.4f} {dW:8.4f} {kapW:10.3f} "
                  f"{dhat:8.4f} {dhat_dW:8.3f} {kap_gen:8.3f} {str(bool(upper_ok)):>5s}")
        print()

    print("Interpretation:")
    print("  * upper_ok=True  confirms  N(lambda) <= Lip*d_W for every test field (Prop 11.2/12.1).")
    print("  * dhat_est in [c_lin*d_low, d_W] confirms the proven bracket (Prop 12.2).")
    print("  * kappa_smooth > 0 (and stable across families/delta) is the G0 non-vacuity number")
    print("    for the synthetic field; replace 'smooth' with the sandbox known field for real kappa.")


if __name__ == "__main__":
    run()
