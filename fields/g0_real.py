"""
G0 real-field measurement: kappa_real for the certificate, using the MAUP-sandbox
KNOWN fine field (Guadalajara 2010 population density, 55,146 base units).

Pipeline:
  1. extract Original_Units.7z -> Tessellated base shapefile (P2010, AREA, POP_DENS).
  2. rasterize POP_DENS onto a [0,1]^2 grid (bbox rescaled) -> lambda*_real  (raw and log1p).
  3. run the SAME certified machinery (g0_numerics) over the stripe / voronoi / nested families,
     measuring  kappa_real = N(lambda*) / (Lip(lambda*) * dhat)  vs the synthetic smooth field.

GO signal: kappa_real > 0 and comparable to the synthetic kappa (~0.2), delta-stable.
This is the real-field instance of check (C) of g0_numerics.py (the single G0 non-vacuity constant).

Run (after the download finishes):  python g0_real.py
Deps: numpy, geopandas, shapely, py7zr, POT  (all in env `zeal`).
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT

import os, glob
import numpy as np
import geopandas as gpd

import g0_numerics as g0   # reuse grid, families, operators, metrics

DATA = paths.SANDBOX_DIR
ARCHIVE = os.path.join(DATA, "Original_Units.7z")


# --------------------------------------------------------------------------- #
def ensure_extracted():
    shp = glob.glob(os.path.join(DATA, "**", "*essellat*", "*.shp"), recursive=True)
    if not shp:
        shp = glob.glob(os.path.join(DATA, "**", "*.shp"), recursive=True)
    if shp:
        return shp
    import py7zr
    print("extracting", ARCHIVE)
    with py7zr.SevenZipFile(ARCHIVE, "r") as z:
        z.extractall(DATA)
    shp = glob.glob(os.path.join(DATA, "**", "*.shp"), recursive=True)
    return shp


def pick_tessellated(shps):
    for s in shps:
        if "essellat" in s.lower():
            return s
    return shps[0]


def rasterize_density(shp, n=200, value_col=None):
    """Return lambda*_raw (n,n) of POP_DENS on a [0,1]^2 grid (bbox rescaled to unit square)."""
    g = gpd.read_file(shp)
    cols = {c.lower(): c for c in g.columns}
    print("columns:", list(g.columns))
    if "pop_dens" in cols:
        value_col = cols["pop_dens"]
    else:                                    # derive INTENSITY density = P2010 / AREA
        pc = cols.get("p2010"); ac = cols.get("area")
        g["__dens__"] = g[pc] / g[ac].replace(0, np.nan)
        value_col = "__dens__"
    print("using value column (intensity/density):", value_col)

    minx, miny, maxx, maxy = g.total_bounds
    sx, sy = (maxx - minx), (maxy - miny)
    h = 1.0 / n
    xs = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    # grid-point coords back in original CRS
    from shapely.geometry import Point
    px = minx + X.ravel() * sx
    py = miny + Y.ravel() * sy
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(px, py), crs=g.crs)
    j = gpd.sjoin(pts, g[[value_col, "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated(keep="first")]            # one polygon per point
    field = j[value_col].to_numpy(dtype=float).reshape(n, n)
    field = np.nan_to_num(field, nan=0.0)               # points outside study area -> 0
    cover = np.mean(field > 0)
    print(f"raster {n}x{n}, study-area coverage {cover:.1%}, "
          f"density range [{field.min():.1f}, {field.max():.1f}]")
    return field


# --------------------------------------------------------------------------- #
def run():
    shps = ensure_extracted()
    if not shps:
        print("No shapefile found; is the download complete?", DATA); return
    shp = pick_tessellated(shps)
    print("base shapefile:", shp)

    from scipy.ndimage import gaussian_filter
    n, h, X, Y, coords = g0.make_grid(n=180)
    dens = rasterize_density(shp, n=n)
    logd = np.log1p(dens)                                 # intensity on log scale (softplus-like head)

    # The certificate is about the SMOOTH LEARNED field (SIREN + softplus + smoothness loss),
    # NOT the raw piecewise-constant data (whose ||grad||_inf is dominated by inter-unit jumps).
    # Smoothing logd at model-resolution scales is the faithful proxy for lambda*_learned.
    smooth = g0.field_smooth_random(X, Y, n_modes=6, seed=3, lip_target=1.0)
    fields = {
        "real_raw":   logd,                               # unsmoothed: Lip huge -> kappa tiny (expected)
        "real_s4":    gaussian_filter(logd, 4.0),         # learned-field proxy, sigma=4px
        "real_s8":    gaussian_filter(logd, 8.0),         # learned-field proxy, sigma=8px
        "synthetic":  smooth,                             # reference
    }

    families = {
        "voronoi_40":    (g0.part_voronoi(n, 40, 0), g0.part_voronoi(n, 40, 1)),
        "voronoi_120":   (g0.part_voronoi(n, 120, 0), g0.part_voronoi(n, 120, 1)),
        "nested_120>30": g0.part_nested(n, 120, 30, 0),
    }

    for dpx in (3.0, 6.0):
        print(f"\n========  delta = {dpx} px  ========")
        print(f"{'family':14s} {'field':10s} {'Lip':>9s} {'N':>9s} {'dhat':>8s} {'kappa':>7s}")
        for fam, (Z, Zp) in families.items():
            # operator-norm proxy dhat from the linear/synthetic basis (field-independent geometry)
            basis = {"lx": g0.field_linear(X, Y, [1, 0]),
                     "ly": g0.field_linear(X, Y, [0, 1]),
                     "lxy": g0.field_linear(X, Y, [1, 1]),
                     "syn": smooth}
            dhat = max(g0.certificate_N(f, Z, Zp, dpx, h) / (g0.lipschitz(f, h) + 1e-12)
                       for f in basis.values())
            for fname, fld in fields.items():
                Lip = g0.lipschitz(fld, h)
                Nv = g0.certificate_N(fld, Z, Zp, dpx, h)
                kappa = Nv / (Lip * dhat + 1e-12)
                print(f"{fam:14s} {fname:10s} {Lip:9.2f} {Nv:9.4f} {dhat:8.4f} {kappa:7.3f}")
    print("\nGO iff kappa(dens_raw/dens_log) > 0 and comparable to synthetic, delta-stable.")


if __name__ == "__main__":
    run()
