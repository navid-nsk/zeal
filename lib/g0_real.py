"""
Helpers to rasterize the MAUP-sandbox KNOWN fine field (Guadalajara 2010
population density, 55,146 base units) onto a [0,1]^2 grid.

The derived 160x160 density is cached as data/sandbox_density.npz (shipped with the
data record) and loaded by zeal_m1.load_sandbox(); this module is only needed to
REBUILD that cache from the raw tessellated shapefile (see make_sandbox_density.py).
The raw archive is not redistributed -- place Original_Units.7z (or its extracted
shapefile) under <repo>/raw/sandbox/.

Deps: numpy, geopandas, shapely, py7zr.
"""

import os, glob
import numpy as np
import geopandas as gpd

import paths
import g0_numerics as g0   # reuse grid, families, operators, metrics

DATA = os.path.join(paths.RAW, "sandbox")
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
