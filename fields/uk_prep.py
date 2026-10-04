"""Build the Greater Manchester ladder OA -> LSOA -> MSOA -> LAD as a raster (British National Grid) with the Output
Areas as the fine, fully observed truth.

Outcomes (additive counts from Census 2021, so every aggregation is exact):
  q4      share of residents aged 16+ whose highest qualification is Level 4 or above (TS067); weight = residents 16+
  bad     share of residents in bad or very bad health (TS037); weight = all residents
Writes <data root>/uk_gm/gm_raster.npz with: mask, oa_idx, lsoa_idx, msoa_idx, lad_idx (int32, -1 outside), the per-pixel
weight density for each outcome (pixels of a unit sum to the unit weight), the OA-truth outcome rasters and the exact
LSOA / MSOA / LAD mean rasters; and gm_units.csv with the per-OA table. Usage: python uk_prep.py [--N 1024]
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, sys, json, numpy as np, pandas as pd, geopandas as gpd
from rasterio import features
from affine import Affine
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--N", type=int, default=1024); p.add_argument("--region", default="gm"); a = p.parse_args()
D = paths.uk_dir(a.region)

g = gpd.read_file(D + "oa_boundaries.geojson").set_crs(27700, allow_override=True)
lu = pd.read_csv(D + "lookup.csv")
q = pd.read_csv(D + "ts067_oa.csv"); h = pd.read_csv(D + "ts037_oa.csv")
qc = [c for c in q.columns if c.startswith("Highest level")]; hc = [c for c in h.columns if c.startswith("General health")]
qt = pd.DataFrame({"OA21CD": q["geography code"], "w_q4": q[qc[0]], "n_q4": q[[c for c in qc if "Level 4" in c][0]]})
ht = pd.DataFrame({"OA21CD": h["geography code"], "w_bad": h[hc[0]],
                   "n_bad": h[[c for c in hc if c.endswith("Bad health")][0]] + h[[c for c in hc if "Very bad" in c][0]]})
u = g[["OA21CD", "LSOA21CD", "geometry"]].merge(lu[["OA21CD", "MSOA21CD", "LAD23CD", "LAD23NM"]], on="OA21CD").merge(qt, on="OA21CD").merge(ht, on="OA21CD")
u["q4"] = u["n_q4"] / u["w_q4"].clip(lower=1); u["bad"] = u["n_bad"] / u["w_bad"].clip(lower=1)
u["area_km2"] = u.geometry.area / 1e6
for lvl in ("LSOA21CD", "MSOA21CD", "LAD23CD"):
    u[lvl + "_i"] = pd.factorize(u[lvl], sort=True)[0]
u["OA_i"] = np.arange(len(u))
print(f"{len(u):,} OAs | {u['LSOA21CD'].nunique():,} LSOAs | {u['MSOA21CD'].nunique()} MSOAs | {u['LAD23CD'].nunique()} LADs | "
      f"residents {u['w_bad'].sum():,.0f} | q4 share {u['n_q4'].sum() / u['w_q4'].sum():.3f} | bad-health share {u['n_bad'].sum() / u['w_bad'].sum():.4f} | "
      f"OA area km2 median {u['area_km2'].median():.3f}, max {u['area_km2'].max():.1f}")

# ---- raster ----
x0, y0, x1, y1 = u.total_bounds; side = max(x1 - x0, y1 - y0) * 1.02; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
res = side / a.N; T = Affine(res, 0, cx - side / 2, 0, -res, cy + side / 2)
print(f"raster {a.N}^2, {res:.1f} m per pixel, extent {side / 1000:.1f} km")
oa = features.rasterize(((geom, i) for geom, i in zip(u.geometry, u["OA_i"])), out_shape=(a.N, a.N), transform=T, fill=-1, dtype="int32", all_touched=False)
mask = oa >= 0
missing = sorted(set(u["OA_i"]) - set(np.unique(oa[mask])))
if missing:                                                      # OAs smaller than a pixel: stamp their centroid pixel
    for i in missing:
        c = u.geometry.iloc[i].centroid; col, row = ~T * (c.x, c.y); r_, c_ = int(row), int(col)
        if 0 <= r_ < a.N and 0 <= c_ < a.N:
            oa[r_, c_] = i
    mask = oa >= 0
    print(f"  {len(missing):,} OAs below one pixel were stamped at their centroid; {len(set(u['OA_i']) - set(np.unique(oa[mask])))} still missing")
npix = np.bincount(oa[mask], minlength=len(u))
lsoa = np.where(mask, u["LSOA21CD_i"].values[np.maximum(oa, 0)], -1); msoa = np.where(mask, u["MSOA21CD_i"].values[np.maximum(oa, 0)], -1)
lad = np.where(mask, u["LAD23CD_i"].values[np.maximum(oa, 0)], -1)
out = dict(mask=mask, oa_idx=oa.astype(np.int32), lsoa_idx=lsoa.astype(np.int32), msoa_idx=msoa.astype(np.int32), lad_idx=lad.astype(np.int32),
           m_per_px=np.array([res]), transform=np.array(T.to_gdal()), n_units=np.array([len(u), u["LSOA21CD"].nunique(), u["MSOA21CD"].nunique(), u["LAD23CD"].nunique()]))
for oc, wc in (("q4", "w_q4"), ("bad", "w_bad")):
    dens = np.where(mask, (u[wc].values / np.maximum(npix, 1))[np.maximum(oa, 0)], 0.0)       # weight per pixel
    out[f"weight_{oc}"] = dens.astype(np.float32)
    out[f"truth_{oc}"] = np.where(mask, u[oc].values[np.maximum(oa, 0)], np.nan).astype(np.float32)
    for lvl, idx in (("lsoa", lsoa), ("msoa", msoa), ("lad", lad)):
        key = {"lsoa": "LSOA21CD_i", "msoa": "MSOA21CD_i", "lad": "LAD23CD_i"}[lvl]; K = u[key].max() + 1
        W = np.bincount(u[key], weights=u[wc], minlength=K); V = np.bincount(u[key], weights=u[wc] * u[oc], minlength=K) / np.maximum(W, 1e-9)
        out[f"{lvl}_{oc}"] = np.where(mask, V[np.maximum(idx, 0)], np.nan).astype(np.float32)
    # exact population-weighted movements between rungs (from the unit table, not the raster)
    def mov(a_, b_):
        return float(np.sqrt(np.average((a_ - b_) ** 2, weights=u[wc])))
    Vl = u.groupby("LSOA21CD_i").apply(lambda d: np.average(d[oc], weights=d[wc]))[u["LSOA21CD_i"]].values
    Vm = u.groupby("MSOA21CD_i").apply(lambda d: np.average(d[oc], weights=d[wc]))[u["MSOA21CD_i"]].values
    Vd = u.groupby("LAD23CD_i").apply(lambda d: np.average(d[oc], weights=d[wc]))[u["LAD23CD_i"]].values
    print(f"{oc}: movement OA->LSOA {mov(u[oc].values, Vl):.4f} | LSOA->MSOA {mov(Vl, Vm):.4f} | MSOA->LAD {mov(Vm, Vd):.4f} | OA->LAD {mov(u[oc].values, Vd):.4f} "
          f"| OA s.d. {np.sqrt(np.average((u[oc] - np.average(u[oc], weights=u[wc])) ** 2, weights=u[wc])):.4f}")
np.savez_compressed(D + f"{a.region}_raster.npz", **out)
u.drop(columns="geometry").to_csv(D + f"{a.region}_units.csv", index=False)
# OA adjacency (shared boundary) for contiguous zonings
sj = gpd.sjoin(u[["OA_i", "geometry"]], u[["OA_i", "geometry"]], predicate="touches")
adj = sj[sj["OA_i_left"] < sj["OA_i_right"]][["OA_i_left", "OA_i_right"]].drop_duplicates()
adj.to_csv(D + f"{a.region}_adjacency.csv", index=False)
print(f"adjacency: {len(adj):,} OA pairs | pixels in state {int(mask.sum()):,} | OAs on grid {int((npix > 0).sum()):,}")
print("saved", D + f"{a.region}_raster.npz")
