"""Build a known-truth ladder for central-west Mexico: Relative Wealth Index tiles (2.4 km; Chi et al. 2022) ->
municipios (GADM level 2) -> states (GADM level 1), with WorldPop 2020 population (1 km) as the weight.
The RWI tile is the fine, observed truth; the analyst is assumed to see only municipio (and state) population-weighted
means. Raster: 1024^2 in the Mexico LCC projection (EPSG:6372), about 0.9 km per pixel.
Output: <data root>/mexico/mx_raster.npz (mask, unit_idx = RWI tile, muni_idx, state_idx, weight = population per pixel,
truth = RWI per pixel, muni/state mean rasters, m_per_px) and mx_units.csv, mx_adjacency.csv.
Usage: python mx_prep.py [--N 1024] [--states "Jalisco,Guanajuato,..."]
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, os, sys, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from affine import Affine
from scipy.spatial import cKDTree
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--N", type=int, default=1024)
p.add_argument("--states", default="Jalisco,Guanajuato,Michoacán,Aguascalientes,Colima,Nayarit,Zacatecas,Querétaro,SanLuisPotosí")
a = p.parse_args(); D = paths.SOURCES_DIR; OUT = paths.MX_DIR; os.makedirs(OUT, exist_ok=True); CRS = 6372

adm2 = gpd.read_file(D + "gadm/gadm41_MEX.gpkg", layer="ADM_ADM_2")
want = [s.replace("SanLuisPotosí", "San Luis Potosí") for s in a.states.split(",")]
adm2 = adm2[adm2["NAME_1"].isin(want)].to_crs(CRS).reset_index(drop=True)
adm2["muni_i"] = np.arange(len(adm2)); adm2["state_i"] = pd.factorize(adm2["GID_1"], sort=True)[0]
print(f"{len(adm2)} municipios in {adm2['state_i'].nunique()} states: {sorted(adm2['NAME_1'].unique())}")
x0, y0, x1, y1 = adm2.total_bounds; side = max(x1 - x0, y1 - y0) * 1.02; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
res = side / a.N; T = Affine(res, 0, cx - side / 2, 0, -res, cy + side / 2); print(f"raster {a.N}^2, {res / 1000:.2f} km per pixel, extent {side / 1000:.0f} km")
muni = features.rasterize(((g, i) for g, i in zip(adm2.geometry, adm2["muni_i"])), out_shape=(a.N, a.N), transform=T, fill=-1, dtype="int32")
mask = muni >= 0; state = np.where(mask, adm2["state_i"].values[np.maximum(muni, 0)], -1)

# population: WorldPop 1 km counts (WGS84) -> density on the LCC grid -> count per pixel
with rasterio.open(D + "worldpop/mex_ppp_2020_1km_Aggregated_UNadj.tif") as src:
    cnt = src.read(1).astype("float64"); cnt[cnt < 0] = 0; nd = src.nodata
    if nd is not None:
        cnt[cnt == nd] = 0
    # area of a source cell (30 arcsec) depends on latitude: convert counts to density (per km^2) before resampling
    lat = np.array([src.xy(r, 0)[1] for r in range(src.height)]); cell_km2 = (111.32 / 120) ** 2 * np.cos(np.deg2rad(lat))[:, None]
    dens = cnt / cell_km2
    dst = np.zeros((a.N, a.N), dtype="float64")
    reproject(dens, dst, src_transform=src.transform, src_crs=src.crs, dst_transform=T, dst_crs=f"EPSG:{CRS}", resampling=Resampling.bilinear)
weight = np.where(mask, dst * (res / 1000) ** 2, 0.0)                       # people per pixel
print(f"population on grid: {weight.sum() / 1e6:.2f} M (WorldPop total in the selected states)")

# RWI tiles -> nearest-tile assignment of pixels (tiles are ~2.4 km; cap at 2.0 km from the tile centre)
rwi = pd.read_csv(D + "rwi/MEX_rwi.csv"); pts = gpd.GeoSeries(gpd.points_from_xy(rwi.longitude, rwi.latitude), crs=4326).to_crs(CRS)
ii, jj = np.nonzero(mask); px, py = T * (jj + 0.5, ii + 0.5)
tree = cKDTree(np.c_[pts.x, pts.y]); dist, tid = tree.query(np.c_[px, py]); near = dist <= 2000.0
unit = np.full((a.N, a.N), -1, dtype=np.int32); unit[ii[near], jj[near]] = tid[near]
used = np.unique(unit[unit >= 0]); remap = -np.ones(len(rwi), dtype=np.int64); remap[used] = np.arange(len(used))
unit = np.where(unit >= 0, remap[np.maximum(unit, 0)], -1).astype(np.int32)
truth = np.where(unit >= 0, rwi["rwi"].values[used][np.maximum(unit, 0)], np.nan).astype(np.float32)
err = np.where(unit >= 0, rwi["error"].values[used][np.maximum(unit, 0)], np.nan).astype(np.float32)
w_obs = np.where(unit >= 0, weight, 0.0)                                      # population with an observed RWI
print(f"RWI tiles used: {len(used):,} | population with a tile within 2 km: {w_obs.sum() / weight.sum():.3f}")
# unit table and exact population-weighted municipio / state means (truth aggregated)
K = len(used); u_w = np.bincount(unit[unit >= 0], weights=w_obs[unit >= 0], minlength=K)
u_muni = np.zeros(K, dtype=int); np.maximum.at(u_muni, unit[unit >= 0], muni[unit >= 0])              # dominant by max (tiles rarely straddle)
cnt_mm = pd.DataFrame({"u": unit[unit >= 0], "m": muni[unit >= 0], "w": w_obs[unit >= 0]}).groupby(["u", "m"])["w"].sum().reset_index()
u_muni = cnt_mm.sort_values("w").groupby("u")["m"].last().reindex(range(K)).fillna(0).astype(int).values
units = pd.DataFrame({"unit_i": np.arange(K), "rwi": rwi["rwi"].values[used], "rwi_error": rwi["error"].values[used], "pop": u_w,
                      "muni_i": u_muni, "state_i": adm2["state_i"].values[u_muni], "muni_name": adm2["NAME_2"].values[u_muni], "state_name": adm2["NAME_1"].values[u_muni]})
Wm = np.bincount(muni[unit >= 0], weights=w_obs[unit >= 0], minlength=len(adm2)); Vm = np.bincount(muni[unit >= 0], weights=(w_obs * truth)[unit >= 0], minlength=len(adm2)) / np.maximum(Wm, 1e-9)
Ws = np.bincount(state[unit >= 0], weights=w_obs[unit >= 0]); Vs = np.bincount(state[unit >= 0], weights=(w_obs * truth)[unit >= 0]) / np.maximum(Ws, 1e-9)
muni_r = np.where(mask, Vm[np.maximum(muni, 0)], np.nan).astype(np.float32); state_r = np.where(mask, Vs[np.maximum(state, 0)], np.nan).astype(np.float32)
wr = lambda x, y, w: float(np.sqrt(np.average((x - y) ** 2, weights=w)))
sel = unit >= 0
print(f"movement tile->muni {wr(truth[sel], muni_r[sel], w_obs[sel]):.4f} | muni->state {wr(muni_r[sel], state_r[sel], w_obs[sel]):.4f} | tile s.d. {np.sqrt(np.average((truth[sel] - np.average(truth[sel], weights=w_obs[sel])) ** 2, weights=w_obs[sel])):.4f} | median RWI s.e. {np.median(units.rwi_error):.3f}")
np.savez_compressed(OUT + "mx_raster.npz", mask=mask, unit_idx=unit, muni_idx=muni.astype(np.int32), state_idx=state.astype(np.int32), weight=w_obs.astype(np.float32),
                    weight_all=weight.astype(np.float32), truth=truth, truth_se=err, muni_mean=muni_r, state_mean=state_r, m_per_px=np.array([res]), transform=np.array(T.to_gdal()))
units.to_csv(OUT + "mx_units.csv", index=False)
# tile adjacency through pixel neighbours
A = set()
for dy, dx in ((0, 1), (1, 0)):
    a_, b_ = unit[:-dy or None, :-dx or None] if (dy or dx) else unit, unit[dy:, dx:]
    m_ = (a_ >= 0) & (b_ >= 0) & (a_ != b_); A |= set(zip(np.minimum(a_[m_], b_[m_]).tolist(), np.maximum(a_[m_], b_[m_]).tolist()))
pd.DataFrame(sorted(A), columns=["unit_a", "unit_b"]).to_csv(OUT + "mx_adjacency.csv", index=False)
print(f"adjacency: {len(A):,} tile pairs | saved {OUT}mx_raster.npz")
