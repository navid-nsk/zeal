"""
Rasterize the whole-state nested ladder onto a grid for field training + certification.
Outputs (npz): mask (in-state), coords in [-1,1]^2, per-pixel tract/bg/county labels,
the Atlas kfr_p25 target (from tract), and the count weight (pooled_pooled_count, from tract).

Usage:  python geo_raster.py 13 512
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import numpy as np, geopandas as gpd
import rasterio.features as rf
from rasterio.transform import from_bounds

STATE = sys.argv[1] if len(sys.argv) > 1 else "13"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512

tr = gpd.read_file(paths.geom(f"geo_st{STATE}_tract.gpkg"))
bg = gpd.read_file(paths.geom(f"geo_st{STATE}_bg.gpkg"))
cnt = tr.dissolve(by="county").reset_index()

minx, miny, maxx, maxy = tr.total_bounds
transform = from_bounds(minx, miny, maxx, maxy, N, N)

def rasterize_idx(gdf):                          # burn row index+1 (0 = background/outside)
    return rf.rasterize(((g, i + 1) for i, g in enumerate(gdf.geometry)),
                        out_shape=(N, N), transform=transform, fill=0, dtype="int32")

tract_idx = rasterize_idx(tr)
bg_idx = rasterize_idx(bg)
county_idx = rasterize_idx(cnt)
mask = tract_idx > 0

# per-pixel target + weight from tract
kfr_vals = tr["kfr_pooled_pooled_p25"].to_numpy()
cnt_vals = tr["pooled_pooled_count"].to_numpy()
ti = np.clip(tract_idx - 1, 0, len(tr) - 1)
kfr = np.where(mask, kfr_vals[ti], np.nan).astype(np.float32)
weight = np.where(mask, cnt_vals[ti], 0.0).astype(np.float32)

# coords in [-1,1]^2 (image convention: row=x, col=y to match zeal_m1 grid)
xs = np.linspace(-1, 1, N).astype(np.float32)
X, Y = np.meshgrid(xs, xs, indexing="ij")

cov = mask.mean()
print(f"state {STATE} grid {N}x{N}: in-state coverage {cov:.1%}")
print(f"  unique tracts on grid {len(np.unique(tract_idx))-1}/{len(tr)}, "
      f"bg {len(np.unique(bg_idx))-1}/{len(bg)}, county {len(np.unique(county_idx))-1}/{len(cnt)}")
print(f"  kfr per-pixel range [{np.nanmin(kfr):.3f},{np.nanmax(kfr):.3f}]")
# how many cells get >=1 pixel (resolution check)
for name, idx in [("tract", tract_idx), ("bg", bg_idx)]:
    u, c = np.unique(idx[idx > 0], return_counts=True)
    print(f"  {name}: median pixels/cell {np.median(c):.0f}, cells with <3 px: {(c<3).mean():.1%}")

suf = "" if N == 512 else f"_{N}"                       # 512 = standard name; other res get a suffix
np.savez_compressed(paths.raster(f"geo_st{STATE}_raster{suf}.npz"),
                    mask=mask, X=X, Y=Y, tract_idx=tract_idx, bg_idx=bg_idx,
                    county_idx=county_idx, kfr=kfr, weight=weight)
print(f"saved geo_st{STATE}_raster{suf}.npz")
