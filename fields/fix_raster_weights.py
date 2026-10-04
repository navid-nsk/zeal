"""Make the raster weight layer a population DENSITY (children per pixel) instead of the tract count replicated on
every pixel, so that every count-weighted quantity in the pipeline (tract weights Wt = sum of pixel weights, pixel-level
population shares, population-fat windows, the L^2(rho) movements) is weighted by population and not by population
times area. The tract count is kept in a new layer 'count' (per pixel, replicated as before) and the original file is
backed up once as <name>.countxarea.npz. Idempotent: a file that already carries 'weight_is_density' is left alone.
Usage: python fix_raster_weights.py [paths...]  (default: every <data root>/rasters/geo_st*_raster*.npz)
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import sys, glob, os, shutil
import numpy as np

paths = sys.argv[1:] or sorted(glob.glob(paths.data("rasters", "geo_st*_raster*.npz")))
for p in paths:
    if p.endswith(".countxarea.npz"):
        continue
    d = dict(np.load(p))
    if "weight_is_density" in d:
        print(f"{os.path.basename(p)}: already a density layer, skipped"); continue
    mask = d["mask"].astype(bool); tr = d["tract_idx"]; w = d["weight"].astype(np.float64)
    K = int(tr.max()) + 1
    npix = np.bincount(tr[mask], minlength=K).astype(np.float64)            # pixels per tract on the grid
    dens = np.zeros_like(w)
    dens[mask] = w[mask] / np.maximum(npix[tr[mask]], 1.0)
    bak = p[:-4] + ".countxarea.npz"
    if not os.path.exists(bak):
        shutil.copy(p, bak)
    d["count"] = d["weight"].astype(np.float32)                              # tract count on every pixel (as before)
    d["weight"] = dens.astype(np.float32)                                   # children per pixel: sums to the tract count
    d["weight_is_density"] = np.array([1])
    np.savez_compressed(p, **d)
    Wt_old = np.bincount(tr[mask], weights=w[mask], minlength=K); Wt_new = np.bincount(tr[mask], weights=dens[mask], minlength=K)
    ok = npix > 0
    print(f"{os.path.basename(p)}: tracts on grid {int(ok.sum())}, total weight {Wt_old[ok].sum():.4g} -> {Wt_new[ok].sum():.4g} "
          f"(= sum of tract counts), max |Wt_new - count| = {np.abs(Wt_new[ok] - w[mask][np.unique(tr[mask], return_index=True)[1]]).max():.3g}; backup {os.path.basename(bak)}")
