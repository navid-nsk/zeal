"""Rebuild the cached MAUP-sandbox density (data/sandbox_density.npz) from the raw
Guadalajara 2010 tessellated shapefile.

This is OPTIONAL: the derived cache ships with the data record and is what the
certifier experiments load (via zeal_m1.load_sandbox). Run this only to regenerate
it from scratch. The raw archive (Original_Units.7z, ~128 MB) is not redistributed;
place it under <repo>/raw/sandbox/ first. The output exactly matches the standardized
160x160 field used throughout: lam = log1p(density) / std.

Run:  python data_prep/make_sandbox_density.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import numpy as np
import g0_real

N = 160
shp = g0_real.pick_tessellated(g0_real.ensure_extracted())
print("base shapefile:", shp)
dens = g0_real.rasterize_density(shp, n=N)
lam = np.log1p(dens)
lam = lam / (lam.std() + 1e-9)           # scale-normalized; KEEP >= 0 (softplus/Tobler positivity)
out = paths.data("sandbox_density.npz")
np.savez_compressed(out, lam=lam.astype(np.float32))
print(f"wrote {out}  shape={lam.shape}  mean={lam.mean():.4f}  std={lam.std():.4f}")
