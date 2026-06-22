"""D2: sliver-robustness. A naive cell-mean E[lambda|cell] is ill-conditioned on a thin sliver (its value
   swings wildly as the sliver shifts), while the mollified aggregator R_Z^delta = phi_delta * E stays bounded.
   Sweep sliver width w; at each w place the sliver at many positions and measure the SPREAD (instability) of
   naive vs mollified aggregate. Saves zeal/data/sliver.json."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import json, numpy as np
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(0)
N = 512
xs = np.linspace(0, 1, N); X, Y = np.meshgrid(xs, xs, indexing="ij")
# structured field = smooth bumps + SUB-delta high-frequency texture (in the sliver-width direction).
# The mollifier washes out the sub-delta part; a naive thin-sliver mean does NOT -> it is unstable.
field = np.zeros((N, N))
for _ in range(6):
    cx, cy = rng.uniform(0.15, 0.85, 2); s = rng.uniform(0.05, 0.12); a = rng.uniform(0.5, 1.5)
    field += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2))
field = field / field.max()
delta = 0.025
field = field + 0.45 * np.sin(2 * np.pi * 45 * Y)                 # sub-delta texture (wavelength 0.022 < delta)
field_moll = gaussian_filter(field, sigma=delta * N)              # phi_delta * lambda removes the sub-delta part

widths = [0.004, 0.008, 0.016, 0.032, 0.064, 0.128]
L = 0.5; npos = 60
rows = []
for w in widths:
    naive, moll = [], []
    for _ in range(npos):
        y0 = rng.uniform(0.2, 0.8); x0 = rng.uniform(0.05, 0.45)
        sl = (X >= x0) & (X < x0 + L) & (Y >= y0) & (Y < y0 + w)   # thin horizontal sliver
        if sl.sum() < 3:
            # ensure at least a 1-px band
            j = int(y0 * N); sl = np.zeros((N, N), bool); sl[int(x0*N):int((x0+L)*N), j:j+1] = True
        naive.append(field[sl].mean())
        moll.append(field_moll[sl].mean())
    rows.append({"w": w, "naive_spread": float(np.std(naive)), "moll_spread": float(np.std(moll)),
                 "naive_mean": float(np.mean(naive)), "moll_mean": float(np.mean(moll))})
    print(f"w={w:.3f}  naive_spread={rows[-1]['naive_spread']:.4f}  moll_spread={rows[-1]['moll_spread']:.4f}", flush=True)

# also save a small field tile + one sliver mask for the inset
ins = field[140:380, 120:400]
np.savez(paths.data("sliver.npz"), field_tile=ins, delta=delta)
json.dump({"delta": delta, "rows": rows}, open(paths.data("sliver.json"), "w"))
print("saved sliver.json + sliver.npz")
