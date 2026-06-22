"""
Spatially-blocked cross-validation (controls for autocorrelation leakage in a random split).
Partition GA counties into K spatially-contiguous blocks (KMeans on county centroids); leave-one-block-out:
train the field to reconstruct tract means on the retained blocks (+smoothness), evaluate reconstruction R2
on the HELD-OUT block's tracts. Reports blocked R2 (honest spatial generalization) vs in-sample, and that
the no-artifact decomposition (signed B_red <= 0) holds out-of-sample. Usage: python geo_m4_spatialcv.py [K] [STATE]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import sys, json, numpy as np, torch, time
from sklearn.cluster import KMeans
from zeal_m1 import dev
import geo_m4_multistate as M

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
STATE = sys.argv[2] if len(sys.argv) > 2 else "13"
s = M.S(STATE)

# county centroids (in [-1,1]^2) -> K spatial blocks
cen = torch.zeros(s.Kco, 2, device=dev)
cnt = torch.zeros(s.Kco, device=dev).index_add_(0, s.inv_co, torch.ones_like(s.w))
cen.index_add_(0, s.inv_co, s.coords); cen /= cnt[:, None].clamp_min(1)
block_of_county = torch.tensor(KMeans(K, n_init=4, random_state=0).fit_predict(cen.cpu().numpy()), device=dev)
block_pix = block_of_county[s.inv_co]                                  # block label per pixel
B_irr = s.wrms(s.Dt, s.county_of(s.Dt, s.tc))

print(f"spatially-blocked CV: state {STATE}, {K} folds (leave-one-spatial-block-out), beta=0.5")
print(f"  county blocks sizes: {[int((block_of_county==b).sum()) for b in range(K)]}")
print(f"{'fold':>4s} {'held_cties':>10s} {'R2_insamp':>10s} {'R2_blocked':>11s} {'B_red_held':>11s}")
rows = []
for f in range(K):
    t = time.time()
    keep_pix = (block_pix != f).float()                               # train on all but block f
    held_pix = (block_pix == f)
    torch.manual_seed(0); net = s.train_masked(0.5, keep_pix)
    with torch.no_grad():
        pred = net(s.coords)
        dtr = s.wmean(pred, s.inv_tr, s.Ktr)
        # in-sample R2 (trained blocks) and blocked R2 (held-out block) -- count-weighted
        def wr2_sub(mask):
            m = mask.float(); wm = s.w * m
            mu = (wm * s.tgt_tr).sum() / wm.sum()
            return float(1 - (wm * (dtr - s.tgt_tr) ** 2).sum() / ((wm * (s.tgt_tr - mu) ** 2).sum() + 1e-12))
        r2_in, r2_held = wr2_sub(block_pix != f), wr2_sub(held_pix)
        # decomposition on held-out tracts only
        Vt = torch.zeros(s.Ktr, device=dev).index_add_(0, s.inv_tr, pred * s.w) / s.Wt.clamp_min(1e-9)
        held_tr = torch.zeros(s.Ktr, device=dev, dtype=torch.bool); held_tr[s.inv_tr[held_pix]] = True
        Wk = s.Wt * held_tr.float()
        mov_held = float((Wk * (Vt - s.county_of(Vt, s.tc)) ** 2).sum().div(Wk.sum().clamp_min(1e-9)).sqrt())
        Birr_held = float((Wk * (s.Dt - s.county_of(s.Dt, s.tc)) ** 2).sum().div(Wk.sum().clamp_min(1e-9)).sqrt())
    rows.append(dict(fold=f, r2_in=r2_in, r2_held=r2_held, B_red_held=mov_held - Birr_held))
    print(f"{f:4d} {int((block_of_county==f).sum()):10d} {r2_in:10.3f} {r2_held:11.3f} {mov_held - Birr_held:+11.4f}   ({time.time()-t:.0f}s)", flush=True)

a = lambda k: np.array([r[k] for r in rows])
print(f"\nBlocked R2 (spatial generalization): {a('r2_held').mean():.3f} +- {a('r2_held').std():.3f}  "
      f"(in-sample {a('r2_in').mean():.3f})")
print(f"Held-out signed B_red: {a('B_red_held').mean():+.4f} +- {a('B_red_held').std():.4f}  "
      f"({(a('B_red_held')<=0).sum()}/{K} folds <= 0 => no manufactured artifact out-of-sample)")
json.dump({"K": K, "state": STATE, "rows": rows}, open(paths.data("geo_m4_spatialcv.json"), "w"))
