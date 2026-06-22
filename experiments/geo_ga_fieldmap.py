"""Prep the GA field maps for Fig 7a: the continuous decoded field + its county-aggregated view + the data,
   on the Georgia grid. Saves zeal/data/ga_fieldmap.npz."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import numpy as np, torch
import geo_m4_multistate as M
from zeal_m1 import dev

s = M.S("13")
torch.manual_seed(0)
net = s.train(0.5)
N = s.N; mi = s.mask.reshape(-1)
with torch.no_grad():
    full = net(s.Xf).view(N, N).cpu().numpy()                 # continuous field on full grid
    pred = net(s.coords)
    Ptr = s.wmean(pred, s.inv_tr, s.Ktr).cpu().numpy()         # tract-decoded (per in-state pixel)
    Pco = s.wmean(pred, s.inv_co, s.Kco).cpu().numpy()         # county-decoded
d = np.load(paths.raster("geo_st13_raster.npz"))
kfr_grid = np.where(s.mask, np.nan_to_num(d["kfr"]), np.nan)

def to_grid(vals):
    g = np.full(N * N, np.nan); g[mi] = vals; return g.reshape(N, N)

# per-tract / per-county predicted vs true (for the 7b scatter)
with torch.no_grad():
    Vt = (torch.zeros(s.Ktr, device=dev).index_add_(0, s.inv_tr, pred * s.w) / s.Wt.clamp_min(1e-9)).cpu().numpy()
    tgt_t = (torch.zeros(s.Ktr, device=dev).index_add_(0, s.inv_tr, s.kfr * s.w) / s.Wt.clamp_min(1e-9)).cpu().numpy()
    Wc = torch.zeros(s.Kco, device=dev).index_add_(0, s.inv_co, s.w)
    Vc = (torch.zeros(s.Kco, device=dev).index_add_(0, s.inv_co, pred * s.w) / Wc.clamp_min(1e-9)).cpu().numpy()
    tgt_c = (torch.zeros(s.Kco, device=dev).index_add_(0, s.inv_co, s.kfr * s.w) / Wc.clamp_min(1e-9)).cpu().numpy()

np.savez(paths.data("ga_fieldmap.npz"),
         field=np.where(s.mask, full, np.nan), kfr=kfr_grid,
         tr=to_grid(Ptr), co=to_grid(Pco), mask=s.mask,
         tract_pred=Vt, tract_true=tgt_t, county_pred=Vc, county_true=tgt_c)
r2tr = s.wr2(s.wmean(torch.tensor(net(s.coords).detach()), s.inv_tr, s.Ktr) if False else s.wmean(net(s.coords).detach(), s.inv_tr, s.Ktr), s.tgt_tr)
print(f"saved ga_fieldmap.npz  (N={N}, R2_tract={r2tr:.3f}, field range "
      f"[{np.nanmin(kfr_grid):.3f},{np.nanmax(kfr_grid):.3f}])")
