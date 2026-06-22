"""
Core experiment: train the certified ZEAL field to reconstruct the count-weighted Opportunity-Atlas
kfr_p25 on the REAL Georgia nested ladder, then measure cross-zoning stability.

  field lambda(x) = softplus(MLP_tanh(Fourier(x)))  (the certified architecture)
  intensive count-weighted aggregation:  pred_a = sum_{x in a} w(x) lambda(x) / sum_{x in a} w(x),
  w = pooled_pooled_count.  Train so the TRACT (and COUNTY) count-weighted means match the data.

Reports per beta (smoothness): tract R^2, county R^2 (ACE), and PSI = how much the decoded prediction
moves across the BG/tract/county ladder (the real-vs-artifact signal).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import numpy as np, torch, time
import zeal_m1 as M
from zeal_m1 import Field, dev

d = np.load(paths.raster("geo_st13_raster.npz"))
mask = d["mask"]; N = mask.shape[0]; mi = mask.reshape(-1)
Xf = torch.tensor(np.stack([d["X"].reshape(-1), d["Y"].reshape(-1)], 1), device=dev)   # full grid
coords = Xf[mi]                                                                          # in-state pixels
kfr = torch.tensor(np.nan_to_num(d["kfr"]).reshape(-1)[mi], device=dev)
w = torch.tensor(d["weight"].reshape(-1)[mi], device=dev)

def labels(idx):
    lab = idx.reshape(-1)[mi]; uniq, inv = np.unique(lab, return_inverse=True)
    return torch.tensor(inv, device=dev, dtype=torch.long), len(uniq)
inv_bg, Kbg = labels(d["bg_idx"]); inv_tr, Ktr = labels(d["tract_idx"]); inv_co, Kco = labels(d["county_idx"])

def wmean(vals, inv, K):                                  # count-weighted cell mean, broadcast to pixels
    num = torch.zeros(K, device=dev).index_add_(0, inv, vals * w)
    den = torch.zeros(K, device=dev).index_add_(0, inv, w)
    return (num / den.clamp_min(1e-9))[inv]
tgt_tr = wmean(kfr, inv_tr, Ktr)                          # = tract kfr (constant within tract)
tgt_co = wmean(kfr, inv_co, Kco)                          # exact county aggregate

def wr2(pred_cellmean, tgt):                              # count-weighted R^2
    ss_res = (w * (pred_cellmean - tgt) ** 2).sum()
    mu = (w * tgt).sum() / w.sum()
    ss_tot = (w * (tgt - mu) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-12))

def smooth_full(net):
    g = net(Xf).view(N, N)
    return ((g[1:] - g[:-1]) ** 2).mean() + ((g[:, 1:] - g[:, :-1]) ** 2).mean()

def train(beta, iters=3000, lr=2e-3):
    net = Field(m=128, sigma=6.0, w=128, depth=3, act="tanh").to(dev)
    opt = torch.optim.Adam(net.parameters(), lr)
    for it in range(iters):
        opt.zero_grad()
        pred = net(coords)
        rec = ((w * (wmean(pred, inv_tr, Ktr) - tgt_tr) ** 2).sum() / w.sum()
               + (w * (wmean(pred, inv_co, Kco) - tgt_co) ** 2).sum() / w.sum())
        (rec + beta * smooth_full(net)).backward(); opt.step()
    return net

if __name__ == "__main__":
  print(f"GA ladder: bg {Kbg}, tract {Ktr}, county {Kco}; in-state pixels {coords.shape[0]}")
  print(f"{'beta':>6s} {'R2_tract':>9s} {'R2_county':>10s} {'PSI_ladder':>11s} {'Lip_emp':>8s}")
  for beta in (0.05, 0.5, 5.0):
    t = time.time()
    net = train(beta)
    with torch.no_grad():
        pred = net(coords)
        dbg, dtr, dco = wmean(pred, inv_bg, Kbg), wmean(pred, inv_tr, Ktr), wmean(pred, inv_co, Kco)
        r2tr, r2co = wr2(dtr, tgt_tr), wr2(dco, tgt_co)
        # PSI: count-weighted mean over pixels of Var across the 3 ladder readouts / Var of the tract field
        stack = torch.stack([dbg, dtr, dco], 0)
        psi = float(((w * stack.var(0)).sum() / w.sum()) / (((w * (dtr - (w*dtr).sum()/w.sum())**2).sum()/w.sum()) + 1e-12))
        grid = net(Xf).view(N, N).cpu().numpy()
    import g0_numerics as g0
    Lip_emp = g0.lipschitz(grid, 2.0 / N)
    print(f"{beta:6.2f} {r2tr:9.3f} {r2co:10.3f} {psi:11.4f} {Lip_emp:8.1f}   ({time.time()-t:.0f}s)", flush=True)
  print("\nPSI = fraction of prediction variance that is zoning-dependent across BG/tract/county.")
  print("Low PSI + high R2 => opportunity surface is stable under re-aggregation (granularity ~ real, not artifact).")
