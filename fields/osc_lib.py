"""Certified VALUE maps for a ZEAL field: per-tile CROWN bounds through the analytic sin/cos encoder box,
refined until every in-state tile's certified width is below eps, rasterized to sound pixel LO/HI maps.
Shared by geo_osc_cert.py (oscillation certificate) and geo_corr_cert.py (correlation / regression)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import time, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from zeal_m1 import dev


class _Pre(nn.Module):                                          # pre-softplus field h(x) = g(gamma(x)), 2-D input
    def __init__(self, f): super().__init__(); self.f = f
    def forward(self, x): return self.f.net(self.f.feat(x))


def rasterize(XL, XH, lo, hi, N):
    """Sound pixel maps: LO = min of lo over tiles touching the pixel, HI = max of hi (vectorized)."""
    h = 2.0 / (N - 1)
    il = np.clip(np.floor((XL[:, 0] + 1) / h - 0.5), 0, N - 1).astype(np.int64); ih = np.clip(np.ceil((XH[:, 0] + 1) / h + 0.5), 1, N).astype(np.int64)
    jl = np.clip(np.floor((XL[:, 1] + 1) / h - 0.5), 0, N - 1).astype(np.int64); jh = np.clip(np.ceil((XH[:, 1] + 1) / h + 0.5), 1, N).astype(np.int64)
    LO = np.full(N * N, np.inf); HI = np.full(N * N, -np.inf)
    ni, nj = ih - il, jh - jl
    for di in range(int(ni.max())):                              # tiles touch at most a few pixels: loop over offsets, not tiles
        for dj in range(int(nj.max())):
            sel = (ni > di) & (nj > dj)
            if not sel.any(): continue
            flat = (il[sel] + di) * N + (jl[sel] + dj)
            np.minimum.at(LO, flat, lo[sel]); np.maximum.at(HI, flat, hi[sel])
    return LO.reshape(N, N), HI.reshape(N, N)
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from zeal_manifold_cert import encoder_box


def value_bounder(net, mode="2d", method="CROWN", batch=2048, device=None):
    """Per-tile certified VALUE bounds of the softplus field over axis-aligned 2-D tiles [XL, XH] (rows of [T,2]).
    mode 2d: CROWN on the whole map x -> h(x) with the 2-D location as the perturbed input (relaxations stay
    linear in x, no transverse slack); mode box: CROWN on the MLP over the analytic 256-D encoder box.
    Returns vb(XL, XH) -> (lo, hi) with the (monotone) softplus folded exactly onto the pre-softplus bounds.
    Shared by certify_value_maps (refined value maps) and value_bound_tightness.py (fixed-width tile sweep)."""
    device = dev if device is None else device
    B, m, g = net.B.to(device), net.m, net.net.to(device).eval()
    if mode == "2d":
        bm = BoundedModule(_Pre(net).to(device).eval(), torch.zeros(batch, 2, device=device), device=device)
    else:
        bm = BoundedModule(g, torch.zeros(batch, 2 * m, device=device), device=device)

    def vb(XL, XH):
        T = XL.shape[0]; lo = torch.zeros(T, device=device); hi = torch.zeros(T, device=device)
        for i in range(0, T, batch):
            xl, xh = XL[i:i + batch], XH[i:i + batch]; n = xl.shape[0]
            if n < batch:
                xl = torch.cat([xl, xl[-1:].expand(batch - n, 2)]); xh = torch.cat([xh, xh[-1:].expand(batch - n, 2)])
            if mode == "2d":
                zb = BoundedTensor((xl + xh) / 2, PerturbationLpNorm(norm=np.inf, x_L=xl, x_U=xh))
            else:
                _, _, _, _, zL, zH = encoder_box(B, m, xl, xh)
                zb = BoundedTensor((zL + zH) / 2, PerturbationLpNorm(norm=np.inf, x_L=zL, x_U=zH))
            glb, gub = bm.compute_bounds(x=(zb,), method=method)
            lo[i:i + n] = F.softplus(glb.detach().reshape(-1))[:n]; hi[i:i + n] = F.softplus(gub.detach().reshape(-1))[:n]
        return lo, hi
    return vb


def certify_value_maps(net, mask, eps=0.003, k0=128, budget=3_000_000, batch=2048, log=print, mode="2d", method="CROWN"):
    """mode 2d: CROWN on the whole map x -> h(x) with the 2-D location as the perturbed input (relaxations stay
    linear in x, no transverse slack); mode box: CROWN on the MLP over the analytic 256-D encoder box."""
    N = mask.shape[0]; h = 2.0 / (N - 1); t0 = time.time()
    vb = value_bounder(net, mode=mode, method=method, batch=batch)
    II = torch.zeros(N + 1, N + 1, device=dev); II[1:, 1:] = torch.tensor(mask.astype("float32"), device=dev).cumsum(0).cumsum(1)

    def overlaps(XL, XH):
        il = ((XL[:, 0] + 1) / h - 0.5).floor().clamp(0, N).long(); ih = ((XH[:, 0] + 1) / h + 0.5).ceil().clamp(0, N).long()
        jl = ((XL[:, 1] + 1) / h - 0.5).floor().clamp(0, N).long(); jh = ((XH[:, 1] + 1) / h + 0.5).ceil().clamp(0, N).long()
        return (II[ih, jh] - II[il, jh] - II[ih, jl] + II[il, jl]) > 0.5

    e = torch.linspace(-1, 1, k0 + 1, device=dev)
    XL = torch.stack(torch.meshgrid(e[:-1], e[:-1], indexing="ij"), -1).reshape(-1, 2)
    XH = torch.stack(torch.meshgrid(e[1:], e[1:], indexing="ij"), -1).reshape(-1, 2)
    keep = overlaps(XL, XH); XL, XH = XL[keep], XH[keep]
    lo, hi = vb(XL, XH); it = 0
    while True:
        act = (hi - lo) > eps
        if act.sum() == 0 or XL.shape[0] + 3 * int(act.sum()) > budget: break
        aL, aH = XL[act], XH[act]; mid = (aL + aH) / 2
        x0, y0, x1, y1, mx, my = aL[:, 0], aL[:, 1], aH[:, 0], aH[:, 1], mid[:, 0], mid[:, 1]
        nL = torch.cat([torch.stack([x0, y0], 1), torch.stack([mx, y0], 1), torch.stack([x0, my], 1), torch.stack([mx, my], 1)])
        nH = torch.cat([torch.stack([mx, my], 1), torch.stack([x1, my], 1), torch.stack([mx, y1], 1), torch.stack([x1, y1], 1)])
        ok = overlaps(nL, nH); nL, nH = nL[ok], nH[ok]
        nlo, nhi = vb(nL, nH)
        XL = torch.cat([XL[~act], nL]); XH = torch.cat([XH[~act], nH]); lo = torch.cat([lo[~act], nlo]); hi = torch.cat([hi[~act], nhi])
        it += 1
        log(f"   refine {it}: {XL.shape[0]} tiles, width median {(hi-lo).median():.4f} max {(hi-lo).max():.4f} ({time.time()-t0:.0f}s)")

    XLn, XHn, lon, hin = XL.cpu().numpy(), XH.cpu().numpy(), lo.cpu().numpy(), hi.cpu().numpy()
    LO, HI = rasterize(XLn, XHn, lon, hin, N)
    assert np.isfinite(LO[mask]).all() and np.isfinite(HI[mask]).all(), "every in-state pixel must be covered"
    return LO, HI, dict(tiles=int(XL.shape[0]), width_median=float((hi - lo).median()), width_max=float((hi - lo).max()),
                        mode=mode, method=method, seconds=time.time() - t0), (XLn, XHn, lon, hin)
