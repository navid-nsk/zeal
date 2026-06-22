"""
Ablation of the manifold-aware certifier: build up from ambient alpha-CROWN to the final, toggling
  (1) 2D input tiling, (2) coupled Jacobian (J_gamma^T projection vs decoupled L_enc*||J_g||),
  (3) analytic sin/cos z-box (exact interval) vs crude Lipschitz z-box (|d sin|<=|d theta|).
Reports cert / S_star at a FIXED tiling budget so each row isolates one improvement.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import math, time, numpy as np, torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import zeal_m1 as M
from zeal_manifold_cert import encoder_box, project_bound, _JacW

dev = "cuda"


def encoder_box_crude(B, m, xl, xh):                      # CRUDE z-box: |sin(t)-sin(tc)| <= |t-tc|
    Bp, Bn = B.clamp(min=0), B.clamp(max=0)
    th_lo = 2 * math.pi * (xl @ Bp.t() + xh @ Bn.t())
    th_hi = 2 * math.pi * (xh @ Bp.t() + xl @ Bn.t())
    thc, dth = (th_lo + th_hi) / 2, (th_hi - th_lo) / 2
    sc, cc = torch.sin(thc), torch.cos(thc)
    sL, sH = (sc - dth).clamp(-1, 1), (sc + dth).clamp(-1, 1)
    cL, cH = (cc - dth).clamp(-1, 1), (cc + dth).clamp(-1, 1)
    zL = torch.cat([sL, cL], 1) / math.sqrt(m); zH = torch.cat([sH, cH], 1) / math.sqrt(m)
    return sL, sH, cL, cH, zL, zH


def cert(net, budget, sincos, coupling, single=False, batch=1024, frac=0.6, k0=16):
    B, m, g = net.B.to(dev), net.m, net.net.to(dev).eval()
    L_enc = (2 * math.pi / math.sqrt(m)) * torch.linalg.svdvals(B)[0].item()
    bm = BoundedModule(_JacW(g), torch.zeros(batch, 2 * m, device=dev), device=dev,
                       bound_opts={"sparse_intermediate_bounds": False})
    encf = encoder_box if sincos == "analytic" else encoder_box_crude

    def tb(XL, XH):
        T = XL.shape[0]; out = torch.zeros(T, device=dev)
        for i in range(0, T, batch):
            xl, xh = XL[i:i + batch], XH[i:i + batch]; n = xl.shape[0]
            if n < batch:
                xl = torch.cat([xl, xl[-1:].expand(batch - n, 2)]); xh = torch.cat([xh, xh[-1:].expand(batch - n, 2)])
            sL, sH, cL, cH, zL, zH = encf(B, m, xl, xh)
            zb = BoundedTensor((zL + zH) / 2, PerturbationLpNorm(norm=np.inf, x_L=zL, x_U=zH))
            lb, ub = bm.compute_jacobian_bounds((zb,), optimize=False)
            dgL, dgH = lb.detach().reshape(batch, -1), ub.detach().reshape(batch, -1)
            if coupling == "coupled":
                pb = project_bound(B, m, sL, sH, cL, cH, dgL, dgH)
            else:
                pb = L_enc * torch.sqrt(torch.maximum(dgL ** 2, dgH ** 2).sum(1))
            out[i:i + n] = pb[:n]
        return out

    if single:
        return tb(torch.tensor([[-1., -1.]], device=dev), torch.tensor([[1., 1.]], device=dev)).max().item()
    e = torch.linspace(-1, 1, k0 + 1, device=dev)
    XL = torch.stack(torch.meshgrid(e[:-1], e[:-1], indexing="ij"), -1).reshape(-1, 2)
    XH = torch.stack(torch.meshgrid(e[1:], e[1:], indexing="ij"), -1).reshape(-1, 2)
    bnd = tb(XL, XH)
    while XL.shape[0] < budget:
        gmax = bnd.max().item(); act = bnd >= frac * gmax
        if act.sum() == 0: break
        aL, aH = XL[act], XH[act]; mid = (aL + aH) / 2
        x0, y0, x1, y1, mx, my = aL[:, 0], aL[:, 1], aH[:, 0], aH[:, 1], mid[:, 0], mid[:, 1]
        nL = torch.cat([torch.stack([x0, y0], 1), torch.stack([mx, y0], 1), torch.stack([x0, my], 1), torch.stack([mx, my], 1)])
        nH = torch.cat([torch.stack([mx, my], 1), torch.stack([x1, my], 1), torch.stack([mx, y1], 1), torch.stack([x1, y1], 1)])
        XL = torch.cat([XL[~act], nL]); XH = torch.cat([XH[~act], nH]); bnd = torch.cat([bnd[~act], tb(nL, nH)])
    return bnd.max().item()


if __name__ == "__main__":
    ck = torch.load(paths.data("zeal_full_field.pt"), map_location=dev, weights_only=False)
    net = M.Field(m=ck["m"], sigma=ck["sigma"], w=ck["w"], depth=ck["depth"], act="tanh").to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    lin = torch.linspace(-1, 1, 200, device=dev); gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).requires_grad_(True)
    S_star = torch.autograd.grad(net.net(net.feat(X)).sum(), X)[0].norm(dim=1).max().item()
    BUD = 60000
    print(f"S_star = {S_star:.1f}; tiling budget = {BUD}\n{'variant':32s} {'cert':>9s} {'gap':>7s}")
    rows = [("ambient alpha-CROWN (no tiling, decoupled)", dict(budget=1, sincos="analytic", coupling="decoupled", single=True)),
            ("+ 2D tiling (crude z-box, decoupled)",       dict(budget=BUD, sincos="crude", coupling="decoupled")),
            ("+ coupled Jacobian (crude z-box)",           dict(budget=BUD, sincos="crude", coupling="coupled")),
            ("+ analytic sin/cos  (= final)",              dict(budget=BUD, sincos="analytic", coupling="coupled"))]
    for name, kw in rows:
        t = time.time(); c = cert(net, **kw)
        print(f"{name:32s} {c:9.1f} {c/S_star:6.1f}x   ({time.time()-t:.0f}s)", flush=True)
