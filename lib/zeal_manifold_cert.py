"""
Route 1 prototype: manifold-aware SOUND certifier for
    S = sup_{x in Omega} || J_gamma(x)^T grad g(gamma(x)) ||_2          (true manifold Lipschitz)
WITHOUT forming the ambient 2m-D box.  gamma(x)=[sin(2pi Bx); cos(2pi Bx)]/sqrt(m), g = Tanh MLP.

Per input-tile T (a small box in the 2-D x-domain):
  1. encoder z-box: theta = 2pi B x over T -> per-coord sin/cos INTERVALS (analytic, exact).
  2. [dg_lo, dg_hi] = auto_LiRPA Jacobian bounds of g over that z-box (Tanh supported; sound, ~tight).
  3. analytic projection: bound || J_gamma(x)^T dg ||_2 over (x in T, dg in [dg_lo,dg_hi]) using the EXACT
     J_gamma structure with the cos/sin intervals -> per-tile sound upper bound.
S_cert = max over tiles.  The J_gamma^T projection keeps the 2-D-tangent structure (kills transverse slack).
"""
import math, numpy as np, torch, torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators.jacobian import JacobianOP


# ---------------- analytic sin/cos interval bounds (SOUND) ----------------
def _hits(p, lo, hi):                         # does p + 2*pi*k lie in [lo,hi] for some integer k?
    tp = 2 * math.pi
    return torch.floor((hi - p) / tp) >= torch.ceil((lo - p) / tp)

def sin_iv(lo, hi):
    sl, sh = torch.sin(lo), torch.sin(hi)
    mn, mx = torch.minimum(sl, sh), torch.maximum(sl, sh)
    mx = torch.where(_hits(math.pi / 2, lo, hi), torch.ones_like(mx), mx)
    mn = torch.where(_hits(-math.pi / 2, lo, hi), -torch.ones_like(mn), mn)
    return mn, mx

def cos_iv(lo, hi):
    return sin_iv(lo + math.pi / 2, hi + math.pi / 2)

def iv_prod(al, ah, bl, bh):                  # interval product [al,ah]*[bl,bh]
    p = torch.stack([al * bl, al * bh, ah * bl, ah * bh])
    return p.amin(0), p.amax(0)


# ---------------- encoder z-box over a batch of input-tiles ----------------
def encoder_box(B, m, xl, xh):
    Bp, Bn = B.clamp(min=0), B.clamp(max=0)
    th_lo = 2 * math.pi * (xl @ Bp.t() + xh @ Bn.t())     # [T,m]
    th_hi = 2 * math.pi * (xh @ Bp.t() + xl @ Bn.t())
    sL, sH = sin_iv(th_lo, th_hi)
    cL, cH = cos_iv(th_lo, th_hi)
    zL = torch.cat([sL, cL], 1) / math.sqrt(m)
    zH = torch.cat([sH, cH], 1) / math.sqrt(m)
    return sL, sH, cL, cH, zL, zH


# ---------------- auto_LiRPA Jacobian bounds of g over z-boxes ----------------
class _JacW(nn.Module):
    def __init__(self, g): super().__init__(); self.g = g
    def forward(self, z): return JacobianOP.apply(self.g(z), z)

def dg_bounds(g, zL, zH, device, optimize=False):
    zc = (zL + zH) / 2
    bm = BoundedModule(_JacW(g), zc, device=device,
                       bound_opts={"sparse_intermediate_bounds": False})
    zb = BoundedTensor(zc, PerturbationLpNorm(norm=np.inf, x_L=zL, x_U=zH))
    lb, ub = bm.compute_jacobian_bounds((zb,), optimize=optimize)
    T = zc.shape[0]
    return lb.detach().reshape(T, -1), ub.detach().reshape(T, -1)


# ---------------- exact J_gamma^T projection bound ----------------
def project_bound(B, m, sL, sH, cL, cH, dgL, dgH):
    dgsL, dgsH = dgL[:, :m], dgH[:, :m]
    dgcL, dgcH = dgL[:, m:], dgH[:, m:]
    al, ah = iv_prod(cL, cH, dgsL, dgsH)       # cos(theta)*dg_sin
    bl, bh = iv_prod(sL, sH, dgcL, dgcH)       # sin(theta)*dg_cos
    uL, uH = al - bh, ah - bl                  # u_j = cos*dg_s - sin*dg_c
    coef = 2 * math.pi / math.sqrt(m)
    comps = []
    for k in range(2):
        Bk = B[:, k]
        Bp, Bn = Bk.clamp(min=0), Bk.clamp(max=0)
        cl = coef * ((uL * Bp).sum(1) + (uH * Bn).sum(1))   # [T]
        ch = coef * ((uH * Bp).sum(1) + (uL * Bn).sum(1))
        comps.append((cl, ch))
    (c1l, c1h), (c2l, c2h) = comps
    m1 = torch.maximum(c1l ** 2, c1h ** 2)
    m2 = torch.maximum(c2l ** 2, c2h ** 2)
    return torch.sqrt(m1 + m2)                 # [T] per-tile sound UB on ||J_gamma^T dg||


def manifold_cert(net, k, device="cuda", optimize=False, batch=1024):
    B, m, g = net.B.to(device), net.m, net.net.to(device).eval()
    e = torch.linspace(-1, 1, k + 1, device=device)
    lo2 = torch.stack(torch.meshgrid(e[:-1], e[:-1], indexing="ij"), -1).reshape(-1, 2)
    hi2 = torch.stack(torch.meshgrid(e[1:], e[1:], indexing="ij"), -1).reshape(-1, 2)
    S = 0.0
    for i in range(0, lo2.shape[0], batch):
        XL, XH = lo2[i:i + batch], hi2[i:i + batch]
        sL, sH, cL, cH, zL, zH = encoder_box(B, m, XL, XH)
        dgL, dgH = dg_bounds(g, zL, zH, device, optimize)
        S = max(S, project_bound(B, m, sL, sH, cL, cH, dgL, dgH).max().item())
    return S


# ---------------- branch-and-bound refinement (build bm ONCE) ----------------
def manifold_cert_bab(net, k0=16, budget=300000, device="cuda", optimize=False, batch=1024, frac=0.6):
    B, m, g = net.B.to(device), net.m, net.net.to(device).eval()
    bm = BoundedModule(_JacW(g), torch.zeros(batch, 2 * m, device=device), device=device,
                       bound_opts={"sparse_intermediate_bounds": False})

    def tile_bounds(XL, XH):
        T = XL.shape[0]; out = torch.zeros(T, device=device)
        for i in range(0, T, batch):
            xl, xh = XL[i:i + batch], XH[i:i + batch]; n = xl.shape[0]
            if n < batch:
                xl = torch.cat([xl, xl[-1:].expand(batch - n, 2)])
                xh = torch.cat([xh, xh[-1:].expand(batch - n, 2)])
            sL, sH, cL, cH, zL, zH = encoder_box(B, m, xl, xh)
            zb = BoundedTensor((zL + zH) / 2, PerturbationLpNorm(norm=np.inf, x_L=zL, x_U=zH))
            lb, ub = bm.compute_jacobian_bounds((zb,), optimize=optimize)
            dgL, dgH = lb.detach().reshape(batch, -1), ub.detach().reshape(batch, -1)
            out[i:i + n] = project_bound(B, m, sL, sH, cL, cH, dgL, dgH)[:n]
        return out

    e = torch.linspace(-1, 1, k0 + 1, device=device)
    XL = torch.stack(torch.meshgrid(e[:-1], e[:-1], indexing="ij"), -1).reshape(-1, 2)
    XH = torch.stack(torch.meshgrid(e[1:], e[1:], indexing="ij"), -1).reshape(-1, 2)
    bnd = tile_bounds(XL, XH)
    while XL.shape[0] < budget:
        gmax = bnd.max().item()
        act = bnd >= frac * gmax
        if act.sum() == 0:
            break
        aL, aH = XL[act], XH[act]; mid = (aL + aH) / 2
        x0, y0, x1, y1, mx, my = aL[:, 0], aL[:, 1], aH[:, 0], aH[:, 1], mid[:, 0], mid[:, 1]
        nL = torch.cat([torch.stack([x0, y0], 1), torch.stack([mx, y0], 1),
                        torch.stack([x0, my], 1), torch.stack([mx, my], 1)])
        nH = torch.cat([torch.stack([mx, my], 1), torch.stack([x1, my], 1),
                        torch.stack([mx, y1], 1), torch.stack([x1, y1], 1)])
        XL = torch.cat([XL[~act], nL]); XH = torch.cat([XH[~act], nH])
        bnd = torch.cat([bnd[~act], tile_bounds(nL, nH)])
    return bnd.max().item(), XL.shape[0]


# ---------------- full-FIELD cert (folds the outer softplus per tile) + optimize ----------------
def manifold_cert_field(net, k0=16, budget=300000, device="cuda", optimize=False,
                        fold_softplus=True, batch=1024, frac=0.6):
    """Sound UB on  sup_x ||grad lambda(x)||,  lambda = softplus(g(gamma(x))).
       ||grad lambda|| = sigmoid(g(gamma(x))) * ||J_gamma^T grad g||  (softplus' = sigmoid).
       Per tile: sigmoid(g_hi_tile) * project_bound, with g_hi from a forward CROWN pass."""
    B, m, g = net.B.to(device), net.m, net.net.to(device).eval()
    ex = torch.zeros(batch, 2 * m, device=device)
    bm_jac = BoundedModule(_JacW(g), ex, device=device, bound_opts={"sparse_intermediate_bounds": False})
    bm_fwd = BoundedModule(g, ex, device=device) if fold_softplus else None

    def tile_bounds(XL, XH):
        T = XL.shape[0]; out = torch.zeros(T, device=device)
        for i in range(0, T, batch):
            xl, xh = XL[i:i + batch], XH[i:i + batch]; n = xl.shape[0]
            if n < batch:
                xl = torch.cat([xl, xl[-1:].expand(batch - n, 2)])
                xh = torch.cat([xh, xh[-1:].expand(batch - n, 2)])
            sL, sH, cL, cH, zL, zH = encoder_box(B, m, xl, xh)
            zb = BoundedTensor((zL + zH) / 2, PerturbationLpNorm(norm=np.inf, x_L=zL, x_U=zH))
            lb, ub = bm_jac.compute_jacobian_bounds((zb,), optimize=optimize)
            pb = project_bound(B, m, sL, sH, cL, cH,
                               lb.detach().reshape(batch, -1), ub.detach().reshape(batch, -1))
            if fold_softplus:
                _, gub = bm_fwd.compute_bounds(x=(zb,), method="CROWN")
                pb = pb * torch.sigmoid(gub.detach().reshape(-1))
            out[i:i + n] = pb[:n]
        return out

    e = torch.linspace(-1, 1, k0 + 1, device=device)
    XL = torch.stack(torch.meshgrid(e[:-1], e[:-1], indexing="ij"), -1).reshape(-1, 2)
    XH = torch.stack(torch.meshgrid(e[1:], e[1:], indexing="ij"), -1).reshape(-1, 2)
    bnd = tile_bounds(XL, XH)
    while XL.shape[0] < budget:
        gmax = bnd.max().item()
        act = bnd >= frac * gmax
        if act.sum() == 0:
            break
        aL, aH = XL[act], XH[act]; mid = (aL + aH) / 2
        x0, y0, x1, y1, mx, my = aL[:, 0], aL[:, 1], aH[:, 0], aH[:, 1], mid[:, 0], mid[:, 1]
        nL = torch.cat([torch.stack([x0, y0], 1), torch.stack([mx, y0], 1),
                        torch.stack([x0, my], 1), torch.stack([mx, my], 1)])
        nH = torch.cat([torch.stack([mx, my], 1), torch.stack([x1, my], 1),
                        torch.stack([mx, y1], 1), torch.stack([x1, y1], 1)])
        XL = torch.cat([XL[~act], nL]); XH = torch.cat([XH[~act], nH])
        bnd = torch.cat([bnd[~act], tile_bounds(nL, nH)])
    return bnd.max().item(), XL.shape[0]


# ---------------- mask-restricted cert (certify only over an in-domain mask) ----------------
def manifold_cert_masked(net, mask, budget=120000, device="cuda", batch=1024, frac=0.6, k0=24, fold=True):
    """Like manifold_cert_field but only over tiles overlapping `mask` (N x N bool, domain on [-1,1]^2),
       via an integral-image overlap test + BaB pruning of out-of-domain tiles."""
    B, m, g = net.B.to(device), net.m, net.net.to(device).eval()
    ex = torch.zeros(batch, 2 * m, device=device)
    bm_j = BoundedModule(_JacW(g), ex, device=device, bound_opts={"sparse_intermediate_bounds": False})
    bm_f = BoundedModule(g, ex, device=device) if fold else None
    Ng = mask.shape[0]
    II = torch.zeros(Ng + 1, Ng + 1, device=device)
    II[1:, 1:] = torch.tensor(np.asarray(mask, dtype="float32"), device=device).cumsum(0).cumsum(1)

    def overlaps(XL, XH):
        il = ((XL[:, 0] + 1) / 2 * (Ng - 1)).floor().clamp(0, Ng).long()
        ih = (((XH[:, 0] + 1) / 2 * (Ng - 1)).ceil() + 1).clamp(0, Ng).long()
        jl = ((XL[:, 1] + 1) / 2 * (Ng - 1)).floor().clamp(0, Ng).long()
        jh = (((XH[:, 1] + 1) / 2 * (Ng - 1)).ceil() + 1).clamp(0, Ng).long()
        return (II[ih, jh] - II[il, jh] - II[ih, jl] + II[il, jl]) > 0.5

    def tb(XL, XH):
        T = XL.shape[0]; out = torch.zeros(T, device=device)
        for i in range(0, T, batch):
            xl, xh = XL[i:i + batch], XH[i:i + batch]; n = xl.shape[0]
            if n < batch:
                xl = torch.cat([xl, xl[-1:].expand(batch - n, 2)]); xh = torch.cat([xh, xh[-1:].expand(batch - n, 2)])
            sL, sH, cL, cH, zL, zH = encoder_box(B, m, xl, xh)
            zb = BoundedTensor((zL + zH) / 2, PerturbationLpNorm(norm=np.inf, x_L=zL, x_U=zH))
            lb, ub = bm_j.compute_jacobian_bounds((zb,), optimize=False)
            pb = project_bound(B, m, sL, sH, cL, cH, lb.detach().reshape(batch, -1), ub.detach().reshape(batch, -1))
            if fold:
                _, gub = bm_f.compute_bounds(x=(zb,), method="CROWN"); pb = pb * torch.sigmoid(gub.detach().reshape(-1))
            out[i:i + n] = pb[:n]
        return out * overlaps(XL, XH).float()

    e = torch.linspace(-1, 1, k0 + 1, device=device)
    XL = torch.stack(torch.meshgrid(e[:-1], e[:-1], indexing="ij"), -1).reshape(-1, 2)
    XH = torch.stack(torch.meshgrid(e[1:], e[1:], indexing="ij"), -1).reshape(-1, 2)
    keep = overlaps(XL, XH); XL, XH = XL[keep], XH[keep]
    bnd = tb(XL, XH)
    while XL.shape[0] < budget:
        gmax = bnd.max().item(); act = bnd >= frac * gmax
        if act.sum() == 0: break
        aL, aH = XL[act], XH[act]; mid = (aL + aH) / 2
        x0, y0, x1, y1, mx, my = aL[:, 0], aL[:, 1], aH[:, 0], aH[:, 1], mid[:, 0], mid[:, 1]
        nL = torch.cat([torch.stack([x0, y0], 1), torch.stack([mx, y0], 1), torch.stack([x0, my], 1), torch.stack([mx, my], 1)])
        nH = torch.cat([torch.stack([mx, my], 1), torch.stack([x1, my], 1), torch.stack([mx, y1], 1), torch.stack([x1, y1], 1)])
        ok = overlaps(nL, nH); nL, nH = nL[ok], nH[ok]
        XL = torch.cat([XL[~act], nL]); XH = torch.cat([XH[~act], nH]); bnd = torch.cat([bnd[~act], tb(nL, nH)])
    return bnd.max().item(), XL.shape[0]


# ---------------- validation on a TINY net ----------------
if __name__ == "__main__":
    import time
    import zeal_m1 as M
    dev = "cuda"
    torch.manual_seed(0)
    # interval soundness self-test
    lo = torch.rand(2000) * 20 - 10; hi = lo + torch.rand(2000) * 3
    mn, mx = sin_iv(lo, hi)
    ok = True
    for _ in range(50):
        t = lo + torch.rand(2000) * (hi - lo)
        ok &= bool((torch.sin(t) >= mn - 1e-6).all() and (torch.sin(t) <= mx + 1e-6).all())
    print("sin_iv sound:", ok)

    net = M.Field(m=8, sigma=6.0, w=16, depth=2, act="tanh").to(dev).eval()
    # empirical S_star = sup_x ||grad_x (g o gamma)||
    lin = torch.linspace(-1, 1, 400, device=dev); gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).requires_grad_(True)
    h = net.net(net.feat(X)).sum()
    S_star = torch.autograd.grad(h, X)[0].norm(dim=1).max().item()
    # ambient box bound  L_enc * sup_box ||grad g||
    mf = net.m; r = 1 / math.sqrt(mf)
    L_enc = (2 * math.pi / math.sqrt(mf)) * torch.linalg.svdvals(net.B)[0].item()
    zb = ((2 * torch.rand(80000, 2 * mf, device=dev) - 1) * r).requires_grad_(True)
    S_ambient = L_enc * torch.autograd.grad(net.net(zb).sum(), zb)[0].norm(dim=1).max().item()
    print(f"empirical S_star = {S_star:.3f}   ambient-box bound = {S_ambient:.1f} ({S_ambient/S_star:.0f}x)")
    for k in (8, 16, 32, 64):
        t = time.time()
        S = manifold_cert(net, k, device=dev, optimize=False)
        print(f"  k={k:3d} tiles={k*k:5d}: S_cert={S:8.3f}  cert/emp={S/S_star:6.2f}x  "
              f"sound={S >= S_star - 1e-3}  ({time.time()-t:.1f}s)", flush=True)
