"""
Empirical validation of the Transverse Slack theorem on the trained ZEAL field.
Decomposes the box-cert looseness  rho = S_box/S_star  into  rho_off * rho_tan * rho_cond.

  S_star = sup_x ||J_gamma(x)^T grad g(gamma(x))||      (true manifold Lipschitz, pre-softplus)
  G_M    = sup_x ||grad g(gamma(x))||                   (full MLP grad ON the manifold)
  T_M    = sup_x ||P_{T_x} grad g(gamma(x))||           (tangential part)
  G_B    = sup_{z in box} ||grad g(z)||                 (full MLP grad over the ambient box)
  L_gamma= sigma_max(J_gamma)  (= L_enc, constant since J_gamma^T J_gamma = (2pi/sqrt(mf))^2 B^T B)

Key identity used:  ||P_{T_x} grad g||^2 = (J_gamma^T grad g)^T M^{-1} (J_gamma^T grad g) = ||M^{-1/2} grad h||^2,
  M = J_gamma^T J_gamma (2x2 constant), grad h = J_gamma^T grad g  (the field gradient).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import math, numpy as np, torch
import zeal_m1 as M

dev = M.dev
ck = torch.load(paths.data("zeal_full_field.pt"), map_location=dev, weights_only=False)
net = M.Field(m=ck["m"], sigma=ck["sigma"], w=ck["w"], depth=ck["depth"], act="tanh").to(dev)
net.load_state_dict(ck["state"]); net.eval()
mf, B, g = net.m, net.B, net.net                       # g = pre-softplus MLP z->scalar

Mmat = (2 * math.pi / math.sqrt(mf)) ** 2 * (B.t() @ B)  # [2,2], constant
ev, evec = torch.linalg.eigh(Mmat)
Minv_sqrt = evec @ torch.diag(ev.clamp_min(1e-20).rsqrt()) @ evec.t()
L_gamma = math.sqrt(ev.max().item())                    # sigma_max(J_gamma)

lin = torch.linspace(-1, 1, 160, device=dev); gx, gy = torch.meshgrid(lin, lin, indexing="ij")
X = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)

# field gradient grad h = J_gamma^T grad g  (pre-softplus)
Xr = X.clone().requires_grad_(True)
h = g(net.feat(Xr)).squeeze(-1)
gradh = torch.autograd.grad(h.sum(), Xr)[0]             # [N,2]
S_star = gradh.norm(dim=1).max().item()
T_M = (gradh @ Minv_sqrt.t()).norm(dim=1).max().item()  # sup ||P_T grad g||

# full MLP grad ON manifold
zr = net.feat(X).detach().requires_grad_(True)
G_M = torch.autograd.grad(g(zr).sum(), zr)[0].norm(dim=1).max().item()

# full MLP grad over the ambient box: random sample + PGA refine
r = 1.0 / math.sqrt(mf)
zb = ((2 * torch.rand(40000, 2 * mf, device=dev) - 1) * r).requires_grad_(True)
gb = torch.autograd.grad(g(zb).sum(), zb)[0].norm(dim=1)
G_B = gb.max().item()
seeds = zb[gb.topk(256).indices].detach()
for _ in range(80):
    s = seeds.clone().requires_grad_(True)
    gn = torch.autograd.grad(g(s).sum(), s, create_graph=True)[0]
    step = torch.autograd.grad((gn * gn).sum(), s)[0]
    seeds = (seeds + 1e-3 * step).clamp(-r, r).detach()
    G_B = max(G_B, gn.detach().norm(dim=1).max().item())

S_box = L_gamma * G_B
print(f"L_gamma (sigma_max J_gamma) = {L_gamma:.2f}")
print(f"S_star  (manifold Lipschitz)= {S_star:.2f}")
print(f"T_M     (tangential sup)    = {T_M:.3f}")
print(f"G_M     (||grad g|| on mfld)= {G_M:.2f}")
print(f"G_B     (||grad g|| box)    = {G_B:.2f}")
print(f"S_box   = L_gamma*G_B       = {S_box:.1f}\n")
print(f"rho total   S_box/S_star          = {S_box/S_star:7.1f}x")
print(f"  rho_off   G_B/G_M  (off-manifold) = {G_B/G_M:7.2f}x")
print(f"  rho_tan   G_M/T_M  (TRANSVERSE)   = {G_M/T_M:7.2f}x   [generic sqrt(m/d)={math.sqrt(2*mf/2):.1f}x]")
print(f"  rho_cond  L_gamma*T_M/S_star      = {L_gamma*T_M/S_star:7.2f}x   (immersion conditioning)")
print(f"  product check                     = {(G_B/G_M)*(G_M/T_M)*(L_gamma*T_M/S_star):7.1f}x")
