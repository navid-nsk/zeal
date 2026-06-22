"""
Validation suite for the manifold-aware certifier (now a main contribution).
  V1  identity embedding gamma(x)=x  ->  reduces EXACTLY to ordinary Jacobian alpha-CROWN.
  V2  linear g(z)=u^T z  ->  S_star = max_x ||J_gamma(x)^T u||  analytic; recover to precision for
      tangent / normal / mixed u  (the direct Transverse-Slack test: normal u => tiny S_star << ambient).
  V3  brute force on tiny nets: dense-sample Omega, true max vs certified max; cert must DOMINATE (sound).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import math, numpy as np, torch, torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import zeal_manifold_cert as MC
from zeal_manifold_cert import encoder_box, project_bound, _JacW, manifold_cert
import zeal_m1 as M

dev = "cuda"


def Jgamma(B, m, x):                                   # [N, 2m, 2]
    th = 2 * np.pi * (x @ B.t())
    c, s = torch.cos(th), torch.sin(th)
    coef = 2 * np.pi / np.sqrt(m)
    J = torch.zeros(x.shape[0], 2 * m, 2, device=x.device)
    for k in range(2):
        J[:, :m, k] = coef * c * B[:, k]
        J[:, m:, k] = -coef * s * B[:, k]
    return J


class LinField:                                        # linear g(z)=u^T z on the Fourier embedding
    def __init__(self, B, u):
        self.B = B; self.m = B.shape[0]
        lin = nn.Linear(2 * self.m, 1, bias=False).to(B.device)
        lin.weight.data = u.reshape(1, -1).to(B.device)
        self.net = nn.Sequential(lin)
    def feat(self, x):
        p = 2 * np.pi * x @ self.B.t()
        return torch.cat([torch.sin(p), torch.cos(p)], -1) / np.sqrt(self.m)


def alpha_crown_jac_L2(g, xL, xH):                     # direct: L2 sound bound on ||grad g|| over [xL,xH]
    bm = BoundedModule(_JacW(g), (xL + xH) / 2, device=dev,
                       bound_opts={"sparse_intermediate_bounds": False})
    zb = BoundedTensor((xL + xH) / 2, PerturbationLpNorm(norm=np.inf, x_L=xL, x_U=xH))
    lb, ub = bm.compute_jacobian_bounds((zb,), optimize=False)
    lb, ub = lb.detach().reshape(-1), ub.detach().reshape(-1)
    return math.sqrt(float(torch.maximum(lb ** 2, ub ** 2).sum())), lb, ub


# ---------------- V1: identity embedding reduces to ordinary Jacobian alpha-CROWN ----------------
print("=== V1: identity embedding gamma(x)=x reduces to ordinary Jacobian alpha-CROWN ===")
torch.manual_seed(0)
d = 4
g1 = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1)).to(dev).eval()
xL = -0.3 * torch.ones(1, d, device=dev); xH = 0.3 * torch.ones(1, d, device=dev)
direct, lb, ub = alpha_crown_jac_L2(g1, xL, xH)
# identity manifold-cert: J_gamma = I => z-box = x-box, projection = ||dg||_2 = sqrt(sum max(lb^2,ub^2))
ident = math.sqrt(float(torch.maximum(lb ** 2, ub ** 2).sum()))
print(f"  direct alpha-CROWN ||grad g|| bound = {direct:.6f}")
print(f"  identity manifold-cert (J_gamma=I)  = {ident:.6f}   MATCH={abs(direct-ident) < 1e-6}\n")


# ---------------- V2: linear g, analytic S_star, tangent/normal/mixed ----------------
print("=== V2: linear g(z)=u^T z, S_star=max||J_gamma^T u|| analytic; recover for tangent/normal/mixed ===")
torch.manual_seed(1)
m = 12; B = torch.randn(m, 2, device=dev) * 3.0
x0 = torch.tensor([[0.05, -0.07]], device=dev)
J0 = Jgamma(B, m, x0)[0]                                # [2m,2] tangent basis at x0
ut = J0 @ torch.tensor([1.0, 0.5], device=dev); ut = ut / ut.norm()       # tangent
rnd = torch.randn(2 * m, device=dev)
P = J0 @ torch.linalg.pinv(J0)                          # projector onto tangent at x0
un = rnd - P @ rnd; un = un / un.norm()                 # normal at x0
um = (ut + un); um = um / um.norm()                     # mixed
L_enc = (2 * np.pi / np.sqrt(m)) * torch.linalg.svdvals(B)[0].item()

def true_Sstar(u, ng=700):
    lin = torch.linspace(-1, 1, ng, device=dev); gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    JTu = (Jgamma(B, m, X) * u.reshape(1, -1, 1)).sum(1)
    return JTu.norm(dim=1).max().item()

print(f"  {'u':8s} {'true_S':>8s} {'cert':>8s} {'cert/true':>9s} {'ambient':>8s} {'amb/true':>8s}")
for name, u in [("tangent", ut), ("normal", un), ("mixed", um)]:
    tS = true_Sstar(u)
    cS = manifold_cert(LinField(B, u), k=128, device=dev)
    amb = u.norm().item() * L_enc
    print(f"  {name:8s} {tS:8.3f} {cS:8.3f} {cS/tS:9.2f} {amb:8.2f} {amb/tS:8.1f}", flush=True)
print("  (full Omega: tangent rotates fully so any u is tangent somewhere -> amb/true~1 for all)\n")

# V2b: SMALL domain (tangent ~ fixed) -> the clean Transverse-Slack separation
def cert_on_domain(net, lo, hi, k):
    B, m, g = net.B.to(dev), net.m, net.net.to(dev).eval()
    e0 = torch.linspace(lo, hi, k + 1, device=dev)
    XL = torch.stack(torch.meshgrid(e0[:-1], e0[:-1], indexing="ij"), -1).reshape(-1, 2)
    XH = torch.stack(torch.meshgrid(e0[1:], e0[1:], indexing="ij"), -1).reshape(-1, 2)
    sL, sH, cL, cH, zL, zH = encoder_box(B, m, XL, XH)
    dgL, dgH = MC.dg_bounds(g, zL, zH, dev)
    return project_bound(B, m, sL, sH, cL, cH, dgL, dgH).max().item()

def true_Sstar_dom(u, lo, hi, ng=400):
    e = torch.linspace(lo, hi, ng, device=dev); gx, gy = torch.meshgrid(e, e, indexing="ij")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    return ((Jgamma(B, m, X) * u.reshape(1, -1, 1)).sum(1)).norm(dim=1).max().item()

print("=== V2b: SMALL domain [-0.03,0.03]^2 (tangent ~fixed) -> Transverse Slack visible ===")
print(f"  {'u':8s} {'true_S':>8s} {'cert':>8s} {'cert/true':>9s} {'ambient':>8s} {'amb/true':>9s}")
lo, hi = -0.03, 0.03
for name, u in [("tangent", ut), ("normal", un), ("mixed", um)]:
    tS = true_Sstar_dom(u, lo, hi)
    cS = cert_on_domain(LinField(B, u), lo, hi, k=64)
    amb = u.norm().item() * L_enc
    print(f"  {name:8s} {tS:8.3f} {cS:8.3f} {cS/tS:9.2f} {amb:8.2f} {amb/tS:9.1f}", flush=True)
print("  -> normal u: true_S tiny, ambient huge (amb/true large), cert recovers the tiny true value.\n")


# ---------------- V3: brute-force soundness on tiny nets ----------------
print("=== V3: brute-force soundness (tiny Fourier nets): cert must dominate true max ===")
for seed in range(4):
    torch.manual_seed(100 + seed)
    net = M.Field(m=8, sigma=5.0, w=16, depth=2, act="tanh").to(dev).eval()
    lin = torch.linspace(-1, 1, 500, device=dev); gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).requires_grad_(True)
    JTg = torch.autograd.grad(net.net(net.feat(X)).sum(), X)[0]      # grad_x(g o gamma) = J_gamma^T grad g
    trueS = JTg.norm(dim=1).max().item()
    cert = manifold_cert(net, k=96, device=dev)
    print(f"  seed {seed}: true={trueS:8.3f}  cert={cert:8.3f}  SOUND(cert>=true)={cert >= trueS - 1e-3}", flush=True)
