"""
alpha-CROWN certified LOCAL Lipschitz for the ZEAL field (Route B, verified plan).

  lambda(x) = softplus( g( gamma(x) ) ),   g = Tanh MLP,  gamma = RFF encoder.
  Lip_Omega = L_enc (EXACT) * Lip_box(g over [-1/sqrt(m),1/sqrt(m)]^{2m}) * s_outer (<=1).

L_enc          = (2 pi / sqrt(m)) * sigma_max(B)                 [equality, sin^2+cos^2=1]
Lip_box(g)     = sqrt( UB_{box} || grad_z g ||_2^2 )            [auto_LiRPA Jacobian path, GradNorm(2)]
s_outer        = sigmoid( UB_{box} g(z) ) <= 1                   [outer softplus monotone factor]

Soundness chain (verifier-confirmed): ||grad lambda|| <= sup softplus' * sup_box ||J_g|| * sigma_max(J_gamma).
"""
import math, numpy as np, torch, torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
try:
    from auto_LiRPA.operators.jacobian import JacobianOP, GradNorm
except Exception:
    from auto_LiRPA.jacobian import JacobianOP, GradNorm


class LipWrapper(nn.Module):
    """z -> ||grad_z g(z)||_2^2   (g scalar). GradNorm(norm=2) returns the SQUARED norm; sqrt is ours.
       (No output mask: g is already scalar, so JacobianOP gives the [B,1,2m] gradient directly.)"""
    def __init__(self, mlp_scalar):
        super().__init__()
        self.mlp = mlp_scalar
        self.grad_norm = GradNorm(norm=2)

    def forward(self, z):
        y = self.mlp(z)                       # [B,1]
        jac = JacobianOP.apply(y, z)          # [B,1,2m]
        return self.grad_norm(jac)            # [B] = ||grad||^2


def lip_box_mlp(mlp_scalar, m, device="cuda", optimize=True):
    """SOUND UB on sup_{z in box} ||grad_z g(z)||_2,  box = [-1/sqrt(m), 1/sqrt(m)]^{2m}."""
    r, d = 1.0 / math.sqrt(m), 2 * m
    z0 = torch.zeros(1, d, device=device)
    bm = BoundedModule(LipWrapper(mlp_scalar).to(device), (z0,),
                       bound_opts={"sparse_intermediate_bounds": False}, device=device)
    ptb = PerturbationLpNorm(norm=np.inf, x_L=-r * torch.ones(1, d, device=device),
                             x_U=r * torch.ones(1, d, device=device))
    zb = BoundedTensor(z0, ptb)
    lb, ub = bm.compute_jacobian_bounds((zb,), optimize=optimize, bound_lower=False)
    return float(ub.max().clamp_min(0).sqrt().item())


def mlp_output_upper_bound(mlp_scalar, m, device="cuda", optimize=True):
    """SOUND UB on g(z) over the same box (for the outer-softplus sigmoid factor)."""
    r, d = 1.0 / math.sqrt(m), 2 * m
    z0 = torch.zeros(1, d, device=device)
    bm = BoundedModule(mlp_scalar.to(device), z0, device=device)
    ptb = PerturbationLpNorm(norm=np.inf, x_L=-r * torch.ones(1, d, device=device),
                             x_U=r * torch.ones(1, d, device=device))
    zb = BoundedTensor(z0, ptb)
    method = "CROWN-Optimized" if optimize else "CROWN"
    lb, ub = bm.compute_bounds(x=(zb,), method=method)
    return float(ub.max().item())


def lip_omega_certificate(field, device="cuda", optimize=True):
    m = field.m
    sigma_max_B = torch.linalg.svdvals(field.B.to(device))[0].item()
    L_enc = (2.0 * math.pi / math.sqrt(m)) * sigma_max_B            # EXACT
    mlp = field.mlp_pre_softplus().to(device).eval()
    L_mlp_box = lip_box_mlp(mlp, m, device=device, optimize=optimize)
    z_U = mlp_output_upper_bound(mlp, m, device=device, optimize=optimize)
    s_outer = float(torch.sigmoid(torch.tensor(z_U)).item())       # <= 1
    return {"Lip_Omega": L_enc * L_mlp_box * s_outer, "L_enc": L_enc,
            "L_mlp_box": L_mlp_box, "s_outer": s_outer, "z_U": z_U, "sigma_max_B": sigma_max_B}


if __name__ == "__main__":
    # SMOKE TEST on a tiny tanh MLP (fast; surfaces any API error in seconds)
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = 4
    g = nn.Sequential(nn.Linear(2 * m, 8), nn.Tanh(), nn.Linear(8, 1)).to(dev).eval()
    t = time.time()
    L = lip_box_mlp(g, m, device=dev, optimize=True)
    # cross-check: empirical max grad-norm over random points in the box
    r = 1.0 / math.sqrt(m)
    z = (2 * torch.rand(20000, 2 * m, device=dev) - 1) * r
    z.requires_grad_(True)
    gr = torch.autograd.grad(g(z).sum(), z)[0].norm(dim=1).max().item()
    print(f"SMOKE OK  Lip_box(cert)={L:.4f}  emp_max_grad={gr:.4f}  ratio={L/gr:.2f}  ({time.time()-t:.1f}s)")
    assert L >= gr - 1e-4, "UNSOUND: cert below empirical!"
    print("sound (cert >= empirical).")
