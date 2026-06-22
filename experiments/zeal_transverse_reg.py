"""
Test (B): manifold-aligned (certifiability-aware) regularization.
Penalize the TRANSVERSE MLP gradient  ||n(x)||^2 = ||grad g(gamma(x))||^2 - ||P_T grad g||^2
during training -> should drive rho_tan -> 1 -> make the EXISTING alpha-CROWN cert tight.
Track rho_off too (does the slack migrate off-manifold instead?). R2 should hold (transverse
gradient is irrelevant to lambda on the manifold).

||P_T grad g||^2 = ||M^{-1/2} grad h||^2,  M = J_gamma^T J_gamma = (2pi/sqrt(mf))^2 B^T B (2x2 const),
grad h = grad_x (g o gamma) = J_gamma^T grad g.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import math, time, numpy as np, torch
import zeal_m1 as M, g0_numerics as g0
M.load_sandbox()
from zeal_lip_alpha_crown import lip_omega_certificate

dev = M.dev
Z, Zp = M.test_Z[0], M.test_Z[1]
basis = [g0.field_linear(M.XX, M.YY, u) for u in ([1, 0], [0, 1], [1, 1])]


def Minv_sqrt_of(net):
    Mmat = (2 * math.pi / math.sqrt(net.m)) ** 2 * (net.B.t() @ net.B)
    ev, evec = torch.linalg.eigh(Mmat)
    return evec @ torch.diag(ev.clamp_min(1e-20).rsqrt()) @ evec.t(), math.sqrt(ev.max().item())


def train(lam_perp, beta=2.0, iters=3200, lr=2e-3, nsub=4096, warmup=1000, ramp_iters=700):
    net = M.Field(m=128, sigma=6.0, w=128, depth=3, act="tanh").to(dev)
    Minv_sqrt, _ = Minv_sqrt_of(net)
    opt = torch.optim.Adam(net.parameters(), lr)
    Ncoord = M.coords.shape[0]
    for it in range(iters):
        opt.zero_grad()
        pred = net(M.coords)
        rec = sum(((M.cell_mean(pred, inv, K) - tm) ** 2).mean()
                  for inv, K, tm in M.train_idx) / len(M.train_idx)
        loss = rec + beta * M.smoothness(pred.view(M.N, M.N))
        if lam_perp > 0 and it >= warmup:                       # warm up fit, then ramp penalty in
            w = lam_perp * min(1.0, (it - warmup) / ramp_iters)
            xs = M.coords[torch.randint(0, Ncoord, (nsub,), device=dev)]
            xr = xs.clone().requires_grad_(True)
            gradh = torch.autograd.grad(net.net(net.feat(xr)).sum(), xr, create_graph=True)[0]
            pt2 = (gradh @ Minv_sqrt.t()).pow(2).sum(1)
            zr = net.feat(xs).detach().requires_grad_(True)
            full2 = torch.autograd.grad(net.net(zr).sum(), zr, create_graph=True)[0].pow(2).sum(1)
            loss = loss + w * (full2 - pt2).clamp_min(0).mean()
        loss.backward(); opt.step()
        if it == 350:
            with torch.no_grad():
                if M.r2(net(M.coords), M.test_idx) < 0.1:
                    print(f"  lam={lam_perp}: SANITY FAIL @350 -> skip", flush=True); return None
    return net


def measure(net):
    Minv_sqrt, L_gamma = Minv_sqrt_of(net)
    with torch.no_grad():
        R2 = M.r2(net(M.coords), M.test_idx)
        grid = net(M.coords).view(M.N, M.N).cpu().numpy()
    Lip_emp = g0.lipschitz(grid, M.H_COORD)
    Xr = M.coords.clone().requires_grad_(True)
    gradh = torch.autograd.grad(net.net(net.feat(Xr)).sum(), Xr)[0]
    S_star = gradh.norm(dim=1).max().item()
    T_M = (gradh @ Minv_sqrt.t()).norm(dim=1).max().item()
    zr = net.feat(M.coords).detach().requires_grad_(True)
    G_M = torch.autograd.grad(net.net(zr).sum(), zr)[0].norm(dim=1).max().item()
    r = 1.0 / math.sqrt(net.m)
    zb = ((2 * torch.rand(40000, 2 * net.m, device=dev) - 1) * r).requires_grad_(True)
    gb = torch.autograd.grad(net.net(zb).sum(), zb)[0].norm(dim=1); G_B = gb.max().item()
    seeds = zb[gb.topk(256).indices].detach()
    for _ in range(60):
        s = seeds.clone().requires_grad_(True)
        gn = torch.autograd.grad(net.net(s).sum(), s, create_graph=True)[0]
        step = torch.autograd.grad((gn * gn).sum(), s)[0]
        seeds = (seeds + 1e-3 * step).clamp(-r, r).detach()
        G_B = max(G_B, gn.detach().norm(dim=1).max().item())
    Lc = lip_omega_certificate(net, device=dev, optimize=True)["Lip_Omega"]

    def kap(Lip, dpx=3.0):
        dhat = max(g0.certificate_N(f, Z, Zp, dpx, M.H_COORD) / (g0.lipschitz(f, M.H_COORD) + 1e-12) for f in basis)
        return g0.certificate_N(grid, Z, Zp, dpx, M.H_COORD) / (Lip * dhat)
    return dict(R2=R2, Lip_emp=Lip_emp, S_star=S_star, rho_tan=G_M / T_M, rho_off=G_B / G_M,
                rho_cond=L_gamma * T_M / S_star, Lc=Lc, kap_emp=kap(Lip_emp), kap_cert=kap(Lc))


print(f"{'lam_perp':>8s} {'R2':>6s} {'kap_emp':>7s} {'Lip_emp':>7s} {'rho_tan':>7s} {'rho_off':>7s} "
      f"{'L_cert':>9s} {'cert/emp':>8s} {'kap_cert':>8s}", flush=True)
for lam in (0.0, 0.01, 0.1, 0.5, 2.0):
    t = time.time()
    net = train(lam)
    if net is None:
        continue
    r = measure(net)
    print(f"{lam:8.1f} {r['R2']:6.3f} {r['kap_emp']:7.4f} {r['Lip_emp']:7.2f} {r['rho_tan']:7.2f} "
          f"{r['rho_off']:7.2f} {r['Lc']:9.1f} {r['Lc']/r['Lip_emp']:8.1f} {r['kap_cert']:8.4f}  "
          f"({time.time()-t:.0f}s)", flush=True)
print("\nGO if rho_tan->~1 AND cert/emp single-digit AND kap_cert non-vacuous, with R2 holding.", flush=True)
print("WATCH rho_off: if it grows, the transverse slack just migrated off-manifold (cert still loose).", flush=True)
