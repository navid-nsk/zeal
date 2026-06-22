"""How the ambient-box looseness scales with embedding dimension. For Fourier widths m
   (ambient dim = 2m), train the field and measure on-manifold vs ambient-box gradient bounds.
   rho_off = G_B/G_M (ambient/manifold ratio), rho_tan = G_M/T_M (transverse term). Multi-seed.
   Saves zeal/data/dimsweep.json."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import math, json, time, numpy as np, torch
import zeal_m1 as M
M.load_sandbox()
dev = M.dev

def Minv_sqrt_of(net):
    Mmat = (2 * math.pi / math.sqrt(net.m)) ** 2 * (net.B.t() @ net.B)
    ev, evec = torch.linalg.eigh(Mmat)
    return evec @ torch.diag(ev.clamp_min(1e-20).rsqrt()) @ evec.t()

def train(m, seed, iters=1500, beta=2.0, lr=2e-3):
    torch.manual_seed(seed)
    net = M.Field(m=m, sigma=6.0, w=128, depth=3).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr)
    for it in range(iters):
        opt.zero_grad(); pred = net(M.coords)
        rec = sum(((M.cell_mean(pred, inv, K) - tm) ** 2).mean() for inv, K, tm in M.train_idx) / len(M.train_idx)
        (rec + beta * M.smoothness(pred.view(M.N, M.N))).backward(); opt.step()
    return net

def measure(net):
    Minv = Minv_sqrt_of(net)
    Xr = M.coords.clone().requires_grad_(True)
    gradh = torch.autograd.grad(net.net(net.feat(Xr)).sum(), Xr)[0]
    S_star = gradh.norm(dim=1).max().item()                      # on-manifold input gradient (the truth)
    T_M = (gradh @ Minv.t()).norm(dim=1).max().item()
    zr = net.feat(M.coords).detach().requires_grad_(True)
    G_M = torch.autograd.grad(net.net(zr).sum(), zr)[0].norm(dim=1).max().item()   # feature grad on manifold
    r = 1.0 / math.sqrt(net.m)
    zb = ((2 * torch.rand(40000, 2 * net.m, device=dev) - 1) * r).requires_grad_(True)
    gb = torch.autograd.grad(net.net(zb).sum(), zb)[0].norm(dim=1); G_B = gb.max().item()
    seeds = zb[gb.topk(256).indices].detach()
    for _ in range(60):
        sd = seeds.clone().requires_grad_(True)
        gn = torch.autograd.grad(net.net(sd).sum(), sd, create_graph=True)[0]
        step = torch.autograd.grad((gn * gn).sum(), sd)[0]
        seeds = (seeds + 1e-3 * step).clamp(-r, r).detach()
        G_B = max(G_B, gn.detach().norm(dim=1).max().item())     # ambient-box worst-case (PGD)
    return dict(S_star=S_star, G_M=G_M, T_M=T_M, G_B=G_B,
                rho_off=G_B / G_M, rho_tan=G_M / T_M, ratio_box=G_B / S_star)

ms = [8, 16, 32, 64, 128, 256]; NSEED = 3
rows = []; t0 = time.time()
print(f"{'m':>4s} {'dim':>5s} {'rho_off':>8s} {'rho_tan':>8s} {'box/true':>9s}")
for m in ms:
    acc = {k: [] for k in ("rho_off", "rho_tan", "ratio_box")}
    for sd in range(NSEED):
        r = measure(train(m, sd))
        for k in acc: acc[k].append(r[k])
    row = {"m": m, "dim": 2 * m, **{k: float(np.mean(v)) for k, v in acc.items()},
           **{k + "_std": float(np.std(v)) for k, v in acc.items()}}
    rows.append(row)
    print(f"{m:4d} {2*m:5d} {row['rho_off']:8.1f} {row['rho_tan']:8.1f} {row['ratio_box']:9.1f}   ({time.time()-t0:.0f}s)", flush=True)
json.dump({"rows": rows, "nseed": NSEED}, open(paths.data("dimsweep.json"), "w"))
print("saved dimsweep.json")
