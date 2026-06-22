"""
Multi-state breadth. Replicate the real-vs-artifact decomposition across several states
(GA/CO/IL by default), multi-seed, so the headline numbers become DISTRIBUTIONS, not n=1.

Self-contained (does not mutate geo_m4's GA-specific module state). Per (state, seed): R2_tract, R2_county,
PSI, signed B_red, permutation z. L_cert (mask-restricted) computed once per state (seed 0).
Usage: python geo_m4_multistate.py [STATES_CSV] [NSEED]   e.g.  python geo_m4_multistate.py 13,08,17 5
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import json, numpy as np, torch, time
from zeal_m1 import Field, dev
from zeal_manifold_cert import manifold_cert_masked

STATES = (sys.argv[1] if len(sys.argv) > 1 else "13,08,17").split(",")
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 5
NPERM = 400


class S:                                                              # one state's ladder, on the GPU
    def __init__(self, st, N=512):
        suf = "" if N == 512 else f"_{N}"
        d = np.load(paths.raster(f"geo_st{st}_raster{suf}.npz"))
        self.st = st; self.mask = d["mask"]; self.N = self.mask.shape[0]; mi = self.mask.reshape(-1)
        self.Xf = torch.tensor(np.stack([d["X"].reshape(-1), d["Y"].reshape(-1)], 1), device=dev)
        self.coords = self.Xf[mi]
        self.kfr = torch.tensor(np.nan_to_num(d["kfr"]).reshape(-1)[mi], device=dev)
        self.w = torch.tensor(d["weight"].reshape(-1)[mi], device=dev)
        def lab(idx):
            l = idx.reshape(-1)[mi]; u, inv = np.unique(l, return_inverse=True)
            return torch.tensor(inv, device=dev, dtype=torch.long), len(u)
        self.inv_bg, self.Kbg = lab(d["bg_idx"]); self.inv_tr, self.Ktr = lab(d["tract_idx"]); self.inv_co, self.Kco = lab(d["county_idx"])
        self.tgt_tr = self.wmean(self.kfr, self.inv_tr, self.Ktr)
        self.tgt_co = self.wmean(self.kfr, self.inv_co, self.Kco)
        self.Wt = torch.zeros(self.Ktr, device=dev).index_add_(0, self.inv_tr, self.w)
        self.tc = torch.zeros(self.Ktr, device=dev, dtype=torch.long); self.tc[self.inv_tr] = self.inv_co
        self.Dt = torch.zeros(self.Ktr, device=dev).index_add_(0, self.inv_tr, self.kfr * self.w) / self.Wt.clamp_min(1e-9)

    def wmean(self, vals, inv, K):
        num = torch.zeros(K, device=dev).index_add_(0, inv, vals * self.w)
        den = torch.zeros(K, device=dev).index_add_(0, inv, self.w)
        return (num / den.clamp_min(1e-9))[inv]
    def wr2(self, p, t):
        mu = (self.w * t).sum() / self.w.sum()
        return float(1 - (self.w * (p - t) ** 2).sum() / ((self.w * (t - mu) ** 2).sum() + 1e-12))
    def smooth_full(self, net):
        g = net(self.Xf).view(self.N, self.N)
        return ((g[1:] - g[:-1]) ** 2).mean() + ((g[:, 1:] - g[:, :-1]) ** 2).mean()
    def train(self, beta, iters=3000, lr=2e-3):
        net = Field(m=128, sigma=6.0, w=128, depth=3, act="tanh").to(dev)
        opt = torch.optim.Adam(net.parameters(), lr)
        for it in range(iters):
            opt.zero_grad(); pred = net(self.coords)
            rec = ((self.w * (self.wmean(pred, self.inv_tr, self.Ktr) - self.tgt_tr) ** 2).sum() / self.w.sum()
                   + (self.w * (self.wmean(pred, self.inv_co, self.Kco) - self.tgt_co) ** 2).sum() / self.w.sum())
            (rec + beta * self.smooth_full(net)).backward(); opt.step()
        return net
    def train_masked(self, beta, keep_pix, iters=3000, lr=2e-3):     # fit recon only where keep_pix (spatial CV)
        net = Field(m=128, sigma=6.0, w=128, depth=3, act="tanh").to(dev)
        opt = torch.optim.Adam(net.parameters(), lr); wk = self.w * keep_pix
        for it in range(iters):
            opt.zero_grad(); pred = net(self.coords)
            rec = (wk * (self.wmean(pred, self.inv_tr, self.Ktr) - self.tgt_tr) ** 2).sum() / wk.sum().clamp_min(1e-9)
            (rec + beta * self.smooth_full(net)).backward(); opt.step()
        return net
    def county_of(self, valt, county):
        Wc = torch.zeros(self.Kco, device=dev).index_add_(0, county, self.Wt)
        Vc = torch.zeros(self.Kco, device=dev).index_add_(0, county, valt * self.Wt) / Wc.clamp_min(1e-9)
        return Vc[county]
    def wrms(self, a, b): return float((self.Wt * (a - b) ** 2).sum().div(self.Wt.sum()).sqrt())


def perm_z(s, valt, real):
    g = torch.Generator(device=dev); null = torch.empty(NPERM, device=dev)
    for k in range(NPERM):
        g.manual_seed(1000 + k)
        null[k] = s.wrms(valt, s.county_of(valt, s.tc[torch.randperm(s.Ktr, generator=g, device=dev)]))
    return (real - float(null.mean())) / (float(null.std()) + 1e-12)


def run_state(st, nseed=NSEED, N=512):
    s = S(st, N=N); B_irr = s.wrms(s.Dt, s.county_of(s.Dt, s.tc)); rows = []
    for seed in range(nseed):
        t = time.time(); torch.manual_seed(seed); net = s.train(0.5)
        with torch.no_grad():
            pred = net(s.coords)
            Pbg, Ptr, Pco = s.wmean(pred, s.inv_bg, s.Kbg), s.wmean(pred, s.inv_tr, s.Ktr), s.wmean(pred, s.inv_co, s.Kco)
            r2tr, r2co = s.wr2(Ptr, s.tgt_tr), s.wr2(Pco, s.tgt_co)
            Vt = torch.zeros(s.Ktr, device=dev).index_add_(0, s.inv_tr, pred * s.w) / s.Wt.clamp_min(1e-9)
            mov = s.wrms(Vt, s.county_of(Vt, s.tc))
            stack = torch.stack([Pbg, Ptr, Pco], 0)
            psi = float(((s.w * stack.var(0)).sum() / s.w.sum()) / (((s.w * (Ptr - (s.w * Ptr).sum() / s.w.sum()) ** 2).sum() / s.w.sum()) + 1e-12))
        pz = perm_z(s, Vt, mov)
        Lc = (manifold_cert_masked(net, s.mask, budget=60000)[0] if seed == 0 else None)
        rows.append(dict(seed=seed, r2tr=r2tr, r2co=r2co, psi=psi, mov=mov, B_red_signed=mov - B_irr, perm_z=pz, L_cert=Lc))
        print(f"{st:>5s} {seed:4d} {r2tr:6.3f} {r2co:6.3f} {psi:6.3f} {mov:7.4f} {mov - B_irr:+8.4f} {pz:+7.2f} "
              f"{(Lc if Lc else float('nan')):8.1f}   ({time.time()-t:.0f}s)", flush=True)
    a = lambda k: np.array([r[k] for r in rows])
    print(f"  -> state {st}: R2_tr {a('r2tr').mean():.3f}+-{a('r2tr').std():.3f}  PSI {a('psi').mean():.3f}+-{a('psi').std():.3f}  "
          f"B_red {a('B_red_signed').mean():+.4f}+-{a('B_red_signed').std():.4f}  perm_z {a('perm_z').mean():+.2f}", flush=True)
    return dict(B_irr=B_irr, Ktr=s.Ktr, Kco=s.Kco, Kbg=s.Kbg, rows=rows)


if __name__ == "__main__":
  print(f"multi-state: states {STATES}, {NSEED} seeds each, beta=0.5")
  print(f"{'state':>5s} {'seed':>4s} {'R2_tr':>6s} {'R2_co':>6s} {'PSI':>6s} {'mov':>7s} {'B_red':>8s} {'perm_z':>7s} {'L_cert':>8s}")
  out = {}
  for st in STATES:
    out[st] = run_state(st)
  json.dump(out, open(paths.data("geo_m4_multistate.json"), "w"))
  print("\nSaved geo_m4_multistate.json")
