"""LOCAL cross-zoning certificate on Georgia (tract -> county).

For a location x in tract a inside county b, Kantorovich-Rubinstein duality gives the pointwise bound
    | <lam>_a - <lam>_b |  <=  L_loc(b) * W1(mu_a, mu_b),
with mu_a, mu_b the count-weighted cell measures and L_loc(b) a certified bound on ||grad lam|| over the
convex hull of b (we use the bounding box, a superset). L_loc(b) comes from the SAME manifold-aware
branch-and-bound as the global L_cert, but keeping every tile's certified bound instead of only the max,
and refining the tiles that bind each county's constant (a local branch-and-bound).
From the local envelope e(x) follow: the max-local-change certificate sup_x e(x); a certified hot-spot
map (certified-in / certified-out / undetermined); a per-location stability radius; and rank intervals.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import sys, json, time, numpy as np, torch, ot
import geo_m4_multistate as M
from zeal_m1 import dev
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from zeal_manifold_cert import _JacW, encoder_box, project_bound

K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 256
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 250000
FRAC = 0.6
OUT = paths.field_file("local_cert")

t0 = time.time()
s = M.S("13"); N = s.N
torch.manual_seed(0); net = s.train(0.5)
with torch.no_grad():
    pred = net(s.coords)
print(f"[{time.time()-t0:.0f}s] field trained (beta=0.5, seed 0)", flush=True)


def make_tb(net, batch=1024):
    B, m, g = net.B.to(dev), net.m, net.net.to(dev).eval()
    bm_j = BoundedModule(_JacW(g), torch.zeros(batch, 2 * m, device=dev), device=dev,
                         bound_opts={"sparse_intermediate_bounds": False})
    bm_f = BoundedModule(g, torch.zeros(batch, 2 * m, device=dev), device=dev)

    def tb(XL, XH):                                        # sound per-tile bound on ||grad lam||, softplus folded
        T = XL.shape[0]; out = torch.zeros(T, device=dev)
        for i in range(0, T, batch):
            xl, xh = XL[i:i + batch], XH[i:i + batch]; n = xl.shape[0]
            if n < batch:
                xl = torch.cat([xl, xl[-1:].expand(batch - n, 2)]); xh = torch.cat([xh, xh[-1:].expand(batch - n, 2)])
            sL, sH, cL, cH, zL, zH = encoder_box(B, m, xl, xh)
            zb = BoundedTensor((zL + zH) / 2, PerturbationLpNorm(norm=np.inf, x_L=zL, x_U=zH))
            lb, ub = bm_j.compute_jacobian_bounds((zb,), optimize=False)
            pb = project_bound(B, m, sL, sH, cL, cH, lb.detach().reshape(batch, -1), ub.detach().reshape(batch, -1))
            _, gub = bm_f.compute_bounds(x=(zb,), method="CROWN")
            out[i:i + n] = (pb * torch.sigmoid(gub.detach().reshape(-1)))[:n]
        return out
    return tb


tb = make_tb(net)

# ---- county bounding boxes (superset of the convex hull -> sound domain for the local Lipschitz constant)
co_lo = torch.full((s.Kco, 2), 9.0, device=dev).scatter_reduce(0, s.inv_co[:, None].expand(-1, 2), s.coords, "amin")
co_hi = torch.full((s.Kco, 2), -9.0, device=dev).scatter_reduce(0, s.inv_co[:, None].expand(-1, 2), s.coords, "amax")
h = 2.0 / (N - 1); co_lo -= h; co_hi += h                       # pad by one pixel


def county_tiles(XL, XH):                                     # [Kco, T] bool: tile intersects county bbox
    return ((XL[None, :, 0] <= co_hi[:, None, 0]) & (XH[None, :, 0] >= co_lo[:, None, 0]) &
            (XL[None, :, 1] <= co_hi[:, None, 1]) & (XH[None, :, 1] >= co_lo[:, None, 1]))


def local_L(XL, XH, bnd):
    ct = county_tiles(XL, XH)
    return torch.where(ct, bnd[None, :], torch.zeros_like(bnd)[None, :]).max(1).values, ct


# ---- uniform tile grid, then LOCAL branch-and-bound: split tiles that bind ANY county's constant
e = torch.linspace(-1, 1, K0 + 1, device=dev)
XL = torch.stack(torch.meshgrid(e[:-1], e[:-1], indexing="ij"), -1).reshape(-1, 2)
XH = torch.stack(torch.meshgrid(e[1:], e[1:], indexing="ij"), -1).reshape(-1, 2)
bnd = tb(XL, XH)
Lloc0, _ = local_L(XL, XH, bnd)
print(f"[{time.time()-t0:.0f}s] uniform k0={K0}: {XL.shape[0]} tiles, global max {bnd.max():.1f}, "
      f"L_loc median {Lloc0.median():.2f} min {Lloc0.min():.2f} max {Lloc0.max():.1f}", flush=True)
it = 0
while XL.shape[0] < BUDGET:
    Lloc, ct = local_L(XL, XH, bnd)
    thr = torch.where(ct, Lloc[:, None], torch.zeros_like(ct, dtype=bnd.dtype)).max(0).values   # per-tile threshold
    act = (bnd >= FRAC * thr) & (thr > 0)
    if act.sum() == 0: break
    aL, aH = XL[act], XH[act]; mid = (aL + aH) / 2
    x0, y0, x1, y1, mx, my = aL[:, 0], aL[:, 1], aH[:, 0], aH[:, 1], mid[:, 0], mid[:, 1]
    nL = torch.cat([torch.stack([x0, y0], 1), torch.stack([mx, y0], 1), torch.stack([x0, my], 1), torch.stack([mx, my], 1)])
    nH = torch.cat([torch.stack([mx, my], 1), torch.stack([x1, my], 1), torch.stack([mx, y1], 1), torch.stack([x1, y1], 1)])
    XL = torch.cat([XL[~act], nL]); XH = torch.cat([XH[~act], nH]); bnd = torch.cat([bnd[~act], tb(nL, nH)])
    it += 1
    if it % 3 == 0:
        Ll, _ = local_L(XL, XH, bnd)
        print(f"   bab it {it}: {XL.shape[0]} tiles, split {int(act.sum())}, global {bnd.max():.1f}, L_loc median {Ll.median():.2f}", flush=True)
Lloc, ct = local_L(XL, XH, bnd); L_glob = float(bnd.max())
print(f"[{time.time()-t0:.0f}s] local BaB done: {XL.shape[0]} tiles | L_glob {L_glob:.1f} | "
      f"L_loc: min {Lloc.min():.2f} q25 {Lloc.quantile(.25):.2f} median {Lloc.median():.2f} q75 {Lloc.quantile(.75):.2f} max {Lloc.max():.1f}", flush=True)

# ---- exact count-weighted W1(mu_tract, mu_county) per tract
coords = s.coords.cpu().numpy().astype(np.float64); w = s.w.cpu().numpy().astype(np.float64)
inv_tr = s.inv_tr.cpu().numpy(); inv_co = s.inv_co.cpu().numpy(); tc = s.tc.cpu().numpy()
pix_tr = [np.where(inv_tr == a)[0] for a in range(s.Ktr)]; pix_co = [np.where(inv_co == b)[0] for b in range(s.Kco)]
np.savez_compressed(OUT + "_tiles.npz", XL=XL.cpu().numpy(), XH=XH.cpu().numpy(), bnd=bnd.cpu().numpy(), Lloc=Lloc.cpu().numpy())
def meas(p):                                                   # count-weighted cell measure (uniform if the cell has no count)
    wp = w[p]; return wp / wp.sum() if wp.sum() > 0 else np.full(len(p), 1.0 / len(p))
W1 = np.zeros(s.Ktr)
for a in range(s.Ktr):
    pa, pb = pix_tr[a], pix_co[tc[a]]
    Mc = ot.dist(coords[pa], coords[pb], metric="euclidean")
    W1[a] = ot.emd2(meas(pa), meas(pb), Mc)
print(f"[{time.time()-t0:.0f}s] W1 per tract: median {np.median(W1):.4f} max {W1.max():.4f} (coordinate units)", flush=True)

# ---- values, movements, envelopes
Vt = (torch.zeros(s.Ktr, device=dev).index_add_(0, s.inv_tr, pred * s.w) / s.Wt.clamp_min(1e-9))
Ct = s.county_of(Vt, s.tc); Dt = s.Dt; DCt = s.county_of(Dt, s.tc)
Vt, Ct, Dt, DCt = [x.cpu().numpy() for x in (Vt, Ct, Dt, DCt)]
Lloc_np = Lloc.cpu().numpy(); Wt = s.Wt.cpu().numpy()
mov_loc = np.abs(Vt - Ct); D_loc = np.abs(Dt - DCt)
e_loc = Lloc_np[tc] * W1; e_glob = L_glob * W1
sound = bool(np.all(e_loc >= mov_loc - 1e-9)); viol = int(np.sum(e_loc < mov_loc - 1e-9))
rng = float(Vt.max() - Vt.min())
print(f"\n=== LOCAL ENVELOPE (tract->county), field range {rng:.3f} ===")
print(f"SOUND: {sound} (violations {viol}/{s.Ktr})")
print(f"observed local movement |V_a - C_b|: median {np.median(mov_loc):.4f}  max {mov_loc.max():.4f}")
print(f"data-level local movement |D_a - D_b|: median {np.median(D_loc):.4f}  max {D_loc.max():.4f}")
print(f"local envelope e_loc: median {np.median(e_loc):.4f}  max {e_loc.max():.4f}   (max-local-change certificate)")
print(f"global-L envelope e_glob: median {np.median(e_glob):.4f}  max {e_glob.max():.4f}")
print(f"tightening local vs global-L (median ratio e_glob/e_loc): {np.median(e_glob / e_loc):.1f}x")
print(f"envelope/observed ratio: local median {np.median(e_loc / np.maximum(mov_loc, 1e-9)):.1f}x   global median {np.median(e_glob / np.maximum(mov_loc, 1e-9)):.1f}x")
nv = e_loc < rng
print(f"non-vacuous tracts (e_loc < field range): {nv.sum()}/{s.Ktr} = {100*nv.mean():.1f}% (pop share {100*Wt[nv].sum()/Wt.sum():.1f}%)")

# ---- certified hot-spot map at the population-weighted top-quintile threshold
o = np.argsort(Vt); cw = np.cumsum(Wt[o]) / Wt.sum(); tau = float(Vt[o][np.searchsorted(cw, 0.8)])
cin = (Vt - tau) > e_loc; cout = (tau - Vt) > e_loc; und = ~(cin | cout)
hot = Vt > tau
print(f"\n=== CERTIFIED HOT-SPOT MAP (tau = top-quintile {tau:.4f}) ===")
print(f"hot at tract rung: {hot.sum()} tracts | certified-in {cin.sum()} ({100*cin.sum()/max(hot.sum(),1):.0f}% of hot) | "
      f"certified-out {cout.sum()} ({100*cout.sum()/max((~hot).sum(),1):.0f}% of cold) | undetermined {und.sum()}")
cin_g = (Vt - tau) > e_glob; cout_g = (tau - Vt) > e_glob
print(f"same with the GLOBAL constant: certified-in {cin_g.sum()} certified-out {cout_g.sum()}")
truth_ok = np.all((Ct > tau)[cin]) and np.all((Ct < tau)[cout])
print(f"certified classes agree with the county-rung truth: {bool(truth_ok)}")
r_star = np.abs(Vt - tau) / Lloc_np[tc]                     # stability radius (transport units) for the classification
print(f"stability radius r*: median {np.median(r_star):.4f}  (W1 to county: median {np.median(W1):.4f}); "
      f"tracts with r* > their W1: {(r_star > W1).sum()}")

# ---- rank intervals
diff = np.abs(Vt[:, None] - Vt[None, :]); ee = e_loc[:, None] + e_loc[None, :]
Nrank = (diff <= ee).sum(1) - 1
print(f"\n=== RANK INTERVALS === half-width N(a): median {np.median(Nrank):.0f} of {s.Ktr} tracts; "
      f"tracts with N < 10% of tracts: {(Nrank < 0.1*s.Ktr).sum()}")

# ---- implied lower bound on the TRUE field's local Lipschitz from the data's own movement
L_true_lo = np.zeros(s.Kco)
np.maximum.at(L_true_lo, tc, D_loc / np.maximum(W1, 1e-9))
print(f"\ndata-implied lower bound on the true field local Lipschitz: median {np.median(L_true_lo):.2f} max {L_true_lo.max():.2f}"
      f"  vs certified L_loc of the fitted field: median {np.median(Lloc_np):.2f}")

np.savez_compressed(OUT + ".npz", XL=XL.cpu().numpy(), XH=XH.cpu().numpy(), bnd=bnd.cpu().numpy(), Lloc=Lloc_np,
                    W1=W1, Vt=Vt, Ct=Ct, Dt=Dt, DCt=DCt, Wt=Wt, tc=tc, e_loc=e_loc, e_glob=e_glob, mov_loc=mov_loc,
                    tau=tau, cin=cin, cout=cout, r_star=r_star, Nrank=Nrank, L_true_lo=L_true_lo)
json.dump(dict(k0=K0, budget=BUDGET, tiles=int(XL.shape[0]), L_glob=L_glob, sound=sound, violations=viol, field_range=rng,
               Lloc_q=[float(np.quantile(Lloc_np, q)) for q in (0, .25, .5, .75, 1)],
               W1_median=float(np.median(W1)), mov_loc_median=float(np.median(mov_loc)), mov_loc_max=float(mov_loc.max()),
               D_loc_median=float(np.median(D_loc)), D_loc_max=float(D_loc.max()),
               e_loc_median=float(np.median(e_loc)), e_loc_max=float(e_loc.max()),
               e_glob_median=float(np.median(e_glob)), e_glob_max=float(e_glob.max()),
               tighten_median=float(np.median(e_glob / e_loc)),
               ratio_loc_median=float(np.median(e_loc / np.maximum(mov_loc, 1e-9))),
               nonvacuous_frac=float(nv.mean()), tau=tau, hot=int(hot.sum()), cert_in=int(cin.sum()), cert_out=int(cout.sum()),
               undetermined=int(und.sum()), cert_in_glob=int(cin_g.sum()), cert_out_glob=int(cout_g.sum()),
               r_star_median=float(np.median(r_star)), Nrank_median=float(np.median(Nrank)),
               L_true_lo_median=float(np.median(L_true_lo)), seconds=time.time() - t0),
          open(OUT + ".json", "w"), indent=2)
print(f"\nsaved {OUT}.json / .npz  ({time.time()-t0:.0f}s)")
