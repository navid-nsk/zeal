"""New data for the figure rebuild (GA field). Produces:
  - R6 signed-difference map (county - tract) for the fig7a triptych
  - D7 multi-rung certified envelope vs observed movement (>=5 aggregation transitions), multi-seed
  - D1 Lipschitz-evidence: output drift d_out vs zoning distance d(Z,Z') over many coarsenings
Saves zeal/data/newdata_field.npz (+ .json for scalars)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import numpy as np, torch, json, time
from sklearn.cluster import KMeans
import geo_m4_multistate as M
from zeal_m1 import dev

LCERT_TIGHT = 242.0                                              # mask-restricted L_cert @250K (fig5)
s = M.S("13"); N = s.N; mi = s.mask.reshape(-1)
cen_t = torch.zeros(s.Ktr, 2, device=dev).index_add_(0, s.inv_tr, s.coords * s.w[:, None]) / \
        torch.zeros(s.Ktr, device=dev).index_add_(0, s.inv_tr, s.w)[:, None].clamp_min(1e-9)

def wmean_lab(vals, lab, K, w):
    num = torch.zeros(K, device=dev).index_add_(0, lab, vals * w)
    den = torch.zeros(K, device=dev).index_add_(0, lab, w)
    return (num / den.clamp_min(1e-9))[lab]

def agg_to(labels, K, pred):                                    # decode field onto a zoning, broadcast to pixels
    return wmean_lab(pred, labels, K, s.w)

def d_geo(labA, KA, labB, KB):                                  # geometric transport distance between two zonings
    def cs(lab, K):
        cnt = torch.bincount(lab, minlength=K).clamp_min(1).float()
        cen = torch.zeros(K, 2, device=dev).index_add_(0, lab, s.coords) / cnt[:, None]
        d2 = ((s.coords - cen[lab]) ** 2).sum(1)
        sp = (torch.zeros(K, device=dev).index_add_(0, lab, d2) / cnt).sqrt()
        return cen, sp
    cA, spA = cs(labA, KA); cB, spB = cs(labB, KB)
    return float((((cA[labA] - cB[labB]).norm(dim=1) + spA[labA] + spB[labB]) ** 2).mean().sqrt())

def rms(a, b): return float((s.w * (a - b) ** 2).sum().div(s.w.sum()).sqrt())

# tract-centroid clusterings at many scales -> coarsenings (for D1) computed once (geometry only)
Xc = cen_t.cpu().numpy()
clusterings = {}
for k in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 350]:
    lab = KMeans(k, n_init=3, random_state=0).fit_predict(Xc)
    clusterings[k] = (torch.tensor(lab, device=dev, dtype=torch.long)[s.inv_tr], k)   # per-pixel cluster label

rungs = [("BG↔tract", s.inv_bg, s.Kbg, s.inv_tr, s.Ktr),
         ("tract↔county", s.inv_tr, s.Ktr, s.inv_co, s.Kco),
         ("BG↔county", s.inv_bg, s.Kbg, s.inv_co, s.Kco)]

D7 = {r[0]: {"mov": [], "d": None} for r in rungs}
D1 = {"d": [], "drift": []}                                      # over seeds+clusterings
NSEED = 5
t0 = time.time()
for seed in range(NSEED):
    torch.manual_seed(seed); net = s.train(0.5)
    with torch.no_grad():
        pred = net(s.coords)
    for nm, la, Ka, lb, Kb in rungs:
        mov = rms(agg_to(la, Ka, pred), agg_to(lb, Kb, pred))
        D7[nm]["mov"].append(mov)
        if D7[nm]["d"] is None: D7[nm]["d"] = d_geo(la, Ka, lb, Kb)
    # D1: drift between each coarsening and the tract zoning, vs their geometric distance
    for k, (clab, K) in clusterings.items():
        drift = rms(agg_to(s.inv_tr, s.Ktr, pred), agg_to(clab, K, pred))
        if seed == 0: D1["d"].append(d_geo(s.inv_tr, s.Ktr, clab, K))
        D1["drift"].append((k, D1["d"][list(clusterings).index(k)], drift))
    print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

# R6 signed difference map (seed 0 field already trained last loop is seed4; retrain seed0 for the map)
torch.manual_seed(0); net = s.train(0.5)
with torch.no_grad():
    pred = net(s.coords)
    tr = wmean_lab(pred, s.inv_tr, s.Ktr, s.w).cpu().numpy()
    co = wmean_lab(pred, s.inv_co, s.Kco, s.w).cpu().numpy()
def to_grid(v):
    g = np.full(N * N, np.nan); g[mi] = v; return g.reshape(N, N)
np.savez(paths.data("newdata_field.npz"),
         diff=to_grid(co - tr), tr=to_grid(tr), co=to_grid(co), mask=s.mask)

out = {"LCERT_TIGHT": LCERT_TIGHT,
       "D7": {nm: {"mov_mean": float(np.mean(D7[nm]["mov"])), "mov_std": float(np.std(D7[nm]["mov"])),
                   "d": D7[nm]["d"], "envelope": LCERT_TIGHT * D7[nm]["d"]} for nm in D7},
       "D1": D1["drift"]}
json.dump(out, open(paths.data("newdata_field.json"), "w"))
print("\nD7 multi-rung envelope (L_cert=242):")
for nm in D7:
    e = out["D7"][nm]; print(f"  {nm:14s} movement={e['mov_mean']:.4f}±{e['mov_std']:.4f}  d={e['d']:.3f}  envelope={e['envelope']:.2f}")
print(f"D1 drift-vs-distance: {len(D1['drift'])} points; d∈[{min(p[1] for p in D1['drift']):.3f},{max(p[1] for p in D1['drift']):.3f}] "
      f"drift∈[{min(p[2] for p in D1['drift']):.4f},{max(p[2] for p in D1['drift']):.4f}]")
print("saved newdata_field.npz + .json")
