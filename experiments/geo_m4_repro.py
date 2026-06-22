"""
Multi-seed reproducibility of the Georgia real-vs-artifact decomposition,
with SIGNED B_red (no clamp) + a permutation-null significance test.

Per seed (Fourier B + init): train the ZEAL field on the GA ladder (beta=0.5), then report
  R2_tract, R2_county, PSI, field tract->county movement, data movement B_irr, SIGNED B_red = field - data,
  a permutation p-value (is the field's movement specific to the REAL county geography vs random same-size
  partitions?), and the mask-restricted certified L_cert.
Aggregates to mean +- std across seeds and the permutation result. Usage: python geo_m4_repro.py [NSEED] [CERT_BUDGET] [NPERM]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import sys, json, numpy as np, torch, time
import geo_m4 as G
from zeal_manifold_cert import manifold_cert_masked
dev = G.dev
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CERT_BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 400

# ---- per-tract reductions (tract is the finest identifiable rung; nested tract in county) ----
Wt = torch.zeros(G.Ktr, device=dev).index_add_(0, G.inv_tr, G.w)              # tract weight
tc = torch.zeros(G.Ktr, device=dev, dtype=torch.long); tc[G.inv_tr] = G.inv_co  # tract -> county (nested)
Dt = (torch.zeros(G.Ktr, device=dev).index_add_(0, G.inv_tr, G.kfr * G.w) / Wt.clamp_min(1e-9))  # data kfr per tract

def county_of(valt, county):                                                  # county-aggregate, broadcast to tracts
    Wc = torch.zeros(G.Kco, device=dev).index_add_(0, county, Wt)
    Vc = torch.zeros(G.Kco, device=dev).index_add_(0, county, valt * Wt) / Wc.clamp_min(1e-9)
    return Vc[county]
def wrms(a, b):
    return float((Wt * (a - b) ** 2).sum().div(Wt.sum()).sqrt())

B_irr = wrms(Dt, county_of(Dt, tc))                                            # genuine data movement (fixed)

def perm_null(valt, real_mov):                                                 # movement vs random same-size partitions
    g = torch.Generator(device=dev)
    null = torch.empty(NPERM, device=dev)
    for k in range(NPERM):
        g.manual_seed(1000 + k)
        fake = tc[torch.randperm(G.Ktr, generator=g, device=dev)]              # same county-size multiset, shuffled
        null[k] = wrms(valt, county_of(valt, fake))
    nmean, nstd = float(null.mean()), float(null.std())
    pct = float((null <= real_mov).float().mean())                             # percentile of real within null
    z = (real_mov - nmean) / (nstd + 1e-12)                                    # <0 => real gentler than random
    return pct, nmean, nstd, z

# data movement under real geography vs random partitions (is real admin geography coherent / genuine?)
data_pct, data_nmean, data_nstd, data_z = perm_null(Dt, B_irr)

def one(seed):
    torch.manual_seed(seed)
    net = G.train(beta=0.5)
    with torch.no_grad():
        pred = net(G.coords)
        P_bg, P_tr, P_co = G.wmean(pred, G.inv_bg, G.Kbg), G.wmean(pred, G.inv_tr, G.Ktr), G.wmean(pred, G.inv_co, G.Kco)
        r2tr, r2co = G.wr2(P_tr, G.tgt_tr), G.wr2(P_co, G.tgt_co)
        Vt = (torch.zeros(G.Ktr, device=dev).index_add_(0, G.inv_tr, pred * G.w) / Wt.clamp_min(1e-9))
        mov = wrms(Vt, county_of(Vt, tc))
        stack = torch.stack([P_bg, P_tr, P_co], 0)
        psi = float(((G.w * stack.var(0)).sum() / G.w.sum()) / (((G.w * (P_tr - (G.w * P_tr).sum() / G.w.sum()) ** 2).sum() / G.w.sum()) + 1e-12))
    pct, nmean, nstd, z = perm_null(Vt, mov)
    Lc, _ = manifold_cert_masked(net, G.mask, budget=CERT_BUDGET)
    return dict(seed=seed, r2tr=r2tr, r2co=r2co, psi=psi, mov=mov, B_irr=B_irr,
                B_red_signed=mov - B_irr, perm_pct=pct, perm_z=z, perm_null_mean=nmean, L_cert=Lc)

print(f"GA decomposition reproducibility: {NSEED} seeds, cert budget {CERT_BUDGET}, {NPERM} permutations")
print(f"B_irr (genuine data tract->county movement, fixed) = {B_irr:.4f}")
print(f"  data movement vs random same-size partitions: null={data_nmean:.4f}+-{data_nstd:.4f}  z={data_z:+.2f}  "
      f"(real admin geography is {'gentler/coherent' if data_z < 0 else 'harsher'} than random)")
print(f"{'seed':>4s} {'R2_tr':>6s} {'R2_co':>6s} {'PSI':>6s} {'mov':>7s} {'B_red':>8s} {'perm_z':>7s} {'L_cert':>8s}")
rows = []
for s in range(NSEED):
    t = time.time(); r = one(s); rows.append(r)
    print(f"{s:4d} {r['r2tr']:6.3f} {r['r2co']:6.3f} {r['psi']:6.3f} {r['mov']:7.4f} "
          f"{r['B_red_signed']:+8.4f} {r['perm_z']:+7.2f} {r['L_cert']:8.1f}   ({time.time()-t:.0f}s)", flush=True)

def ms(key):
    a = np.array([r[key] for r in rows]); return a.mean(), a.std()
print("\n=== mean +- std across seeds ===")
for k in ("r2tr", "r2co", "psi", "mov", "B_red_signed", "perm_z", "L_cert"):
    m, s = ms(k); print(f"  {k:14s} {m:+.4f} +- {s:.4f}")
br = np.array([r["B_red_signed"] for r in rows])
print(f"\nSIGNED B_red = {br.mean():+.4f} +- {br.std():.4f};  {(br <= 0).sum()}/{len(br)} seeds <= 0  "
      f"(field movement <= genuine data movement => NO manufactured artifact)")
print(f"Permutation z (field movement vs random partitions): mean {np.array([r['perm_z'] for r in rows]).mean():+.2f}  "
      f"(strongly negative => the field's cross-zoning movement under REAL administrative geography is far gentler "
      f"than random re-partitioning => the geography is coherent and the field tracks it).")
json.dump({"B_irr": B_irr, "data_perm": {"null_mean": data_nmean, "null_std": data_nstd, "z": data_z},
           "rows": rows, "nperm": NPERM}, open(paths.data("geo_m4_repro.json"), "w"))
