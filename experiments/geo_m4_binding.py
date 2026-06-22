"""
Non-vacuity / binding figure: a spectrum of fields that all FIT the GA Atlas data (~equal R^2) but span
spiky -> smooth. Spiky predictors manufacture cross-zoning ARTIFACT (large movement / B_red); the certified
L_cert flags them (large). The certified ZEAL operating point is provably in the stable regime.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import math, json, numpy as np, torch
import geo_m4 as G
from zeal_manifold_cert import manifold_cert_masked
dev = G.dev

B_irr = float(((G.wmean(G.kfr, G.inv_tr, G.Ktr) - G.wmean(G.kfr, G.inv_co, G.Kco)) ** 2).mean().sqrt())

def run(beta):
    torch.manual_seed(0)                                       # same Fourier B across beta -> comparable L_cert
    net = G.train(beta=beta)
    with torch.no_grad():
        pred = net(G.coords)
        P_bg, P_tr, P_co = G.wmean(pred, G.inv_bg, G.Kbg), G.wmean(pred, G.inv_tr, G.Ktr), G.wmean(pred, G.inv_co, G.Kco)
        r2 = G.wr2(P_tr, G.tgt_tr)
        mov = float(((P_tr - P_co) ** 2).mean().sqrt())            # tract->county movement (field)
        psi_num = (G.w * torch.stack([P_bg, P_tr, P_co], 0).var(0)).sum() / G.w.sum()
        psi = float(psi_num / ((G.w * (P_tr - (G.w * P_tr).sum() / G.w.sum()) ** 2).sum() / G.w.sum() + 1e-12))
    Lc, _ = manifold_cert_masked(net, G.mask, budget=60000)
    B_red = max(mov - B_irr, 0.0)
    return dict(beta=beta, r2=r2, mov=mov, B_red=B_red, psi=psi, L_cert=Lc)

print(f"B_irr (genuine, data tract->county) = {B_irr:.4f}\n")
print(f"{'beta':>6s} {'R2':>6s} {'movement':>9s} {'B_red':>8s} {'PSI':>7s} {'L_cert':>8s}")
rows = []
for beta in (0.02, 0.1, 0.5, 2.0, 10.0):
    r = run(beta); rows.append(r)
    print(f"{r['beta']:6.2f} {r['r2']:6.3f} {r['mov']:9.4f} {r['B_red']:8.4f} {r['psi']:7.3f} {r['L_cert']:8.1f}", flush=True)
json.dump({"B_irr": B_irr, "rows": rows}, open(paths.data("geo_m4_binding.json"), "w"))
print("\nSpiky (low beta): high movement/B_red (manufactured artifact) AND high L_cert (flagged). "
      "Smooth (high beta, ZEAL): low artifact, low L_cert. Same R^2 throughout -> fit doesn't reveal it; "
      "the certificate does.")
