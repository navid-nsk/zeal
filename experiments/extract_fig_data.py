"""Gather all per-seed/fold/state arrays + the raw permutation-null distribution into fig_data.json
   for building Fig 7/8/ED-2. No training needed (perm null is data-only)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import json, numpy as np, torch
import geo_m4_multistate as M
from zeal_m1 import dev

def L(p): return json.load(open(paths.data(p)))
out = {}
out["repro"] = L("geo_m4_repro.json")          # 10-seed GA: rows w/ r2tr,r2co,psi,mov,B_red_signed,perm_z,L_cert
out["multistate"] = L("geo_m4_multistate.json")
out["baselines"] = L("geo_m4_baselines.json")
out["spatialcv"] = L("geo_m4_spatialcv.json")
out["finergrid"] = L("geo_m4_finergrid.json")
out["soundness"] = L("zeal_soundness_curve.json")
out["binding"] = L("geo_m4_binding.json")      # discrimination spectrum (PSI vs L_cert)

# raw permutation-null distribution for the DATA movement (Fig 8e histogram) — GA counties
s = M.S("13")
Dt = s.Dt; tc = s.tc; Wt = s.Wt
def wrms(a, b): return float((Wt * (a - b) ** 2).sum().div(Wt.sum()).sqrt())
def county_of(valt, county):
    Wc = torch.zeros(s.Kco, device=dev).index_add_(0, county, Wt)
    Vc = torch.zeros(s.Kco, device=dev).index_add_(0, county, valt * Wt) / Wc.clamp_min(1e-9)
    return Vc[county]
real = wrms(Dt, county_of(Dt, tc))
g = torch.Generator(device=dev); null = []
for k in range(1000):
    g.manual_seed(1000 + k)
    null.append(wrms(Dt, county_of(Dt, tc[torch.randperm(s.Ktr, generator=g, device=dev)])))
out["perm_null"] = {"real": real, "null": null, "n": len(null),
                    "z": (real - float(np.mean(null))) / (float(np.std(null)) + 1e-12)}
json.dump(out, open(paths.data("fig_data.json"), "w"))
print(f"real movement {real:.4f}; null {np.mean(null):.4f}±{np.std(null):.4f}; "
      f"z={out['perm_null']['z']:.1f}; {sum(n<=real for n in null)}/{len(null)} nulls <= real")
print("saved fig_data.json")
