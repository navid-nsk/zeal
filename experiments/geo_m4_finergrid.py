"""
BG-rung resolution convergence: BG under-resolution at 512 (32% of BGs <3px).
Re-run the GA decomposition at 512 vs 1024 and show PSI / signed B_red / R2 are resolution-STABLE
(the cross-zoning findings are not grid artifacts). Usage: python geo_m4_finergrid.py [NSEED]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import sys, json, numpy as np
import geo_m4_multistate as M

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
print(f"finer-grid convergence (GA): 512 vs 1024, {NSEED} seeds each\n")
out = {}
for N in (512, 1024):
    print(f"--- GA @ {N}x{N} ---")
    out[str(N)] = M.run_state("13", nseed=NSEED, N=N)
    print()
json.dump(out, open(paths.data("geo_m4_finergrid.json"), "w"))

a = lambda N, k: np.array([r[k] for r in out[str(N)]["rows"]])
print("=== convergence 512 -> 1024 (means) ===")
for k in ("r2tr", "psi", "B_red_signed"):
    print(f"  {k:14s} 512={a(512, k).mean():+.4f}  1024={a(1024, k).mean():+.4f}  "
          f"Δ={a(1024, k).mean()-a(512, k).mean():+.4f}")
print("Resolution-stable => the cross-zoning verdict is not a grid artifact.")
