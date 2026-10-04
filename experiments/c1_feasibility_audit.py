"""c1_feasibility_audit.py -- feasibility audit of the whole-unit class (Definition S2; results exp2/C1): Georgia tracts, fat focal
class C_{r,p} with the WINDOW defined at pixel level (all raster pixels within r km of the anchor tract's population centroid) and whole
observed units eligible only if WHOLLY contained in the window. Per location and radius we store: anchor contained (all its pixels in the
window), window population mu(W) (pixel-level), eligible-unit mass (sum over wholly contained tracts), the ratio eligible/mu(W), and the
fat-class feasibility = anchor contained AND eligible mass >= p * mu(W) AND >= 2 eligible units (non-trivial; singleton completion by
Definition S2). Compared with the provisional diameter shortcut of c1_datatier_redo.py. Usage: python c1_feasibility_audit.py --out <dir>."""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--radii", default="3.8,7.5,15,30"); p.add_argument("--pfat", type=float, default=0.5); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool)
tr = ras["tract_idx"][mask]; w = ras["weight"][mask].astype(float); X = ras["X"][mask].astype(float); Y = ras["Y"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr); KM_PER_UNIT = 0.94 * (mask.shape[0] - 1) / 2
W = np.bincount(inv, weights=w, minlength=K); cx = np.bincount(inv, weights=w * X, minlength=K) / np.maximum(W, 1e-12); cy = np.bincount(inv, weights=w * Y, minlength=K) / np.maximum(W, 1e-12)
pop = W / W.sum(); diam = np.sqrt(np.bincount(inv, minlength=K)) * 0.94
OUT = {}
for r_km in [float(x) for x in a.radii.split(",")]:
    r = r_km / KM_PER_UNIT; contained = np.zeros(K, bool); muW = np.zeros(K); elig = np.zeros(K); n_elig = np.zeros(K, int)
    for x in range(K):
        inwin = (X - cx[x]) ** 2 + (Y - cy[x]) ** 2 <= r * r                      # pixels in the window
        muW[x] = w[inwin].sum(); cnt_in = np.bincount(inv[inwin], minlength=K); cnt_all = np.bincount(inv, minlength=K)
        whole = (cnt_in == cnt_all) & (cnt_all > 0); contained[x] = whole[x]; elig[x] = W[whole].sum(); n_elig[x] = int(whole.sum())
    feas = contained & (elig >= a.pfat * muW) & (n_elig >= 2); shortcut = (diam <= 2 * r_km) & (W > 0)
    OUT[str(r_km)] = dict(pop_share_contained=float(pop[contained].sum()), pop_share_feasible=float(pop[feas].sum()), pop_share_shortcut=float(pop[shortcut].sum()),
                          eligible_over_window_mass_median=float(np.median(elig[contained] / np.maximum(muW[contained], 1e-12))), n_eligible_median=float(np.median(n_elig[contained])),
                          shortcut_but_infeasible_pop_share=float(pop[shortcut & ~feas].sum()), feasible_but_not_shortcut_pop_share=float(pop[feas & ~shortcut].sum()))
    print(f"r={r_km}: contained {OUT[str(r_km)]['pop_share_contained']:.3f} feasible {OUT[str(r_km)]['pop_share_feasible']:.3f} (shortcut said {OUT[str(r_km)]['pop_share_shortcut']:.3f}); eligible/window mass median {OUT[str(r_km)]['eligible_over_window_mass_median']:.3f}; eligible units median {OUT[str(r_km)]['n_eligible_median']:.0f}; shortcut-but-infeasible {OUT[str(r_km)]['shortcut_but_infeasible_pop_share']:.3f}", flush=True)
    np.savez_compressed(os.path.join(a.out, f"feasibility_r{r_km:g}.npz"), contained=contained, muW=muW, eligible=elig, n_eligible=n_elig, feasible=feas)
json.dump(OUT, open(os.path.join(a.out, "c1_feasibility_audit.json"), "w"), indent=1)
