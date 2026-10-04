"""b2_zoning_vs_scale.py -- data tier (Definitions S1-S2, Supplementary Theorems S1, S18; results exp1/B2): zoning effect vs scale effect
on the Greater Manchester ladder with exact fine truth (8,966 OAs, two outcomes, population weights).

Scale effect: nested movement OA -> LSOA (1,702 cells), OA -> MSOA, LSOA -> MSOA (exact).
Zoning effect, fixed K = #LSOA (same number of cells):
  (a) random population-balanced contiguous partitions of the OAs into K cells (region growing from random seeds with
      balance control; n_part partitions): movement between the alternative zoning and the LSOA zoning (both at OA
      resolution: ||P_Z t - P_LSOA t||), movement OA -> Z, and the per-OA local change distribution; also the share of OAs
      whose hot/not-hot status (top 20 % by population) flips;
  (b) boundary displacement C_delta: starting from the LSOA zoning, reassign boundary OAs to a neighbouring LSOA, with
      delta = number of reassignment sweeps (1 sweep ~ one OA layer ~ 0.25-0.4 km); movement vs delta; the tube identity
      (Theorem S18 (i)) is exact per cell and checked; the delta/h scaling hypothesis is tested as the ratio
      movement(delta) / scale movement against delta / (LSOA diameter).
All data tier: exact arithmetic on OA truth; no certificate. Usage: python b2_zoning_vs_scale.py --out <dir> [--n_part 300] [--seed 9]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, collections, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--n_part", type=int, default=300); p.add_argument("--seed", type=int, default=9); p.add_argument("--balance", type=float, default=0.5); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
D = paths.GM_DIR; u = pd.read_csv(D + "gm_units.csv"); adj = pd.read_csv(D + "gm_adjacency.csv").values; n = len(u)
E = adj[adj[:, 0] != adj[:, 1]]; nbrs = [[] for _ in range(n)]
for x_, y_ in E:
    nbrs[int(x_)].append(int(y_)); nbrs[int(y_)].append(int(x_))
A_ls = u["LSOA21CD_i"].values; A_ms = u["MSOA21CD_i"].values; K = A_ls.max() + 1; KM = A_ms.max() + 1
import geopandas as gpd
g = gpd.read_file(D + "oa_boundaries.geojson").set_index("OA21CD").loc[u["OA21CD"].values]; cx, cy = g.geometry.centroid.x.values / 1000, g.geometry.centroid.y.values / 1000
wm = lambda v, w, gidx, Kk: np.bincount(gidx, weights=w * v, minlength=Kk) / np.maximum(np.bincount(gidx, weights=w, minlength=Kk), 1e-12)
def mov(t, w, g1, g2, K1, K2):
    return float(np.sqrt((w * (wm(t, w, g1, K1)[g1] - wm(t, w, g2, K2)[g2]) ** 2).sum() / w.sum()))


def grow_partition(K_target, w, balance):
    """region growing: K seeds, repeatedly attach an unassigned neighbour to the smallest adjacent region; target mass W/K with tolerance."""
    lab = -np.ones(n, int); seeds = rng.choice(n, K_target, replace=False); lab[seeds] = np.arange(K_target); mass = w[seeds].astype(float).copy(); tgt = w.sum() / K_target
    frontier = collections.defaultdict(set)
    for s_ in seeds:
        for v in nbrs[s_]:
            if lab[v] < 0:
                frontier[lab[s_]].add(v)
    active = set(range(K_target)); unassigned = n - K_target
    while unassigned > 0 and active:
        # pick the region with the smallest mass that still has a frontier
        r = min(active, key=lambda k_: mass[k_])
        cand = [v for v in frontier[r] if lab[v] < 0]
        if not cand:
            active.discard(r); continue
        v = cand[int(rng.integers(len(cand)))]; lab[v] = r; mass[r] += w[v]; unassigned -= 1; frontier[r].discard(v)
        for z in nbrs[v]:
            if lab[z] < 0:
                frontier[r].add(z)
        if mass[r] > tgt * (1 + balance):
            active.discard(r)
    # leftovers (enclosed by full regions): attach to the lightest adjacent region
    for v in np.where(lab < 0)[0]:
        adj_regs = [lab[z] for z in nbrs[v] if lab[z] >= 0]
        if adj_regs:
            r = min(adj_regs, key=lambda k_: mass[k_]); lab[v] = r; mass[r] += w[v]
    for v in np.where(lab < 0)[0]:          # isolated OAs: own cell (counted)
        lab[v] = lab.max() + 1
    return lab


def displace(lab0, w, q, sweeps=1):
    """boundary displacement: in each sweep, every boundary OA (has a neighbour in another cell) is moved with probability q to a random
    neighbouring cell, provided its own cell keeps >= 2 OAs. Returns the new labels and the set of moved OAs."""
    lab = lab0.copy(); moved = set(); counts = np.bincount(lab, minlength=lab.max() + 1)
    for _ in range(sweeps):
        order = rng.permutation(n)
        for v in order:
            other = [lab[z] for z in nbrs[v] if lab[z] != lab[v]]
            if other and rng.random() < q and counts[lab[v]] >= 2:
                counts[lab[v]] -= 1; lab[v] = other[int(rng.integers(len(other)))]; counts[lab[v]] += 1; moved.add(int(v))
    return lab, moved


OUT = {}
for oc, wc in (("q4", "w_q4"), ("bad", "w_bad")):
    t0 = time.time(); w = u[wc].values.astype(float); t = np.where(w > 0, u[oc].values.astype(float), 0.0); w = np.where(w > 0, w, 0.0)
    ident = np.arange(n); res = dict(scale=dict(oa_lsoa=mov(t, w, ident, A_ls, n, K), oa_msoa=mov(t, w, ident, A_ms, n, KM), lsoa_msoa=mov(t, w, A_ls, A_ms, K, KM)))
    # hot status at OA (top 20 % by population) and at LSOA
    o = np.argsort(t); cw = np.cumsum(w[o]) / w.sum(); tau = t[o][np.searchsorted(cw, 0.8)]; hot_oa = t >= tau; res["tau"] = float(tau)
    y_ls = wm(t, w, A_ls, K)[A_ls]; hot_ls = y_ls >= tau
    # (a) same-K balanced contiguous partitions
    movs_z, movs_oa, flips, local = [], [], [], []
    lsoa_diam = np.median([np.sqrt(((cx[A_ls == k_] - cx[A_ls == k_].mean()) ** 2 + (cy[A_ls == k_] - cy[A_ls == k_].mean()) ** 2).max()) * 2 for k_ in range(K)])
    for i in range(a.n_part):
        lab = grow_partition(K, w, a.balance); Kz = lab.max() + 1; yz = wm(t, w, lab, Kz)[lab]
        movs_z.append(float(np.sqrt((w * (yz - y_ls) ** 2).sum() / w.sum()))); movs_oa.append(float(np.sqrt((w * (yz - t) ** 2).sum() / w.sum())))
        flips.append(float((w * ((yz >= tau) != hot_ls)).sum() / w.sum())); local.append(np.quantile(np.abs(yz - y_ls), [0.5, 0.9, 0.99]).tolist())
        if i % 50 == 0:
            print(f"  {oc} partition {i}: cells {Kz} mov(Z,LSOA) {movs_z[-1]:.4f} mov(OA,Z) {movs_oa[-1]:.4f} flips {flips[-1]:.3f} {time.time()-t0:.0f}s", flush=True)
    res["same_K_balanced"] = dict(n=a.n_part, balance=a.balance, mov_Z_vs_LSOA=dict(median=float(np.median(movs_z)), q05=float(np.quantile(movs_z, 0.05)), q95=float(np.quantile(movs_z, 0.95)), max=float(np.max(movs_z))),
                                  mov_OA_vs_Z=dict(median=float(np.median(movs_oa)), q95=float(np.quantile(movs_oa, 0.95))), hot_flip_share=dict(median=float(np.median(flips)), q95=float(np.quantile(flips, 0.95))),
                                  local_change_abs_quantiles_median=np.median(np.array(local), axis=0).tolist(), ratio_zoning_over_scale=float(np.median(movs_z) / res["scale"]["oa_lsoa"]), lsoa_diameter_km_median=float(lsoa_diam))
    # (b) boundary displacement from the LSOA zoning
    disp = []
    for q, sweeps in ((0.02, 1), (0.05, 1), (0.1, 1), (0.2, 1), (0.5, 1), (0.5, 2), (0.5, 4)):
        mv, fl, nm, tube_ok, dd = [], [], [], 0, []
        for rep in range(10):
            lab, moved = displace(A_ls, w, q, sweeps); Kz = lab.max() + 1; yz = wm(t, w, lab, Kz)[lab]
            # displacement distance: for moved OAs, distance from the OA centroid to the centroid of its original LSOA boundary neighbour set ~ OA diameter; we record the moved share and the mean moved-OA diameter
            dd.append(float(np.mean(np.sqrt(u["area_km2"].values[list(moved)]))) if moved else 0.0)
            mv.append(float(np.sqrt((w * (yz - y_ls) ** 2).sum() / w.sum()))); fl.append(float((w * ((yz >= tau) != hot_ls)).sum() / w.sum())); nm.append(len(moved) / n)
            # tube identity check on one changed cell: <f>_{a'} - <f>_a = [int_A (f - <f>_a) - int_D (f - <f>_a)] / mu(a')
            k_ = int(lab[next(iter(moved))]) if moved else 0; a_old = A_ls == k_; a_new = lab == k_
            if a_old.any() and a_new.any():
                ma = (w * a_old * t).sum() / (w * a_old).sum(); mn = (w * a_new * t).sum() / (w * a_new).sum()
                ident_ = ((w * (a_new & ~a_old) * (t - ma)).sum() - (w * (a_old & ~a_new) * (t - ma)).sum()) / (w * a_new).sum(); tube_ok += int(abs((mn - ma) - ident_) < 1e-10)
        # delta: the population share of moved OAs times one OA layer (in km: moved-OA diameter); delta/h with h = median LSOA diameter
        delta_km = float(np.mean(nm)) * sweeps * float(np.mean(dd)) if np.mean(dd) > 0 else 0.0
        disp.append(dict(q=q, sweeps=sweeps, moved_share=float(np.mean(nm)), moved_oa_diam_km=float(np.mean(dd)), delta_km=delta_km, delta_over_h=delta_km / lsoa_diam, mov_median=float(np.median(mv)), mov_over_scale=float(np.median(mv) / res["scale"]["oa_lsoa"]), hot_flip_share=float(np.median(fl)), tube_identity_ok=tube_ok))
        print(f"  {oc} displacement q={q} sweeps={sweeps}: moved {np.mean(nm):.3f} delta/h {delta_km/lsoa_diam:.3f} mov {np.median(mv):.4f} (= {np.median(mv)/res['scale']['oa_lsoa']:.3f} x scale) flips {np.median(fl):.3f} tube ok {tube_ok}/10", flush=True)
    res["displacement"] = disp
    # delta/h hypothesis: log-log slope of mov_over_scale vs delta_over_h
    xs = np.log([d_["moved_share"] for d_ in disp if d_["moved_share"] > 0]); ys = np.log([d_["mov_over_scale"] for d_ in disp if d_["moved_share"] > 0]); res["loglog_slope_mov_vs_moved_share"] = float(np.polyfit(xs, ys, 1)[0])
    res["seconds"] = time.time() - t0; OUT[oc] = res
    print(oc, json.dumps({k: v for k, v in res.items() if k != "displacement"}, indent=1, default=float), flush=True)
    json.dump(OUT, open(os.path.join(a.out, "b2_zoning_vs_scale.json"), "w"), indent=1, default=float)
