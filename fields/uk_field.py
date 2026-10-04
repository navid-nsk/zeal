"""Field tier on the Greater Manchester ladder with the Output Areas as held-out truth.

The field is trained on LSOA and MSOA means only (OA values never enter), then evaluated against the OA truth:
  R^2 at LSOA (fit) and at OA (test); within-LSOA movement of the field vs the truth;
  certified value brackets over windows of half-width r (CROWN with the location as input) and their COVERAGE of the
  true means of random contiguous unions of Output Areas inside the window (cells that cut LSOAs); the same coverage
  for the model-free LSOA bracket (min/max of LSOA values inside the window), which quantifies how often finer
  re-zonings escape a bracket built from coarser data; and the precision of certified hot / not-hot decisions.
Output: data/uk_field_<outcome>.json (+ npz with the value maps). Usage: python uk_field.py [--outcome q4] [--seeds 3]
        [--sigma 20] [--m 256] [--budget 3000000] [--ncells 2000]
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np, pandas as pd, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
p = argparse.ArgumentParser()
p.add_argument("--outcome", default="q4"); p.add_argument("--seeds", type=int, default=3); p.add_argument("--iters", type=int, default=3000)
p.add_argument("--sigma", type=float, default=20.0); p.add_argument("--m", type=int, default=256); p.add_argument("--beta", type=float, default=0.5)
p.add_argument("--budget", type=int, default=3_000_000); p.add_argument("--eps", type=float, default=0.003); p.add_argument("--ncells", type=int, default=2000)
a = p.parse_args()
D = paths.GM_DIR; R = np.load(D + "gm_raster.npz"); u = pd.read_csv(D + "gm_units.csv"); adj = pd.read_csv(D + "gm_adjacency.csv")
oc, wc = a.outcome, {"q4": "w_q4", "bad": "w_bad"}[a.outcome]
N = R["mask"].shape[0]; km = float(R["m_per_px"][0]) / 1000

# ---- a raster in the layout of the Georgia pipeline: tract = LSOA, bg = OA, county = MSOA, kfr = LSOA means ----
tag = f"GM{oc}"; path = paths.raster(tag, N)
if not os.path.exists(path):
    g = np.linspace(-1, 1, N, dtype=np.float32); X, Y = np.meshgrid(g, g)
    np.savez_compressed(path, mask=R["mask"], X=X, Y=Y, tract_idx=R["lsoa_idx"], bg_idx=R["oa_idx"], county_idx=R["msoa_idx"],
                        kfr=np.nan_to_num(R[f"lsoa_{oc}"]), weight=R[f"weight_{oc}"], weight_is_density=np.array([1]))
_argv = sys.argv; sys.argv = _argv[:1]
import geo_m4_multistate as M                                        # parses sys.argv at import
sys.argv = _argv
from zeal_m1 import Field, dev
from osc_lib import certify_value_maps
s = M.S(tag, N=N)
truth = torch.tensor(np.nan_to_num(R[f"truth_{oc}"]).reshape(-1)[s.mask.reshape(-1)], device=dev)
T_oa = s.wmean(truth, s.inv_bg, s.Kbg)                                # OA truth per OA (exact)
wrms_pix = lambda x, y: float(((s.w * (x - y) ** 2).sum() / s.w.sum()).sqrt())          # population-weighted RMS over pixels
mov_truth = wrms_pix(T_oa, s.wmean(truth, s.inv_tr, s.Ktr))                 # wmean returns per-pixel broadcasts


def train(seed):
    torch.manual_seed(seed); net = Field(m=a.m, sigma=a.sigma, w=128, depth=3, act="tanh").to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    for it in range(a.iters):
        opt.zero_grad(); pred = net(s.coords)
        rec = ((s.w * (s.wmean(pred, s.inv_tr, s.Ktr) - s.tgt_tr) ** 2).sum() / s.w.sum()
               + (s.w * (s.wmean(pred, s.inv_co, s.Kco) - s.tgt_co) ** 2).sum() / s.w.sum())
        (rec + a.beta * s.smooth_full(net)).backward(); opt.step()
    return net


out = dict(config=vars(a), truth=dict(mov_oa_lsoa=mov_truth, n_oa=int(s.Kbg), n_lsoa=int(s.Ktr), n_msoa=int(s.Kco)), seeds=[])
nets = []
for seed in range(a.seeds):
    t0 = time.time(); net = train(seed); nets.append(net)
    with torch.no_grad():
        pred = net(s.coords); P_oa, P_tr, P_co = s.wmean(pred, s.inv_bg, s.Kbg), s.wmean(pred, s.inv_tr, s.Ktr), s.wmean(pred, s.inv_co, s.Kco)
        r2_fit = s.wr2(P_tr, s.tgt_tr); r2_test = float(1 - (s.w * (P_oa - T_oa) ** 2).sum() / (s.w * (T_oa - (s.w * T_oa).sum() / s.w.sum()) ** 2).sum())
        mov_field = wrms_pix(P_oa, P_tr)
    row = dict(seed=seed, r2_lsoa_fit=r2_fit, r2_oa_test=r2_test, mov_oa_lsoa_field=mov_field, seconds=time.time() - t0)
    out["seeds"].append(row)
    print(f"seed {seed}: R2 LSOA (fit) {r2_fit:.3f} | R2 OA (held-out truth) {r2_test:.3f} | within-LSOA movement field {mov_field:.4f} vs truth {mov_truth:.4f} ({time.time() - t0:.0f}s)", flush=True)

# ---- certified value maps for seed 0 and the coverage test ----
net = nets[0]; t0 = time.time()
LO, HI, info, _tiles = certify_value_maps(net, s.mask, eps=a.eps, k0=128, budget=a.budget, log=lambda *x: None)
LO, HI = np.asarray(LO), np.asarray(HI); print(f"value maps: {info} ({time.time() - t0:.0f}s)", flush=True)
truth_r = np.nan_to_num(R[f"truth_{oc}"]); lsoa_r = np.nan_to_num(R[f"lsoa_{oc}"]); w_r = R[f"weight_{oc}"].astype(float); oa_r = R["oa_idx"]; ls_r = R["lsoa_idx"]
mask = R["mask"]; ok = (u[wc].values > 0)
# hot-spot threshold from the LSOA data (what the analyst observes) and from the OA truth
def quint(vals, wts): o = np.argsort(vals); cw = np.cumsum(wts[o]) / wts.sum(); return float(vals[o][np.searchsorted(cw, 0.8)])
Vl = np.bincount(u["LSOA21CD_i"], weights=u[wc] * u[oc]) / np.maximum(np.bincount(u["LSOA21CD_i"], weights=u[wc]), 1e-9); Wl = np.bincount(u["LSOA21CD_i"], weights=u[wc])
tau_lsoa = quint(Vl, Wl); tau_oa = quint(u[oc].values[ok], u[wc].values[ok])
rng = np.random.default_rng(0)
nbrs = [[] for _ in range(len(u))]
for i, j in adj.values:
    nbrs[i].append(j); nbrs[j].append(i)
cen = u[["OA_i"]].copy(); ii, jj = np.nonzero(mask); cy = np.zeros(len(u)); cx = np.zeros(len(u)); npx = np.bincount(oa_r[mask], minlength=len(u))
np.add.at(cy, oa_r[mask], ii); np.add.at(cx, oa_r[mask], jj); cy /= np.maximum(npx, 1); cx /= np.maximum(npx, 1)
lo_i = np.full(len(u), N); hi_i = np.full(len(u), -1); lo_j = np.full(len(u), N); hi_j = np.full(len(u), -1)
np.minimum.at(lo_i, oa_r[mask], ii); np.maximum.at(hi_i, oa_r[mask], ii); np.minimum.at(lo_j, oa_r[mask], jj); np.maximum.at(hi_j, oa_r[mask], jj)
cover = []
for r in [9, 18, 37, 74]:
    rows = []
    for _ in range(a.ncells):
        seed_u = int(rng.integers(len(u))); ci, cj = int(round(cy[seed_u])), int(round(cx[seed_u]))
        inside = lambda k: (lo_i[k] >= ci - r) & (hi_i[k] <= ci + r) & (lo_j[k] >= cj - r) & (hi_j[k] <= cj + r)
        if not inside(seed_u) or npx[seed_u] == 0:
            continue
        cell = [seed_u]; frontier = set(k for k in nbrs[seed_u] if inside(k)); target = int(rng.integers(1, 12))
        while frontier and len(cell) < target:
            k = int(rng.choice(sorted(frontier))); cell.append(k); frontier |= {q for q in nbrs[k] if inside(q) and q not in cell}; frontier.discard(k)
        cell = np.array(cell); ws = u[wc].values[cell]
        if ws.sum() <= 0 or len(cell) < 2:
            continue
        t_mean = float((ws * u[oc].values[cell]).sum() / ws.sum())
        i0, i1, j0, j1 = max(ci - r, 0), min(ci + r + 1, N), max(cj - r, 0), min(cj + r + 1, N)
        win = mask[i0:i1, j0:j1]
        f_lo, f_hi = float(np.min(LO[i0:i1, j0:j1][win])), float(np.max(HI[i0:i1, j0:j1][win]))
        # model-free bracket from LSOA values of the LSOAs INSIDE the window
        ls_in = np.unique(ls_r[i0:i1, j0:j1][win]); li, hi_, lj, hj = [np.full(s.Ktr, N), np.full(s.Ktr, -1), np.full(s.Ktr, N), np.full(s.Ktr, -1)]
        np.minimum.at(li, ls_r[mask], ii); np.maximum.at(hi_, ls_r[mask], ii); np.minimum.at(lj, ls_r[mask], jj); np.maximum.at(hj, ls_r[mask], jj)
        ls_ins = [l for l in ls_in if li[l] >= i0 and hi_[l] <= i1 - 1 and lj[l] >= j0 and hj[l] <= j1 - 1]
        m_lo, m_hi = (float(Vl[ls_ins].min()), float(Vl[ls_ins].max())) if ls_ins else (np.nan, np.nan)
        rows.append(dict(n_oa=len(cell), truth=t_mean, f_lo=f_lo, f_hi=f_hi, m_lo=m_lo, m_hi=m_hi, cuts_lsoa=int(len(np.unique(u["LSOA21CD_i"].values[cell])) > 1)))
    df = pd.DataFrame(rows); mf = df.dropna(subset=["m_lo"])
    rec = dict(r_km=r * km, n=len(df), field_cover=float(((df.truth >= df.f_lo) & (df.truth <= df.f_hi)).mean()), field_width=float((df.f_hi - df.f_lo).median()),
               mf_n=len(mf), mf_cover=float(((mf.truth >= mf.m_lo) & (mf.truth <= mf.m_hi)).mean()) if len(mf) else None, mf_width=float((mf.m_hi - mf.m_lo).median()) if len(mf) else None,
               cells_cutting_lsoa=float(df.cuts_lsoa.mean()),
               hot_cert_precision=float((df.truth[df.f_lo > tau_lsoa] > tau_oa).mean()) if (df.f_lo > tau_lsoa).any() else None, n_hot_cert=int((df.f_lo > tau_lsoa).sum()),
               cold_cert_precision=float((df.truth[df.f_hi < tau_lsoa] < tau_oa).mean()) if (df.f_hi < tau_lsoa).any() else None, n_cold_cert=int((df.f_hi < tau_lsoa).sum()))
    cover.append(rec)
    print(f"r {rec['r_km']:4.1f} km | {rec['n']} random OA-unions ({100 * rec['cells_cutting_lsoa']:.0f}% cut an LSOA) | field bracket covers truth {100 * rec['field_cover']:5.1f}% (median width {rec['field_width']:.3f}) | "
          f"LSOA bracket covers truth {100 * (rec['mf_cover'] or 0):5.1f}% of {rec['mf_n']} (width {rec['mf_width'] if rec['mf_width'] is not None else float('nan'):.3f}) | certified hot precision {rec['hot_cert_precision']} (n={rec['n_hot_cert']}) "
          f"not-hot {rec['cold_cert_precision']} (n={rec['n_cold_cert']})", flush=True)
out["coverage"] = cover; out["tau_lsoa"] = tau_lsoa; out["tau_oa"] = tau_oa; out["value_maps"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in info.items()}
json.dump(out, open(paths.field_file(f"uk_field_{oc}.json"), "w", encoding="utf-8"), indent=1)
np.savez_compressed(paths.field_file(f"uk_field_{oc}.npz"), LO=LO, HI=HI)
torch.save(nets[0].state_dict(), paths.field_file(f"uk_field_{oc}_net.pt"))
print("saved", f"data/uk_field_{oc}.json")
