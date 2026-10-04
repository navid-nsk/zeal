"""OSCILLATION certificate: certified local stability at every zoning scale, from VALUE bounds.

For any zoning Z whose cell containing x lies inside the window B(x, r), the aggregated value f_Z(x)
(count-weighted or not) is a mean of lam over a subset of B(x, r), hence lies in
[inf_{B(x,r)} lam, sup_{B(x,r)} lam]. Therefore, for ANY two zonings at scale <= r,
    | f_Z(x) - f_Z'(x) |  <=  osc_r(x) := sup_{B(x,r)} lam - inf_{B(x,r)} lam .
osc_r is certified from per-tile CROWN VALUE bounds of the field through the analytic sin/cos encoder
box (the same manifold-aware pipeline as the Lipschitz certifier, reading function bounds instead of
Jacobian bounds); value relaxations tighten quadratically with tile width, so tiles are refined until
every in-state tile's certified width is below a tolerance. From the certified LO/HI maps follow, at
every scale r: the max-local-change certificate, the certified hot-spot map (certified-in / -out /
undetermined) and its stability scale per location, pairwise rank certificates, and a population-RMS
scale envelope that bounds the global movement under every zoning at scale <= r.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import sys, json, time, numpy as np, torch, geopandas as gpd
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, minimum_filter
import geo_m4_multistate as M
from zeal_m1 import dev

EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.003     # target certified width per tile (field units)
K0 = int(sys.argv[2]) if len(sys.argv) > 2 else 128
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000
D = os.path.join(paths.data("fields"), "")
t0 = time.time()

import os
from zeal_m1 import Field
TAG = os.environ.get("ZEAL_TAG", ""); SUF = ("_" + TAG) if TAG else ""; CKPT = os.environ.get("ZEAL_CKPT", "")
s = M.S("13"); N = s.N; h = 2.0 / (N - 1); mask = s.mask
if CKPT:                                                        # certify a saved field (e.g. the noise-aware fit)
    net = Field(m=128, sigma=6.0, w=128, depth=3, act="tanh").to(dev); net.load_state_dict(torch.load(CKPT)); net.eval()
    print(f"loaded field {CKPT}", flush=True)
else:
    torch.manual_seed(0); net = s.train(0.5)
    torch.save(net.state_dict(), D + "local_cert_net.pt")
with torch.no_grad():
    pred = net(s.coords); lam_grid = net(s.Xf).view(N, N)
print(f"[{time.time()-t0:.0f}s] field trained; in-state range {pred.min():.4f}..{pred.max():.4f}", flush=True)
tb_km = gpd.read_file(paths.GEORGIA_TRACTS).total_bounds
km_per_unit = (tb_km[2] - tb_km[0]) / 2 / 1000.0; km_per_px = km_per_unit * h
print(f"scale: 1 coordinate unit = {km_per_unit:.1f} km, 1 pixel = {km_per_px:.2f} km", flush=True)

# ---- certified VALUE bounds (osc_lib): 2-D-input CROWN by default, refined until width <= EPS, rasterized
MODE = sys.argv[4] if len(sys.argv) > 4 else "2d"; METHOD = sys.argv[5] if len(sys.argv) > 5 else "CROWN"
from osc_lib import certify_value_maps
LO, HI, info, (XLn, XHn, lon, hin) = certify_value_maps(net, mask, eps=EPS, k0=K0, budget=BUDGET, mode=MODE, method=METHOD,
                                                       log=lambda msg: print(msg, flush=True))
with torch.no_grad():                                           # tightness diagnostic on the final tiles
    XL, XH, lo, hi = [torch.tensor(a, device=dev) for a in (XLn, XHn, lon, hin)]
    pts = torch.stack([XL, XH, (XL + XH) / 2, torch.stack([XL[:, 0], XH[:, 1]], 1), torch.stack([XH[:, 0], XL[:, 1]], 1)], 1)
    v = torch.cat([net(pts[i:i + 200000].reshape(-1, 2)).reshape(-1, 5) for i in range(0, XL.shape[0], 200000)])
    samp_w = (v.max(1).values - v.min(1).values); wid = hi - lo
    inside = bool(((v.min(1).values >= lo - 1e-6) & (v.max(1).values <= hi + 1e-6)).all())
print(f"[{time.time()-t0:.0f}s] {info} | certified width median {wid.median():.5f} max {wid.max():.5f}; "
      f"sampled-range median {samp_w.median():.5f} (ratio {float(wid.median()/samp_w.median()):.1f}x); samples inside bounds: {inside}", flush=True)
lam_np = lam_grid.cpu().numpy()
ok = mask & np.isfinite(LO) & np.isfinite(HI)
assert ok[mask].all(), "every in-state pixel must be covered"
assert (LO[mask] <= lam_np[mask] + 1e-6).all() and (HI[mask] >= lam_np[mask] - 1e-6).all(), "pixel maps must bracket the field"
print(f"[{time.time()-t0:.0f}s] pixel maps: HI-LO median {np.median((HI-LO)[mask]):.4f}", flush=True)

# ---- tract / county values (count-weighted), nested check
inv_tr, inv_co, tc = s.inv_tr.cpu().numpy(), s.inv_co.cpu().numpy(), s.tc.cpu().numpy()
Wt = s.Wt.cpu().numpy(); w = s.w.cpu().numpy()
Vt = (torch.zeros(s.Ktr, device=dev).index_add_(0, s.inv_tr, pred * s.w) / s.Wt.clamp_min(1e-9)).cpu().numpy()
Ct = s.county_of(torch.tensor(Vt, device=dev), s.tc).cpu().numpy()
cnt = Wt > 0                                                     # zero-count tracts have undefined means: excluded
mov_loc = np.abs(Vt - Ct); rng = float(Vt[cnt].max() - Vt[cnt].min())
LOm, HIm = LO[mask], HI[mask]
co_lo = np.full(s.Kco, np.inf); co_hi = np.full(s.Kco, -np.inf)
np.minimum.at(co_lo, inv_co, LOm); np.maximum.at(co_hi, inv_co, HIm)
e_co = (co_hi - co_lo)[tc]                                       # certified oscillation over the county
print(f"\n=== NESTED tract->county: certified county oscillation vs observed local movement ===")
ml, ec = mov_loc[cnt], e_co[cnt]
print(f"SOUND: {bool(np.all(ec >= ml - 1e-9))} | observed median {np.median(ml):.4f} max {ml.max():.4f} | "
      f"osc_county median {np.median(ec):.4f} max {ec.max():.4f} | ratio median {np.median(ec/np.maximum(ml,1e-9)):.1f}x "
      f"| max-local-change certificate {ec.max():.4f} vs observed max {ml.max():.4f} ({ec.max()/ml.max():.1f}x)")

# ---- multi-scale oscillation maps: every zoning whose cells have half-width <= r pixels
LOf = np.where(mask, LO, np.inf); HIf = np.where(mask, HI, -np.inf)
o = np.argsort(Vt); cw = np.cumsum(Wt[o]) / Wt.sum(); tau = float(Vt[o][np.searchsorted(cw, 0.8)])
hot_pix = (Vt > tau)[inv_tr]
wpix = w / w.sum()
radii = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
scale_rows = []
print(f"\n=== SCALE CERTIFICATES (tau = top-quintile {tau:.4f}; pixel = {km_per_px:.2f} km) ===")
print(f"{'r px':>5s} {'r km':>6s} {'osc med':>8s} {'osc max':>8s} {'RMS env':>8s} {'hot cert%':>9s} {'cold cert%':>10s} {'undet%':>7s}")
for r in radii:
    sz = 2 * r + 1
    lo_r = minimum_filter(LOf, size=sz, mode="constant", cval=np.inf)[mask]
    hi_r = maximum_filter(HIf, size=sz, mode="constant", cval=-np.inf)[mask]
    osc = hi_r - lo_r
    cin = lo_r > tau; cout = hi_r < tau
    hot_c = wpix[hot_pix & cin].sum() / max(wpix[hot_pix].sum(), 1e-12)
    cold_c = wpix[~hot_pix & cout].sum() / max(wpix[~hot_pix].sum(), 1e-12)
    und = wpix[~(cin | cout)].sum()
    rms_env = float(np.sqrt((wpix * osc ** 2).sum()))
    scale_rows.append(dict(r_px=r, r_km=r * km_per_px, osc_median=float(np.median(osc)), osc_max=float(osc.max()),
                           rms_env=rms_env, hot_cert=float(hot_c), cold_cert=float(cold_c), undetermined=float(und)))
    print(f"{r:5d} {r*km_per_px:6.1f} {np.median(osc):8.4f} {osc.max():8.4f} {rms_env:8.4f} {100*hot_c:8.0f}% {100*cold_c:9.0f}% {100*und:6.0f}%")
print(f"reference: observed tract->county RMS movement = {np.sqrt((Wt*(Vt-Ct)**2).sum()/Wt.sum()):.4f}; field range {rng:.3f}; "
      f"Lipschitz envelope L_cert*d ~ 34.5")

# ---- TIER 1 (model-free): the same scale certificate from the OBSERVED tract values. For every coarsening of the
#      tract zoning whose merged cells fit in the window, the merged value is a convex combination of observed values,
#      so hot-spot status is certified by the oscillation of the DATA in the window. No model, no assumption.
Dt = s.Dt.cpu().numpy(); Dr = np.full((N, N), np.nan); Dr[mask] = np.where(cnt, Dt, np.nan)[inv_tr]
Dlo = np.where(mask & np.isfinite(Dr), Dr, np.inf); Dhi = np.where(mask & np.isfinite(Dr), Dr, -np.inf)
oD = np.argsort(Dt); cwD = np.cumsum(Wt[oD]) / Wt.sum(); tauD = float(Dt[oD][np.searchsorted(cwD, 0.8)])
hotD_pix = ((Dt > tauD) & cnt)[inv_tr]
t1_rows = []
print(f"\n=== TIER-1 MODEL-FREE SCALE CERTIFICATES on the OBSERVED data (tau = top-quintile {tauD:.4f}) ===")
print(f"{'r px':>5s} {'r km':>6s} {'osc med':>8s} {'osc max':>8s} {'RMS env':>8s} {'hot cert%':>9s} {'cold cert%':>10s} {'undet%':>7s}")
for r in radii:
    sz = 2 * r + 1
    lo_r = minimum_filter(Dlo, size=sz, mode="constant", cval=np.inf)[mask]; hi_r = maximum_filter(Dhi, size=sz, mode="constant", cval=-np.inf)[mask]
    osc = hi_r - lo_r; cin = lo_r > tauD; cout = hi_r < tauD
    hot_c = wpix[hotD_pix & cin].sum() / max(wpix[hotD_pix].sum(), 1e-12); cold_c = wpix[~hotD_pix & cout].sum() / max(wpix[~hotD_pix].sum(), 1e-12)
    und = wpix[~(cin | cout)].sum(); rms_env = float(np.sqrt((wpix * osc ** 2).sum()))
    t1_rows.append(dict(r_px=r, r_km=r * km_per_px, osc_median=float(np.median(osc)), osc_max=float(osc.max()), rms_env=rms_env,
                        hot_cert=float(hot_c), cold_cert=float(cold_c), undetermined=float(und)))
    print(f"{r:5d} {r*km_per_px:6.1f} {np.median(osc):8.4f} {osc.max():8.4f} {rms_env:8.4f} {100*hot_c:8.0f}% {100*cold_c:9.0f}% {100*und:6.0f}%")
DCt = s.county_of(torch.tensor(Dt, device=dev), s.tc).cpu().numpy()
print(f"reference: observed DATA tract->county RMS movement (B_irr) = {np.sqrt((Wt*(Dt-DCt)**2).sum()/Wt.sum()):.4f}")

# ---- rank certificates at the county scale (r = 16 px) between tracts: certified if LO_r(x) > HI_r(y)
r = 16; sz = 2 * r + 1
lo_r = minimum_filter(LOf, size=sz, mode="constant", cval=np.inf); hi_r = maximum_filter(HIf, size=sz, mode="constant", cval=-np.inf)
tr_lo = np.full(s.Ktr, np.inf); tr_hi = np.full(s.Ktr, -np.inf)
np.minimum.at(tr_lo, inv_tr, lo_r[mask]); np.maximum.at(tr_hi, inv_tr, hi_r[mask])
cert_above = (tr_lo[:, None] > tr_hi[None, :]) & cnt[:, None] & cnt[None, :]
Nrank = s.Ktr - 1 - cert_above.sum(1) - cert_above.sum(0)
print(f"\nrank certificates at r = {r} px ({r*km_per_px:.0f} km): certified pairs {cert_above.sum()} of {s.Ktr*(s.Ktr-1)//2} "
      f"({100*cert_above.sum()/(s.Ktr*(s.Ktr-1)//2):.0f}%); rank half-width median {np.median(Nrank):.0f} of {s.Ktr}")

np.savez_compressed(D + f"osc_cert{SUF}.npz", LO=LO.astype(np.float32), HI=HI.astype(np.float32), lam=lam_np.astype(np.float32),
                    Vt=Vt, Ct=Ct, Wt=Wt, tc=tc, e_co=e_co, mov_loc=mov_loc, tau=tau, km_per_px=km_per_px)
json.dump(dict(eps=EPS, k0=K0, mode=MODE, method=METHOD, tiles=int(XL.shape[0]), width_median=float(wid.median()), width_max=float(wid.max()),
               sampled_width_median=float(samp_w.median()), samples_inside=inside, km_per_px=km_per_px, field_range=rng, tau=tau,
               nested=dict(sound=bool(np.all(ec >= ml - 1e-9)), mov_median=float(np.median(ml)), mov_max=float(ml.max()),
                           osc_median=float(np.median(ec)), osc_max=float(ec.max()),
                           ratio_median=float(np.median(ec/np.maximum(ml,1e-9)))),
               scales=scale_rows, scales_T1_data=t1_rows, tau_data=tauD, rank_r16=dict(cert_pairs=int(cert_above.sum()), Nrank_median=float(np.median(Nrank))),
               seconds=time.time() - t0), open(D + f"osc_cert{SUF}.json", "w"), indent=2)
print(f"\nsaved osc_cert{SUF}.json / .npz ({time.time()-t0:.0f}s)")
