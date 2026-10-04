"""F1_prep - data layer of Fig. 1 (the certified chain).

Reads ONLY the result files listed in the source table of the data file and writes data/F1.json (every drawn number with
its source path) and F1_source_data.csv (tidy). Provenance codes (experiment folders) appear only in these comments
and in the recorded source paths, never on the figure.

Sources (results package unless stated otherwise; data/... = data package):
  a  geo_st13_raster.npz, data/enclosures/first_order_georgia.npz (Georgia boxes; the experiment scripts transpose both, see
     experiments/c5_production.py l.19-23, pixel side h = 2/(N-1));
     exp2/C5/c5_global_field_georgia_M2.npz (global nodal field of the lifted class, code M = 2 = X^(1), nodal spacing h/2;
     node order = row-major over the active nodes of the (512*2+1)^2 lattice; verified below against the pixel boxes);
     exp2/checker/check_c5_witness_georgia.json (checker bracket);
     exp1/A4/a4_windows.json window 0 (12-OA window; exact slope range and set-partitioning LP endpoints, K = 3, beta = inf);
     law definitions of the four-law family: experiments/c1_familysup.py draws() (gauss, t4/sqrt2, (Gamma(4)-4)/2, Gamma(1)-1);
     lean/certificates_all/lean_c5_log_*.json (map endpoints PASS), exp2/B15/pricing_verify_n30.json (48 configurations x 2
     charges), exp3_ml/rigorous/lean_results.json + exp3_ml_real/rigorous/lean_results_*.json (pooling instances PASS).
  b  exp3_ml_real/jul2014/{nominal_mlp,boxes_mlp,closure_mlp_CROWN_0.01_ZD}.npz (12 clients x 1008 hourly outputs,
     evaluation window 2014-07-01 00:00 + t h; see zeal/ml_transfer/real_common.py), exp3_ml_real/data_report.json (client ids).
  c  exp1/B11/b11_saving_georgia_kvkg.json (per-pair normalized saving G, 30 pairs per budget),
     exp2/B12/b11_georgia_kvkg.json (by_k: U/V median, q10, q90 over 400 pairs).
  d  exp2/B13/b13_real_{georgia,gm_q4,gm_bad,mx_rwi}.json (per-patch ratio_<outer set> = outer width / witness width).
  e  the machine-checked map of the statement document (ZEAL_STATEMENT_DOC; reproduced in the SI), lean/FREEZE_v1.md (build counts),
     the instance-certificate counts above, exp2/checker/check_b15.json (48/48), exp2/checker/check_spectral.json (23/23).
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import glob, os, sys, json, re, csv, glob
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FE = paths.RESULTS_ROOT                     # results package
LEAN = paths.LEAN_DIR                       # lean/ of this repository
STATEMENT_DOC = os.environ.get("ZEAL_STATEMENT_DOC", "")   # optional: the statement document with the machine-checked map (see below)
P = dict(
    raster=paths.RASTER["georgia"],
    boxes=paths.enclosure("georgia"),
    field=os.path.join(FE, "exp2", "C5", "c5_global_field_georgia_M2.npz"),
    witness=os.path.join(FE, "exp2", "checker", "check_c5_witness_georgia.json"),
    a4=os.path.join(FE, "exp1", "A4", "a4_windows.json"),
    laws=paths.repo("experiments", "c1_familysup.py"),
    lean_c5=os.path.join(LEAN, "certificates_all", "lean_c5_log_{}.json"),
    pricing=os.path.join(FE, "exp2", "B15", "pricing_verify_n30.json"),
    lean_pool_syn=os.path.join(FE, "exp3_ml", "rigorous", "lean_results.json"),
    lean_pool_real=os.path.join(FE, "exp3_ml_real", "rigorous", "lean_results_{}.json"),
    nominal=os.path.join(FE, "exp3_ml_real", "jul2014", "nominal_mlp.npz"),
    mlboxes=os.path.join(FE, "exp3_ml_real", "jul2014", "boxes_mlp.npz"),
    closure=os.path.join(FE, "exp3_ml_real", "jul2014", "closure_mlp_CROWN_0.01_ZD.npz"),
    datarep=os.path.join(FE, "exp3_ml_real", "data_report.json"),
    b11=os.path.join(FE, "exp1", "B11", "b11_saving_georgia_kvkg.json"),
    b12=os.path.join(FE, "exp2", "B12", "b11_georgia_kvkg.json"),
    b13=os.path.join(FE, "exp2", "B13", "b13_real_{}.json"),
    v8=STATEMENT_DOC or "SI machine-checked map",
    freeze=os.path.join(LEAN, "FREEZE_v1.md"),
    check_b15=os.path.join(FE, "exp2", "checker", "check_b15.json"),
    check_spec=os.path.join(FE, "exp2", "checker", "check_spectral.json"),
)
REL = {k: (paths.rel(v) if k != "v8" else "SI machine-checked map") for k, v in P.items()}
SETTINGS = ["georgia", "gm_q4", "gm_bad", "mx_rwi"]
OUT = {"_note": "every number drawn in Fig. 1; 'src' gives the source file (results/... = results package, data/... = data package, other paths relative to the code repository)"}
ROWS = []          # tidy source data: panel, element, series, x, y, value, extra


def row(panel, element, series="", x="", y="", value="", extra=""):
    ROWS.append(dict(panel=panel, element=element, series=series, x=x, y=y, value=value, extra=extra))


def r6(a, nd=6):
    a = np.asarray(a, float)
    return [None if not np.isfinite(v) else float(round(v, nd)) for v in a.ravel()]


def qs(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), q25=float(np.quantile(v, 0.25)), q75=float(np.quantile(v, 0.75)), min=float(v.min()), max=float(v.max()), n=int(len(v)))


# ================================================================================= panel a ==
ras = np.load(P["raster"]); F = np.load(P["boxes"])
lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
mask0 = ras["mask"].astype(bool) & np.isfinite(lo0)
N = mask0.shape[0]; h = 2.0 / (N - 1)
# display orientation: raster axis 0 runs north -> south ('X' increases southward), axis 1 west -> east, so the raster
# itself is the north-up map (checked against the state outline: straight northern border top-left, Savannah River slant).
Wd = np.where(mask0, hi0 - lo0, np.nan)
n_tracts = int(len(np.unique(ras["tract_idx"][mask0]))); n_counties = int(len(np.unique(ras["county_idx"][mask0])))
wq = np.nanquantile(Wd, [0.5, 0.99]); wmax = float(np.nanmax(Wd))
vmax_bar = float(np.floor(wq[1] * 100) / 100 + 0.01)        # colour-bar top (rounded up from the 99th percentile)
a1 = dict(src=REL["boxes"] + " (hi - lo) masked by " + REL["raster"], shape=[N, N], pixels=int(mask0.sum()), tracts=n_tracts,
          counties=n_counties, width_median=float(wq[0]), width_q99=float(wq[1]), width_max=wmax, vmin=0.0, vmax=vmax_bar,
          raster=r6(Wd, 5))
for i, j in zip(*np.nonzero(mask0)):
    row("a1", "box width", "Georgia pixel", int(j), int(i), round(float(Wd[i, j]), 6))

# ---- a2: one 1-D strip, coarse value tiles (k_v = 16) + pixel gradient tiles (k_g = 1), lattice closure on the chain
At = {k: F[k].astype(float).T for k in ("lo", "hi", "g1l", "g1h", "g2l", "g2h")}; mT = mask0.T
KV = 16


def pool(A, fn, fill, k):
    B = np.where(mT, A, fill); n1, n2 = B.shape; m1, m2 = -(-n1 // k), -(-n2 // k)
    Pp = np.full((m1 * k, m2 * k), fill); Pp[:n1, :n2] = B; Pp = Pp.reshape(m1, k, m2, k)
    R = fn(fn(Pp, axis=3), axis=1); return np.repeat(np.repeat(R, k, axis=0), k, axis=1)[:n1, :n2]


loK, hiK = pool(At["lo"], np.min, np.inf, KV), pool(At["hi"], np.max, -np.inf, KV)


def chain_closure(l0, u0, cm, cp):
    """Bellman-Ford label correction on the chain p -> p+1 with phi_{p+1} - phi_p in [cm_p, cp_p]: top u*, bottom l*."""
    u, l = u0.copy(), l0.copy()
    for _ in range(len(u) + 2):
        un = u.copy(); un[1:] = np.minimum(un[1:], u[:-1] + cp); un[:-1] = np.minimum(un[:-1], u[1:] - cm)
        ln = l.copy(); ln[1:] = np.maximum(ln[1:], l[:-1] + cm); ln[:-1] = np.maximum(ln[:-1], l[1:] - cp)
        if np.array_equal(un, u) and np.array_equal(ln, l):
            break
        u, l = un, ln
    assert np.all(l <= u + 1e-12), "empty chain polytope"
    return u, l


LS = 48; scans = []
for j in range(8, N - 8, 4):                       # strips along transposed axis 0 (= display row j, west -> east), gradient G1
    idx = np.flatnonzero(mT[:, j]); runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        for s in range(0, len(run) - LS + 1, 16):
            ii = run[s:s + LS]
            lo_s, hi_s = loK[ii, j], hiK[ii, j]; gl, gh = At["g1l"][ii, j], At["g1h"][ii, j]
            cm, cp = h * (gl[:-1] + gl[1:]) / 2, h * (gh[:-1] + gh[1:]) / 2
            u, l = chain_closure(lo_s, hi_s, cm, cp)
            scans.append((float(np.median((u - l) / (hi_s - lo_s))), j, int(ii[0])))
ratios = np.array([s[0] for s in scans]); target = np.quantile(ratios, 0.05)
pick = min(scans, key=lambda s: abs(s[0] - target)); jrow, i0 = pick[1], pick[2]
ii = np.arange(i0, i0 + LS)
lo_s, hi_s = loK[ii, jrow], hiK[ii, jrow]; gl, gh = At["g1l"][ii, jrow], At["g1h"][ii, jrow]
cm, cp = h * (gl[:-1] + gl[1:]) / 2, h * (gh[:-1] + gh[1:]) / 2
u_s, l_s = chain_closure(lo_s, hi_s, cm, cp)
a2 = dict(src=REL["boxes"] + " (value boxes pooled to 16 x 16 tiles as in the tile-budget runs; pixel gradient boxes; tent increments)",
          k_v=KV, k_g=1, length=LS, display_row=int(jrow), display_col0=int(i0), pick_rule="5th percentile of the median closure/box ratio over scanned strips",
          n_scanned=int(len(scans)), ratio_median_all=float(np.median(ratios)), ratio_strip=float(np.median((u_s - l_s) / (hi_s - lo_s))),
          lo=r6(lo_s), hi=r6(hi_s), l_star=r6(l_s), u_star=r6(u_s))
for k in range(LS):
    row("a2", "strip", "value box lo", k, "", round(lo_s[k], 6)); row("a2", "strip", "value box hi", k, "", round(hi_s[k], 6))
    row("a2", "strip", "closure l*", k, "", round(l_s[k], 6)); row("a2", "strip", "closure u*", k, "", round(u_s[k], 6))

# ---- a3: global nodal field of the lifted class X^(1) (nodal spacing h/2) on a 3 x 3 pixel window
G = np.load(P["field"]); x = G["x"]; M = int(G["M"]); I0, J0 = int(G["I0"]), int(G["J0"])
assert (I0, J0) == (0, 0)
act = np.zeros((N * M + 1, N * M + 1), bool)
for (i, j) in np.argwhere(mT):
    act[i * M:i * M + M + 1, j * M:j * M + M + 1] = True
assert act.sum() == len(x)
col = np.full(act.shape, -1, np.int64); col[act] = np.arange(len(x))           # row-major order over active nodes
Xn = np.where(col >= 0, x[np.maximum(col, 0)], np.nan)
viol = 0.0
for (i, j) in np.argwhere(mT)[::37]:
    blk = Xn[i * M:i * M + M + 1, j * M:j * M + M + 1]; viol = max(viol, blk.max() - At["hi"][i, j], At["lo"][i, j] - blk.min())
assert viol < 1e-9, viol                                                  # node order verified against the pixel boxes
best = None
for i in range(0, N - 3, 3):
    for j in range(0, N - 3, 3):
        if not mT[i:i + 3, j:j + 3].all():
            continue
        blk = Xn[i * M:i * M + 3 * M + 1, j * M:j * M + 3 * M + 1]
        rng_ = float(np.ptp(blk))
        if best is None or rng_ > best[0]:
            best = (rng_, i, j)
_, bi, bj = best
blk = Xn[bi * M:bi * M + 3 * M + 1, bj * M:bj * M + 3 * M + 1]          # [a (east), b (south)]
nodes = blk.T                                                              # display: rows = south, cols = east
up = 24; n_n = nodes.shape[0]; t = np.linspace(0, n_n - 1, (n_n - 1) * up + 1)
k0 = np.minimum(np.floor(t).astype(int), n_n - 2); fr = t - k0
img = np.zeros((len(t), len(t)))
for r_, (ka, fa) in enumerate(zip(k0, fr)):                                # exact bilinear on every h/2 sub-element
    rowv = (1 - fa) * nodes[ka] + fa * nodes[ka + 1]
    img[r_] = (1 - fr) * rowv[k0] + fr * rowv[k0 + 1]
wit = json.load(open(P["witness"], encoding="utf-8"))
a3 = dict(src=REL["field"], M_code=M, nodes_total=int(len(x)), checker_nodes=int(wit["nodes"]), block_pixels=[3, 3], block_display_row0=int(bj), block_display_col0=int(bi),
          nodal=r6(nodes), image=r6(img, 6), vmin=float(np.nanmin(nodes)), vmax=float(np.nanmax(nodes)), node_order_check_max_violation=float(viol))
for r_ in range(n_n):
    for c_ in range(n_n):
        row("a3", "nodal value", "global witness field", c_, r_, round(float(nodes[r_, c_]), 6))

# ---- a4: checker bracket (Georgia)
a4 = dict(src=REL["witness"], L=float(wit["bracket"][0]), U=float(wit["bracket"][1]), reference=1.0, pairs=int(wit["pairs"]),
          witness_feasible_exact=bool(wit["witness_feasible_exact"]))
row("a4", "map-level bracket", "global witness (lower)", "", "", a4["L"]); row("a4", "map-level bracket", "lifted ceiling (upper)", "", "", a4["U"])

# ---- a5: identification fiber glyph (schematic; no data): three unit means and two fields with identical unit means
ymeans = np.array([0.42, 0.78, 0.56]); xs = np.linspace(0, 3, 301)
# m(x) = c0 + c1 x + c2 x^2 with exact unit means; the zero-unit-mean perturbation sin(2 pi x) keeps every unit mean
Aq = np.array([[1, (k + 0.5), ((k + 1) ** 3 - k ** 3) / 3] for k in range(3)]); cq = np.linalg.solve(Aq, ymeans)
mx = cq[0] + cq[1] * xs + cq[2] * xs ** 2; amp = 0.17
f1 = mx + amp * np.sin(2 * np.pi * xs); f2 = mx - amp * np.sin(2 * np.pi * xs)
unit_means = [[float(np.mean(f[(xs >= k) & (xs <= k + 1)])) for k in range(3)] for f in (f1, f2)]
a5 = dict(src="schematic glyph (no data): two fields with identical unit means", means=ymeans.tolist(), x=r6(xs, 4), f1=r6(f1, 5), f2=r6(f2, 5),
          numeric_unit_means=unit_means)

# ---- a6: the four declared noise laws (unit variance), exact densities, and their pointwise envelope
z = np.linspace(-3.6, 4.4, 401)
dens = {"Gaussian": stats.norm.pdf(z),
        "t4": stats.t.pdf(z * np.sqrt(2.0), 4) * np.sqrt(2.0),            # t4 / sqrt(2)
        "gamma4": 2.0 * stats.gamma.pdf(2.0 * z + 4.0, 4.0),              # (Gamma(4,1) - 4) / 2
        "gamma1": stats.expon.pdf(z + 1.0)}                                # Gamma(1,1) - 1
env = np.max(np.vstack(list(dens.values())), axis=0)
a6 = dict(src=REL["laws"] + " draws(): gauss, t4/sqrt(2), (Gamma(4)-4)/2, Gamma(1)-1 (all unit variance)", z=r6(z, 4),
          dens={k: r6(v, 5) for k, v in dens.items()}, envelope=r6(env, 5),
          labels={"Gaussian": "Gaussian", "t4": "t, 4 df", "gamma4": "gamma, shape 4", "gamma1": "gamma, shape 1"})
for k, v in dens.items():
    for zz, vv in zip(z[::10], v[::10]):
        row("a6", "density", k, round(float(zz), 3), "", round(float(vv), 5))

# ---- a7: 12-OA window, K = 3, beta = inf (exact enumeration and set-partitioning LP)
A4 = json.load(open(P["a4"], encoding="utf-8")); w0 = A4["windows"][0]; c3 = w0["classes"]["K3_betainf"]
a7 = dict(src=REL["a4"] + " windows[0] classes.K3_betainf / slope_setpart.K3_betainf", n_oa=int(w0["n"]), K=3, partitions=int(c3["count"]),
          slope_exact=[float(v) for v in c3["slope"]], slope_lp=[float(v) for v in w0["slope_setpart"]["K3_betainf"]], sign_exact=c3["slope_sign_exact"],
          glyph_note="cell geometry is not stored in the window file: a 4 x 3 grid glyph of the 12 OAs with one contiguous K = 3 partition (schematic)",
          glyph_zones=[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2]])
row("a7", "slope range", "exact enumeration", "", "", a7["slope_exact"][0], "lower"); row("a7", "slope range", "exact enumeration", "", "", a7["slope_exact"][1], "upper")
row("a7", "slope range", "set-partitioning LP", "", "", a7["slope_lp"][0], "lower"); row("a7", "slope range", "set-partitioning LP", "", "", a7["slope_lp"][1], "upper")

# ---- certificate lane counts
c5n = {}; c5pass = {}
for s in SETTINGS:
    L_ = json.load(open(P["lean_c5"].format(s), encoding="utf-8")); c5n[s] = len(L_); c5pass[s] = int(sum(bool(r["passed"]) for r in L_))
pr = json.load(open(P["pricing"], encoding="utf-8"))
n_cfg = len(pr); n_charges = int(sum((r["upper"]["status"] == "verified") + (r["lower"]["status"] == "verified") for r in pr))
pool_files = [P["lean_pool_syn"]] + [P["lean_pool_real"].format(w) for w in ("jan2014", "jul2014")]
pool_n = 0; pool_pass = 0
for f in pool_files:
    L_ = json.load(open(f, encoding="utf-8")); pool_n += len(L_); pool_pass += int(sum(r["verdict"] == "PASS" for r in L_))
lane = dict(src=[REL["lean_c5"].format("*"), REL["pricing"], REL["lean_pool_syn"], REL["lean_pool_real"].format("*")],
            map_endpoints=int(sum(c5n.values())), map_pass=int(sum(c5pass.values())), map_by_setting=c5pass,
            pricing_configurations=n_cfg, pricing_charges=n_charges, pooling_instances=pool_n, pooling_pass=pool_pass)
assert lane["map_pass"] == lane["map_endpoints"] and lane["pooling_pass"] == lane["pooling_instances"]
for k in ("map_endpoints", "map_pass", "pricing_charges", "pooling_instances", "pooling_pass"):
    row("a", "certificate lane", k, "", "", lane[k])
OUT["a"] = dict(a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7, lane=lane)

# ================================================================================= panel b ==
rep = json.load(open(P["datarep"], encoding="utf-8")); nom = np.load(P["nominal"]); B = np.load(P["mlboxes"]); C = np.load(P["closure"])
J = 2; WEEK = 1; t0, t1 = 168 * WEEK, 168 * (WEEK + 1); sl = slice(t0, t1)
f = nom["f"].astype(float); lo, hi = B["lo|CROWN|0.01"], B["hi|CROWN|0.01"]; u, l = C["u"], C["l"]
hin, lon = B["hi_in|0.01"], B["lo_in|0.01"]; gl1, gh1 = B["glo|CROWN|0.01|1"], B["ghi|CROWN|0.01|1"]
impl_w = (hi[:, 1:] - lo[:, :-1]) - (lo[:, 1:] - hi[:, :-1])            # increment width implied by the value boxes alone
closure_gain_max = float(max(np.max(hi - u), np.max(l - lo)))
b = dict(src=[REL["nominal"], REL["mlboxes"] + " (CROWN, eps = 0.01: lo|hi, glo|ghi lag 1, PGD inner lo_in|hi_in)", REL["closure"], REL["datarep"]],
         client=rep["chosen"][J], client_index=J, window="jul2014", eval_start="2014-07-01 00:00", t0=t0, hours=t1 - t0,
         day0=int(1 + t0 // 24), eps=0.01, method="CROWN",
         f=r6(f[J, sl]), lo=r6(lo[J, sl]), hi=r6(hi[J, sl]), u_star=r6(u[J, sl]), l_star=r6(l[J, sl]),
         width_box=r6(hi[J, sl] - lo[J, sl]), width_closure=r6(u[J, sl] - l[J, sl]), width_inner=r6(hin[J, sl] - lon[J, sl]),
         width_incr1=r6(gh1[J, t0:t1 - 1] - gl1[J, t0:t1 - 1]), width_incr1_implied=r6(impl_w[J, t0:t1 - 1]),
         closure_gain_max_all_clients=closure_gain_max,
         inner_over_box_median=float(np.median((hin[J, sl] - lon[J, sl]) / (hi[J, sl] - lo[J, sl]))),
         incr_over_implied_median=float(np.median((gh1[J, t0:t1 - 1] - gl1[J, t0:t1 - 1]) / impl_w[J, t0:t1 - 1])))
for k in range(t1 - t0):
    for nm in ("f", "lo", "hi", "u_star", "l_star", "width_box", "width_closure", "width_inner"):
        row("b", "series", nm, t0 + k, "", b[nm][k])
    if k < t1 - t0 - 1:
        row("b", "series", "width_incr1", t0 + k, "", b["width_incr1"][k]); row("b", "series", "width_incr1_implied", t0 + k, "", b["width_incr1_implied"][k])
OUT["b"] = b

# ================================================================================= panel c ==
B11 = json.load(open(P["b11"], encoding="utf-8")); B12 = json.load(open(P["b12"], encoding="utf-8"))["by_k"]
budgets = ["1:16", "1:4", "1:1", "4:1", "16:1", "64:1"]
c = dict(src=[REL["b11"], REL["b12"]], budgets=budgets, log2_ratio=[], G={}, UV={}, G_1616=None, UV_1616=None, G_pairs={})
for bud in budgets + ["16:16"]:
    kv, kg = (int(v) for v in bud.split(":"))
    Gs = [p["normalized_saving"] for p in B11[bud]["pairs"]]
    c["G_pairs"][bud] = [float(round(v, 6)) for v in Gs]
    key = f"v{kv}g{kg}"; S = B12[key]
    gq = qs(Gs); uv = dict(median=S["U_over_V_median"], q10=S["U_over_V_q10"], q90=S["U_over_V_q90"], n=S["pairs"])
    if bud == "16:16":
        c["G_1616"], c["UV_1616"] = gq, uv
    else:
        c["G"][bud], c["UV"][bud] = gq, uv; c["log2_ratio"].append(float(np.log2(kv / kg)))
    for p in B11[bud]["pairs"]:
        row("c", "normalized saving G", bud, f"{p['coarse']}/{p['fine']}", "", p["normalized_saving"])
    row("c", "U/V median", bud, "", "", uv["median"], f"q10 {uv['q10']} q90 {uv['q90']} n {uv['n']}")
OUT["c"] = c

# ================================================================================= panel d ==
TAGS = ["PhiC", "13a_4", "X1", "X2", "XL2"]
d = dict(src=[REL["b13"].format(s) for s in SETTINGS], tags=TAGS, order_note="order as in the design: pixel outer, interpolation-completed Q4 + eps4, lifted X^(1), X^(2), X_L^(2)",
         settings={})
for s in SETTINGS:
    R = json.load(open(P["b13"].format(s), encoding="utf-8")); rows_ = R["rows"]; uniq = {}
    for r_ in rows_:
        uniq.setdefault(r_["fine"], r_)
    d["settings"][s] = dict(unique=len(uniq), rows=len(rows_), by_tag={})
    for tg in TAGS:
        v = np.array([r_[f"ratio_{tg}"] for r_ in uniq.values()], float)
        d["settings"][s]["by_tag"][tg] = qs(v)
        for r_ in uniq.values():
            row("d", "outer / witness", s, tg, r_["label"], r_[f"ratio_{tg}"])
        assert int(R["summary"][tg]["violations"]) == 0
OUT["d"] = d

# ================================================================================= panel e ==
# The machine-checked map (statement -> Lean declarations) is part of the statement document, which is not deposited (the SI
# reproduces the map).  With ZEAL_STATEMENT_DOC pointing at it the flags are recomputed; otherwise the stored flags of
# data/F1.json are reused for this panel (every other number of the panel is recomputed from the deposited files).
if STATEMENT_DOC and os.path.exists(STATEMENT_DOC):
    v8 = open(P["v8"], encoding="utf-8").read().splitlines()
    iii = next(i for i, ln in enumerate(v8) if ln.startswith("## Part III-ter")); iv = next(i for i, ln in enumerate(v8) if ln.startswith("## Part IV"))
    table = "\n".join(v8[iii:iv]); notmc = next(ln for ln in v8[iii:iv] if ln.startswith("Not machine-checked")); _MC = None
else:
    _prev = json.load(open(os.path.join(HERE, "data", "F1.json"), encoding="utf-8"))["e"]
    notmc = _prev["not_machine_checked_line"]; _MC = {g_["stage"]: bool(g_["machine_checked"]) for g_ in _prev["grid"]}


def formalized(code):
    """statement code appears in a row of the machine-checked table (and is not in the 'Not machine-checked' list as a whole)"""
    rows_ = [ln for ln in v8[iii:iv] if ln.startswith("| ") and not ln.startswith("| statement") and not ln.startswith("|---")]
    return any(re.search(rf"(?<![\w]){re.escape(code)}(?![\d])", ln.split("|")[1]) for ln in rows_)


# stage -> principal statements (code Tn of the statement document = Supplementary Theorem Sn; provenance only, never on the figure)
STAGES = [("local enclosures", ["T20"]), ("fused bound", ["T3", "T4", "T5"]), ("realizable witness", ["T7", "T8"]),
          ("map-level bracket", ["T10"]), ("identification", ["T11", "T13"]), ("inference under declared law", ["T14", "T15"]),
          ("zoning statistics", ["T16", "T17", "T18"]), ("instance certificates", ["V1/V3"])]
fz = open(P["freeze"], encoding="utf-8").read()
m_src = re.search(r"Source files:\s*(\d+)\s*\((\d+) lines", fz); m_w = re.search(r"warnings (\d+), errors (\d+)", fz)
n_lean_files = len(glob.glob(os.path.join(os.path.dirname(P["freeze"]), "zeal", "Zeal", "**", "*.lean"), recursive=True)) + len(glob.glob(os.path.join(os.path.dirname(P["freeze"]), "zeal", "*.lean"))) + len(glob.glob(os.path.join(os.path.dirname(P["freeze"]), "zeal", "scripts", "*.lean")))   # Lean source files only (the manifest also lists three build files)
b15 = json.load(open(P["check_b15"], encoding="utf-8")); spec = json.load(open(P["check_spec"], encoding="utf-8"))["summary"]
grid = []
for name, codes in STAGES:
    mc = all(formalized(cd) for cd in codes) if _MC is None else _MC[name]
    grid.append(dict(stage=name, statements=codes, machine_checked=bool(mc)))
G_ = {g["stage"]: g for g in grid}
# instance layers (from the counted artifacts above)
for g in grid:
    g.update(verified_checker=None, exact_rational=None, external=True, mc_count=None)
G_["fused bound"].update(verified_checker=lane["pooling_pass"], exact_rational=lane["pooling_instances"])
G_["map-level bracket"].update(verified_checker=lane["map_pass"], exact_rational=lane["map_endpoints"])
wit_all = {f_: bool(json.load(open(P["witness"].replace("georgia", f_), encoding="utf-8"))["witness_feasible_exact"]) for f_ in SETTINGS}
G_["realizable witness"].update(exact_rational="witness" if all(wit_all.values()) else None, witness_exact_by_setting=wit_all)   # exact-feasible global witnesses (no count printed)
G_["zoning statistics"].update(exact_rational=int(b15["n_pass"]), exact_rational_2=int(spec["pass_strict"]))
G_["instance certificates"].update(verified_checker=True, exact_rational=True, external=False, mc_count=lane["pricing_charges"])
e = dict(src=[REL["v8"], REL["freeze"], REL["check_b15"], REL["check_spec"]] + lane["src"],
         grid=grid, lean_files=n_lean_files, lean_lines=int(m_src.group(2)), warnings=int(m_w.group(1)), errors=int(m_w.group(2)),
         not_machine_checked_line=notmc, b15_pass=int(b15["n_pass"]), b15_n=int(b15["n"]), spectral_pass=int(spec["pass_strict"]), spectral_n=int(spec["n"]))
for g in grid:
    row("e", "status", g["stage"], "", "", json.dumps({k: g[k] for k in ("machine_checked", "verified_checker", "exact_rational", "external", "mc_count")}))
OUT["e"] = e

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(OUT, open(os.path.join(HERE, "data", "F1.json"), "w", encoding="utf-8"), indent=None, ensure_ascii=False)
with open(os.path.join(HERE, "F1_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    wtr = csv.DictWriter(fh, fieldnames=["panel", "element", "series", "x", "y", "value", "extra"]); wtr.writeheader(); wtr.writerows(ROWS)
print("F1 data written:", len(ROWS), "tidy rows")
print("a2 strip", jrow, i0, "ratio", a2["ratio_strip"], "median all", a2["ratio_median_all"], "n", a2["n_scanned"])
print("a3 block", bi, bj, "range", a3["vmin"], a3["vmax"], "viol", viol)
print("lane", {k: lane[k] for k in ("map_endpoints", "pricing_charges", "pooling_instances")})
print("b client", b["client"], "closure gain max", closure_gain_max, "inner/box", b["inner_over_box_median"], "incr/implied", b["incr_over_implied_median"])
print("e grid", [(g["stage"], g["machine_checked"]) for g in grid], e["lean_files"], e["lean_lines"])
