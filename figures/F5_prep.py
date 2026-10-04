"""F5_prep - data layer of Fig. 5 (identification and reconstruction, Greater Manchester OAs).

Reads ONLY: exp1/A3/a3_manchester.json, exp2/C3/c3_level_up.json, exp1/B2/b2_zoning_vs_scale.json,
exp1/B2/b2_decomp.json. Writes data/F5.json (every drawn number + source) and F5_source_data.csv.
Provenance (comments only): A3 = identification experiment; C3 = operational level-up shrinkage (replaces the in-sample
kappa_up stored in A3 'field', which results.md marks invalid); B2 = zoning vs scale and the decomposition audit.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FE = paths.RESULTS_ROOT
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
REL = {"a3": "exp1/A3/a3_manchester.json", "c3": "exp2/C3/c3_level_up.json", "b2": "exp1/B2/b2_zoning_vs_scale.json", "b2d": "exp1/B2/b2_decomp.json"}
SRC = {k: "results/" + v for k, v in REL.items()}
a3 = json.load(open(os.path.join(FE, REL["a3"]))); c3 = json.load(open(os.path.join(FE, REL["c3"])))
b2 = json.load(open(os.path.join(FE, REL["b2"]))); b2d = json.load(open(os.path.join(FE, REL["b2d"])))
VARS = ["q4", "bad"]
out = dict(figure="F5", vars=VARS, sources=SRC, panels={})
rows = []

# ---------------- a: identified-set widths (graph Lipschitz / graph total variation budgets) ----------------
pa = {}
for v in VARS:
    ids = a3[v]["identified_sets"]; d = {}
    for cls, base in (("lipschitz_graph", ids["L_min_per_km"]), ("graphTV", ids["tau_min_graphTV"])):
        ent = []
        for k, e in ids[cls].items():
            mult = float(k) / base
            ent.append(dict(budget=float(k), multiple=round(mult, 3), width_median=max(e["width_median"], 0.0), width_mean=max(e["width_mean"], 0.0),
                            width_median_raw=e["width_median"], width_mean_raw=e["width_mean"], truth_compatible_share=e["truth_compatible_share"],
                            decision_certified_share=e["decision_certified_share"], n=e["n"]))
            rows.append(("a", v, f"{cls}_width_median", f"budget={k};multiple={mult:.3f}", e["width_median"], SRC["a3"]))
            rows.append(("a", v, f"{cls}_width_mean", f"budget={k};multiple={mult:.3f}", e["width_mean"], SRC["a3"]))
            rows.append(("a", v, f"{cls}_truth_compatible_share", f"budget={k}", e["truth_compatible_share"], SRC["a3"]))
        d[cls] = sorted(ent, key=lambda z: z["multiple"])
    oa_sd = {e["oa_sd"] for c in ("lipschitz_graph", "graphTV") for e in ids[c].values()}; assert len(oa_sd) == 1
    d.update(within_sd=oa_sd.pop(), L_min=ids["L_min_per_km"], tau_min=ids["tau_min_graphTV"], truth_L=ids["truth_L_graph"], truth_TV=ids["truth_graphTV"],
             truth_L_multiple=ids["truth_L_graph"] / ids["L_min_per_km"], truth_TV_multiple=ids["truth_graphTV"] / ids["tau_min_graphTV"],
             n_sampled=int(next(iter(ids["lipschitz_graph"].values()))["n"]))
    rows.append(("a", v, "within_LSOA_sd", "", d["within_sd"], SRC["a3"]))
    pa[v] = d
out["panels"]["a"] = pa

# ---------------- b: estimators, exact-data R^2 at OA; repeated-noise mean risk -> R^2 scale ----------------
EST = [("painting", "painting"), ("gls_affine_msoa", "affine GLS"), ("energy_exact", "energy"), ("energy_nugget_up", "energy + nugget"),
       ("dasy_ecological", "dasymetric, ecological"), ("dasy_one_level_up", "dasymetric, level-up"), ("neural_field", "neural field")]
pb = {"order": [k for k, _ in EST] + ["field_shrunk_kappa_up"], "labels": dict(EST) | {"field_shrunk_kappa_up": "field, level-up κ"}, "vars": {}}
for v in VARS:
    T = a3[v]["estimators"]; var_t = []
    for k, _ in EST:
        if "risk_exact_data" in T[k]: var_t.append(T[k]["risk_exact_data"] / (1.0 - T[k]["R2_exact_data"]))
    var_t = np.array(var_t); assert np.ptp(var_t) / var_t.mean() < 1e-9, var_t      # one weighted variance of OA truth, R^2 = 1 - risk / var
    vt = float(var_t.mean()); d = {}
    for k, _ in EST:
        e = dict(R2=T[k]["R2_exact_data"], source=SRC["a3"])
        if "risk_noisy_mean" in T[k]:
            e.update(R2_noisy=1.0 - T[k]["risk_noisy_mean"] / vt, R2_noisy_se=T[k]["risk_noisy_se"] / vt, risk_noisy_mean=T[k]["risk_noisy_mean"], risk_noisy_se=T[k]["risk_noisy_se"])
        d[k] = e
        rows.append(("b", v, f"R2_exact_{k}", "", e["R2"], SRC["a3"]))
        if "R2_noisy" in e: rows.append(("b", v, f"R2_noisy_mean_{k}", f"se={e['R2_noisy_se']!r};reps=200", e["R2_noisy"], SRC["a3"]))
    tr = c3[v]["transfer"]
    assert abs(tr["R2_oa_painting"] - T["painting"]["R2_exact_data"]) < 1e-12 and abs(tr["R2_oa_field"] - T["neural_field"]["R2_exact_data"]) < 1e-12
    d["field_shrunk_kappa_up"] = dict(R2=tr["R2_oa_shrunk_transfer"], source=SRC["c3"])
    rows.append(("b", v, "R2_exact_field_shrunk_kappa_up", "", tr["R2_oa_shrunk_transfer"], SRC["c3"]))
    pb["vars"][v] = dict(est=d, var_truth=vt, kappa_up=c3[v]["kappa_up"], kappa_oracle_oa=tr["kappa_oracle_oa"])
    rows.append(("b", v, "kappa_up", "", c3[v]["kappa_up"], SRC["c3"]))
out["panels"]["b"] = pb

# ---------------- c: quadratic risk identity R(k)/R(0) - 1 = -2 k rho m + k^2 m^2 (k* = rho / m) ----------------
pc = {}
for v in VARS:
    f = a3[v]["field"]; m, rho = f["m"], f["rho"]; ks = rho / m
    assert abs(ks - f["kappa_star_oracle"]) < 1e-12 and abs(ks - c3[v]["transfer"]["kappa_oracle_oa"]) < 1e-9
    ku = c3[v]["kappa_up"]; W = a3[v]["within_share_W"]
    rel = lambda k: -2 * k * rho * m + k * k * m * m
    R2p = c3[v]["transfer"]["R2_oa_painting"]; R2s = c3[v]["transfer"]["R2_oa_shrunk_transfer"]
    stored_rel = (1 - R2s) / (1 - R2p) - 1.0                       # relative risk change of the stored transfer result
    pc[v] = dict(m=m, rho=rho, kappa_star=ks, two_kappa_star=2 * ks, kappa_up=ku, within_share_W=W, rel_min_pct=100 * rel(ks), rel_kappa_up_identity_pct=100 * rel(ku),
                 rel_kappa_up_stored_pct=100 * stored_rel, rel_kappa1_pct=100 * rel(1.0), identity_vs_stored_abs_diff_pct=abs(100 * rel(ku) - 100 * stored_rel),
                 formula="R(kappa)/R(0) - 1 = -2 kappa rho m + kappa^2 m^2, kappa* = rho/m (Theorem S13 (iv) with s = field detail, r = within-LSOA truth detail); R(0) = painting risk")
    for k_, v_ in pc[v].items():
        if isinstance(v_, float): rows.append(("c", v, k_, "", v_, SRC["a3"] + " ; " + SRC["c3"]))
out["panels"]["c"] = pc

# ---------------- d: zoning effect vs scale effect; unit-reassignment decomposition ----------------
pd_ = {}
for v in VARS:
    z = b2[v]; sc = z["scale"]["oa_lsoa"]; sk = z["same_K_balanced"]
    left = [dict(key="oa_msoa", label="OA → MSOA", kind="scale", value=z["scale"]["oa_msoa"] / sc),
            dict(key="lsoa_msoa", label="LSOA → MSOA", kind="scale", value=z["scale"]["lsoa_msoa"] / sc),
            dict(key="same_K", label="same-K re-zoning", kind="zoning", value=sk["mov_Z_vs_LSOA"]["median"] / sc, lo=sk["mov_Z_vs_LSOA"]["q05"] / sc, hi=sk["mov_Z_vs_LSOA"]["q95"] / sc, n=sk["n"])]
    for e in z["displacement"]:
        left.append(dict(key=f"disp_q{e['q']}_s{e['sweeps']}", label=f"{100 * e['moved_share']:.0f} % moved", kind="displacement", value=e["mov_over_scale"], q=e["q"], sweeps=e["sweeps"], moved_share=e["moved_share"]))
    for e in left:
        rows.append(("d", v, f"movement_over_scale_{e['key']}", f"lo={e.get('lo', '')};hi={e.get('hi', '')}", e["value"], SRC["b2"]))
    dd = b2d[v]; right = []
    for q, e in dd["by_q"].items():
        assert abs(e["moved_share_of_mov2"] + e["stay_share_of_mov2"] - 1) < 1e-12
        right.append(dict(q=float(q), mu_M=e["mu_M"], moved_share=e["moved_share_of_mov2"], stay_share=e["stay_share_of_mov2"], Cbar_over_within_sd=e["Cbar_over_within_sd"], identity_residual_max=e["identity_residual_max"]))
        rows.append(("d", v, "moved_term_share_of_mov2_median", f"q={q};mu_M_median={e['mu_M']}", e["moved_share_of_mov2"], SRC["b2d"]))
    pd_[v] = dict(scale_oa_lsoa=sc, rows=left, decomp=right, slope_moved=dd["slope_moved_term"], slope_stay=dd["slope_stay_term"], slope_mov2=dd["slope_mov2_vs_muM"],
                  loglog_slope_mov_vs_moved_share=z["loglog_slope_mov_vs_moved_share"], ratio_zoning_over_scale=sk["ratio_zoning_over_scale"],
                  identity_residual_max=max(r["identity_residual_max"] for r in right))
    for k_ in ("slope_moved", "slope_stay"):
        rows.append(("d", v, k_, "", pd_[v][k_], SRC["b2d"]))
out["panels"]["d"] = pd_
RAS = paths.RASTER["gm_q4"]          # OA count = distinct OA labels (bg_idx) inside the GM raster mask
_r = np.load(RAS); out["n_oa"] = int(len(np.unique(_r["bg_idx"][_r["mask"].astype(bool)]))); out["sources"]["n_oa"] = paths.rel(RAS)
json.dump(out, open(os.path.join(DATA, "F5.json"), "w", encoding="utf-8"), indent=1)
with open(os.path.join(HERE, "F5_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["panel", "variable", "quantity", "key", "value", "source"]); w.writerows(rows)
for v in VARS:
    print(v, "c:", {k: round(x, 5) for k, x in pc[v].items() if isinstance(x, float)})
    print(v, "b noisy:", {k: round(e.get("R2_noisy", float("nan")), 4) for k, e in pb["vars"][v]["est"].items()})
