"""F4_prep - data layer for Fig. 4 (zoning statistics: certified slope ranges, hot spots and ranks).

Reads ONLY the canonical result files listed in the source table of the data file for Fig. 4 and writes data/F4.json (every drawn number
with its source path) and F4_source_data.csv (tidy). Provenance codes are kept in comments only.
  a, b : exp2/B15/pricing_verify_n30.json (B15), exp2/B15/enum_counts_n30.json, exp2/checker/check_b15.json,
         lean/certificates/export_log.json, lean/certificates/lean_check_log.md
  c    : exp1/A4/a4_windows.json (A4)
  d    : exp2/C1/c1_georgia_familysup4.json (C1), exp2/C1/c1_focal_counts.json; raster geo_st13_raster.npz (unit count)
  e    : exp2/B3/b3_classes_georgia_v5.json (B3; construction C = pair-specific classes, A = witness-anchored universe)
  f    : exp2/B7/b7_conditional_familysup.json (B7)
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, re, io, csv, json, math, sys
import numpy as np
from scipy.stats import beta as beta_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
FE = paths.RESULTS_ROOT
SRC = {
    "b15": os.path.join(FE, "exp2", "B15", "pricing_verify_n30.json"),
    "b15_enum": os.path.join(FE, "exp2", "B15", "enum_counts_n30.json"),
    "b15_check": os.path.join(FE, "exp2", "checker", "check_b15.json"),
    "lean_export": paths.lean_file(os.path.join("certificates", "export_log.json")),
    "lean_log": paths.lean_file(os.path.join("certificates", "lean_check_log.md")),
    "a4": os.path.join(FE, "exp1", "A4", "a4_windows.json"),
    "c1": os.path.join(FE, "exp2", "C1", "c1_georgia_familysup4.json"),
    "c1_focal": os.path.join(FE, "exp2", "C1", "c1_focal_counts.json"),
    "b3": os.path.join(FE, "exp2", "B3", "b3_classes_georgia_v5.json"),
    "b7": os.path.join(FE, "exp2", "B7", "b7_conditional_familysup.json"),
    "raster": paths.RASTER["georgia"],
}
REL = {k: paths.rel(v) for k, v in SRC.items()}
J = lambda k: json.load(open(SRC[k], encoding="utf-8"))
OUT = dict(sources=REL)
ROWS = []          # tidy source data


def row(panel, series, x, y, value, unit, src, note=""):
    ROWS.append(dict(panel=panel, series=series, x=x, y=y, value=value, unit=unit, source=REL[src], note=note))


# ------------------------------------------------------------------------------------------------ a, b (B15) --
b15 = J("b15"); enum = J("b15_enum"); chk = J("b15_check")
enum_map = {(r["window"], r["K"], float(r["beta"])): r["n_cells"] for r in enum}
chk_map = {(r["window"], r["K"], float(r["beta"])): r for r in chk["rows"]}
cfg = []
for r in sorted(b15, key=lambda z: (z["window"], z["K"], -float(z["beta"]))):
    u, l = r["upper"], r["lower"]
    assert u["status"] == "verified" and l["status"] == "verified"
    key = (r["window"], r["K"], float(r["beta"]))
    # verified slope range = [lower.bound_orig, upper.bound_orig] (Theorem-B1 certificate at the final duals, original slope units);
    # witnessed range = [lower.ip_beta, upper.ip_beta] (best verified feasible partitions, the 'wit' of the run log; the
    # hill-climbing witness hc_beta is never better than ip_beta beyond 1e-15 floating-point ties)
    wlo = l["ip_beta"]; whi = u["ip_beta"]
    assert wlo <= l["hc_beta"] + 1e-12 and whi >= u["hc_beta"] - 1e-12
    c = dict(window=r["window"], window_label=r["window"] + 1, K=r["K"], beta="inf" if math.isinf(float(r["beta"])) else float(r["beta"]),
             n_oa=r["n"], verified=[l["bound_orig"], u["bound_orig"]], witnessed=[wlo, whi], n_cells=r["n_cells"], n_cells_enum=enum_map[key],
             eps_exact=[l["eps_exact"], u["eps_exact"]], eps_used=[l["eps_used"], u["eps_used"]],
             eps_checker=[chk_map[key]["lower"]["eps_checker"], chk_map[key]["upper"]["eps_checker"]],
             checker_mode=[chk_map[key]["lower"]["mode"], chk_map[key]["upper"]["mode"]], checker_pass=bool(chk_map[key]["pass"]))
    assert c["n_cells"] == c["n_cells_enum"]
    c["ratio"] = (c["verified"][1] - c["verified"][0]) / (c["witnessed"][1] - c["witnessed"][0])
    c["both_signs_verified"] = bool(c["verified"][0] < 0 < c["verified"][1]); c["both_signs_witnessed"] = bool(c["witnessed"][0] < 0 < c["witnessed"][1])
    cfg.append(c)
    for s, v in (("verified_lo", c["verified"][0]), ("verified_hi", c["verified"][1]), ("witnessed_lo", wlo), ("witnessed_hi", whi)):
        row("a", s, f"window {c['window_label']}", f"K={c['K']} beta={c['beta']}", v, "slope (share per share)", "b15")
    row("a", "admissible_cells", f"window {c['window_label']}", f"K={c['K']} beta={c['beta']}", c["n_cells"], "cells", "b15_enum")
ratios = [c["ratio"] for c in cfg]
OUT["a"] = dict(configs=cfg, n_config=len(cfg), n_windows=len({c["window"] for c in cfg}), n_oa=cfg[0]["n_oa"], K_set=sorted({c["K"] for c in cfg}),
                both_signs_verified=sum(c["both_signs_verified"] for c in cfg), both_signs_witnessed=sum(c["both_signs_witnessed"] for c in cfg),
                xlim_data=[min(c["verified"][0] for c in cfg), max(c["verified"][1] for c in cfg)],
                src=[REL["b15"], REL["b15_enum"]],
                note="window radius not stored (BFS windows of 30 OAs); fine-unit (OA) slope not stored in any canonical file -> tick omitted")

# kernel checks of the 96 Theorem-B1 charges (export log + Lean check log table)
exp = J("lean_export"); b1 = [e for e in exp if e["kind"] == "b1"]
log = open(SRC["lean_log"], encoding="utf-8").read()
kern = re.findall(r"^\| `(b1_w\d_K\d_beta\w+?_(?:lower|upper))` \| (PASS|FAIL) \|", log, flags=re.M)
eps_exact = [e for c in cfg for e in c["eps_exact"]]; eps_used = [e for c in cfg for e in c["eps_used"]]; eps_chk = [e for c in cfg for e in c["eps_checker"]]
OUT["b"] = dict(ratio=ratios, ratio_median=float(np.median(ratios)), ratio_min=float(min(ratios)), ratio_max=float(max(ratios)),
                eps_exact=eps_exact, eps_checker=eps_chk, eps_used=eps_used, eps_exact_max=float(max(eps_exact)),
                tol_exact=1e-9,   # stopping tolerance of the exact pricing (results.md B15 section: 'until the exact maximum reduced cost <= 10^-9')
                n_directions=len(eps_exact), kernel_pass=sum(1 for _, v in kern if v == "PASS"), kernel_n=len(kern),
                export_b1=len(b1), export_b1_artifact_ge_exact=sum(bool(e["artifact_ge_exact"]) for e in b1),
                checker_pass=int(chk["n_pass"]), checker_n=int(chk["n"]),
                checker_modes={m: sum(1 for c in cfg for x in c["checker_mode"] if x == m) for m in sorted({x for c in cfg for x in c["checker_mode"]})},
                src=[REL["b15"], REL["b15_check"], REL["lean_export"], REL["lean_log"]])
assert OUT["b"]["kernel_n"] == 96 and OUT["b"]["kernel_pass"] == 96 and OUT["b"]["export_b1"] == 96
for c in cfg:
    row("b", "width_ratio", f"window {c['window_label']}", f"K={c['K']} beta={c['beta']}", c["ratio"], "verified / witnessed width", "b15")
    for d_, nm in ((0, "lower"), (1, "upper")):
        row("b", "max_reduced_cost_exact", f"window {c['window_label']} {nm}", f"K={c['K']} beta={c['beta']}", c["eps_exact"][d_], "slope LP units", "b15")
        row("b", "checker_eps", f"window {c['window_label']} {nm}", f"K={c['K']} beta={c['beta']}", c["eps_checker"][d_], "slope LP units", "b15_check")
        row("b", "charged_eps", f"window {c['window_label']} {nm}", f"K={c['K']} beta={c['beta']}", c["eps_used"][d_], "slope LP units", "b15")

# ------------------------------------------------------------------------------------------------ c (A4) --
a4 = J("a4")["windows"]; pts = []; n_cls = 0; n_contain = 0; false_sign = 0; sign_exact = 0; sign_lp = 0
for w in a4:
    for K in (2, 3, 4):
        for b in ("inf", "0.5", "0.25"):
            key = f"K{K}_beta{b}"; cl = w["classes"].get(key)
            if not cl or cl.get("count", 0) == 0:
                continue
            sl = cl["slope"]; sp = w["slope_setpart"][key]
            if not (np.isfinite(sl[0]) and np.isfinite(sl[1]) and np.isfinite(sp[0]) and np.isfinite(sp[1])):
                continue
            n_cls += 1; n_contain += int(bool(cl["setpart_contains"]))
            sign_exact += int(cl["slope_sign_exact"] in ("neg", "pos")); sign_lp += int(cl["setpart_sign"] in ("neg", "pos"))
            false_sign += int(cl["setpart_sign"] in ("neg", "pos") and cl["setpart_sign"] != cl["slope_sign_exact"])
            for e, nm in ((0, "lower"), (1, "upper")):
                pts.append(dict(window=w["window"], K=K, beta=b, end=nm, exact=sl[e], lp=sp[e]))
                row("c", f"beta={b}", f"window {w['window'] + 1} K={K} {nm}", "exact vs LP", f"{sl[e]}|{sp[e]}", "slope (share per share)", "a4")
OUT["c"] = dict(points=pts, n_windows=len(a4), n_oa=int(a4[0]["n"]), n_classes=n_cls, n_endpoints=len(pts), n_contain=n_contain, false_signs=false_sign,
                sign_exact=sign_exact, sign_lp=sign_lp, lim=[min(min(p["exact"], p["lp"]) for p in pts), max(max(p["exact"], p["lp"]) for p in pts)],
                src=[REL["a4"]])

# ------------------------------------------------------------------------------------------------ d (C1) --
c1 = J("c1"); foc = J("c1_focal"); radii = ["3.8", "7.5", "15.0", "30.0"]
ras = np.load(SRC["raster"]); m = ras["mask"].astype(bool); n_tracts = int(len(np.unique(ras["tract_idx"][m][ras["tract_idx"][m] >= 0])))
D = dict(radii_km=[float(r) for r in radii], n_tracts=n_tracts, p=0.5, src=[REL["c1"], REL["c1_focal"], REL["raster"]])
for k in ("noiseless", "gaussian", "family"):
    D[f"not_hot_{k}"] = [c1["radii"][r]["fat"][k]["not_hot"] for r in radii]
    D[f"hot_{k}"] = [c1["radii"][r]["fat"][k]["hot"] for r in radii]
D["feasible"] = [c1["radii"][r]["fat"]["class_feasible_pop_share"] for r in radii]
D["feasible_focal_check"] = [foc[r]["feasible_and_available"] for r in radii]
assert np.allclose(D["feasible"], D["feasible_focal_check"])
D["bonferroni_not_hot"] = [c1["radii"][r]["fat"]["bonferroni"]["not_hot"] for r in radii]
OUT["d"] = D
for i, r in enumerate(radii):
    for k in ("noiseless", "gaussian", "family"):
        row("d", f"certified not-hot share, {k}", r, "population share", D[f"not_hot_{k}"][i], "share", "c1")
    row("d", "certified hot share, family", r, "population share", D["hot_family"][i], "share", "c1")
    row("d", "feasible population share", r, "population share", D["feasible"][i], "share", "c1")
    row("d", "Bonferroni not-hot share (family)", r, "population share", D["bonferroni_not_hot"][i], "share", "c1")

# ------------------------------------------------------------------------------------------------ e (B3) --
b3 = J("b3"); E = dict(radii_km=[7.5, 15.0], src=[REL["b3"]])
# variant used (exp2/results.md, B3 class audit): direct pair statistic with safe fractional endpoints
VAR = "direct_pair_fractional_endpoints"
for k in ("noiseless", "gaussian", "family"):
    E[f"pair_order_{k}"] = [b3["radii"][r][k]["C_pair_specific"][VAR]["ordered_pairs_share_jointfeasible"] for r in ("7.5", "15.0")]
    E[f"universe_pairs_{k}"] = [b3["radii"][r][k]["A_witness_anchored"][VAR]["ordered_pairs_share"] for r in ("7.5", "15.0")]
    E[f"rank_width_median_{k}"] = [b3["radii"][r][k]["A_witness_anchored"][VAR]["rank_width_median"] for r in ("7.5", "15.0")]
    E[f"not_top20_{k}"] = [b3["radii"][r][k]["A_witness_anchored"][VAR]["certified_not_top20_of_universe_pop_share"] for r in ("7.5", "15.0")]
E["n_named"] = [b3["radii"][r]["n_named"] for r in ("7.5", "15.0")]
E["universe_size"] = [b3["radii"][r]["witness_anchored_class"]["universe_size"] for r in ("7.5", "15.0")]
E["joint_feasible_pairs"] = [b3["radii"][r]["pair_classes"]["joint_feasible"] for r in ("7.5", "15.0")]
E["ordered_pairs"] = [b3["radii"][r]["pair_classes"]["ordered_pairs"] for r in ("7.5", "15.0")]
E["note"] = "rank-set widths stored only as medians (no per-location distribution) -> medians drawn, no box plots"
OUT["e"] = E
for i, r in enumerate(("7.5", "15.0")):
    for k in ("noiseless", "gaussian", "family"):
        row("e", f"certified pair order, {k}", r, "share of joint-feasible ordered pairs", E[f"pair_order_{k}"][i], "share", "b3")
        row("e", f"rank-set width median, {k}", r, "witness-anchored universe", E[f"rank_width_median_{k}"][i], "rank share", "b3")
        row("e", f"certified not-top-20 share, {k}", r, "witness-anchored universe", E[f"not_top20_{k}"][i], "population share", "b3")

# ------------------------------------------------------------------------------------------------ f (B7) --
b7 = J("b7"); jt = b7["joint"]; T = int(jt["T"])


def cp(x, n, a=0.05):
    lo = float(beta_dist.ppf(a / 2, x, n - x + 1)) if x > 0 else 0.0
    hi = float(beta_dist.ppf(1 - a / 2, x + 1, n - x)) if x < n else 1.0
    return lo, hi


x_rank = int(round(jt["coverage_rank"] * T)); x_bin = int(round(jt["coverage_conditional_k"] * T))
cpr = cp(x_rank, T); cpb = cp(x_bin, T)
assert abs(cpr[0] - jt["cp_lower_rank"]) < 1e-9 and abs(cpb[0] - jt["cp_lower_conditional_k"]) < 1e-9   # the stored CP lower bounds are reproduced
cd = b7["conditional"]; bf = b7["bonferroni"]
OUT["f"] = dict(T=T, queries=int(b7["queries"]), B=int(b7["B"]), k_rank=int(b7["k_rank"]), k_binomial=int(b7["k_conditional"]),
                joint_rank=dict(coverage=jt["coverage_rank"], cp=list(cpr), guarantee=jt["guarantee_rank"]),
                joint_binomial=dict(coverage=jt["coverage_conditional_k"], cp=list(cpb), guarantee=1 - b7["alpha0"]),
                cond_rank=dict(median=cd["rank_cond_coverage_median"], min=cd["rank_cond_coverage_min"], share_below=cd["share_calibrations_below_1_minus_alpha_rank"], level=1 - b7["alpha"]),
                cond_binomial=dict(median=cd["binomial_cond_coverage_median"], min=cd["binomial_cond_coverage_min"], share_below=cd["share_calibrations_below_1_minus_alpha0_binomial"], level=1 - b7["alpha0"]),
                R=int(cd["R"]), N=int(cd["N"]),
                half_width=dict(tailored=bf["half_width_median_tailored_rank"], bonf_gauss=bf["half_width_median_gauss"], bonf_family=bf["half_width_median_family"]),
                radius_km=15.0,   # r = 15 km: b7_familysup.py default --r_km 15 and results.md B7 section header ('fat class 15 km')
                src=[REL["b7"]], note="CP 95% interval recomputed from (coverage, T) of the file; the lower ends reproduce the stored cp_lower values")
F = OUT["f"]
row("f", "joint coverage, rank", "T=600", "coverage", F["joint_rank"]["coverage"], "probability", "b7", f"CP95 {cpr[0]:.4f}-{cpr[1]:.4f}")
row("f", "joint coverage, binomial", "T=600", "coverage", F["joint_binomial"]["coverage"], "probability", "b7", f"CP95 {cpb[0]:.4f}-{cpb[1]:.4f}")
for nm in ("cond_rank", "cond_binomial"):
    row("f", nm + " median", "R=20", "coverage", F[nm]["median"], "probability", "b7"); row("f", nm + " min", "R=20", "coverage", F[nm]["min"], "probability", "b7")
for nm, v in F["half_width"].items():
    row("f", "half-width median " + nm, "15 km", "half-width", v, "kfr units", "b7")

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(OUT, open(os.path.join(HERE, "data", "F4.json"), "w", encoding="utf-8"), indent=1, default=float)
with open(os.path.join(HERE, "F4_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys())); wr.writeheader(); wr.writerows(ROWS)
print(f"data/F4.json written; {len(ROWS)} source rows")
print(f"a: {OUT['a']['n_config']} configs, both signs verified {OUT['a']['both_signs_verified']}, witnessed {OUT['a']['both_signs_witnessed']}")
print(f"b: ratio median {OUT['b']['ratio_median']:.4f} [{OUT['b']['ratio_min']:.4f}, {OUT['b']['ratio_max']:.4f}]; eps max {OUT['b']['eps_exact_max']:.3e}; kernel {OUT['b']['kernel_pass']}/{OUT['b']['kernel_n']}; checker {OUT['b']['checker_pass']}/{OUT['b']['checker_n']}")
print(f"c: {OUT['c']['n_classes']} classes, {OUT['c']['n_endpoints']} endpoints, contain {OUT['c']['n_contain']}, false signs {OUT['c']['false_signs']}, lim {OUT['c']['lim']}")
print(f"d: tracts {n_tracts}; family not-hot {D['not_hot_family']}; hot family {D['hot_family']}")
print(f"e: pair order family {E['pair_order_family']}; not-top20 {[E['not_top20_' + k] for k in ('noiseless', 'gaussian', 'family')]}")
print(f"f: CP rank {cpr}, CP bin {cpb}")
