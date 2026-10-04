"""F3_prep - data layer of Fig. 3 (transport fusion and realization mechanics).

Reads ONLY the result files listed in the source table of the data file and writes data/F3.json (every drawn number with
its source path) and F3_source_data.csv (tidy). Experiment codes appear only in these comments / recorded paths.

Sources (results package unless stated otherwise; data/... = data package):
  a  exp2/B12/b11_georgia_kvkg.json -> by_k[v{kv}g{kg}]: U_over_V_median, _q10, _q90 over 400 nested pairs (no IQR stored).
  b  exp2/B2_surface/b2s_{georgia,gm_q4}_{remove,swap}.json -> rows (delta, width_X1, tube_bound_width, ...); the plotted
     quantity is width_X1 / tube_bound_width per displaced cell (experiments/b2_surface.py: tube bound = value-only bound
     on the displaced strip, Theorem S18 (i) form; X1 = lifted outer X^(1) with verified duals).
  c  exp2/B14/b14_conditioned.json -> rows (truth, m1/m2: U/L_cond_outer, U/L_cond_inner); exp2/B14/b14_tent.json -> rows.
  d  exp1/A2/a2_real_{field}.json (inner Q_m witnesses m = 1, 2; verified pixel outer U/L_PhiC_ver) and
     exp1/A2/a2_real_{field}_verified_m16.json (witnesses m = 4, 8, 16; certified outer U/L_cert{m} = min{pixel outer,
     verified inner-LP dual + eps_m}, checked here against the stored components); exp1/A2/a2_exact1d.json summary.
  e  exp1/B11/b11_saving_georgia_kvkg.json -> per budget 'pairs' (mass_benefiting_share, normalized_saving, identity_residual).
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FE = paths.RESULTS_ROOT
P = dict(b12=os.path.join(FE, "exp2", "B12", "b11_georgia_kvkg.json"),
         b2s=os.path.join(FE, "exp2", "B2_surface", "b2s_{}_{}.json"),
         b14=os.path.join(FE, "exp2", "B14", "b14_conditioned.json"),
         tent=os.path.join(FE, "exp2", "B14", "b14_tent.json"),
         a2=os.path.join(FE, "exp1", "A2", "a2_real_{}.json"),
         a2v=os.path.join(FE, "exp1", "A2", "a2_real_{}_verified_m16.json"),
         exact1d=os.path.join(FE, "exp1", "A2", "a2_exact1d.json"),
         b11=os.path.join(FE, "exp1", "B11", "b11_saving_georgia_kvkg.json"))
REL = {k: paths.rel(v) for k, v in P.items()}
SETTINGS = ["georgia", "gm_q4", "gm_bad", "mx_rwi"]
OUT = {"_note": "every number drawn in Fig. 3; 'src' gives the source file (results/... = results package, data/... = data package)"}
ROWS = []


def row(panel, element, series="", x="", y="", value="", extra=""):
    ROWS.append(dict(panel=panel, element=element, series=series, x=x, y=y, value=value, extra=extra))


# ================================================================================= panel a ==
B12 = json.load(open(P["b12"], encoding="utf-8"))["by_k"]
KV, KG = [1, 4, 16, 64], [1, 4, 16]
cells = {}
for kv in KV:
    for kg in KG:
        key = f"v{kv}g{kg}"
        if key in B12:
            S = B12[key]; cells[key] = dict(kv=kv, kg=kg, median=S["U_over_V_median"], q10=S["U_over_V_q10"], q90=S["U_over_V_q90"], pairs=S["pairs"],
                                            share_below_095=S["share_U_below_V_5pct"], verified_ok=S["verified_ok"])
            row("a", "U/V", key, kg, kv, S["U_over_V_median"], f"q10 {S['U_over_V_q10']} q90 {S['U_over_V_q90']} n {S['pairs']}")
        else:
            row("a", "U/V", key, kg, kv, "", "not run")
best = min(cells, key=lambda k: cells[k]["median"])
OUT["a"] = dict(src=REL["b12"], kv=KV, kg=KG, cells=cells, best=best, iqr_note="the file stores median, q10 and q90 (no quartiles); q10-q90 is printed")

# ================================================================================= panel b ==
b = dict(src=[REL["b2s"].format(s, m) for s in ("georgia", "gm_q4") for m in ("remove", "swap")], series={})
for s in ("georgia", "gm_q4"):
    for m in ("remove", "swap"):
        R = json.load(open(P["b2s"].format(s, m), encoding="utf-8"))["rows"]; ser = {}
        for dl in sorted({r["delta"] for r in R}):
            rr = [r for r in R if r["delta"] == dl]; v = np.array([r["width_X1"] / r["tube_bound_width"] for r in rr])
            ser[str(dl)] = dict(delta=int(dl), mean=float(v.mean()), sd=float(v.std(ddof=1)), n=int(len(v)), cells=int(len({r["coarse"] for r in rr})),
                                min=float(v.min()), max=float(v.max()))
            for r in rr:
                row("b", "X1 width / tube bound", f"{s} {m}", dl, r["coarse"], r["width_X1"] / r["tube_bound_width"])
        b["series"][f"{s}|{m}"] = ser
OUT["b"] = b

# ================================================================================= panel c ==
C = json.load(open(P["b14"], encoding="utf-8")); T = json.load(open(P["tent"], encoding="utf-8"))
cr = []
for r in C["rows"]:
    e = dict(t=r["t"], K=int(r["K"]), n_pix=r["n_pix"], truth=r["truth"], unit_sd=r["unit_sd"])
    for m in ("m1", "m2"):
        e[m] = {k: r[m][k] for k in ("U_cond_outer", "L_cond_outer", "U_cond_inner", "L_cond_inner", "truth_inside_cond_outer", "inner_inside_outer", "inner_feasible")}
        for k in ("U_cond_outer", "L_cond_outer", "U_cond_inner", "L_cond_inner"):
            row("c", "conditioned bound", f"{m} {k}", r["truth"], r["t"], r[m][k])
    cr.append(e)
tent = [dict(m=r["m"], inner_deficit=r["inner_deficit"], max_mean_outer=r["max_mean_outer"], max_mean_inner=r["max_mean_inner"], nodes=r["nodes"]) for r in T["rows"]]
for r in tent:
    row("c", "tent inner deficit", "tent", r["m"], "", r["inner_deficit"])
Ks = sorted({e["K"] for e in cr})
OUT["c"] = dict(src=[REL["b14"], REL["tent"]], rows=cr, n=len(cr), K_min=Ks[0], K_max=Ks[-1], summary=C["summary"], tent=tent, tent_m_star=T["m_star"])

# ================================================================================= panel d ==
d = dict(src=[REL["a2"].format("*"), REL["a2v"].format("*"), REL["exact1d"]], ms=[1, 2, 4, 8, 16], cert_ms=[4, 8, 16], settings={})
for s in SETTINGS:
    A0 = json.load(open(P["a2"].format(s), encoding="utf-8"))["rows"]; V = json.load(open(P["a2v"].format(s), encoding="utf-8"))["rows"]
    assert [r["label"] for r in A0] == [r["label"] for r in V]
    seen, idx = set(), []
    for i, r in enumerate(V):
        if r["label"] not in seen:
            seen.add(r["label"]); idx.append(i)
    wit = {}
    for m in d["ms"]:
        src = A0 if m in (1, 2) else V
        wit[m] = np.array([src[i][f"U_in{m}"] - src[i][f"L_in{m}"] for i in idx])
    phi = np.array([V[i]["U_PhiC_ver"] - V[i]["L_PhiC_ver"] for i in idx])
    ent = dict(rows=len(V), unique=len(idx), pixel={}, cert={})
    for m in d["ms"]:
        ex = phi / wit[m] - 1.0
        ent["pixel"][str(m)] = dict(median=float(np.median(ex)), min=float(ex.min()), max=float(ex.max()))
        for i, v in zip(idx, ex):
            row("d", "excess over witness, pixel outer", s, m, V[i]["label"], float(v))
    for m in d["cert_ms"]:
        # stored certified outer = min{pixel outer, verified inner-LP dual + eps_m}; check against its components
        chk = max(max(abs(V[i][f"U_cert{m}"] - min(V[i]["U_PhiC_ver"], V[i][f"U_in{m}_lpver"] + V[i][f"eps{m}"])),
                      abs(V[i][f"L_cert{m}"] - max(V[i]["L_PhiC_ver"], V[i][f"L_in{m}_lpver"] - V[i][f"eps{m}"]))) for i in idx)
        assert chk < 1e-9, (s, m, chk); ent.setdefault("cert_component_check_max", {})[str(m)] = float(chk)
        cw = np.array([V[i][f"U_cert{m}"] - V[i][f"L_cert{m}"] for i in idx]); ex = cw / wit[m] - 1.0
        ent["cert"][str(m)] = dict(median=float(np.median(ex)), min=float(ex.min()), max=float(ex.max()))
        for i, v in zip(idx, ex):
            row("d", "excess over witness, certified bracket", s, m, V[i]["label"], float(v))
    d["settings"][s] = ent
E1 = json.load(open(P["exact1d"], encoding="utf-8"))["summary"]
d["exact1d"] = dict(instances=int(E1["instances"]), max_err_UPhi=float(E1["max_err_UPhi"]))
row("d", "exact 1-D theorem", "max |LP - closed form|", "", "", E1["max_err_UPhi"], f"{E1['instances']} instances")
OUT["d"] = d

# ================================================================================= panel e ==
B11 = json.load(open(P["b11"], encoding="utf-8"))
e = dict(src=REL["b11"], focus="1:1", budgets={}, residual_max={})
for bud, blk in B11.items():
    pr = blk["pairs"]
    e["budgets"][bud] = dict(mass=[float(p["mass_benefiting_share"]) for p in pr], G=[float(p["normalized_saving"]) for p in pr], n=len(pr))
    e["residual_max"][bud] = float(max(p["identity_residual"] for p in pr))
    for p in pr:
        row("e", "saving identity", bud, p["mass_benefiting_share"], p["normalized_saving"], p["identity_residual"], f"pair {p['coarse']}/{p['fine']}")
e["residual_max_focus"] = e["residual_max"]["1:1"]; e["residual_max_all"] = max(e["residual_max"].values())
allm = np.concatenate([e["budgets"][k]["mass"] for k in e["budgets"]]); allg = np.concatenate([e["budgets"][k]["G"] for k in e["budgets"]])
e["G_le_mass_all"] = bool(np.all(allg <= allm + 1e-12)); e["min_positive_mass"] = float(allm[allm > 0].min()); e["min_positive_G"] = float(allg[allg > 0].min())
e["zero_G_pairs"] = int(np.sum(allg <= 0)); e["zero_mass_pairs"] = int(np.sum(allm <= 0))
OUT["e"] = e

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(OUT, open(os.path.join(HERE, "data", "F3.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(os.path.join(HERE, "F3_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["panel", "element", "series", "x", "y", "value", "extra"]); w.writeheader(); w.writerows(ROWS)
print("F3 data written:", len(ROWS), "rows; best cell", best, cells[best]["median"])
print("e residual 1:1", e["residual_max_focus"], "all", e["residual_max_all"], "G<=mass", e["G_le_mass_all"], "zeros G/mass", e["zero_G_pairs"], e["zero_mass_pairs"], e["min_positive_G"], e["min_positive_mass"])
print("d unique", {s: d["settings"][s]["unique"] for s in SETTINGS}, "exact1d", d["exact1d"])
print("c K range", Ks, "tent", [(r["m"], r["inner_deficit"]) for r in tent])
