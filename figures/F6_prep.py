"""F6_prep - data layer for Fig. 6 (temporal pooling of a learned hourly risk).

Reads ONLY the pooling-transfer result files (paths relative to the results package):
  exp3_ml/summary_{mlp,feat,feat24}.json, exp3_ml/boxes_{model}.npz                      (synthetic feeders)
  exp3_ml_real/{jul2014,jan2014}/summary_{model}.json, boxes_{model}.npz, nominal_mlp.npz (electricity clients)
  exp3_ml_real/jul2014/s3_certify_mlp.json  (declared class constants; client-0 sampled choices / arg-max pair)
  exp3_ml_real/data_report.json             (client identifiers)
Provenance codes (comments only): S1a global range, S1b look-back range, S2 before/after sign, S3 binned trend, S4 top-3
ranking, S5 intra-day contrasts; arc set ZD = dyadic lags; V = value-only, Z = fused (transport LP).
Writes data/F6.json (every drawn number + its source path) and F6_source_data.csv (tidy).
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, sys, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FE = paths.RESULTS_ROOT
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
DS_DIR = {"synthetic": "exp3_ml", "jul2014": "exp3_ml_real/jul2014", "jan2014": "exp3_ml_real/jan2014"}
DATASETS = ["synthetic", "jul2014", "jan2014"]
MODELS = ["mlp", "feat", "feat24"]
EPS = [0.01, 0.02, 0.05, 0.1]
LAGS = [1, 2, 4, 8, 16, 32, 64, 128]
ROWS = []          # tidy source-data rows


def rel(p):
    return paths.rel(p)


def P(ds, name):
    return os.path.join(FE, DS_DIR[ds], name)


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def add(panel, dataset, model, quantity, k1, k2, value, src):
    ROWS.append(dict(panel=panel, dataset=dataset, model=model, quantity=quantity, key1=k1, key2=k2, value=value, source=rel(src)))


SUM = {(ds, m): jload(P(ds, f"summary_{m}.json")) for ds in DATASETS for m in MODELS}

# ============================================================== a: one real series and the six window classes
CL = 0                                     # client index 0 of 12 (largest mean training load)
H0, H1 = 168, 504                          # 14 days: evaluation hours [168, 504) = 2014-07-08 00:00 .. 2014-07-21 23:00
f_nom = np.load(P("jul2014", "nominal_mlp.npz"))["f"].astype(np.float64)
Bj = np.load(P("jul2014", "boxes_mlp.npz"))
lo, hi = Bj["lo|CROWN|0.01"], Bj["hi|CROWN|0.01"]
rep = jload(os.path.join(FE, "exp3_ml_real", "data_report.json"))
C3 = jload(P("jul2014", "s3_certify_mlp.json"))
decl = C3["declared"]
pf0 = C3["configs"]["CROWN|0.01|ZD"]["per_feeder"][CL]
t = np.arange(H0, H1)
src_nom, src_box, src_s3 = P("jul2014", "nominal_mlp.npz"), P("jul2014", "boxes_mlp.npz"), P("jul2014", "s3_certify_mlp.json")
for i in t:
    add("a", "jul2014", "mlp", "nominal_f", int(i), "", float(f_nom[CL, i]), src_nom)
    add("a", "jul2014", "mlp", "CROWN_lo_eps0.01", int(i), "", float(lo[CL, i]), src_box)
    add("a", "jul2014", "mlp", "CROWN_hi_eps0.01", int(i), "", float(hi[CL, i]), src_box)


def first_fit(choices, ok):
    for c in choices:
        if ok(c):
            return [int(x) for x in c]
    raise ValueError("no sampled choice fits the stretch")


inst = {S: {r["inst"]: r for r in pf0[S]} for S in ("S2", "S3", "S5")}
lg_min, lg_max = decl["LGRID"]
# look-back range (S1b): report time e = 480 (2014-07-21 00:00); certified arg-max pair of look-back lengths at e
s1b_row = [x for x in pf0["S1b"] if x["e"] == 480][0]
s1b_L = sorted({lg_min, lg_max, *s1b_row["Z_argmax_pair"]})
# before/after sign (S2): instance tau0 = 336; the first sampled choice (tau, L) whose two blocks lie inside the stretch
tau0 = 336; s2c = first_fit(inst["S2"][tau0]["choices_sample"], lambda c: c[0] - c[1] >= H0 and c[0] + c[1] <= H1)
# binned trend (S3): instance base 168; the first sampled choice (s0, K) with K >= 14 (several bins visible)
s3c = first_fit(inst["S3"][168]["choices_sample"], lambda c: c[1] >= 14)
s3_bin = decl["S3_SPAN"] // s3c[1]
s3_bins = [[s3c[0] + k * s3_bin, s3c[0] + (k + 1) * s3_bin] for k in range(s3c[1]) if s3c[0] + k * s3_bin < H1]
# intra-day contrasts (S5): week 1 = hours [168, 336); the first sampled choice (w, h0, b) with b >= 2
s5c = first_fit(inst["S5"][1]["choices_sample"], lambda c: c[2] >= 2)
w, h0, b = s5c
s5_pairs = [[168 * w + 24 * d + h0 - b, 168 * w + 24 * d + h0, 168 * w + 24 * d + h0 + b] for d in range(7)]
# top-3 ranking (S4): report time e = 408 (2014-07-18 00:00); common look-back L in [24, 168]
s4_e = 408
DAY0 = 1                                   # hour 0 of the window = 2014-07-01 00:00, i.e. July day 1
a = dict(client_index=CL, client_id=rep["chosen"][CL], window="jul2014", model="mlp", method="CROWN", eps=0.01,
         hour_start=H0, hour_end=H1, date_start="2014-07-08 00:00", date_end="2014-07-21 23:00", t0_date="2014-07-01 00:00",
         t=t.tolist(), f=f_nom[CL, t].tolist(), lo=lo[CL, t].tolist(), hi=hi[CL, t].tolist(),
         box_width_median=float(np.median(hi[CL, t] - lo[CL, t])),
         xticks=[h for h in range(H0, H1 + 1, 72)], xticklabels=[str(DAY0 + h // 24) for h in range(H0, H1 + 1, 72)],
         classes=dict(
             S1a=dict(span=[H0, H1], L_range=[lg_min, lg_max]),
             S1b=dict(e=480, L=s1b_L, argmax_pair=s1b_row["Z_argmax_pair"], settled=s1b_row["Z_settled"]),
             S2=dict(tau0=tau0, tau_range=[tau0 + decl["S2_DT"][0], tau0 + decl["S2_DT"][1]], tau=s2c[0], L=s2c[1],
                     blocks=[[s2c[0] - s2c[1], s2c[0]], [s2c[0], s2c[0] + s2c[1]]]),
             S3=dict(base=168, s0=s3c[0], K=s3c[1], bin_h=s3_bin, bins=s3_bins),
             S4=dict(e=s4_e, L_range=[lg_min, lg_max], top=3, n_clients=12),
             S5=dict(week=w, h0=h0, b=b, pairs=s5_pairs)),
         source=dict(series=rel(src_nom), boxes=rel(src_box), classes=rel(src_s3), client=rel(os.path.join(FE, "exp3_ml_real", "data_report.json"))))
for k, v in a["classes"].items():
    add("a", "jul2014", "mlp", f"class_{k}", "", "", json.dumps(v), src_s3)

# ============================================================== b: verifier widths vs eps and kappa_k vs lag
b = dict(eps=EPS, lags=LAGS, widths={}, kappa={}, n_series=int(lo.shape[0]), n_hours=int(lo.shape[1]))
for ds in DATASETS:
    b["widths"][ds] = {}; b["kappa"][ds] = {}
    for m in MODELS:
        S = SUM[(ds, m)]; Bx = np.load(P(ds, f"boxes_{m}.npz")); sp, bp = P(ds, f"summary_{m}.json"), P(ds, f"boxes_{m}.npz")
        W = {}
        for meth in ("IBP", "CROWN", "alpha-CROWN"):
            if meth not in S["verifier"]["0.01"]:
                continue
            med, q25, q75 = [], [], []
            for e in EPS:
                wv = Bx[f"hi|{meth}|{e}"] - Bx[f"lo|{meth}|{e}"]
                ms = S["verifier"][str(e)][meth]["value_width_median"]
                assert abs(float(np.median(wv)) - ms) < 1e-12 * max(1.0, ms), (ds, m, meth, e)
                med.append(ms); q25.append(float(np.quantile(wv, 0.25))); q75.append(float(np.quantile(wv, 0.75)))
                add("b", ds, m, f"value_width_median|{meth}", e, "", ms, sp)
                add("b", ds, m, f"value_width_q25|{meth}", e, "", q25[-1], bp); add("b", ds, m, f"value_width_q75|{meth}", e, "", q75[-1], bp)
            W[meth] = dict(median=med, q25=q25, q75=q75)
        med, q25, q75 = [], [], []
        for e in EPS:
            wi = Bx[f"hi_in|{e}"] - Bx[f"lo_in|{e}"]; ms = S["verifier"][str(e)]["inner_value_width_median"]
            assert abs(float(np.median(wi)) - ms) < 1e-12 * max(1.0, ms)
            med.append(ms); q25.append(float(np.quantile(wi, 0.25))); q75.append(float(np.quantile(wi, 0.75)))
            add("b", ds, m, "value_width_median|PGD inner", e, "", ms, sp)
            add("b", ds, m, "value_width_q25|PGD inner", e, "", q25[-1], bp); add("b", ds, m, "value_width_q75|PGD inner", e, "", q75[-1], bp)
        W["inner"] = dict(median=med, q25=q25, q75=q75)
        b["widths"][ds][m] = W
        # kappa_k (CROWN, eps = 0.01) = median over outputs of (increment-box width at lag k) / (value width), as in s2_verify.stats
        vw = Bx["hi|CROWN|0.01"] - Bx["lo|CROWN|0.01"]; km, k25, k75 = [], [], []
        for k in LAGS:
            r = (Bx[f"ghi|CROWN|0.01|{k}"] - Bx[f"glo|CROWN|0.01|{k}"]) / np.maximum(vw[:, :-k], 1e-12)
            ms = S["verifier"]["0.01"]["CROWN"]["kappa_median"][str(k)]
            assert abs(float(np.median(r)) - ms) < 1e-9, (ds, m, k)
            km.append(ms); k25.append(float(np.quantile(r, 0.25))); k75.append(float(np.quantile(r, 0.75)))
            add("b", ds, m, "kappa_median|CROWN|eps0.01", k, "", ms, sp)
            add("b", ds, m, "kappa_q25|CROWN|eps0.01", k, "", k25[-1], bp); add("b", ds, m, "kappa_q75|CROWN|eps0.01", k, "", k75[-1], bp)
        b["kappa"][ds][m] = dict(median=km, q25=k25, q75=k75)
b["source"] = "summary_{model}.json -> verifier[eps][method] (medians); boxes_{model}.npz (IQR, same per-output definition)"

# ============================================================== c: normalized saving G by window class (CROWN, eps 0.01, ZD)
TAGS = {"S1b": "S1b_top", "S2": "S2_sample", "S3": "S3_sample", "S5": "S5_sample"}
c = dict(classes=list(TAGS), G={}, config="CROWN|0.01|ZD")
for ds in DATASETS:
    c["G"][ds] = {}
    for m in MODELS:
        g = SUM[(ds, m)]["configs"]["CROWN|0.01|ZD"]["G"]; sp = P(ds, f"summary_{m}.json"); c["G"][ds][m] = {}
        for cl, tag in TAGS.items():
            c["G"][ds][m][cl] = dict(median=g[tag]["G_median"], q90=g[tag]["G_q90"], n=g[tag]["n"])
            add("c", ds, m, "G_median", cl, "CROWN|0.01|ZD", g[tag]["G_median"], sp); add("c", ds, m, "G_q90", cl, "CROWN|0.01|ZD", g[tag]["G_q90"], sp)
            add("c", ds, m, "G_n", cl, "CROWN|0.01|ZD", g[tag]["n"], sp)

# ============================================================== d: outer/inner excess value-only -> fused
d = dict(classes=["S5", "S2"], excess={}, config="CROWN|0.01|ZD")
for ds in DATASETS:
    d["excess"][ds] = {}
    for m in MODELS:
        oi = SUM[(ds, m)]["configs"]["CROWN|0.01|ZD"]["outer_over_inner_excess"]; sp = P(ds, f"summary_{m}.json"); d["excess"][ds][m] = {}
        for cl in d["classes"]:
            d["excess"][ds][m][cl] = dict(V=oi[cl]["V"], Z=oi[cl]["Z"], n=oi[cl]["n"])
            for bt in ("V", "Z"):
                for q in ("median", "q10", "q90"):
                    add("d", ds, m, f"excess_{bt}_{q}", cl, "CROWN|0.01|ZD", oi[cl][bt][q], sp)
            add("d", ds, m, "excess_n", cl, "CROWN|0.01|ZD", oi[cl]["n"], sp)

# ============================================================== e: decision certificates (counts of instances)
CAT = {"invariant": ["certified_invariant"], "sensitive": ["certified_sensitive"],
       "witnessed": ["witnessed_sensitive_nominal", "witnessed_flip_perturbation"], "open": ["open"]}
e_ = dict(eps=[0.01, 0.02], classes=["S2", "S3", "S5"], categories=list(CAT), counts={}, raw={})
for ds in DATASETS:
    e_["counts"][ds] = {}; e_["raw"][ds] = {}
    for m in MODELS:
        e_["counts"][ds][m] = {}; e_["raw"][ds][m] = {}; sp = P(ds, f"summary_{m}.json")
        for ep in e_["eps"]:
            dec = SUM[(ds, m)]["configs"][f"CROWN|{ep}|ZD"]["decisions"]; e_["counts"][ds][m][str(ep)] = {}; e_["raw"][ds][m][str(ep)] = {}
            for cl in e_["classes"]:
                e_["counts"][ds][m][str(ep)][cl] = {}; e_["raw"][ds][m][str(ep)][cl] = {}
                for bt in ("V", "Z"):
                    cnt = dec[cl][bt]; row = {k: int(sum(cnt[x] for x in v)) for k, v in CAT.items()}; row["n"] = int(cnt["n"])
                    assert sum(row[k] for k in CAT) == row["n"]
                    e_["counts"][ds][m][str(ep)][cl][bt] = row; e_["raw"][ds][m][str(ep)][cl][bt] = cnt
                    for k, v in cnt.items():
                        add("e", ds, m, f"decision_{bt}_{k}", cl, f"CROWN|{ep}|ZD", v, sp)

# ============================================================== f: top-3 ranking across report times (CROWN, eps 0.01; ZD = V here)
f_ = dict(config="CROWN|0.01|ZD", S4={}, n_report=28, n_clients=12, n_L=145)
for ds in DATASETS:
    f_["S4"][ds] = {}
    for m in MODELS:
        s4 = sorted(SUM[(ds, m)]["configs"]["CROWN|0.01|ZD"]["S4"], key=lambda r: r["e"]); sp = P(ds, f"summary_{m}.json")
        rows = []
        for r in s4:
            # certified invariant: nominal top-3 beats every other client for all L and all perturbations;
            # certified sensitive: >= 2 distinct top-3 sets each certified at some L; witnessed: the nominal top-3 changes with L
            st = ("invariant" if r["top3_certified_invariant"] else "sensitive" if r["distinct_certified_top3_sets"] > 1
                  else "witnessed" if r["nominal_distinct_top3"] > 1 else "open")
            assert not (r["top3_certified_invariant"] and (r["distinct_certified_top3_sets"] > 1 or r["nominal_distinct_top3"] > 1))
            rows.append(dict(e=r["e"], day=r["e"] // 24, state=st, distinct_certified_top3_sets=r["distinct_certified_top3_sets"],
                             nominal_distinct_top3=r["nominal_distinct_top3"],
                             n_L_with_certified_top3=r["n_L_with_certified_top3"], certified_pairs=r["certified_pairs"]))
            add("f", ds, m, "S4_state", r["e"], "CROWN|0.01|ZD", st, sp)
            add("f", ds, m, "S4_distinct_certified_top3_sets", r["e"], "CROWN|0.01|ZD", r["distinct_certified_top3_sets"], sp)
        f_["S4"][ds][m] = dict(rows=rows, **{f"n_{k}": sum(x["state"] == k for x in rows) for k in ("invariant", "sensitive", "witnessed", "open")})

OUT = dict(stem="F6", note="Fig. 6 data; every number from the pooling-transfer result files (paths in 'sources' and in F6_source_data.csv)",
           datasets=DATASETS, models=MODELS,
           sources={ds: {"summary": rel(P(ds, "summary_{model}.json")), "boxes": rel(P(ds, "boxes_{model}.npz"))} for ds in DATASETS},
           a=a, b=b, c=c, d=d, e=e_, f=f_,
           moved_to_ED3="panel g (rigorous enclosure vs the verifier) -> Extended Data Fig. 3f (225 mm height limit)")
json.dump(OUT, open(os.path.join(DATA, "F6.json"), "w", encoding="utf-8"), indent=1)
with open(os.path.join(HERE, "F6_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(ROWS[0])); wr.writeheader(); wr.writerows(ROWS)
print("F6.json written;", len(ROWS), "source rows; client", a["client_id"], "S1b L", s1b_L, "S2", s2c, "S3", s3c, "S5", s5c)
