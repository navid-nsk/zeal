"""ED3_prep - data layer for Extended Data Fig. 3 (temporal pooling: models, verifiers, search cost, exactness, rigorous enclosure).

Reads ONLY the pooling-transfer result files (paths relative to the results package):
  exp3_ml/s1_train_mlp_mlp_tc_feat_mlp7_feat7.json, exp3_ml/s1_train_feat24.json, exp3_ml_real/s1_train_real.json,
  exp3_ml_real/data_report.json                                         (a: AUC, label rates)
  {exp3_ml, exp3_ml_real/jul2014, exp3_ml_real/jan2014}/summary_{model}.json, boxes_{model}.npz   (b, c, d, e)
  {...}/s3_certify_{model}.json                                         (d: per-case look-back LP counts)
  {...}/closure_{model}_{method}_{eps}_{arcs}.npz                       (e: closure u*, l* against the boxes)
  exp3_ml/rigorous/{r1_summary.json, lean_results.json, rigorous_boxes_{model}_{eps}.npz},
  exp3_ml_real/rigorous/{r1_summary_{w}.json, lean_results_{w}.json, rigorous_boxes_{w}_{model}_{eps}.npz}   (f)
Provenance codes (comments only): S1b = look-back range class; Z1 / ZD = lag-1 / dyadic arc sets.
Writes data/ED3.json and ED3_source_data.csv.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, json, csv, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FE = paths.RESULTS_ROOT
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
DS_DIR = {"synthetic": "exp3_ml", "jul2014": "exp3_ml_real/jul2014", "jan2014": "exp3_ml_real/jan2014"}
DATASETS = ["synthetic", "jul2014", "jan2014"]; MODELS = ["mlp", "feat", "feat24"]; EPS = [0.01, 0.02, 0.05, 0.1]
LAGS = [1, 2, 4, 8, 16, 32, 64, 128]; MARGIN = 1e-9          # outward margin of the stored verifier boxes (r1_summary 'declared')
ROWS = []


def rel(p):
    return paths.rel(p)


def P(ds, name):
    return os.path.join(FE, DS_DIR[ds], name)


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def add(panel, dataset, model, quantity, k1, k2, value, src):
    ROWS.append(dict(panel=panel, dataset=dataset, model=model, quantity=quantity, key1=k1, key2=k2, value=value, source=rel(src)))


SUM = {(ds, m): jload(P(ds, f"summary_{m}.json")) for ds in DATASETS for m in MODELS}

# ============================================================== a: model quality
fs1 = os.path.join(FE, "exp3_ml", "s1_train_mlp_mlp_tc_feat_mlp7_feat7.json"); fs2 = os.path.join(FE, "exp3_ml", "s1_train_feat24.json")
fr = os.path.join(FE, "exp3_ml_real", "s1_train_real.json"); fdr = os.path.join(FE, "exp3_ml_real", "data_report.json")
S1, S2, SR, DR = jload(fs1), jload(fs2), jload(fr), jload(fdr)
a = dict(auc={}, label_rate={}, source=dict(synthetic=[rel(fs1), rel(fs2)], real=rel(fr), clients=rel(fdr)))
for m in MODELS:
    rs = (S2 if m == "feat24" else S1)[m]; src = fs2 if m == "feat24" else fs1
    a["auc"].setdefault("synthetic", {})[m] = dict(eval=rs["eval_auc"], early_stop=rs["val_auc"])
    add("a", "synthetic", m, "eval_auc", "", "", rs["eval_auc"], src); add("a", "synthetic", m, "early_stop_auc", "", "", rs["val_auc"], src)
    for w in ("jul2014", "jan2014"):
        a["auc"].setdefault(w, {})[m] = dict(eval=SR[m][w]["eval_auc"], early_stop=SR[m]["val_auc"])
        add("a", w, m, "eval_auc", "", "", SR[m][w]["eval_auc"], fr); add("a", w, m, "early_stop_auc", "", "", SR[m]["val_auc"], fr)
dh = S1["data_h24"]
a["label_rate"] = dict(synthetic=dict(train=dh["pos_rate_train"], early_stop=dh["pos_rate_val"], eval=dh["pos_rate_eval"]),
                       jul2014=dict(train=SR["pos_rate_train"], early_stop=SR["pos_rate_val"], eval=SR["pos_rate_jul2014"]),
                       jan2014=dict(train=SR["pos_rate_train"], early_stop=SR["pos_rate_val"], eval=SR["pos_rate_jan2014"]))
for ds, v in a["label_rate"].items():
    for k, x in v.items():
        add("a", ds, "", f"label_rate_{k}", "", "", x, fs1 if ds == "synthetic" else fr)
a["n_clients_raw"], a["n_eligible"], a["n_chosen"] = DR["n_clients"], DR["n_eligible"], len(DR["chosen"])

# ============================================================== b: verifier box / PGD-inner width (value, lag-1 increment)
b = dict(eps=EPS, ratio={}, methods=["IBP", "CROWN", "alpha-CROWN"], n_series=None, n_hours=None)
for ds in DATASETS:
    b["ratio"][ds] = {}
    for m in MODELS:
        S = SUM[(ds, m)]; Bx = np.load(P(ds, f"boxes_{m}.npz")); sp, bp = P(ds, f"summary_{m}.json"), P(ds, f"boxes_{m}.npz"); R = {}
        b["n_series"], b["n_hours"] = [int(x) for x in Bx["lo|CROWN|0.01"].shape]
        for meth in b["methods"]:
            if meth not in S["verifier"]["0.01"]:
                continue
            R[meth] = {"value": dict(median=[], q25=[], q75=[]), "incr1": dict(median=[], q25=[], q75=[])}
            for e in EPS:
                ck = S["verifier"][str(e)][meth]["check"]
                vin = np.maximum(Bx[f"hi_in|{e}"] - Bx[f"lo_in|{e}"], 1e-12); gin = np.maximum(Bx[f"ghi_in|{e}|1"] - Bx[f"glo_in|{e}|1"], 1e-12)
                rv = (Bx[f"hi|{meth}|{e}"] - Bx[f"lo|{meth}|{e}"]) / vin; rg = (Bx[f"ghi|{meth}|{e}|1"] - Bx[f"glo|{meth}|{e}|1"]) / gin
                for kind, arr, ms in (("value", rv, ck["value_width_ratio_median"]), ("incr1", rg, ck["incr1_width_ratio_median"])):
                    assert abs(float(np.median(arr)) - ms) < 1e-9 * max(1.0, ms), (ds, m, meth, e, kind)
                    R[meth][kind]["median"].append(ms); R[meth][kind]["q25"].append(float(np.quantile(arr, 0.25))); R[meth][kind]["q75"].append(float(np.quantile(arr, 0.75)))
                    add("b", ds, m, f"{kind}_box_over_inner_median|{meth}", e, "", ms, sp)
                    add("b", ds, m, f"{kind}_box_over_inner_q25|{meth}", e, "", R[meth][kind]["q25"][-1], bp)
                    add("b", ds, m, f"{kind}_box_over_inner_q75|{meth}", e, "", R[meth][kind]["q75"][-1], bp)
        b["ratio"][ds][m] = R

# ============================================================== c: kappa_k heat maps (CROWN, eps 0.01 and 0.1)
c = dict(eps=[0.01, 0.1], lags=LAGS, kappa={})
for ds in DATASETS:
    c["kappa"][ds] = {}
    for m in MODELS:
        S = SUM[(ds, m)]; c["kappa"][ds][m] = {}
        for e in c["eps"]:
            km = S["verifier"][str(e)]["CROWN"]["kappa_median"]; c["kappa"][ds][m][str(e)] = [km[str(k)] for k in LAGS]
            for k in LAGS:
                add("c", ds, m, "kappa_median|CROWN", e, k, km[str(k)], P(ds, f"summary_{m}.json"))

# ============================================================== d: look-back search (lazy LP), CROWN ZD
d = dict(eps=EPS, lazy_cap=None, search={})
for ds in DATASETS:
    d["search"][ds] = {}
    for m in MODELS:
        C3 = jload(P(ds, f"s3_certify_{m}.json")); d["lazy_cap"] = C3["declared"]["LAZY_CAP"]; out = dict(settled=[], n=[], lp_median=[], lp_q10=[], lp_q90=[], n_lp=[], lp_seconds=[])
        for e in EPS:
            s1b = SUM[(ds, m)]["configs"][f"CROWN|{e}|ZD"]["S1b"]; cfg = SUM[(ds, m)]["configs"][f"CROWN|{e}|ZD"]
            nl = np.array([x["n_lp"] for r in C3["configs"][f"CROWN|{e}|ZD"]["per_feeder"] for x in r["S1b"]], float)
            assert len(nl) == s1b["n"] and float(np.median(nl)) == s1b["lp_median"], (ds, m, e)
            out["settled"].append(s1b["settled"]); out["n"].append(s1b["n"]); out["lp_median"].append(s1b["lp_median"])
            out["lp_q10"].append(float(np.quantile(nl, 0.1))); out["lp_q90"].append(float(np.quantile(nl, 0.9)))
            out["n_lp"].append(cfg["n_lp"]); out["lp_seconds"].append(cfg["lp_seconds"])
            sp = P(ds, f"summary_{m}.json")
            for k in ("settled", "n", "lp_median"):
                add("d", ds, m, f"S1b_{k}", e, "CROWN|ZD", s1b[k], sp)
            add("d", ds, m, "S1b_lp_q10", e, "CROWN|ZD", out["lp_q10"][-1], P(ds, f"s3_certify_{m}.json"))
            add("d", ds, m, "S1b_lp_q90", e, "CROWN|ZD", out["lp_q90"][-1], P(ds, f"s3_certify_{m}.json"))
            add("d", ds, m, "n_lp_all_classes", e, "CROWN|ZD", cfg["n_lp"], sp); add("d", ds, m, "lp_seconds_all_classes", e, "CROWN|ZD", cfg["lp_seconds"], sp)
        d["search"][ds][m] = out

# ============================================================== e: closure gain and the exact 1-D check
e_ = dict(eps=EPS, exact1d={}, exact1d_alpha={}, closure_gain_median_max={}, closure_gain_max_all_outputs={}, n_configs=0)
for ds in DATASETS:
    e_["exact1d"][ds] = {}; e_["exact1d_alpha"][ds] = {}; e_["closure_gain_median_max"][ds] = {}; e_["closure_gain_max_all_outputs"][ds] = {}
    for m in MODELS:
        S = SUM[(ds, m)]; sp = P(ds, f"summary_{m}.json"); Bx = np.load(P(ds, f"boxes_{m}.npz"))
        e_["exact1d"][ds][m] = [S["configs"][f"CROWN|{e}|Z1"]["exact1d_check_max_abs_diff"] for e in EPS]
        if f"alpha-CROWN|0.01|Z1" in S["configs"]:
            e_["exact1d_alpha"][ds][m] = [S["configs"][f"alpha-CROWN|{e}|Z1"]["exact1d_check_max_abs_diff"] for e in EPS]
        for k, v in S["configs"].items():
            if k.endswith("|Z1"):
                add("e", ds, m, "exact1d_check_max_abs_diff", k, "", v["exact1d_check_max_abs_diff"], sp)
        gm, gx = 0.0, 0.0
        for k, v in S["configs"].items():
            if k.endswith("|ZD") or k.endswith("|Z1"):
                meth, ep, arc = k.split("|"); e_["n_configs"] += 1
                gm = max(gm, v["closure_gain_median"]); add("e", ds, m, "closure_gain_median", k, "", v["closure_gain_median"], sp)
                cf = P(ds, f"closure_{m}_{meth}_{ep}_{arc}.npz"); z = np.load(cf)
                g = max(float(np.max(Bx[f"hi|{meth}|{ep}"] - z["u"])), float(np.max(z["l"] - Bx[f"lo|{meth}|{ep}"])))
                gx = max(gx, g); add("e", ds, m, "closure_gain_max_over_outputs", k, "", g, cf)
        e_["closure_gain_median_max"][ds][m] = gm; e_["closure_gain_max_all_outputs"][ds][m] = gx

# ============================================================== f: rigorous enclosure vs the stored verifier (CROWN) boxes
RIG = {"synthetic": (os.path.join(FE, "exp3_ml", "rigorous"), "", "r1_summary.json", "lean_results.json"),
       "jul2014": (os.path.join(FE, "exp3_ml_real", "rigorous"), "jul2014_", "r1_summary_jul2014.json", "lean_results_jul2014.json"),
       "jan2014": (os.path.join(FE, "exp3_ml_real", "rigorous"), "jan2014_", "r1_summary_jan2014.json", "lean_results_jan2014.json")}
rng = np.random.default_rng(20261004)
N_SUB = 20000
f_ = dict(eps=[0.01, 0.05], datasets={}, n_sub=N_SUB, margin=MARGIN, bins_log10=np.arange(-24.0, -8.75, 0.25).tolist())
wa_all, wr_all, ds_all = [], [], []
for ds in DATASETS:
    rdir, tag, r1f, lf = RIG[ds]; R1 = jload(os.path.join(rdir, r1f)); LR = jload(os.path.join(rdir, lf))
    n_tot = proven = margin_needed = not_proven = proven_remaining = 0; over = []
    for m in MODELS:
        Bx = np.load(P(ds, f"boxes_{m}.npz"))
        for e in f_["eps"]:
            rec = R1[f"{m}|{e}"]; allc = rec["compare_with_auto_LiRPA"]["CROWN"]["ALL"]
            n_tot += allc["n"]; proven += allc["proven_sound"]; margin_needed += allc["margin_needed"]; not_proven += allc["not_proven"]
            fix = {}
            if "remaining_proven_at_auto_slopes" in rec:
                rem = rec["remaining_proven_at_auto_slopes"]; proven_remaining += rem["n_proven"]
                for r in rem["records"]:          # the second rigorous box (auto_LiRPA's own sigmoid slopes) proves these outputs
                    fix[(r["k"], r["j"], r["t"])] = (max(r["rig_at_auto_slopes"][0], r["rig_default"][0]), min(r["rig_at_auto_slopes"][1], r["rig_default"][1]))
            Rb = np.load(os.path.join(rdir, f"rigorous_boxes_{tag}{m}_{e}.npz"))
            pairs = [(Rb["lo"], Rb["hi"], Bx[f"lo|CROWN|{e}"], Bx[f"hi|CROWN|{e}"], 0)] + \
                    [(Rb[f"glo|{k}"], Rb[f"ghi|{k}"], Bx[f"glo|CROWN|{e}|{k}"], Bx[f"ghi|CROWN|{e}|{k}"], k) for k in LAGS]
            for rl, rh, al, ah, k in pairs:
                rl, rh = rl.copy(), rh.copy()
                for (kk, j, t), (l2, h2) in fix.items():
                    if kk == k:
                        rl[j, t], rh[j, t] = l2, h2
                assert np.all(rl >= al) and np.all(rh <= ah), (ds, m, e, k)        # every stored box contains a rigorous enclosure
                ov = np.maximum(al + MARGIN - rl, rh - ah + MARGIN).ravel()           # rigorous endpoint beyond the RAW float box
                over.append(ov[ov > 0])
                wa_all.append((ah - al).ravel()); wr_all.append((rh - rl).ravel()); ds_all.append(np.full(al.size, DATASETS.index(ds), np.int8))
    over = np.concatenate(over)
    hist, _ = np.histogram(np.log10(over), bins=f_["bins_log10"])
    assert hist.sum() == len(over)
    lean = dict(n=len(LR), passed=sum(x["verdict"] == "PASS" and x["exit_code"] == 0 for x in LR), max_seconds=max(x["seconds"] for x in LR),
                rows=sorted({x["rows"] for x in LR}))
    f_["datasets"][ds] = dict(n_boxes=n_tot, proven_default=proven, proven_at_auto_slopes=proven_remaining, sound=proven + proven_remaining,
                              margin_needed_r1=margin_needed, not_proven_default=not_proven, n_overshoot=int(len(over)),
                              overshoot_max=float(over.max()), overshoot_min=float(over.min()), overshoot_hist=hist.tolist(), lean=lean,
                              source=dict(r1=rel(os.path.join(rdir, r1f)), lean=rel(os.path.join(rdir, lf)), boxes=rel(os.path.join(rdir, f"rigorous_boxes_{tag}{{model}}_{{eps}}.npz"))))
    for k in ("n_boxes", "sound", "proven_at_auto_slopes", "n_overshoot", "overshoot_max"):
        add("f", ds, "", k, "", "", f_["datasets"][ds][k], os.path.join(rdir, r1f))
    add("f", ds, "", "lean_pass", "", "", lean["passed"], os.path.join(rdir, lf)); add("f", ds, "", "lean_n", "", "", lean["n"], os.path.join(rdir, lf))
wa_all = np.concatenate(wa_all); wr_all = np.concatenate(wr_all); ds_all = np.concatenate(ds_all)
idx = np.sort(rng.choice(len(wa_all), N_SUB, replace=False))
f_["sub"] = dict(verifier_width=wa_all[idx].tolist(), rigorous_width=wr_all[idx].tolist(), dataset=ds_all[idx].tolist())
f_["ratio_all"] = dict(max=float(np.max(wr_all / wa_all)), n_le_1=int(np.sum(wr_all <= wa_all)), n=int(len(wa_all)),
                       median={ds: float(np.median((wr_all / wa_all)[ds_all == i])) for i, ds in enumerate(DATASETS)})
f_["totals"] = dict(n_boxes=int(sum(v["n_boxes"] for v in f_["datasets"].values())), sound=int(sum(v["sound"] for v in f_["datasets"].values())),
                    lean_pass=int(sum(v["lean"]["passed"] for v in f_["datasets"].values())), lean_n=int(sum(v["lean"]["n"] for v in f_["datasets"].values())),
                    n_overshoot=int(sum(v["n_overshoot"] for v in f_["datasets"].values())), overshoot_max=float(max(v["overshoot_max"] for v in f_["datasets"].values())))

OUT = dict(stem="ED3", note="Extended Data Fig. 3 data; every number from the pooling-transfer result files (paths in each block and ED3_source_data.csv)",
           datasets=DATASETS, models=MODELS, a=a, b=b, c=c, d=d, e=e_, f=f_,
           sources={ds: {"summary": rel(P(ds, "summary_{model}.json")), "boxes": rel(P(ds, "boxes_{model}.npz")), "s3": rel(P(ds, "s3_certify_{model}.json"))} for ds in DATASETS})
json.dump(OUT, open(os.path.join(DATA, "ED3.json"), "w", encoding="utf-8"), indent=1)
with open(os.path.join(HERE, "ED3_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(ROWS[0])); wr.writeheader(); wr.writerows(ROWS)
    for i in range(N_SUB):
        wr.writerow(dict(panel="f", dataset=DATASETS[f_["sub"]["dataset"][i]], model="", quantity="subsample_verifier_width|rigorous_width", key1=i, key2="",
                         value=f"{f_['sub']['verifier_width'][i]!r}|{f_['sub']['rigorous_width'][i]!r}", source="rigorous_boxes_*.npz + boxes_*.npz (seeded subsample)"))
print("ED3.json written;", len(ROWS), "rows; totals", f_["totals"], "ratio", f_["ratio_all"])
for ds in DATASETS:
    v = f_["datasets"][ds]; print(ds, v["n_boxes"], v["sound"], v["n_overshoot"], f"{v['overshoot_min']:.2e} {v['overshoot_max']:.2e}", v["lean"])
print("closure max", e_["closure_gain_max_all_outputs"], "n_configs", e_["n_configs"])
