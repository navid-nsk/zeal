"""s5_tables.py -- print the markdown tables of findings.md from exp3_ml/summary_<model>.json, s2_verify_<model>.json, s1_train_*.json.
Usage: python s5_tables.py [--models mlp,feat,feat24] [--dir DIR] > tables.md   (DIR holds summary_<model>.json; default exp3_ml)"""
import os, sys, json, argparse, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlt_common import OUT

ap = argparse.ArgumentParser(); ap.add_argument("--models", default="mlp,feat,feat24")
ap.add_argument("--dir", default=OUT, help="directory of summary_<model>.json (default exp3_ml); s2_verify_/s1_train_ always from exp3_ml"); a = ap.parse_args()
M = a.models.split(",")
S = {m: json.load(open(os.path.join(a.dir, f"summary_{m}.json"), encoding="utf-8")) for m in M}
V = {m: json.load(open(os.path.join(OUT, f"s2_verify_{m}.json"), encoding="utf-8")) for m in M}
TR = {}
for f in glob.glob(os.path.join(OUT, "s1_train_*.json")):
    for k, v in json.load(open(f, encoding="utf-8")).items():
        if isinstance(v, dict) and "eval_auc" in v:
            TR[k] = v
f3 = lambda x: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"
f2 = lambda x: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
EPS = sorted({float(k) for m in M for k in S[m]["verifier"]})

print("### Models\n\n| model | eval AUC | val AUC | mean abs hourly step of f | sd of f |\n|---|---|---|---|---|")
for m in TR:
    t = TR[m]; print(f"| {m} | {f3(t['eval_auc'])} | {f3(t['val_auc'])} | {f3(t['mean_abs_step'])} | {f3(t['f_eval_sd'])} |")

print("\n### Verifier: value width, kappa_k = (increment-box width at lag k) / (value width), soundness and tightness (medians over 12 x 1008)\n")
print("| model | eps | method | value width | kappa_1 | kappa_4 | kappa_16 | kappa_128 | box / PGD-inner (value) | box / PGD-inner (lag-1 incr) | max violation, 64 random series |\n|---|---|---|---|---|---|---|---|---|---|---|")
for m in M:
    for e in EPS:
        for meth in ("IBP", "CROWN", "alpha-CROWN"):
            r = V[m]["runs"].get(f"{meth}|{e}")
            if r is None:
                continue
            c = V[m][f"check|{e}"]; vi = c["verifier_over_inner"].get(meth, {}); km = r["kappa_median"]
            print(f"| {m} | {e} | {meth} | {f3(r['value_width_median'])} | {f2(km['1'])} | {f2(km['4'])} | {f2(km['16'])} | {f2(km['128'])} | "
                  f"{f2(vi.get('value_width_ratio_median'))} | {f2(vi.get('incr1_width_ratio_median'))} | {c['max_violation_random_series'].get(meth, float('nan')):.1e} |")
    print(f"| {m} | | PGD inner (per box) | " + ", ".join(f"{e}: {f3(V[m][f'check|{e}']['inner_value_width_median'])}" for e in EPS)
          + " | lag-1 inner/value: " + ", ".join(f"{V[m][f'check|{e}']['inner_incr1_width_median'] / V[m][f'check|{e}']['inner_value_width_median']:.2f}" for e in EPS) + " | | | | | | |")


def rows(m, arcs=("Z1", "ZD")):
    for e in EPS:
        for meth in ("CROWN", "alpha-CROWN"):
            for arc in arcs:
                k = f"{meth}|{e}|{arc}"
                if k in S[m]["configs"]:
                    yield e, meth, arc, S[m]["configs"][k]


print("\n### Normalized saving G = (V* - Z)/(V* - d.psi) (regime theorem), median [q90] over the evaluated choices; V* = V here (closure gain)\n")
print("| model | eps | method | arcs | S1b top pair | S2 sample | S3 sample | S5 sample (post hoc) | median (u*-l*)/(hi-lo) |\n|---|---|---|---|---|---|---|---|---|")
for m in M:
    for e, meth, arc, r in rows(m):
        g = r["G"]; cell = lambda t: f"{f3(g[t]['G_median'])} [{f3(g[t]['G_q90'])}]" if t in g else "n/a"
        print(f"| {m} | {e} | {meth} | {arc} | {cell('S1b_top')} | {cell('S2_sample')} | {cell('S3_sample')} | {cell('S5_sample')} | {f3(r['width_ratio_median'])} |")

print("\n### Outer / inner: excess ratios (U - nominal)/(W - nominal), W = PGD witness; medians (S1: per feeder / report time; S2, S3, S5: validation choices, both directions)\n")
print("| model | eps | method | arcs | S1a V | S1a V* | S1b V | S1b Z | S2 V | S2 Z | S3 V | S3 Z | S5 V | S5 Z |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in M:
    for e, meth, arc, r in rows(m):
        oi = r["outer_over_inner_excess"]
        print(f"| {m} | {e} | {meth} | {arc} | {f2(r['S1a']['excess_ratio_V'])} | {f2(r['S1a']['excess_ratio_Vs'])} | {f2(r['S1b']['excess_ratio']['V'])} | {f2(r['S1b']['excess_ratio']['Z'])} | "
              + " | ".join(f"{f2(oi[s_]['V']['median'])} | {f2(oi[s_]['Z']['median'])}" for s_ in ("S2", "S3", "S5")) + " |")

print("\n### S1 ranges (medians): nominal, witness W, value-only V, ZEAL Z (S1a: bracket [LP at argmax, V*]); S1b settled = lazy LP closed the sup over all 145^2 pairs\n")
print("| model | eps | method | arcs | S1a nominal | S1a W | S1a V | S1a Z bracket | S1b nominal | S1b W | S1b V | S1b Z | S1b (Z-nom)/(V-nom) | S1b settled |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in M:
    for e, meth, arc, r in rows(m):
        a1, b1 = r["S1a"]["median"], r["S1b"]["median"]
        print(f"| {m} | {e} | {meth} | {arc} | {f3(a1['nominal'])} | {f3(a1['witness'])} | {f3(a1['V'])} | [{f3(a1['Z_lo'])}, {f3(a1['Z_hi'])}] | {f3(b1['nominal'])} | {f3(b1['witness'])} | {f3(b1['V'])} | {f3(b1['Z'])} | {f3(r['S1b']['Z_over_V'])} | {r['S1b']['settled']}/{r['S1b']['n']} |")

print("\n### Decisions (instances): CI certified invariant / CS certified aggregation-sensitive / WSn witnessed sensitive (nominal signs differ) / WFp witnessed flip under perturbation (PGD) / open\n")
print("| model | eps | method | arcs | stat | n | nominal mixed | V: CI/CS/WSn/WFp/open | Z: CI/CS/WSn/WFp/open | LP calls |\n|---|---|---|---|---|---|---|---|---|---|")
for m in M:
    for e, meth, arc, r in rows(m):
        for s_ in ("S2", "S3", "S5"):
            d = r["decisions"][s_]; c = lambda b: "/".join(str(d[b][k]) for k in ("certified_invariant", "certified_sensitive", "witnessed_sensitive_nominal", "witnessed_flip_perturbation", "open"))
            print(f"| {m} | {e} | {meth} | {arc} | {s_} | {d['V']['n']} | {d['nominal_mixed_sign_instances']} | {c('V')} | {c('Z')} | {d['lp_calls']} |")

print("\n### S4 ranking (28 report times): certified pairs (of 132 ordered), mean certified rank-interval width, mean nominal rank range over L, report times with certified-invariant top-3, report times with >1 nominal top-3 set\n")
print("| model | eps | config | certified pairs (median) | rank-interval width (mean) | nominal rank range (mean) | top-3 certified invariant | top-3 changes with L (nominal) | top-3 certified at >=1 L and >1 distinct certified top-3 |\n|---|---|---|---|---|---|---|---|---|")
for m in M:
    for e in EPS:
        for ck in (f"IBP|{e}|V", f"CROWN|{e}|V", f"CROWN|{e}|ZD", f"alpha-CROWN|{e}|V", f"alpha-CROWN|{e}|ZD"):
            if ck not in S[m]["configs"]:
                continue
            s4 = S[m]["configs"][ck]["S4"]
            print(f"| {m} | {e} | {ck} | {np.median([x['certified_pairs'] for x in s4]):.0f} | {np.mean([x['rank_interval_width_mean'] for x in s4]):.2f} | "
                  f"{np.mean([x['nominal_rank_range_mean'] for x in s4]):.2f} | {sum(x['top3_certified_invariant'] for x in s4)} | {sum(x['nominal_distinct_top3'] > 1 for x in s4)} | "
                  f"{sum(x['distinct_certified_top3_sets'] > 1 for x in s4)} |")

print("\n### Exactness checks\n")
for m in M:
    xs = [r.get("exact1d_check_max_abs_diff") for k, r in S[m]["configs"].items() if k.endswith("Z1")]
    print(f"- {m}: discrete exact 1-D theorem (gradient-only lag-1 LP vs closed form sum_s s_G(B_s)), max |diff| over configs = {max(xs):.1e}")
for m in M:                                       # LP-failure guard counts (only in summaries produced by the guarded s3_certify.py)
    if "lp_failures_total" in S[m]:
        print(f"- {m}: certificate LPs {S[m]['n_lp_total']}, LP failures {S[m]['lp_failures_total']} (exact-1-D check LPs failed: {S[m]['lp_failures_exact1d_total']})")
