"""compare_rerun_guarded.py -- number-by-number comparison of the stored synthetic exp3_ml results with the LP-guarded rerun
(exp3_ml/rerun_guarded, produced by the guarded s3_certify.py --out, s4_witness.py --out, s5_tables.py --dir).
Read-only on the stored files.  Writes the tables of exp3_ml/rerun_guarded/COMPARISON_tables.md (the verdict paragraph of
COMPARISON.md is written by hand from this output) and prints the leaf-diff statistics.
Usage: python compare_rerun_guarded.py [--models mlp,feat,feat24]"""
import os, sys, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlt_common import OUT, EPS_GRID

RR = os.path.join(OUT, "rerun_guarded")
IGN = {"seconds", "lp_seconds"}                                       # wall-clock timings: not results
GUARD_KEYS = {"lp_failures", "lp_failures_by_class", "lp_failed", "lp_failed_U", "lp_failed_L", "n_lp_total", "lp_failures_total",
              "lp_failures_exact1d_total", "lp_failures_by_config"}  # keys added by the guard (absent from the stored files)
REL = 1e-9


def jl(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def leafdiff(a, b, path, st):
    """walk the STORED tree b; compare with rerun a; st collects counts and the largest relative difference"""
    if isinstance(b, dict):
        if not isinstance(a, dict):
            st["struct"].append(path); return
        for k in a:
            if k not in b:
                (st["new_guard_keys"] if k in GUARD_KEYS else st["new_other_keys"]).add(k)
        for k, v in b.items():
            if k in IGN:
                continue
            if k not in a:
                st["missing"].append(path + "/" + k); continue
            leafdiff(a[k], v, path + "/" + k, st)
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            st["struct"].append(path); return
        for i, (x, y) in enumerate(zip(a, b)):
            leafdiff(x, y, f"{path}[{i}]", st)
    else:
        st["n"] += 1
        if isinstance(b, bool) or b is None or isinstance(b, str):
            if a != b:
                st["neq"].append((path, a, b))
            else:
                st["exact"] += 1
            return
        x, y = float(a), float(b)
        if (np.isnan(x) and np.isnan(y)) or x == y:
            st["exact"] += 1; return
        r = abs(x - y) / max(abs(x), abs(y), 1e-300) if np.isfinite(x) and np.isfinite(y) else np.inf
        st["maxrel"] = max(st["maxrel"], r); st["neq"].append((path, x, y))


def newst():
    return dict(n=0, exact=0, maxrel=0.0, neq=[], missing=[], struct=[], new_guard_keys=set(), new_other_keys=set())


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    return "nan" if np.isnan(x) else repr(x)


def same(x, y):
    if isinstance(x, (str, bool)) or x is None:
        return x == y
    x, y = float(x), float(y)
    return (np.isnan(x) and np.isnan(y)) or x == y


def rel(x, y):
    if isinstance(x, (str, bool)) or x is None:
        return 0.0 if x == y else np.inf
    x, y = float(x), float(y)
    if (np.isnan(x) and np.isnan(y)) or x == y:
        return 0.0
    return abs(x - y) / max(abs(x), abs(y), 1e-300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="mlp,feat,feat24"); a = ap.parse_args(); M = a.models.split(",")
    L = []; P = lambda s="": L.append(s); ALLROWS = []          # ALLROWS: (quantity, stored, rerun) for the global verdict
    # ------------------------------------------------------------------ whole-file leaf diffs
    P("## A. Whole-file comparison (every leaf; wall-clock timings excluded)\n")
    P("| file | numeric/bool leaves compared | bit-identical | differing | max rel. diff | missing in rerun | keys added by the guard |")
    P("|---|---|---|---|---|---|---|")
    tot = dict(n=0, neq=0)
    for m in M:
        for f in (f"s3_certify_{m}.json", f"summary_{m}.json", f"s4_witness_{m}.json"):
            st = newst(); leafdiff(jl(os.path.join(RR, f)), jl(os.path.join(OUT, f)), "", st)
            tot["n"] += st["n"]; tot["neq"] += len(st["neq"])
            P(f"| {f} | {st['n']} | {st['exact']} | {len(st['neq'])} | {st['maxrel']:.1e} | {len(st['missing']) + len(st['struct'])} | {', '.join(sorted(st['new_guard_keys'])) or '-'} |")
            if st["neq"]:
                print(f, "first diffs:", st["neq"][:10])
            if st["new_other_keys"] or st["missing"] or st["struct"]:
                print(f, "UNEXPECTED keys/structure:", st["new_other_keys"], st["missing"][:5], st["struct"][:5])
        nz = 0; nzeq = 0; nel = 0
        for meth in ("CROWN", "alpha-CROWN"):
            for e in EPS_GRID:
                for arc in ("Z1", "ZD"):
                    f = f"closure_{m}_{meth}_{e}_{arc}.npz"; A = np.load(os.path.join(RR, f)); B = np.load(os.path.join(OUT, f)); nz += 1
                    ok = all(np.array_equal(A[k], B[k]) for k in B.files) and set(A.files) == set(B.files); nzeq += ok; nel += sum(B[k].size for k in B.files)
        P(f"| closure_{m}_*.npz ({nz} files) | {nel} | {nel if nzeq == nz else 'see below'} | {0 if nzeq == nz else nz - nzeq} files | {'0' if nzeq == nz else '>0'} | 0 | - |")
    ta = open(os.path.join(OUT, "tables.md"), encoding="utf-8").read().splitlines(); tb = open(os.path.join(RR, "tables.md"), encoding="utf-8").read().splitlines()
    extra = [x for x in tb if x not in ta]; lost = [x for x in ta if x not in tb]
    P(f"| tables.md | {len(ta)} lines | {sum(1 for x, y in zip(ta, tb) if x == y)} identical lines in order | {len(lost)} stored lines absent from rerun | - | - | {len(extra)} added line(s): the LP-failure lines |")
    P(f"\nTotal over the JSON files: {tot['n']} leaves compared, {tot['neq']} differ.\n")
    for x in extra:
        P(f"    rerun-only line in tables.md: {x}")
    # ------------------------------------------------------------------ LP volume and failures
    S = {m: (jl(os.path.join(OUT, f"summary_{m}.json")), jl(os.path.join(RR, f"summary_{m}.json"))) for m in M}
    C = {m: jl(os.path.join(RR, f"s3_certify_{m}.json")) for m in M}
    Cs = {m: jl(os.path.join(OUT, f"s3_certify_{m}.json")) for m in M}
    P("\n## B. LP volume and LP failures (the key number)\n")
    P("| model | certificate LPs, stored (sum of per-feeder n_lp) | certificate LPs, rerun | of which S1b look-back LPs (stored / rerun) | LP failures, stored run | LP failures, rerun | exact-1-D check LPs (rerun) | exact-1-D check failures (rerun) |")
    P("|---|---|---|---|---|---|---|---|")
    for m in M:
        ns = sum(r["n_lp"] for k, c in Cs[m]["configs"].items() if "per_feeder" in c for r in c["per_feeder"])
        nr = C[m]["n_lp_total"]
        s1bs = sum(x["n_lp"] for k, c in Cs[m]["configs"].items() if "per_feeder" in c for r in c["per_feeder"] for x in r["S1b"])
        s1br = sum(x["n_lp"] for k, c in C[m]["configs"].items() if "per_feeder" in c for r in c["per_feeder"] for x in r["S1b"])
        n1d = sum(1 for k, c in C[m]["configs"].items() if k.endswith("Z1") for r in c["per_feeder"])
        P(f"| {m} | {ns} | {nr} | {s1bs} / {s1br} | not recorded | **{C[m]['lp_failures_total']}** | {n1d} | {C[m]['lp_failures_exact1d_total']} |")
        ALLROWS.append((f"{m} n_lp", ns, nr)); ALLROWS.append((f"{m} S1b n_lp", s1bs, s1br))
    P("\nPer-class failure counts (rerun, every configuration CROWN/alpha-CROWN x eps x Z1/ZD summed):\n")
    P("| model | S1a | S1b (all 28 report times) | S2 | S3 | S5 | exact-1-D | total |\n|---|---|---|---|---|---|---|---|")
    for m in M:
        bc = C[m]["lp_failures_by_config"]; s = {c: sum(v[c] for v in bc.values()) for c in ("S1a", "S1b", "S2", "S3", "S5", "exact1d")}
        P(f"| {m} | {s['S1a']} | {s['S1b']} | {s['S2']} | {s['S3']} | {s['S5']} | {s['exact1d']} | {sum(s.values())} |")
        e_max = max((x["lp_failures"] for c in C[m]["configs"].values() if "per_feeder" in c for r in c["per_feeder"] for x in r["S1b"]), default=0)
        P(f"|   {m}: max failures at any single (config, feeder, report time) | | {e_max} | | | | | |")
    # direct scan of the STORED files: a failed LP would have left NaN in the recorded LP outputs (old lp() passed NaN through)
    P("\nDirect scan of the stored s3_certify files for the LP outputs that were recorded (old code: a failed LP left NaN there):\n")
    P("| model | S1a Z_at_argmax non-finite (of n) | S1b Z non-finite (of n) | G-record Z non-finite (of n) | sample UZ/LZ non-finite (of n) |\n|---|---|---|---|---|")
    for m in M:
        pf = [r for c in Cs[m]["configs"].values() if "per_feeder" in c for r in c["per_feeder"]]
        za = [r["S1a"]["Z_at_argmax"] for r in pf]; zb = [x["Z"] for r in pf for x in r["S1b"]]; zg = [g["Z"] for r in pf for g in r["G_records"]]
        zs = [b[k] for r in pf for S_ in ("S2", "S3", "S5") for i in r[S_] for b in i["sample_bounds"] for k in ("UZ", "LZ")]
        nf = lambda v: int(sum(not np.isfinite(float(x)) for x in v))
        P(f"| {m} | {nf(za)} of {len(za)} | {nf(zb)} of {len(zb)} | {nf(zg)} of {len(zg)} | {nf(zs)} of {len(zs)} |")
    # ------------------------------------------------------------------ reported quantities, stored vs rerun
    def row(q, s_, r_):
        ALLROWS.append((q, s_, r_)); q = q.replace("|", " ")            # config keys 'meth|eps|arcs' -> 'meth eps arcs' (markdown cells)
        return f"| {q} | {fmt(s_)} | {fmt(r_)} | {'yes' if same(s_, r_) else 'NO (rel %.1e)' % rel(s_, r_)} |"
    P("\n## C. Reported quantities, stored vs rerun (every value printed at full float64 precision)\n")
    P("### C1. Normalized saving G: median and q90 per class (S1b top pair, S2, S3, S5 validation samples), all eps, CROWN and alpha-CROWN, Z1 and ZD\n")
    P("| quantity | stored | rerun | identical |\n|---|---|---|---|")
    for m in M:
        for meth in ("CROWN", "alpha-CROWN"):
            for arc in ("ZD", "Z1"):
                for e in EPS_GRID:
                    k = f"{meth}|{e}|{arc}"; gs, gr = S[m][0]["configs"][k]["G"], S[m][1]["configs"][k]["G"]
                    for t in ("S1b_top", "S2_sample", "S3_sample", "S5_sample"):
                        for st_ in ("G_median", "G_q90"):
                            P(row(f"{m} {k} {t} {st_}", gs[t][st_], gr[t][st_]))
    P("\n### C2. Outer/inner excess ratios (U - nominal)/(W - nominal), medians (and S1b q90), CROWN and alpha-CROWN, ZD and Z1, all eps\n")
    P("| quantity | stored | rerun | identical |\n|---|---|---|---|")
    for m in M:
        for meth in ("CROWN", "alpha-CROWN"):
            for arc in ("ZD", "Z1"):
                for e in EPS_GRID:
                    k = f"{meth}|{e}|{arc}"; rs, rr = S[m][0]["configs"][k], S[m][1]["configs"][k]
                    P(row(f"{m} {k} S1a V", rs["S1a"]["excess_ratio_V"], rr["S1a"]["excess_ratio_V"]))
                    P(row(f"{m} {k} S1a V*", rs["S1a"]["excess_ratio_Vs"], rr["S1a"]["excess_ratio_Vs"]))
                    for b in ("V", "Z"):
                        P(row(f"{m} {k} S1b {b} median", rs["S1b"]["excess_ratio"][b], rr["S1b"]["excess_ratio"][b]))
                        P(row(f"{m} {k} S1b {b} q90", rs["S1b"]["excess_ratio_q90"][b], rr["S1b"]["excess_ratio_q90"][b]))
                    for s_ in ("S2", "S3", "S5"):
                        for b in ("V", "Z"):
                            P(row(f"{m} {k} {s_} {b} median", rs["outer_over_inner_excess"][s_][b]["median"], rr["outer_over_inner_excess"][s_][b]["median"]))
                    for b in ("nominal", "witness", "V", "Z"):
                        P(row(f"{m} {k} S1b median {b}", rs["S1b"]["median"][b], rr["S1b"]["median"][b]))
                    P(row(f"{m} {k} S1b settled (of {rs['S1b']['n']})", rs["S1b"]["settled"], rr["S1b"]["settled"]))
                    P(row(f"{m} {k} S1b (Z-nom)/(V-nom)", rs["S1b"]["Z_over_V"], rr["S1b"]["Z_over_V"]))
    P("\n### C3. Decisions CI/CS/WSn/WFp/open per class, V and Z, all eps (eps = 0.01 and 0.02 are the reported ones), CROWN and alpha-CROWN, ZD and Z1\n")
    P("| quantity | stored | rerun | identical |\n|---|---|---|---|")
    cats = ("certified_invariant", "certified_sensitive", "witnessed_sensitive_nominal", "witnessed_flip_perturbation", "open")
    for m in M:
        for meth in ("CROWN", "alpha-CROWN"):
            for arc in ("ZD", "Z1"):
                for e in EPS_GRID:
                    k = f"{meth}|{e}|{arc}"; ds, dr = S[m][0]["configs"][k]["decisions"], S[m][1]["configs"][k]["decisions"]
                    for s_ in ("S2", "S3", "S5"):
                        for b in ("V", "Z"):
                            P(row(f"{m} {k} {s_} {b} CI/CS/WSn/WFp/open", "/".join(str(ds[s_][b][c]) for c in cats), "/".join(str(dr[s_][b][c]) for c in cats)))
                        P(row(f"{m} {k} {s_} nominal-mixed instances", ds[s_]["nominal_mixed_sign_instances"], dr[s_]["nominal_mixed_sign_instances"]))
                        P(row(f"{m} {k} {s_} LP calls", ds[s_]["lp_calls"], dr[s_]["lp_calls"]))
    P("\n### C4. S4 ranking states (28 report times), every configuration\n")
    P("| quantity | stored | rerun | identical |\n|---|---|---|---|")
    for m in M:
        for e in EPS_GRID:
            for ck in [f"IBP|{e}|V", f"CROWN|{e}|V", f"alpha-CROWN|{e}|V"] + [f"{mm}|{e}|{aa}" for mm in ("CROWN", "alpha-CROWN") for aa in ("Z1", "ZD")]:
                s4s, s4r = S[m][0]["configs"][ck]["S4"], S[m][1]["configs"][ck]["S4"]
                P(row(f"{m} {ck} certified pairs (median)", float(np.median([x["certified_pairs"] for x in s4s])), float(np.median([x["certified_pairs"] for x in s4r]))))
                P(row(f"{m} {ck} rank-interval width (mean)", float(np.mean([x["rank_interval_width_mean"] for x in s4s])), float(np.mean([x["rank_interval_width_mean"] for x in s4r]))))
                P(row(f"{m} {ck} nominal rank range (mean)", float(np.mean([x["nominal_rank_range_mean"] for x in s4s])), float(np.mean([x["nominal_rank_range_mean"] for x in s4r]))))
                for fld, lab in (("top3_certified_invariant", "top-3 certified invariant"), ("nominal_distinct_top3", "nominal top-3 changes with L"),
                                 ("distinct_certified_top3_sets", ">1 distinct certified top-3")):
                    fs = (lambda x: x[fld]) if fld == "top3_certified_invariant" else (lambda x: x[fld] > 1)
                    P(row(f"{m} {ck} {lab} (of 28)", int(sum(fs(x) for x in s4s)), int(sum(fs(x) for x in s4r))))
                eq28 = json.dumps(s4s, sort_keys=True) == json.dumps(s4r, sort_keys=True); ALLROWS.append((f"{m} {ck} S4 records", True, eq28))
                P(f"| {m} {ck.replace('|', ' ')} per-report-time S4 records (all fields, all 28) | 28 records | 28 records | {'yes' if eq28 else 'NO'} |")
    P("\n### C5. Closure gain, width ratio, LP counts per configuration, exact-1-D check\n")
    P("| quantity | stored | rerun | identical |\n|---|---|---|---|")
    for m in M:
        for meth in ("CROWN", "alpha-CROWN"):
            for arc in ("ZD", "Z1"):
                for e in EPS_GRID:
                    k = f"{meth}|{e}|{arc}"; rs, rr = S[m][0]["configs"][k], S[m][1]["configs"][k]
                    P(row(f"{m} {k} closure gain (median)", rs["closure_gain_median"], rr["closure_gain_median"]))
                    P(row(f"{m} {k} (u*-l*)/(hi-lo) (median)", rs["width_ratio_median"], rr["width_ratio_median"]))
                    P(row(f"{m} {k} n_lp", rs["n_lp"], rr["n_lp"]))
                    if arc == "Z1":
                        P(row(f"{m} {k} exact-1-D max abs diff", rs["exact1d_check_max_abs_diff"], rr["exact1d_check_max_abs_diff"]))
    nbad = [x for x in ALLROWS if not same(x[1], x[2])]
    P(f"\nRows in sections B-C: {len(ALLROWS)}; not identical: {len(nbad)}; max relative difference: {max([rel(x[1], x[2]) for x in ALLROWS] + [0.0]):.1e}.")
    # ------------------------------------------------------------------ compact headline (CROWN, ZD: the configuration quoted in findings.md)
    H = []; Q = H.append; f3 = lambda x: "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.3f}"; f2 = lambda x: f"{float(x):.2f}"
    Q("## 0. Headline view (CROWN, dyadic arcs ZD; the configuration quoted in findings.md). Values = stored; 'identical' = every underlying float64 equal in the rerun\n")
    Q("| model | eps | G median [q90]: S1b top / S2 / S3 / S5 | excess V -> Z: S1b / S2 / S3 / S5 (S1a V) | decisions V -> Z, CI/CS/WSn/WFp/open: S2 / S3 / S5 | S4 (CROWN eps V): pairs, width, nominal range, top3-inv, top3-changes, >1 cert top3 | closure gain (median) | LPs | identical |")
    Q("|---|---|---|---|---|---|---|---|---|")
    cats_ = ("certified_invariant", "certified_sensitive", "witnessed_sensitive_nominal", "witnessed_flip_perturbation", "open")
    for m in M:
        for e in EPS_GRID:
            k = f"CROWN|{e}|ZD"; rs, rr = S[m][0]["configs"][k], S[m][1]["configs"][k]; kv = f"CROWN|{e}|V"
            g = " / ".join(f"{f3(rs['G'][t]['G_median'])} [{f3(rs['G'][t]['G_q90'])}]" for t in ("S1b_top", "S2_sample", "S3_sample", "S5_sample"))
            oi = rs["outer_over_inner_excess"]
            ex = f"{f2(rs['S1b']['excess_ratio']['V'])}->{f2(rs['S1b']['excess_ratio']['Z'])} / " + " / ".join(f"{f2(oi[s_]['V']['median'])}->{f2(oi[s_]['Z']['median'])}" for s_ in ("S2", "S3", "S5")) + f" ({f2(rs['S1a']['excess_ratio_V'])})"
            dc = " / ".join("/".join(str(rs["decisions"][s_]["V"][c]) for c in cats_) + " -> " + "/".join(str(rs["decisions"][s_]["Z"][c]) for c in cats_) for s_ in ("S2", "S3", "S5"))
            s4 = S[m][0]["configs"][kv]["S4"]
            s4t = (f"{np.median([x['certified_pairs'] for x in s4]):.0f}, {np.mean([x['rank_interval_width_mean'] for x in s4]):.2f}, {np.mean([x['nominal_rank_range_mean'] for x in s4]):.2f}, "
                   f"{sum(x['top3_certified_invariant'] for x in s4)}, {sum(x['nominal_distinct_top3'] > 1 for x in s4)}, {sum(x['distinct_certified_top3_sets'] > 1 for x in s4)}")
            ident = (json.dumps({kk: rs[kk] for kk in rs if kk not in IGN}, sort_keys=True) == json.dumps({kk: rr[kk] for kk in rr if kk not in IGN}, sort_keys=True)
                     and json.dumps(s4, sort_keys=True) == json.dumps(S[m][1]["configs"][kv]["S4"], sort_keys=True))
            Q(f"| {m} | {e} | {g} | {ex} | {dc} | {s4t} | {rs['closure_gain_median']:.1e} | {rs['n_lp']} | {'yes' if ident else 'NO'} |")
    Q("")
    open(os.path.join(RR, "COMPARISON_tables.md"), "w", encoding="utf-8").write("\n".join(H + L) + "\n")
    print("rows", len(ALLROWS), "not identical", len(nbad), nbad[:10])
    print("JSON leaves", tot)
