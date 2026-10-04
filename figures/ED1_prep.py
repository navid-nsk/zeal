"""ED1_prep - data layer of Extended Data Fig. 1 (machine-checked development and checker cost).

Reads ONLY: lean/freeze_manifest_v1.json, lean/FREEZE_v1.md, the Lean sources lean/zeal/** (line, declaration and
#print-axioms counts), the optional statement document ZEAL_STATEMENT_DOC (statement headings, machine-checked map),
lean/certificates_all/lean_c5_log_*.json, results/exp2/A7_verified/a7_verified_poincare.json.
Writes data/ED1.json and ED1_source_data.csv.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, re, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEAN = paths.LEAN_DIR; LZ = paths.LEAN_PROJECT; FE = paths.RESULTS_ROOT
STATEMENT_DOC = os.environ.get("ZEAL_STATEMENT_DOC", "")   # optional: the statement document (see panel b)
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
S = lambda p: ("results/" + p[len("foundation_experiment/"):]) if p.startswith("foundation_experiment/") else p
SRC = dict(manifest=S("lean/freeze_manifest_v1.json"), freeze=S("lean/FREEZE_v1.md"), lean_src=S("lean/zeal/"), foundation="SI machine-checked map",
           lean_log=S("lean/certificates_all/lean_c5_log_{field}.json"), a7=S("foundation_experiment/exp2/A7_verified/a7_verified_poincare.json"))
rows = []
out = dict(figure="ED1", sources=SRC)

# ---------------- a: module map ----------------
man = json.load(open(os.path.join(LEAN, "freeze_manifest_v1.json"), encoding="utf-8"))
freeze_md = open(os.path.join(LEAN, "FREEZE_v1.md"), encoding="utf-8").read()
MODULE = [("Transport", lambda p: p.startswith("Zeal/Transport/")), ("Realization", lambda p: p.startswith("Zeal/Realization")),
          ("Zoning", lambda p: p.startswith("Zeal/Zoning/")), ("Basic", lambda p: p.startswith("Zeal/Basic")),
          ("Certificates", lambda p: p.startswith("Zeal/Certificates/")), ("checker scripts", lambda p: p.startswith("scripts/")),
          ("Identification", lambda p: p.startswith("Zeal/Identification/")), ("main theorems", lambda p: p in ("MainTheorems.lean", "MainTheorems_M1.lean", "Zeal.lean"))]
lean_files = [p for p in man["manifest"] if p.endswith(".lean")]; other_files = [p for p in man["manifest"] if not p.endswith(".lean")]
decl_file = {}; nthm = {}; nlines = {}
strip = lambda src: re.sub(r"--[^\n]*", "", re.sub(r"/-.*?-/", "", src, flags=re.S))
for p in lean_files:
    src = open(os.path.join(LZ, p), encoding="utf-8").read(); nlines[p] = src.count("\n") + (0 if src.endswith("\n") else 1)
    body = strip(src)
    nthm[p] = len(re.findall(r"^(?:private\s+|protected\s+)*(?:theorem|lemma)\s+\S+", body, re.M))
    for m in re.finditer(r"^(?:private\s+|protected\s+|noncomputable\s+)*(?:theorem|lemma|def|abbrev|structure|instance)\s+([^\s(:{\[]+)", body, re.M):
        decl_file.setdefault(m.group(1).split(".")[-1], p)
assert sum(nlines.values()) == man["source_lines"], (sum(nlines.values()), man["source_lines"])
reported = set()
for p in ("MainTheorems.lean", "MainTheorems_M1.lean", "Zeal/Realization.lean"):
    reported |= set(re.findall(r"#print axioms\s+(\S+)", strip(open(os.path.join(LZ, p), encoding="utf-8").read())))
n_print_cmds = sum(len(re.findall(r"#print axioms\s+(\S+)", strip(open(os.path.join(LZ, p), encoding="utf-8").read()))) for p in ("MainTheorems.lean", "MainTheorems_M1.lean", "Zeal/Realization.lean"))
exp_file = {}
for nm in reported:
    f = decl_file.get(nm.split(".")[-1]); exp_file[nm] = f
mods = []
for name, pred in MODULE:
    files = sorted([p for p in lean_files if pred(p)], key=lambda p: -nlines[p])
    mods.append(dict(name=name, files=[dict(path=p, lines=nlines[p], theorems=nthm[p]) for p in files], n_files=len(files), lines=sum(nlines[p] for p in files),
                     theorems=sum(nthm[p] for p in files), exported=sum(1 for nm, f in exp_file.items() if f in files)))
    for p in files: rows.append(("a", name, "lines", p, nlines[p], SRC["lean_src"] + p))
assert sum(m["n_files"] for m in mods) == len(lean_files)
m_ax = re.search(r"axiom reports (\d+), non-standard (\d+)", freeze_md); m_w = re.search(r"warnings (\d+), errors (\d+)", freeze_md)
out["a"] = dict(modules=mods, n_lean_files=len(lean_files), n_manifest_files=man["n_source_files"], non_lean_manifest_files=other_files, lines_total=man["source_lines"],
                theorems_total=sum(nthm.values()), exported_distinct=len(reported), print_axioms_commands=n_print_cmds,
                build=dict(warnings=man["build"]["warnings"], errors=man["build"]["errors"], axiom_reports=man["build"]["axiom_reports"], nonstandard=len(man["build"]["nonstandard_axiom_reports"]),
                           toolchain=man["toolchain"], freeze_md_axiom_reports=int(m_ax.group(1)), freeze_md_nonstandard=int(m_ax.group(2)), freeze_md_warnings=int(m_w.group(1))),
                note="exported = distinct theorem names in the uncommented #print axioms commands of MainTheorems.lean, MainTheorems_M1.lean and Zeal/Realization.lean, attributed to the declaring file; theorems = theorem/lemma declarations outside comments")

# ---------------- b: statement coverage (statement families; machine-checked map of the statement document) ----------------
_HAVE_DOC = bool(STATEMENT_DOC) and os.path.exists(STATEMENT_DOC)
fv8 = open(STATEMENT_DOC, encoding="utf-8").read().splitlines() if _HAVE_DOC else []
if _HAVE_DOC:
    ter = "\n".join(fv8[93:112])            # machine-checked map, lines 94-112
    heads = {m.group(1): m.group(2) for m in re.finditer(r"\*\*(T\d+) \(([^)]*)\)", "\n".join(fv8[10:52]))}
    # code Tn = Supplementary Theorem Sn; descriptive names (no codes on the figure); level: 2 = machine-checked statement, 1 = finite/discrete core only, 0 = not formalized
    FAM = [("T1", "projection identities", 1, "finite, any ordered field; convex order not in the map"),
           ("T2", "sound discretization", 0, "listed as not machine-checked (continuous tent identity)"),
           ("T3", "grounded-flow closure", 1, "representation full theorem; lattice closure for finite bounds only"),
           ("T4", "value-face criterion", 1, "criterion and Wasserstein reduction; parts (iv)-(v) not in the map"),
           ("T5", "saving identity, regime theorem", 2, "over R"),
           ("T6", "interpolation-completed bracket", 0, "continuum statement, not machine-checked"),
           ("T7", "lifted outer–inner theorem", 1, "discrete core; continuous inclusions and lens rows not formalized"),
           ("T8", "exact one-dimensional theorem", 1, "discrete identity and quadrature value; continuum class not formalized"),
           ("T9", "conditioned classes", 0, "continuum statement, not machine-checked"),
           ("T10", "map-level brackets", 2, "universal forms"),
           ("T11", "fibers and nullspace", 1, "finite-dimensional"),
           ("T12", "one-mean interval", 0, "listed as not machine-checked"),
           ("T13", "reconstruction identities", 1, "projection identities and shrinkage; parts (v)-(vii) not in the map"),
           ("T14", "simultaneous envelope", 0, "probability, not machine-checked"),
           ("T15", "bias-plus-noise intervals", 0, "probability, not machine-checked"),
           ("T16", "pair differences, rank sets", 2, "pair bounds for j != k; non-empty focal families"),
           ("T17", "charged pricing bound", 2, "full theorem (column generation not formalized)"),
           ("T18", "unit-reassignment decomposition", 1, "decomposition only; smooth boundary-perturbation part not in the map"),
           ("T19", "certified Poincaré constant", 0, "listed as not machine-checked"),
           ("T20", "certifier enclosures", 0, "listed as not machine-checked"),
           ("V1", "certificate rule, weak duality", 2, "universal inequality (verification specification)"),
           ("V1/V3", "checker soundness", 2, "soundness kernel-checked; instances by the compiled checker")]
    # Lean declarations named for each family in the machine-checked map (counted from the backticked names of that family's row(s))
    DECL_ROWS = {"T1": ["T1 (projection identities)"], "T3": ["T3 (lattice closure", "T3 (iii) Theorem A"], "T4": ["T4 value-face criterion"], "T5": ["T5 saving identity"],
                 "T7": ["D3/T7 lifted set"], "T8": ["T8 exact one-dimensional"], "T10": ["T10 map-level bracket"], "T11": ["T11 identification"],
                 "T13": ["T13 reconstruction identities"], "T16": ["T16 rank sets"], "T17": ["T17 set-partitioning LP"], "T18": ["T18 unit-reassignment"],
                 "V1": ["V1 checker rule / weak duality"], "V1/V3": ["V1/V3 instance certificates"]}
    fam_out = []
    for code, name, lvl, scope in FAM:
        decl = []
        for key in DECL_ROWS.get(code, []):
            ln = next(l for l in ter.splitlines() if key in l)
            decl += [d for d in re.findall(r"`([^`]+)`", ln.split("|")[2]) if not d.endswith(".lean") and "/" not in d]
        assert (lvl > 0) == bool(decl), code
        fam_out.append(dict(code=code, heading=heads.get(code, ""), name=name, level=lvl, scope=scope, n_decl=len(decl), decl=decl))
        rows.append(("b", name, "coverage_level", code, lvl, SRC["foundation"] + " Part III-ter"))
    not_line = next(l for l in ter.splitlines() if l.startswith("Not machine-checked"))
    for code in ("T2", "T12", "T14", "T15", "T19", "T20"):
        assert re.search(rf"\b{code}\b|T14–T15|T19–T20", not_line), code
    out["b"] = dict(families=fam_out, not_machine_checked_line=not_line, n_level=[sum(f["level"] == k for f in fam_out) for k in (2, 1, 0)],
                    n_level_T=[sum(f["level"] == k for f in fam_out if f["code"].startswith("T")) for k in (2, 1, 0)], n_T=sum(f["code"].startswith("T") for f in fam_out))
else:   # the statement document is not deposited (the SI reproduces the map): reuse the stored panel b
    out["b"] = json.load(open(os.path.join(DATA, "ED1.json"), encoding="utf-8"))["b"]
    for f_ in out["b"]["families"]:
        rows.append(("b", f_["name"], "coverage_level", f_["code"], f_["level"], SRC["foundation"] + " Part III-ter"))

# ---------------- c: checker cost (Lean checker log) ----------------
ORDER = ["georgia", "gm_q4", "gm_bad", "mx_rwi"]; c = {}
for f in ORDER:
    lg = json.load(open(os.path.join(LEAN, "certificates_all", f"lean_c5_log_{f}.json")))
    c[f] = dict(rows=[r["rows"] for r in lg], check_s=[r["check_s"] for r in lg], export_s=[r["export_s"] for r in lg], MB=[r["bytes"] / 1e6 for r in lg],
                n=len(lg), n_pass=sum(r["passed"] for r in lg), total_s_median=float(np.median([r["check_s"] + r["export_s"] for r in lg])),
                MB_median=float(np.median([r["bytes"] / 1e6 for r in lg])), MB_max=float(max(r["bytes"] for r in lg) / 1e6), check_s_max=float(max(r["check_s"] for r in lg)))
    for r in lg: rows.append(("c", f, "endpoint", r["name"], f"rows={r['rows']};export_s={r['export_s']};check_s={r['check_s']};MB={r['bytes'] / 1e6}", SRC["lean_log"].format(field=f)))
allMB = np.concatenate([c[f]["MB"] for f in ORDER]); edges = np.arange(np.floor(np.log10(allMB.min()) * 8) / 8, np.ceil(np.log10(allMB.max()) * 8) / 8 + 1e-9, 0.125)
out["c"] = dict(order=ORDER, settings=c, MB_log10_edges=edges.tolist(), MB_counts={f: np.histogram(np.log10(c[f]["MB"]), edges)[0].tolist() for f in ORDER},
                n_total=sum(c[f]["n"] for f in ORDER), n_pass=sum(c[f]["n_pass"] for f in ORDER), MB_total_GB=float(allMB.sum() / 1e3))

# ---------------- d: verified spectral Poincare constants ----------------
a7 = json.load(open(os.path.join(FE, "exp2", "A7_verified", "a7_verified_poincare.json")))
TOYN = {"two_pixels": "two pixels", "chain20": "chain 20", "square6": "square 6×6", "square15": "square 15×15", "bottleneck": "bottleneck"}
cells = []
for r in a7["toy"]:
    cells.append(dict(kind="toy", name=TOYN[r["shape"]], n=r["n"], lam2=r["lambda2_numerical"], t=r["t_verified"], e=r["details"]["e"], rho=r["details"]["rho"], strict=r["details"]["strict"], verified=r["verified"], margin=r["details"]["margin"]))
skipped = []
for r in a7["real"]:
    if "t_verified" not in r: skipped.append(dict(county=r["county"], n=r["n"], note=r["note"])); continue
    cells.append(dict(kind="county", name=f"county {r['county']}", n=r["n"], lam2=r["lambda2_numerical"], t=r["t_verified"], e=r["details"]["e"], rho=r["details"]["rho"], strict=r["details"]["strict"], verified=r["verified"], margin=r["details"]["margin"]))
for x in cells:
    assert x["verified"] and x["t"] < x["lam2"] and x["rho"] >= x["e"]
    rows.append(("d", x["name"], "lambda2_numerical|t_verified|e|rho", f"n={x['n']}", f"{x['lam2']!r}|{x['t']!r}|{x['e']!r}|{x['rho']!r}", SRC["a7"]))
cells.sort(key=lambda x: x["lam2"])
out["d"] = dict(cells=cells, skipped_not_side_connected=skipped, n_county=sum(x["kind"] == "county" for x in cells), n_toy=sum(x["kind"] == "toy" for x in cells),
                n_strict=sum(x["strict"] for x in cells), e_max_county=max(x["e"] for x in cells if x["kind"] == "county"), n_real=a7["n_real"], n_verified_real=a7["n_verified_real"])
json.dump(out, open(os.path.join(DATA, "ED1.json"), "w", encoding="utf-8"), indent=1)
with open(os.path.join(HERE, "ED1_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["panel", "item", "quantity", "key", "value", "source"]); w.writerows(rows)
for m in mods: print(f"{m['name']:16s} files {m['n_files']:2d} lines {m['lines']:5d} thm {m['theorems']:3d} exported {m['exported']:3d}")
print("lean files", len(lean_files), "manifest files", man["n_source_files"], "non-lean", other_files, "theorems", sum(nthm.values()), "exported distinct", len(reported), "print cmds", n_print_cmds)
print("coverage levels (2,1,0):", out["b"]["n_level"], [(f["code"], f["n_decl"]) for f in out["b"]["families"]])
print("c:", {f: (c[f]["n"], round(c[f]["MB_median"], 2), round(c[f]["total_s_median"], 2)) for f in ORDER}, out["c"]["MB_total_GB"])
print("d:", out["d"]["n_county"], out["d"]["n_toy"], out["d"]["n_strict"], out["d"]["e_max_county"], skipped)
