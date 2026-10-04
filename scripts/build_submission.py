"""build_submission.py -- assemble the journal submission package under <manuscript dir>/submission/:
  01_manuscript/            manuscript.md, manuscript.docx (figures embedded at their legends; Extended Data figures after them)
  02_figures/               Fig1..Fig6 (.png 600 dpi, .pdf, .svg)
  03_extended_data/         ED_Fig1..3 (.png, .pdf, .svg) and Extended_Data_Table1..4.xlsx (one file per table)
  04_source_data/           SourceData.xlsx (one sheet per figure panel)
  05_supplementary_information/   SI.md, SI.docx (when the SI source exists)
Run with the Python of the zeal environment:  python scripts/build_submission.py
The manuscript and SI sources are not part of this repository: set ZEAL_MANUSCRIPT_DIR to the folder that holds
the manuscript and SI Markdown sources (paths MAN and SI below; default folder: ../manuscript next to the repository). The figures are taken from
figures/figures/ of this repository (written by figures/F*.py and figures/ED*.py), the Source Data workbook from
figures/figures/SourceData.xlsx (figures/make_source_data.py).
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, re, shutil, json
import pandas as pd, pypandoc
HERE = os.path.abspath(os.environ.get("ZEAL_MANUSCRIPT_DIR") or paths.repo("..", "manuscript")); SUB = os.path.join(HERE, "submission")
FIG = paths.repo("figures", "figures"); MAN = os.path.join(HERE, "manuscript", "manuscript_v6.md"); SI = os.path.join(HERE, "SI", "SI_v6.md")
D = {k: os.path.join(SUB, k) for k in ("01_manuscript", "02_figures", "03_extended_data", "04_source_data", "05_supplementary_information")}
for d in D.values(): os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------------------------------ figures --
for n in range(1, 7):
    for ext in ("png", "pdf", "svg"): shutil.copy(os.path.join(FIG, f"F{n}.{ext}"), os.path.join(D["02_figures"], f"Fig{n}.{ext}"))
for n in range(1, 4):
    for ext in ("png", "pdf", "svg"): shutil.copy(os.path.join(FIG, f"ED{n}.{ext}"), os.path.join(D["03_extended_data"], f"ED_Fig{n}.{ext}"))
shutil.copy(os.path.join(FIG, "SourceData.xlsx"), os.path.join(D["04_source_data"], "SourceData.xlsx"))

# ------------------------------------------------------------------------------- Extended Data tables --
csvF6 = pd.read_csv(paths.repo("figures", "F6_source_data.csv"), dtype=str, keep_default_na=False)
csvE3 = pd.read_csv(paths.repo("figures", "ED3_source_data.csv"), dtype=str, keep_default_na=False)
csvF2 = pd.read_csv(paths.repo("figures", "F2_source_data.csv"), dtype=str, keep_default_na=False)

def g(src, p, ds, m, q, k1=None, k2="CROWN|0.01|ZD"):
    s = src[(src.panel == p) & (src.dataset == ds) & (src.model == m) & (src.quantity == q)]
    if k1 is not None: s = s[s.key1 == k1]
    if k2 is not None and "key2" in s.columns: s = s[s.key2 == k2]
    return s.value.tolist()

t1 = pd.DataFrame([
    ["Georgia, Opportunity Atlas income rank", "1,937 tracts → 159 counties", "1,931", "tract estimates with published s.e.", "tract estimates", "512², 166,017 pixels", "0.018 (4.3 %)"],
    ["Greater Manchester, Level-4 qualification share", "1,702 LSOAs → 353 MSOAs (8,966 OAs)", "1,702", "OA counts (Census 2021)", "LSOA and MSOA means", "1,024², 54 m, 434,168 pixels", "0.060 (5.1 %)"],
    ["Greater Manchester, bad-health share", "1,702 LSOAs → 353 MSOAs (8,966 OAs)", "1,702", "OA counts (Census 2021)", "LSOA and MSOA means", "1,024², 54 m, 434,168 pixels", "0.018 (3.7 %)"],
    ["Mexico, Relative Wealth Index", "459 municipalities → 9 states (21,245 tiles)", "459", "2.4-km tiles", "municipality and state means", "1,024², 2.4-km tiles, 119,776 pixels", "0.080 (3.4 %)"],
], columns=["Setting", "Fine → coarse units", "Nested pairs", "Fine truth", "Field fitted to", "Raster", "Median value-enclosure width (share of field range)"])

def f2(setting, q):
    s = csvF2[(csvF2.panel == "b") & (csvF2.setting == setting) & (csvF2.quantity == q)]; return float(s.value.iloc[0])
names = {"georgia": "Georgia", "gm_q4": "Greater Manchester, qualification", "gm_bad": "Greater Manchester, bad health", "mx_rwi": "Mexico"}
ctop = {"georgia": 1.019, "gm_q4": 1.108, "gm_bad": 1.190, "mx_rwi": 1.155}; nimp = {"georgia": 310, "gm_q4": 706, "gm_bad": 706, "mx_rwi": 18}; mesh = {"georgia": 2, "gm_q4": 2, "gm_bad": 2, "mx_rwi": 1}
signs = {"georgia": (1150, 690), "gm_q4": (426, 1144), "gm_bad": (304, 1246), "mx_rwi": (329, 88)}; npairs = {"georgia": 1931, "gm_q4": 1702, "gm_bad": 1702, "mx_rwi": 459}
medratio = {"georgia": 1.125, "gm_q4": 1.256, "gm_bad": 1.231, "mx_rwi": 1.575}
rows = []
for k in ("georgia", "gm_q4", "gm_bad", "mx_rwi"):
    L, U = f2(k, "global_witness_checker_L"), f2(k, "ceiling_checker_U"); inv, sen = signs[k]
    rows.append([names[k], ctop[k], round(L, 4), round(f2(k, "S_separable_witness_x"), 3), round(U, 4), mesh[k], round(f2(k, "S_PhiC_x"), 3), round(f2(k, "S_V_x"), 3), round(U - L, 3), nimp[k], inv, sen, npairs[k] - inv - sen, medratio[k]])
t2 = pd.DataFrame(rows, columns=["Setting", "Lattice top", "Global witness L (checker)", "Separable witness", "Lifted outer guarantee U (checker)", "Mesh parameter m", "Pixel outer bound", "Value-only bound", "Certified width U − L", "Accepted cellwise improvements", "Pairs certified invariant", "Pairs certified sensitive", "Pairs open", "Median outer/witness width per pair"])

t3 = pd.DataFrame([
    ["Slope configurations (8 neighbourhoods × K ∈ {3, 5, 8} × β ∈ {∞, ½})", "48"],
    ["Certified ranges containing both signs / witnessed both signs", "48 / 47"],
    ["OA-level slopes negative", "8 of 8 neighbourhoods"],
    ["Certified / witnessed width: median (range)", "1.071 (1.009 to 1.245)"],
    ["Admissible cells enumerated per family", "4.3 × 10³ to 2.5 × 10⁸"],
    ["Largest reduced cost after exact pricing", "≤ 3.5 × 10⁻¹⁶"],
    ["Charge certificates kernel-checked", "96 / 96"],
    ["12-OA neighbourhoods: classes with the exact range inside the certificate / false signs", "146 / 146; 0"],
    ["Certified not-hot population share at 3.8 / 7.5 / 15 / 30 km, exact data", "0.31 / 0.47 / 0.58 / 0.71"],
    ["Certified not-hot population share, Gaussian law (joint 0.95)", "0.260 / 0.407 / 0.490 / 0.544"],
    ["Certified not-hot population share, four-law family (joint 0.95)", "0.224 / 0.358 / 0.444 / 0.472"],
    ["Feasible population share", "0.34 / 0.56 / 0.73 / 0.91"],
    ["Certified hot share, four-law family", "0 / 0 / 0.0003 / 0"],
    ["Certified pair order at 7.5 km (15 km): exact / Gaussian / family", "0.34 (0.26) / 0.15 (0.10) / 0.075 (0.062)"],
    ["Witness-anchored universe at 7.5 km (15 km): locations", "102 (106)"],
    ["Certified not-top-20-per-cent share, exact / Gaussian / family, 7.5 km (15 km)", "0.63 (0.47) / 0.21 (0.07) / 0.036 (0.015)"],
    ["Joint coverage of the exchangeable-rank envelope (guarantee 0.950)", "0.970 over 600 trials"],
    ["Median half-width at 15 km: tailored / Bonferroni Gaussian / Bonferroni family", "0.0232 / 0.0667 / 0.2452"],
], columns=["Quantity", "Value"])

coh = {}
c = csvE3[(csvE3.panel == "c") & (csvE3.key1 == "0.01")]
for ds in ("synthetic", "jul2014", "jan2014"):
    for m in ("mlp", "feat", "feat24"):
        ks = sorted((int(r.key2), float(r.value)) for r in c[(c.dataset == ds) & (c.model == m)].itertuples()); coh[(ds, m)] = next((k for k, v in ks if v >= 1), None)
dsn = {"synthetic": "Synthetic feeders", "jul2014": "Electricity clients, July 2014", "jan2014": "Electricity clients, January 2014"}; mn = {"mlp": "raw-lag", "feat": "block-mean", "feat24": "day-block"}
rows = []
for ds in ("synthetic", "jul2014", "jan2014"):
    for m in ("mlp", "feat", "feat24"):
        auc = float(g(csvE3, "a", ds, m, "eval_auc", k2=None)[0]); G = float(g(csvF6, "c", ds, m, "G_median", "S5")[0])
        exV = float(g(csvF6, "d", ds, m, "excess_V_median", "S5")[0]); exZ = float(g(csvF6, "d", ds, m, "excess_Z_median", "S5")[0])
        s2 = int(g(csvF6, "e", ds, m, "decision_Z_certified_sensitive", "S2")[0]); s3 = int(g(csvF6, "e", ds, m, "decision_Z_certified_invariant", "S3")[0])
        s5V = int(g(csvF6, "e", ds, m, "decision_V_certified_sensitive", "S5")[0]); s5Z = int(g(csvF6, "e", ds, m, "decision_Z_certified_sensitive", "S5")[0])
        st = g(csvF6, "f", ds, m, "S4_state", None); inv = sum(x == "invariant" for x in st); two = sum(x == "sensitive" for x in st)
        rows.append([dsn[ds], mn[m], round(auc, 3), coh[(ds, m)], round(G, 2), round(exV, 1), round(exZ, 1), s2, s3, inv, two, s5V, s5Z])
t4 = pd.DataFrame(rows, columns=["Dataset", "Network", "AUC (evaluation window)", "Coherence length (h)", "G, intra-day contrasts (median)", "Excess over witness, intra-day, value-only (median)", "Excess over witness, intra-day, fused (median)", "Before/after sign certified sensitive (of 36)", "Binned trend certified invariant (of 36)", "Top-3 ranking certified invariant (of 28 report times)", "Top-3 ranking: two certified sets (of 28)", "Intra-day contrasts certified sensitive, value-only (of 72)", "Intra-day contrasts certified sensitive, fused (of 72)"])
notes = {
    "ED Table 1": "Settings, inputs and enclosures. Field: multilayer perceptron with random Fourier features; enclosures by CROWN with the pixel as the perturbed input. Reference movement = 1 (box-midpoint field).",
    "ED Table 2": "Map-level certificates (× reference movement). Endpoints L and U are the independent checker's exact recomputations; all 11,588 endpoints passed the verified checker. Certified invariant: outer range excludes zero; certified sensitive: realizable witnesses of both signs.",
    "ED Table 3": "Zoning statistics: Greater Manchester slope certificates (health share on qualification share) and Georgia hot-spot and rank certificates (Opportunity Atlas). Certified shares are exact-data statistics or decisions conditional on the declared law family; no sampling error.",
    "ED Table 4": "Temporal pooling with CROWN enclosures, ε = 0.01 training s.d., dyadic lags 1 to 128 h. Coherence length = first lag at which the increment-to-value width ratio reaches one. Rigorous enclosure: 1,904,472 of 1,904,472 stored boxes contain their rigorous enclosure (overshoot ≤ 1.4 × 10⁻¹²); 288 of 288 end-to-end certificates passed.",
}
for i, (t, name) in enumerate([(t1, "ED Table 1"), (t2, "ED Table 2"), (t3, "ED Table 3"), (t4, "ED Table 4")], 1):
    xl = os.path.join(D["03_extended_data"], f"Extended_Data_Table{i}.xlsx")
    with pd.ExcelWriter(xl, engine="openpyxl") as xw:
        t.to_excel(xw, sheet_name=f"Extended Data Table {i}", index=False, startrow=1)
        xw.sheets[f"Extended Data Table {i}"].cell(row=1, column=1, value=f"Extended Data Table {i} | {notes[name]}")
    print("wrote", xl)
old_xl = os.path.join(D["03_extended_data"], "Extended_Data_Tables.xlsx")
if os.path.exists(old_xl): os.remove(old_xl)

# ------------------------------------------------------------------------------------- manuscript docx --
md = open(MAN, encoding="utf-8").read()
def embed(md, pattern, path_fn):
    def rep(m):
        n = int(m.group(1)); p = path_fn(n).replace("\\", "/")
        return f"![]({p}){{width=100%}}\n\n" + m.group(0)
    return re.sub(pattern, rep, md)
md = embed(md, r"\*\*Fig\. (\d) \|", lambda n: os.path.join(D["02_figures"], f"Fig{n}.png"))
md = embed(md, r"\*\*Extended Data Fig\. (\d) \|", lambda n: os.path.join(D["03_extended_data"], f"ED_Fig{n}.png"))
# Extended Data tables appended as text tables for the reader of the docx
for i, t in enumerate((t1, t2, t3, t4), 1):
    md += f"\n\n**Extended Data Table {i} |** {notes[f'ED Table {i}']}\n\n" + t.to_markdown(index=False) + "\n"
tmp = os.path.join(D["01_manuscript"], "_manuscript_with_figures.md"); open(tmp, "w", encoding="utf-8").write(md)
shutil.copy(MAN, os.path.join(D["01_manuscript"], "manuscript.md"))
pypandoc.convert_file(tmp, "docx", outputfile=os.path.join(D["01_manuscript"], "manuscript.docx"), extra_args=["--from=markdown+tex_math_dollars+pipe_tables", "--wrap=none", f"--resource-path={D['02_figures']};{D['03_extended_data']}"])
os.remove(tmp); print("wrote manuscript.docx")

# ---------------------------------------------------------------------------------------------- SI --
if os.path.exists(SI):
    shutil.copy(SI, os.path.join(D["05_supplementary_information"], "SI.md"))
    pypandoc.convert_file(SI, "docx", outputfile=os.path.join(D["05_supplementary_information"], "SI.docx"), extra_args=["--from=markdown+tex_math_dollars+pipe_tables", "--wrap=none"])
    print("wrote SI.docx")
else:
    print("SI source not present; skipped")
print("done ->", SUB)
