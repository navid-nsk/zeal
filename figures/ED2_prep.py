"""ED2_prep - data layer for Extended Data Fig. 2 (class audits behind the rank and hot-spot certificates).

Reads ONLY the result files below and the Georgia raster; writes data/ED2.json (incl. the category raster of
panel c as row strings and the county-boundary runs) and ED2_source_data.csv. Provenance codes appear in comments only.
  a : exp2/C1/c1_feasibility_audit.json (+ its per-tract arrays exp2/C1/feasibility_r{r}.npz, same run; the arrays reproduce
      every number of the JSON, checked below), exp2/C1/c1_focal_counts.json, exp2/C1/c1_georgia_familysup4.json (C1)
  b : exp2/B3/b3_classes_georgia_v5.json (B3), exp2/checker/check_b3_incompat.json (7.5 km),
      exp2/checker/check_b3_incompat_7.5_15.json (15 km, the run on the producer's sample)
  c : geo_st13_raster.npz (mask, tract_idx, county_idx) + exp2/C1/feasibility_r{r}.npz
  d : exp2/C1/c1_georgia_familysup4.json (per-law critical values, half-widths), exp2/B3/b3_classes_georgia_v5.json (pair statistic)
  e : exp2/C1/c1_focal_counts.json, exp2/C1/c1_georgia_familysup4.json
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, io, csv, json, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
E2 = paths.EXP2
RAD = ["3.8", "7.5", "15", "30"]
SRC = {
    "audit": os.path.join(E2, "C1", "c1_feasibility_audit.json"),
    "focal": os.path.join(E2, "C1", "c1_focal_counts.json"),
    "c1": os.path.join(E2, "C1", "c1_georgia_familysup4.json"),
    "b3": os.path.join(E2, "B3", "b3_classes_georgia_v5.json"),
    "chk75": os.path.join(E2, "checker", "check_b3_incompat.json"),
    "chk15": os.path.join(E2, "checker", "check_b3_incompat_7.5_15.json"),
    "raster": paths.RASTER["georgia"],
}
for r in RAD:
    SRC[f"feas{r}"] = os.path.join(E2, "C1", f"feasibility_r{r}.npz")
REL = {k: paths.rel(v) for k, v in SRC.items()}
J = lambda k: json.load(open(SRC[k], encoding="utf-8"))
OUT = dict(sources=REL); ROWS = []


def row(panel, series, x, value, unit, src, note=""):
    ROWS.append(dict(panel=panel, series=series, x=x, value=value, unit=unit, source=REL[src], note=note))


key = lambda r: str(float(r))          # JSON keys '3.8', '7.5', '15.0', '30.0'
audit, focal, c1, b3 = J("audit"), J("focal"), J("c1"), J("b3")

# raster: tracts and population weights (same construction as the producers: units = sorted tract ids inside the mask)
ras = np.load(SRC["raster"]); mask = ras["mask"].astype(bool); tidx = ras["tract_idx"]; cidx = ras["county_idx"]
tr = tidx[mask]; w = ras["weight"][mask].astype(float)
units = np.unique(tr[tr >= 0]); K = len(units); inv = np.searchsorted(units, tr)
W = np.bincount(inv, weights=w, minlength=K); pop = W / W.sum()
n_counties = int(len(np.unique(cidx[mask][cidx[mask] >= 0])))

# ------------------------------------------------------------------------------------------------ a --
A_ = dict(radii_km=[float(r) for r in RAD], p=0.5, n_tracts=int(K),
          contained=[audit[key(r)]["pop_share_contained"] for r in RAD],
          mass_feasible=[focal[key(r)]["feasible"] for r in RAD],
          class_feasible=[focal[key(r)]["feasible_and_available"] for r in RAD],
          two_distinct=[focal[key(r)]["feasible_two_distinct_cells"] for r in RAD],
          ratio_median_json=[audit[key(r)]["eligible_over_window_mass_median"] for r in RAD],
          n_eligible_median=[audit[key(r)]["n_eligible_median"] for r in RAD], ratio=[], ratio_q=[], n_contained=[],
          src=[REL["audit"], REL["focal"], REL["c1"]] + [REL[f"feas{r}"] for r in RAD])
feas = {}
for i, r in enumerate(RAD):
    f = np.load(SRC[f"feas{r}"]); c = f["contained"]; ratio = f["eligible"][c] / np.maximum(f["muW"][c], 1e-12)
    # the per-tract arrays reproduce the audit JSON exactly
    assert abs(pop[c].sum() - audit[key(r)]["pop_share_contained"]) < 1e-12 and abs(pop[f["feasible"]].sum() - audit[key(r)]["pop_share_feasible"]) < 1e-12
    assert abs(np.median(ratio) - audit[key(r)]["eligible_over_window_mass_median"]) < 1e-12
    assert abs(c1["radii"][key(r)]["fat"]["class_feasible_pop_share"] - A_["class_feasible"][i]) < 1e-12
    feas[r] = f["feasible"].astype(bool)
    q = np.quantile(ratio, [0.05, 0.25, 0.5, 0.75, 0.95])
    A_["ratio"].append([float(v) for v in ratio]); A_["ratio_q"].append([float(v) for v in q]); A_["n_contained"].append(int(c.sum()))
    for s in ("contained", "mass_feasible", "class_feasible", "two_distinct"):
        row("a", s, r, A_[s][i], "population share", "focal" if s != "contained" else "audit")
    row("a", "eligible / window mass, q05 q25 q50 q75 q95", r, " ".join(f"{v:.4f}" for v in q), "ratio", f"feas{r}", f"over {int(c.sum())} anchor-contained tracts")
OUT["a"] = A_

# ------------------------------------------------------------------------------------------------ b --
chk = {"7.5": J("chk75")["7.5"], "15.0": J("chk15")["15.0"]}
B_ = dict(radii_km=[7.5, 15.0], src=[REL["b3"], REL["chk75"], REL["chk15"]])
for r in ("7.5", "15.0"):
    pc = b3["radii"][r]["pair_classes"]; ck = chk[r]
    assert ck["agree"] and ck["script_count"] == pc["empty"]
    B_[r] = dict(ordered_pairs=pc["ordered_pairs"], producer=dict(fractional=pc["empty_proofs"]["fractional"], subset_sums=pc["empty_proofs"]["brute"], total=pc["empty"]),
                 checker=dict(fractional=ck["by_proof_type"]["fractional"], subset_sums=ck["by_proof_type"]["subset_sums"], total=ck["proved_incompatible"]),
                 undecided=ck["undecided_large_overlap"], agree=bool(ck["agree"]), all_named_empty=bool(ck["C_N_empty_proved"] and b3["radii"][r]["all_named_class"]["empty_proven_by_pairs"]),
                 n_named=b3["radii"][r]["n_named"])
    for who in ("producer", "checker"):
        for t in ("fractional", "subset_sums"):
            row("b", f"{who}, {t}", r, B_[r][who][t], "ordered named pairs", "b3" if who == "producer" else ("chk75" if r == "7.5" else "chk15"))
    row("b", "undecided (large overlap, not counted)", r, B_[r]["undecided"], "ordered named pairs", "chk75" if r == "7.5" else "chk15")
OUT["b"] = B_

# ------------------------------------------------------------------------------------------------ c --
# smallest radius at which the audited class is feasible for the tract (index into RAD; len(RAD) = not feasible at 30 km)
cat = np.full(K, len(RAD), dtype=np.int16)
for i in reversed(range(len(RAD))):
    cat[feas[RAD[i]]] = i
grid = np.full(mask.shape, -1, dtype=np.int16); grid[mask] = cat[inv]
ys, xs = np.where(mask)
crop = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]
G = grid[crop[0]:crop[1], crop[2]:crop[3]]; CTY = np.where(mask, cidx, -1)[crop[0]:crop[1], crop[2]:crop[3]]
grid_rows = ["".join("." if v < 0 else str(int(v)) for v in r_) for r_ in G]          # '.' = outside the state mask
edges = []                     # county-boundary runs between pixel rows/columns i and i+1: [i, start, end, horizontal(1)/vertical(0)]
for i in range(G.shape[0] - 1):
    for diff, horiz in ((CTY[i + 1, :] != CTY[i, :], 1), (CTY[:, i + 1] != CTY[:, i], 0)):
        on = np.flatnonzero(np.diff(np.r_[0, diff.astype(int), 0]))
        edges += [[i, int(s_), int(e_), horiz] for s_, e_ in zip(on[::2], on[1::2])]
C_ = dict(categories=[f"{r} km" for r in RAD] + ["none"], tract_counts=[int((cat == i).sum()) for i in range(len(RAD) + 1)],
          pop_shares=[float(pop[cat == i].sum()) for i in range(len(RAD) + 1)], n_tracts=int(K), n_counties=n_counties, raster_px=list(mask.shape),
          crop=crop, grid_rows=grid_rows, county_edges=edges, src=[REL["raster"]] + [REL[f"feas{r}"] for r in RAD])
OUT["c"] = C_
for i, nm in enumerate(C_["categories"]):
    row("c", "tracts by smallest feasible radius", nm, C_["tract_counts"][i], "tracts", "raster", f"population share {C_['pop_shares'][i]:.4f}")

# ------------------------------------------------------------------------------------------------ d --
LAWS = ["gauss", "skew", "skew2", "t4"]
D_ = dict(laws=LAWS, law_labels=["Gaussian", "gamma, shape 4", "exponential", "Student t, 4 d.f."],
          columns=[dict(stat="endpoint", r=float(r)) for r in RAD] + [dict(stat="pair", r=7.5), dict(stat="pair", r=15.0)],
          c=[[c1["radii"][key(r)]["fat"]["c"][l] for r in RAD] + [b3["radii"]["7.5"]["c_pair"][l], b3["radii"]["15.0"]["c_pair"][l]] for l in LAWS],
          half_width_gaussian=[c1["radii"][key(r)]["fat"]["gaussian"]["half_width_median"] for r in RAD],
          half_width_family=[c1["radii"][key(r)]["fat"]["family"]["half_width_median"] for r in RAD],
          B=999, src=[REL["c1"], REL["b3"]])
D_["family_c_endpoint"] = [c1["radii"][key(r)]["fat"]["family"]["c"] for r in RAD]
assert all(abs(D_["family_c_endpoint"][j] - max(D_["c"][i][j] for i in range(len(LAWS)))) < 1e-12 for j in range(len(RAD)))
OUT["d"] = D_
for i, l in enumerate(LAWS):
    for j, col in enumerate(D_["columns"]):
        row("d", f"critical value, {col['stat']} statistic, {D_['law_labels'][i]}", f"{col['r']:g}", D_["c"][i][j], "standardized units", "c1" if col["stat"] == "endpoint" else "b3")
for j, r in enumerate(RAD):
    row("d", "median half-width, Gaussian", r, D_["half_width_gaussian"][j], "kfr", "c1")
    row("d", "median half-width, four-law family", r, D_["half_width_family"][j], "kfr", "c1")

# ------------------------------------------------------------------------------------------------ e --
E_ = dict(radii_km=[float(r) for r in RAD], single_cell_only=[focal[key(r)]["single_cell_only"] for r in RAD],
          log2_median=[focal[key(r)]["log2_cells_lower_bound_median_over_feasible"] for r in RAD], src=[REL["focal"], REL["c1"]])
for k in ("noiseless", "gaussian", "family"):
    E_[f"not_hot_{k}"] = [c1["radii"][key(r)]["fat"][k]["not_hot"] for r in RAD]
    E_[f"not_hot_two_{k}"] = [c1["radii"][key(r)]["fat"][k]["not_hot_two_distinct"] for r in RAD]
OUT["e"] = E_
for j, r in enumerate(RAD):
    row("e", "single admissible focal cell only", r, E_["single_cell_only"][j], "population share", "focal")
    row("e", "median log2 lower bound on admissible focal cells", r, E_["log2_median"][j], "log2 count", "focal")
    for k in ("noiseless", "gaussian", "family"):
        row("e", f"certified not-hot share, {k}, all locations", r, E_[f"not_hot_{k}"][j], "population share", "c1")
        row("e", f"certified not-hot share, {k}, >=2 distinct cells", r, E_[f"not_hot_two_{k}"][j], "population share", "c1")

json.dump(OUT, open(os.path.join(HERE, "data", "ED2.json"), "w", encoding="utf-8"), indent=1, default=float)
with open(os.path.join(HERE, "ED2_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys())); wr.writeheader(); wr.writerows(ROWS)
print(f"data/ED2.json written; {len(ROWS)} rows; tracts {K}, counties {n_counties}")
print("a contained", [round(v, 3) for v in A_["contained"]], "two distinct", [round(v, 3) for v in A_["two_distinct"]], "ratio medians", [round(q[2], 3) for q in A_["ratio_q"]])
print("b", {r: (B_[r]["producer"], B_[r]["checker"], B_[r]["undecided"]) for r in ("7.5", "15.0")})
print("c tract counts", C_["tract_counts"], "pop", [round(v, 3) for v in C_["pop_shares"]], "crop", crop)
print("d c", [[round(v, 2) for v in row_] for row_ in D_["c"]])
print("e", E_["single_cell_only"], E_["log2_median"])
