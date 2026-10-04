"""F2_prep - data layer of Fig. 2 (map-level certified brackets on three geographies).

Reads ONLY the result files listed in the source table of the data file and the input rasters; writes data/F2.json (every drawn
number with its source path), data/F2_maps.npz (cropped fine-cell label rasters and coarse-boundary segments: the geometry
needed to paint the per-fine-cell values; the values themselves are per cell in F2.json) and F2_source_data.csv (tidy).
Provenance (comments only): production = exp2 C5 / C5_M4full / C5_M2full; global witness = exp2 C5 global-witness runs;
checker = exp2 checker; Lean log = lean certificates_all.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import os, json, csv
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FE = paths.RESULTS_ROOT
LEAN = paths.LEAN_DIR
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)

ORDER = ["georgia", "gm_q4", "gm_bad", "mx_rwi"]
SRC = {
    "georgia": dict(prod="exp2/C5/c5_georgia.json", raster=paths.RASTER["georgia"]),
    "gm_q4": dict(prod="exp2/C5_M4full/c5_gm_q4.json", raster=paths.RASTER["gm_q4"]),
    "gm_bad": dict(prod="exp2/C5_M4full/c5_gm_bad.json", raster=paths.RASTER["gm_bad"]),
    "mx_rwi": dict(prod="exp2/C5_M2full/c5_mx_rwi.json", raster=paths.RASTER["mx_rwi"]),
}
UNITS = {"georgia": ("tract", "county"), "gm_q4": ("LSOA", "MSOA"), "gm_bad": ("LSOA", "MSOA"), "mx_rwi": ("municipality", "state")}
for f in ORDER:
    SRC[f]["glob"] = f"exp2/C5/c5_global_witness_{f}.json"
    SRC[f]["checker"] = f"exp2/checker/check_c5_witness_{f}.json"
    SRC[f]["lean"] = f"lean/certificates_all/lean_c5_log_{f}.json"


def P(rel):
    return paths.lean_file(rel[5:]) if rel.startswith("lean/") else os.path.join(FE, rel)


def rel_src(rel):
    return rel if rel.startswith("lean/") else ("results/" + rel)


def merged_segments(lab):
    """Boundary segments between pixels of different label (label -1 = outside), merged into straight runs.
    Pixel (i, j) occupies [j, j+1] x [i, i+1] in (x = column, y = row) coordinates. Returns int32 array [x0, y0, x1, y1]."""
    H, W = lab.shape; pad = np.full((H + 2, W + 2), -1, lab.dtype); pad[1:-1, 1:-1] = lab; segs = []
    # horizontal edges at y = i (between padded rows i and i+1 -> original rows i-1 and i)
    dh = (pad[:-1, 1:-1] != pad[1:, 1:-1]) & ((pad[:-1, 1:-1] >= 0) | (pad[1:, 1:-1] >= 0))   # shape (H+1, W)
    for i in range(dh.shape[0]):
        row = dh[i]; j = 0
        while j < W:
            if row[j]:
                k = j
                while k < W and row[k]: k += 1
                segs.append((j, i, k, i)); j = k
            else:
                j += 1
    dv = (pad[1:-1, :-1] != pad[1:-1, 1:]) & ((pad[1:-1, :-1] >= 0) | (pad[1:-1, 1:] >= 0))   # shape (H, W+1)
    for j in range(dv.shape[1]):
        col = dv[:, j]; i = 0
        while i < H:
            if col[i]:
                k = i
                while k < H and col[k]: k += 1
                segs.append((j, i, j, k)); i = k
            else:
                i += 1
    return np.asarray(segs, np.int32)


out = dict(figure="F2", settings={}, order=ORDER, sources={})
maps_npz = {}
rows_csv = []
all_vals = []
for f in ORDER:
    s = SRC[f]; prod = json.load(open(P(s["prod"]))); S = prod["summary"]; pairs = prod["pairs"]
    glob = json.load(open(P(s["glob"]))); G = glob["summary"]; hist = glob["history"]
    chk = json.load(open(P(s["checker"])))
    lean = json.load(open(P(s["lean"])))
    ras = np.load(s["raster"]); mask = ras["mask"].astype(bool); fine = np.where(mask, ras["tract_idx"], -1).astype(np.int32); coarse = np.where(mask, ras["county_idx"], -1).astype(np.int32)
    mov = float(S["movement_mid"]); M_code = int(S["M"]); m = M_code // 2
    assert prod["summary"]["pairs"] == len(pairs) == chk["pairs"] == G["pairs"], f
    assert S["cropped_pairs"] == 0

    # ---------------- units (counts from the raster indices and the pairs) ----------------
    fine_ids = np.unique(fine[fine >= 0]); coarse_ids = np.unique(coarse[coarse >= 0])
    pf = {int(p["fine"]) for p in pairs}; pc = {int(p["coarse"]) for p in pairs}
    # nesting check: every pair's fine cell lies inside exactly its coarse cell
    nest_bad = 0
    for p in pairs:
        cs = np.unique(coarse[fine == p["fine"]])
        nest_bad += int(not (len(cs) == 1 and cs[0] == p["coarse"]))
    assert nest_bad == 0, f

    # ---------------- (a) per-fine-cell certified pair-range width / reference movement ----------------
    width = {int(p["fine"]): (p["U_X"] - p["L_X"]) / mov for p in pairs}
    all_vals += list(width.values())
    # crop: bounding box of the mask components holding >= 1 % of the mask pixels (offshore specks outside are recorded)
    lab_cc, ncc = ndimage.label(mask); sizes = ndimage.sum(mask, lab_cc, index=np.arange(1, ncc + 1))
    keep = np.isin(lab_cc, 1 + np.nonzero(sizes >= 0.01 * mask.sum())[0])
    ii, jj = np.nonzero(keep); r0, r1, c0, c1 = int(ii.min()), int(ii.max()) + 1, int(jj.min()), int(jj.max()) + 1
    dropped_px = int(mask.sum() - mask[r0:r1, c0:c1].sum())
    fine_c = fine[r0:r1, c0:c1].copy(); coarse_c = coarse[r0:r1, c0:c1].copy()
    segs = merged_segments(coarse_c)
    maps_npz[f"fine_{f}"] = fine_c.astype(np.int32); maps_npz[f"segs_{f}"] = segs
    unpaired = sorted(int(x) for x in set(fine_ids.tolist()) - pf)
    unpaired_px = int(np.isin(fine, unpaired).sum())

    # ---------------- (b) map-level brackets ----------------
    br = dict(L_checker=float(chk["S_witness_lower"]), U_checker=float(chk["S_ceiling_upper"]), bracket=[float(x) for x in chk["bracket"]],
              S_global_witness_x=float(G["S_global_witness_x"]), S_X_x=float(S["S_X_x"]), S_separable_witness_x=float(S["S_witness_x"]),
              S_PhiC_x=float(S["S_PhiC_x"]), S_V_x=float(S["S_V_x"]), S_lattice_top_x=float(G["S_global_lattice_top_x"]),
              witness_feasible_exact=bool(chk["witness_feasible_exact"]), checker_M_witness=int(chk["M"]))
    assert abs(br["bracket"][0] - br["L_checker"]) < 1e-15 and abs(br["bracket"][1] - br["U_checker"]) < 1e-15

    # ---------------- (c) per-pair outer / witness width ratio and sign decisions ----------------
    ratio = [p["width_X"] / p["width_wit"] for p in pairs]
    assert all(p["width_wit"] > 1e-9 for p in pairs)
    n_inv = sum(p["sign_outer"] != "both" for p in pairs)                     # outer range excludes 0: sign certified invariant
    n_sens = sum(p["sign_wit"] == "both" for p in pairs)                      # witnessed range straddles 0: both signs realized
    n_open = len(pairs) - n_inv - n_sens
    assert n_inv == S["sign_resolved_outer"] and len(pairs) - n_sens == S["sign_resolved_wit"]
    assert sum(p["sign_outer"] != "both" and p["sign_wit"] == "both" for p in pairs) == 0     # outer-resolved => witness-resolved

    # ---------------- (d) global witness construction: cumulative share of the summed cellwise gains ----------------
    dq = np.array([h["Qb_new"] - h["Qb_old"] for h in hist]); sw = np.array([h["sweep"] for h in hist])
    xs = [0.0]; ys = [0.0]; cum = np.cumsum(dq) / dq.sum()
    for s_ in sorted(set(sw.tolist())):
        idx = np.nonzero(sw == s_)[0]
        for k, ix in enumerate(idx):
            xs.append(float(s_ + (k + 1) / len(idx))); ys.append(float(100.0 * cum[ix]))
    share_sweep1 = float(100.0 * dq[sw == 0].sum() / dq.sum())
    gw = dict(x=xs, y=ys, n_accepted=int(len(hist)), n_sweep=[int((sw == s_).sum()) for s_ in sorted(set(sw.tolist()))], min_dQ=float(dq.min()),
              share_sweep1_pct=share_sweep1, S_lattice_top_x=br["S_lattice_top_x"], S_global_witness_x=br["S_global_witness_x"], nodes=int(G["nodes"]), mesh_M_code=int(G["mesh_M"]))

    # ---------------- (e) Lean checker log ----------------
    ex = np.array([r["bound_exact"] for r in lean]); ar = np.array([r["artifact"] for r in lean]); sgn = np.array([1.0 if r["dir"] == "U" else -1.0 for r in lean])
    slack = sgn * (ar - ex) / np.abs(ex)                                           # outward relative slack: U: (reported - exact)/|exact|; L: (exact - reported)/|exact|
    assert np.all(slack > 0) and all(r["passed"] and r["artifact_certified"] and r["verdict"].startswith("PASS") for r in lean)
    le = dict(n=len(lean), n_pass=int(sum(r["passed"] for r in lean)), n_certified=int(sum(r["artifact_certified"] for r in lean)),
              exact=ex.tolist(), producer=ar.tolist(), dir=[r["dir"] for r in lean], slack=slack.tolist(),
              rows=[int(r["rows"]) for r in lean], check_s=[float(r["check_s"]) for r in lean], export_s=[float(r["export_s"]) for r in lean], MB=[r["bytes"] / 1e6 for r in lean],
              slack_min=float(slack.min()), slack_max=float(slack.max()), slack_median=float(np.median(slack)))

    st = dict(fine_unit=UNITS[f][0], coarse_unit=UNITS[f][1], n_fine_raster=int(len(fine_ids)), n_coarse_raster=int(len(coarse_ids)),
              n_fine_pairs=len(pf), n_coarse_pairs=len(pc), n_pairs=len(pairs), unpaired_fine=unpaired, unpaired_px=unpaired_px,
              M_code=M_code, m=m, movement_mid=mov, crop=[r0, r1, c0, c1], crop_dropped_px=dropped_px, raster_shape=list(mask.shape), mask_px=int(mask.sum()),
              orientation="native raster arrays, row 0 = north (verified on the Georgia outline: Atlanta NW, Savannah on the east coast, straight Florida border at the bottom); drawn with origin='upper'; no transpose",
              width_by_fine=width, width_min=float(min(width.values())), width_max=float(max(width.values())), width_median=float(np.median(list(width.values()))),
              bracket=br, ratio=ratio, ratio_median=float(np.median(ratio)), ratio_min=float(min(ratio)), ratio_max=float(max(ratio)),
              ratio_median_summary=float(S["width_X_over_wit_median"]), sign=dict(invariant=n_inv, sensitive=n_sens, open=n_open),
              global_witness=gw, lean=le,
              sources={k: rel_src(v) for k, v in s.items() if k != "raster"} | {"raster": paths.rel(s["raster"])})
    assert abs(st["ratio_median"] - st["ratio_median_summary"]) < 1e-12
    out["settings"][f] = st

    # tidy source data
    src_p, src_c, src_g, src_l = rel_src(s["prod"]), rel_src(s["checker"]), rel_src(s["glob"]), rel_src(s["lean"])
    for fi, v in sorted(width.items()):
        rows_csv.append(("a", f, "pair_range_width_over_reference", f"fine={fi}", v, src_p))
    for k, v in (("global_witness_checker_L", br["L_checker"]), ("ceiling_checker_U", br["U_checker"])):
        rows_csv.append(("b", f, k, "", v, src_c))
    for k in ("S_separable_witness_x", "S_PhiC_x", "S_V_x", "S_X_x"):
        rows_csv.append(("b", f, k, "", br[k], src_p))
    rows_csv.append(("b", f, "S_global_witness_x_producer", "", br["S_global_witness_x"], src_g))
    for p, r_ in zip(pairs, ratio):
        rows_csv.append(("c", f, "width_outer_over_width_witness", f"coarse={p['coarse']};fine={p['fine']}", r_, src_p))
    for k, v in st["sign"].items():
        rows_csv.append(("c", f, f"sign_{k}_pairs", "", v, src_p))
    for x_, y_ in zip(gw["x"], gw["y"]):
        rows_csv.append(("d", f, "cumulative_cellwise_gain_pct", f"progress={x_:.6f}", y_, src_g))
    rows_csv.append(("d", f, "S_lattice_top_x", "", br["S_lattice_top_x"], src_g))
    for r, s_l in zip(lean, slack):
        rows_csv.append(("e", f, "endpoint", f"{r['name']}", f"exact={r['bound_exact']!r};producer={r['artifact']!r};rel_slack={s_l!r};rows={r['rows']};check_s={r['check_s']};verdict={r['verdict'][:4]}", src_l))

# shared colour range of the maps (outward to two significant digits)
lo = float(min(all_vals)); hi = float(max(all_vals))
out["map_range"] = [float(np.floor(lo * 100) / 100), float(np.ceil(hi * 10) / 10)]
out["lean_total"] = dict(n=sum(out["settings"][f]["lean"]["n"] for f in ORDER), n_pass=sum(out["settings"][f]["lean"]["n_pass"] for f in ORDER))
# panel-e histogram of the relative outward slack (shared log bins)
lg = np.log10(np.concatenate([out["settings"][f]["lean"]["slack"] for f in ORDER])); edges = np.arange(np.floor(lg.min() * 4) / 4, np.ceil(lg.max() * 4) / 4 + 1e-9, 0.25)
out["slack_hist"] = dict(log10_edges=edges.tolist(), counts={f: np.histogram(np.log10(out["settings"][f]["lean"]["slack"]), edges)[0].tolist() for f in ORDER})
out["notes"] = dict(
    transpose="The first-order arrays (data/enclosures/first_order_*.npz) are aligned with the raster arrays natively (isfinite(lo) == mask); the experiment scripts transpose both jointly, which does not change cell correspondence. The maps use only the raster label arrays and are drawn in native orientation (row 0 = north).",
    d_running_value="The history stores per-cell Qb_old/Qb_new; summing the cellwise changes does not reproduce S_global exactly (cross-cell coupling at shared nodes: Georgia 1.2022 vs 1.2033, GM q4 1.4619 vs 1.4583, GM bad 1.5949 vs 1.5919, Mexico 1.29019 vs 1.28999), so panel d plots the cumulative share of the summed cellwise gains, with the exact start (lattice top) and end (global witness) values printed.",
    e_slack="relative outward slack = (producer - exact)/|exact| for upper endpoints and (exact - producer)/|exact| for lower endpoints; all > 0.",
)
json.dump(out, open(os.path.join(DATA, "F2.json"), "w", encoding="utf-8"), indent=0)
np.savez_compressed(os.path.join(DATA, "F2_maps.npz"), **maps_npz)
with open(os.path.join(HERE, "F2_source_data.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["panel", "setting", "quantity", "key", "value", "source"]); w.writerows(rows_csv)
for f in ORDER:
    st = out["settings"][f]
    print(f, "fine", st["n_fine_raster"], "/", st["n_fine_pairs"], "coarse", st["n_coarse_raster"], "/", st["n_coarse_pairs"], "unpaired", st["unpaired_fine"], st["unpaired_px"], "px; crop", st["crop"], "dropped", st["crop_dropped_px"],
          "segs", len(maps_npz[f"segs_{f}"]), "sign", st["sign"], "ratio med", round(st["ratio_median"], 4), "gain sweep1 %", round(st["global_witness"]["share_sweep1_pct"], 1))
print("map range", out["map_range"], "lean", out["lean_total"])
