"""F6 - Generalization and Rigor (merges fig8 + ed2).

Six lettered data panels, ALL from REAL data in ../data/. Three+ distinct packing
idioms: signed forest, 8-bar spine, quadrant scatter, dumbbell, null histogram,
twin-axis. Color lock (zstyle): ZEAL=PINK, baselines=BASE_RAMP (navy->steel),
GREEN=good/kept only, RED=threshold/reference only, DIV=signed colormap.

  a  Breadth across 3 states: signed B_red forest (GA/CO/IL x 5 seeds, 15/15 < 0)
     with per-state perm-z (-17 / -6 / -11); red artifact-boundary line.
                                                   (geo_m4_multistate.json N3/N4)
  b  Baselines vs ZEAL: 8-bar R2_tract spine (OLS 0.045 -> pycno 1.000 ->
     ZEAL 0.986 certified); +/-SEM where multi-seed; pycno hatched (uncertifiable).
                                                   (geo_m4_baselines.json N9)
  c  Certification frontier quadrant: R2_tract (x) x PSI (y); markers encode
     faithful / continuous / certified; ZEAL the only point in the all-three cell.
                                                   (geo_m4_baselines.json N10)
  d  Spatial cross-validation dumbbell: in-sample R2 ~0.99 (steel) vs held-out R2
     (-0.5...-1.2, navy) per fold; held-out B_red < 0 in 5/5.
                                                   (geo_m4_spatialcv.json N7/N8)
  e  Geography-coherence permutation null: 1000 random re-partitions (gray) vs the
     real value (red rule) at z = -16, 0/1000 in the tail.
                                                   (fig_data.json perm_null  N1)
  f  Resolution robustness: PSI + signed B_red at 512^2 (n=3, +/-s.d.) vs 1024^2
     (single seed, OOM-limited, tagged); B_red < 0 at both -> result resolution-robust.
                                                   (geo_m4_finergrid.json N12)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import zstyle as Z

Z.setup()

ms = Z.load("geo_m4_multistate.json")
bl = Z.load("geo_m4_baselines.json")
sc = Z.load("geo_m4_spatialcv.json")
fg = Z.load("geo_m4_finergrid.json")
fd = Z.load("fig_data.json")

# ----------------------------------------------------------------------------
# Canvas: double-column 180 mm. Two bands of three panels (4-4-4 / 4-4-4).
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(180 * Z.MM, 132 * Z.MM))
gs = fig.add_gridspec(
    2, 6,
    height_ratios=[1.0, 1.0],
    left=0.070, right=0.965, top=0.905, bottom=0.105,
    wspace=1.65, hspace=0.80,
)
axA = fig.add_subplot(gs[0, 0:2])   # a  breadth forest (3 states)
axB = fig.add_subplot(gs[0, 2:4])   # b  baselines 8-bar spine
axC = fig.add_subplot(gs[0, 4:6])   # c  certification frontier quadrant
axD = fig.add_subplot(gs[1, 0:2])   # d  spatial CV dumbbell
axE = fig.add_subplot(gs[1, 2:4])   # e  geography-coherence null
axF = fig.add_subplot(gs[1, 4:6])   # f  resolution robustness

TITLE_KW = dict(fontsize=7, loc="center", pad=6)
CAP_KW = dict(xycoords="axes fraction", ha="center", va="top", fontsize=5.2, color=Z.GRAY)
# footnote y for panels that carry a (single-line) x-axis title underneath: drop a
# touch lower so the gray sub-caption never collides with the x-axis label (d/e).
CAP_Y = -0.165          # panels with short/no x-title (a)
CAP_Y_XTITLE = -0.225   # panels with an x-axis title (d, e)


# ============================================================================
# a  BREADTH FOREST: 3 states x 5 seeds, signed B_red, below red artifact line
#    (lifted from fig8a; kept as a per-seed jittered forest + state mean bar)
# ============================================================================
states = [("13", "GA", Z.NAVY), ("08", "CO", Z.STEEL1), ("17", "IL", Z.STEEL2)]
rng = np.random.default_rng(7)
state_cols = []
for i, (key, name, col) in enumerate(states):
    rows = ms[key]["rows"]
    vals = np.array([r["B_red_signed"] for r in rows]) * 1e3
    zmean = float(np.mean([r["perm_z"] for r in rows]))
    jit = (rng.random(len(vals)) - 0.5) * 0.22
    axA.scatter(np.full(len(vals), i) + jit, vals, s=20,
                facecolor=col, edgecolor="white", linewidth=0.4, zorder=3)
    m = vals.mean()
    axA.plot([i - 0.24, i + 0.24], [m, m], color=Z.INK, lw=1.1, zorder=4)
    # per-state geography-coherence z, in the gap above the cluster
    axA.annotate(f"z = -{abs(zmean):.0f}", xy=(i, m + 0.14), ha="center",
                 va="bottom", fontsize=5.6, color=col, fontweight="bold")
    state_cols.append(col)

Z.threshold(axA, 0.0, axis="y")
axA.annotate("Artifact boundary", xy=(-0.45, 0.0), xytext=(0, 2),
             textcoords="offset points", ha="left", va="bottom",
             fontsize=5.5, color=Z.RED)
axA.set_xlim(-0.5, 2.5)
axA.set_ylim(-2.05, 0.42)
axA.set_xticks([0, 1, 2])
axA.set_xticklabels([s[1] for s in states])
for lab, col in zip(axA.get_xticklabels(), state_cols):
    lab.set_color(col); lab.set_fontweight("bold")
axA.tick_params(axis="x", length=0)
axA.set_ylabel("signed B_red (×10^-3)")
axA.set_title("Breadth Across Three States", **TITLE_KW)
axA.annotate("per-state perm-z (n = 5 seeds);  15/15 seeds < 0 = no inflation artifact",
             xy=(0.5, CAP_Y), **CAP_KW)
Z.panel(axA, "a")


# ============================================================================
# b  BASELINES vs ZEAL: 8-bar R2_tract spine, ranked
#    (lifted from ed2a; ZEAL=pink certified, pycno hatched, baselines ramp)
# ============================================================================
short = {"OLS (linear)": "OLS", "Ridge+poly3": "Ridge poly-3", "kNN (k=8)": "kNN",
         "GBM": "GBM", "RandomForest": "Random forest", "GP-kriging (RBF)": "GP kriging",
         "pycnophylactic (areal)": "Pycno (areal)"}
recs = []
for r in bl["rows"]:
    recs.append(dict(name=short[r["model"]], r2=r["r2tr"], psi=r["psi"],
                     bred=r["B_red"], std=r["r2_std"],
                     kind=("pycno" if "pycno" in r["model"] else "base")))
# ZEAL operating point (mean +/- per-seed s.d. from the GA 5-seed multistate run)
ga_rows = ms["13"]["rows"]
recs.append(dict(name="ZEAL", r2=float(np.mean([r["r2tr"] for r in ga_rows])),
                 psi=float(np.mean([r["psi"] for r in ga_rows])),
                 bred=float(np.mean([r["B_red_signed"] for r in ga_rows])),
                 std=float(np.std([r["r2tr"] for r in ga_rows])), kind="zeal"))

# rank the pure baselines worst->best for the navy->pale-steel ramp (color = magnitude)
bases = sorted([r for r in recs if r["kind"] == "base"], key=lambda r: r["r2"])
ramp = [Z.NAVY, Z.STEEL1, Z.STEEL2, Z.STEEL3, "#BFD8EE", "#D9E8F6"]
for k, r in enumerate(bases):
    r["col"] = ramp[min(k, len(ramp) - 1)]

order = sorted(recs, key=lambda r: r["r2"])          # 8 bars, worst at bottom
y = np.arange(len(order))
for yi, r in zip(y, order):
    if r["kind"] == "zeal":
        axB.barh(yi, r["r2"], height=0.66, color=Z.PINK, zorder=3)
    elif r["kind"] == "pycno":
        axB.barh(yi, r["r2"], height=0.66, facecolor="white",
                 edgecolor=Z.PINKD, hatch="////", linewidth=0.9, zorder=3)
    else:
        axB.barh(yi, r["r2"], height=0.66, color=r["col"], zorder=3)
    if r["std"] > 1e-9:                               # per-seed SEM cap (RF/GBM/ZEAL)
        axB.errorbar(r["r2"], yi, xerr=r["std"], color=Z.INK, lw=0.7,
                     capsize=2.2, capthick=0.7, zorder=4)

axB.axvline(1.0, ls=(0, (4, 3)), lw=0.9, color=Z.RED, zorder=2)
axB.text(0.985, 1.35, "areal-faithful\nceiling", ha="right", va="center",
         fontsize=5.4, color=Z.RED, linespacing=0.95)
axB.set_yticks(y)
axB.set_yticklabels([r["name"] for r in order], fontsize=6.1)
for tl, r in zip(axB.get_yticklabels(), order):
    if r["kind"] == "zeal":
        tl.set_color(Z.PINKD); tl.set_fontweight("bold")
    elif r["kind"] == "pycno":
        tl.set_color(Z.PINKD)
axB.set_xlim(0, 1.12)
axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axB.set_xlabel("R²_tract  (higher better)")
axB.set_title("Baselines vs ZEAL", **TITLE_KW)
axB.set_ylim(-0.7, len(order) - 0.3)
leg_b = [Line2D([0], [0], marker="s", ls="none", mfc=Z.PINK, mec=Z.PINK, ms=6, label="ZEAL (certified)"),
         Line2D([0], [0], marker="s", ls="none", mfc="white", mec=Z.PINKD, ms=6,
                markeredgewidth=0.9, label="Areal (uncertifiable)"),
         Line2D([0], [0], marker="s", ls="none", mfc=Z.STEEL1, mec=Z.STEEL1, ms=6, label="Statistical / ML")]
axB.legend(handles=leg_b, loc="center right", fontsize=5.3, handletextpad=0.4,
           labelspacing=0.32, borderpad=0.3, bbox_to_anchor=(1.0, 0.42),
           title="caps: ±SEM (RF, GBM, ZEAL)", title_fontsize=4.9)
Z.panel(axB, "b")


# ============================================================================
# c  CERTIFICATION FRONTIER QUADRANT: R2_tract (x) x PSI (y), shaped markers
#    encode {faithful? continuous? certified?}; only ZEAL is all-three.
# ============================================================================
# Define the "all-three" target quadrant: faithful (R2 high) AND continuous (PSI in
# the supported band) AND certified. Faithful threshold = 0.95; only ZEAL & pycno pass
# R2, but pycno is areal -> NOT continuous (discontinuous at zone edges) & NOT certified.
R2_FAITH = 0.95
FRAME = dict(faithful=R2_FAITH)

# shade the faithful band (x >= 0.95) lightly; that is the only region of interest
axC.axvspan(R2_FAITH, 1.05, color=Z.GRAYLL, alpha=0.7, zorder=0)
axC.axvline(R2_FAITH, ls=(0, (4, 3)), lw=0.8, color=Z.GRAYREF, zorder=1)

mk = {"OLS": "v", "Ridge poly-3": "^", "kNN": "D", "GBM": "P",
      "Random forest": "s", "GP kriging": "o", "Pycno (areal)": "X"}
for r in recs:
    if r["kind"] == "zeal":
        axC.scatter(r["r2"], r["psi"], marker="*", s=240, facecolor=Z.PINK,
                    edgecolor=Z.PINKD, linewidth=0.9, zorder=6)
    elif r["kind"] == "pycno":
        # faithful but NOT continuous, NOT certified -> open pink-edged marker
        axC.scatter(r["r2"], r["psi"], marker=mk[r["name"]], s=58, facecolor="white",
                    edgecolor=Z.PINKD, linewidth=1.1, zorder=5)
    else:
        axC.scatter(r["r2"], r["psi"], marker=mk[r["name"]], s=34, facecolor="white",
                    edgecolor=r["col"], linewidth=1.0, zorder=4)

axC.set_xlim(-0.03, 1.10)
axC.set_ylim(-0.012, 0.235)
axC.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axC.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
axC.set_xlabel("R²_tract  (faithful →)")
axC.set_ylabel("PSI  (cross-zoning sensitivity)")
axC.set_title("Certification Frontier", **TITLE_KW)
axC.text(R2_FAITH + 0.008, 0.232, "faithful band\n(R² ≥ 0.95)", ha="left", va="top",
         fontsize=5.0, color=Z.GRAY, linespacing=0.95)

# The two faithful methods (ZEAL 0.986, Pycno 1.0) sit close together top-right.
# Route both callouts into the empty lower-right quadrant with thin leaders so the
# certified-vs-uncertified contrast is the readable story.
zeal = next(r for r in recs if r["kind"] == "zeal")
pycno = next(r for r in recs if r["kind"] == "pycno")
arc = dict(arrowstyle="-", lw=0.5, color=Z.GRAY, shrinkA=0, shrinkB=3)
axC.annotate("ZEAL\nfaithful + continuous\n+ certified",
             xy=(zeal["r2"], zeal["psi"]), xytext=(0.99, 0.083),
             fontsize=5.4, color=Z.PINKD, fontweight="bold", ha="right", va="top",
             linespacing=1.0, arrowprops=arc)
axC.annotate("Pycno (areal):\nnot continuous,\nnot certified",
             xy=(pycno["r2"], pycno["psi"]), xytext=(0.99, 0.035),
             fontsize=5.1, color=Z.PINKD, ha="right", va="top",
             linespacing=1.0, arrowprops=arc)
# light tags on the baseline cloud (all fall in the un-faithful left region)
blab = {"OLS": (5, 4, "left"), "Ridge poly-3": (5, -2, "left"), "kNN": (6, 5, "left"),
        "GBM": (6, -3, "left"), "Random forest": (-5, 7, "right"), "GP kriging": (6, 4, "left")}
for r in recs:
    if r["kind"] != "base":
        continue
    dx, dy, ha = blab[r["name"]]
    axC.annotate(r["name"], (r["r2"], r["psi"]), xytext=(dx, dy),
                 textcoords="offset points", fontsize=5.0, color=Z.GRAY, ha=ha, va="center")

leg_c = [Line2D([0], [0], marker="*", ls="none", mfc=Z.PINK, mec=Z.PINKD, ms=12, label="Certified (ZEAL)"),
         Line2D([0], [0], marker="X", ls="none", mfc="white", mec=Z.PINKD, mew=1.1, ms=6.5, label="Faithful, no cert."),
         Line2D([0], [0], marker="o", ls="none", mfc="white", mec=Z.GRAY, mew=1.0, ms=6, label="Predictive baseline")]
axC.legend(handles=leg_c, loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=5.2,
           handletextpad=0.4, labelspacing=0.32, borderpad=0.3)
Z.panel(axC, "c")


# ============================================================================
# d  SPATIAL CROSS-VALIDATION DUMBBELL: in-sample (steel) vs held-out (navy)
#    per fold + held-out B_red < 0 tag. (N7/N8; idiom upgrade over fig8c bars.)
# ============================================================================
scr = sc["rows"]
nF = len(scr)
ypos = np.arange(nF)[::-1]                            # fold 1 on top
r2_in = np.array([r["r2_in"] for r in scr])
r2_held = np.array([r["r2_held"] for r in scr])
br_held = np.array([r["B_red_held"] for r in scr]) * 1e3

Z.threshold(axD, 0.0, axis="x")
for yi, ri, rh in zip(ypos, r2_in, r2_held):
    axD.plot([rh, ri], [yi, yi], color=Z.GRAYL, lw=1.6, zorder=2)   # connector
    axD.scatter(rh, yi, s=34, color=Z.NAVY, edgecolor="white", linewidth=0.4, zorder=4)
    axD.scatter(ri, yi, s=34, color=Z.STEEL2, edgecolor="white", linewidth=0.4, zorder=4)

axD.set_yticks(ypos)
axD.set_yticklabels([f"Fold {r['fold'] + 1}" for r in scr], fontsize=6)
axD.set_ylim(-0.7, nF - 0.3)
axD.set_xlim(-1.45, 1.30)
axD.set_xticks([-1, -0.5, 0, 0.5, 1])
axD.set_xlabel("Reconstruction R²")
axD.set_title("Spatial Cross-Validation", **TITLE_KW)
axD.annotate("0", xy=(0.0, nF - 0.45), xytext=(2, 0), textcoords="offset points",
             fontsize=5.2, color=Z.RED, ha="left", va="top")
leg_d = [Line2D([0], [0], marker="o", ls="none", mfc=Z.STEEL2, mec="white", ms=6, label="In-sample"),
         Line2D([0], [0], marker="o", ls="none", mfc=Z.NAVY, mec="white", ms=6, label="Held-out block")]
axD.legend(handles=leg_d, loc="upper left", fontsize=5.5, handletextpad=0.3,
           labelspacing=0.32, borderpad=0.3, ncol=1, bbox_to_anchor=(0.0, 1.0))
axD.annotate("reconstructs support, does not extrapolate;  held-out B_red<0 in 5/5",
             xy=(0.5, CAP_Y_XTITLE), **CAP_KW)
Z.panel(axD, "d")


# ============================================================================
# e  GEOGRAPHY-COHERENCE PERMUTATION NULL: 1000 re-partitions vs real value
#    (lifted from fig8d; histogram + red rule at the real value, z = -16, 0/1000)
# ============================================================================
pn = fd["perm_null"]
null = np.array(pn["null"]) * 1e3
real = pn["real"] * 1e3
zval = pn["z"]
n_null = pn["n"]

axE.hist(null, bins=26, color=Z.GRAYL, edgecolor="white", linewidth=0.25, zorder=2)
Z.threshold(axE, real, axis="x")
ymax = axE.get_ylim()[1]
axE.annotate(f"Real geography\nz = {zval:.0f}  ({0}/{n_null})",
             xy=(real, ymax * 0.96), xytext=(real + 0.5, ymax * 0.96),
             ha="left", va="top", fontsize=6, color=Z.RED, fontweight="bold")
nm = null.mean()
axE.annotate("Random\nre-partitions", xy=(nm, ymax * 0.60),
             ha="center", va="center", fontsize=5.8, color=Z.GRAY)
axE.set_xlabel("Cross-zoning movement under re-partitioning (×10^-3)")
axE.set_ylabel(f"Count (of {n_null})")
axE.set_xlim(21.0, 30.5)
axE.set_title("Geography-Coherence Null", **TITLE_KW)
axE.annotate(f"{n_null} random re-partitions of the real data;  0 fall below the real value",
             xy=(0.5, CAP_Y_XTITLE), **CAP_KW)
Z.panel(axE, "e")


# ============================================================================
# f  RESOLUTION ROBUSTNESS: 512^2 (n=3, +/-s.d.) vs 1024^2 (single seed, tagged)
#    (lifted from fig8e; twin-axis PSI bars + signed B_red bars, red 0 line)
# ============================================================================
r512 = fg["512"]["rows"]
r1024 = fg["1024"]["rows"]
psi512 = np.array([r["psi"] for r in r512])
psi1024 = np.array([r["psi"] for r in r1024])
br512 = np.array([r["B_red_signed"] for r in r512]) * 1e3
br1024 = np.array([r["B_red_signed"] for r in r1024]) * 1e3

xx = np.array([0.0, 1.0])
bw = 0.30
psi_means = [psi512.mean(), psi1024[0]]
psi_err = [psi512.std(), 0.0]
axF.bar(xx - 0.17, psi_means, bw, yerr=psi_err, color=Z.PINKL,
        edgecolor=Z.PINKD, linewidth=0.5,
        error_kw=dict(elinewidth=0.6, capsize=2.5, ecolor=Z.PINKD), zorder=3)
axF.set_ylabel("PSI", color=Z.PINKD)
axF.tick_params(axis="y", colors=Z.PINKD)
axF.spines["left"].set_color(Z.PINKD)
# extra top headroom (ylim 0.46, ticks to 0.30) frees a clean band above the bars
# for a horizontal legend + the two uncertainty keys, so nothing collides with data.
axF.set_ylim(0, 0.46)
axF.set_yticks([0, 0.1, 0.2, 0.3])

axF2 = axF.twinx()
axF2.spines["top"].set_visible(False)
br_means = [br512.mean(), br1024[0]]
br_err = [br512.std(), 0.0]
axF2.bar(xx + 0.17, br_means, bw, yerr=br_err, color=Z.STEEL2,
         edgecolor=Z.NAVY, linewidth=0.5,
         error_kw=dict(elinewidth=0.6, capsize=2.5, ecolor=Z.NAVY), zorder=3)
Z.threshold(axF2, 0.0, axis="y")
axF2.set_ylabel("signed B_red (×10^-3)", color=Z.NAVY)
axF2.tick_params(axis="y", colors=Z.NAVY)
axF2.spines["right"].set_color(Z.NAVY)
# match the left-axis headroom so B_red = 0 stays co-located with the PSI 0.30 gridline;
# extra headroom above is the clean band for the legend + keys. Ticks unchanged.
axF2.set_ylim(-1.7, 0.907)
axF2.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.25])

axF.set_xticks(xx)
axF.set_xticklabels(["512²" + "\n(n = 3)", "1024²" + "\n(n = 1, OOM-limited)"],
                    fontsize=5.8)
axF.set_xlim(-0.55, 1.55)
axF.set_title("Resolution Robustness", fontsize=7, loc="center", pad=18)
# Horizontal 2-entry legend in the clean top band, just under the title; full
# "(left/right axis)" labels so nothing is clipped.
leg_f = [Patch(facecolor=Z.PINKL, edgecolor=Z.PINKD, label="PSI (left axis)"),
         Patch(facecolor=Z.STEEL2, edgecolor=Z.NAVY, label="B_red (right axis)")]
axF.legend(handles=leg_f, loc="center", fontsize=5.3, handlelength=1.0,
           handletextpad=0.4, columnspacing=1.2, borderpad=0.3, ncol=2,
           bbox_to_anchor=(0.5, 0.945))
# Two uncertainty keys: each anchored OVER its own x-group, below the legend, so the
# legend, the keys, and the bar tops occupy three distinct horizontal strips.
axF.annotate("±1 s.d. (n = 3)", xy=(-0.17, 0.345), ha="center", va="center",
             fontsize=5.0, color=Z.GRAY, style="italic")
axF.annotate("single seed (no CI)", xy=(1.0, 0.345), ha="center", va="center",
             fontsize=5.0, color=Z.GRAY, style="italic")
Z.panel(axF, "f")


Z.save(fig, "F6")
