"""F5 — "Georgia Verdict". Six dense panels, all from real data in ../data/.
   Panels a-f.

   a  4-rung choropleth row: the same continuous mobility field re-aggregated onto three
      real Georgia zonings (BG / tract / county, shared single-hue SEQ colorbar) + the signed
      tract-county re-aggregation shift (blue-white-red DIV, 0 tick printed).   [ga_fieldmap.npz]
   b  Field reconstruction residual (field - kfr, DIV) + the pixelwise |tract - county| movement
      heatmap: literally where the prediction moves when the map is redrawn.   [ga_fieldmap.npz, N14/N15]
   c  pred-vs-true scatter: 1937 tracts (1935 finite) + 159 county pts, count-weighted R^2 0.986 / 0.9997
      (aggregation identity), n~2000.                                          [ga_fieldmap.npz, N16]
   d  Signed B_red + permutation null: 1000 permuted-county movements, real Georgia at z=-16.06 in the
      far-left tail, 0/1000.                                                    [fig_data.json perm_null, N1]
   e  Non-vacuity spectrum reprise (twin-axis): across equal-fitting beta fields PSI spans 2.6x and the
      certified L_cert 3.3x while R^2 stays flat.                              [geo_m4_binding.json, N18]
   f  Certified envelope vs observed movement per rung (BG-tract, tract-county, BG-county): envelope
      >> movement with mov +/- s.d. -- sound but conservative.                 [newdata_field.json D7, N27]
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import zstyle as Z
Z.setup()

# ================================================================ data
G = np.load(Z.DATA + "/ga_fieldmap.npz")
mask = G["mask"]
field = G["field"]; kfr = G["kfr"]


def mfield(a):
    return np.ma.masked_where(~(mask & np.isfinite(a) & (a > 0)), a)


bg = mfield(field)                                   # BG-resolution continuous field
tr = mfield(G["tr"]); co = mfield(G["co"])           # re-aggregated onto tract / county zonings
valid = mask & np.isfinite(G["tr"]) & np.isfinite(G["co"]) & (G["tr"] > 0) & (G["co"] > 0)
shift = np.ma.masked_where(~valid, G["tr"] - G["co"])         # signed re-aggregation shift
mov_map = np.ma.masked_where(~valid, np.abs(G["tr"] - G["co"]))   # |tr - co| movement map (N14)
resid = np.ma.masked_where(~(mask & np.isfinite(field) & np.isfinite(kfr)), field - kfr)  # N15

tt, tp = G["tract_true"], G["tract_pred"]
ct, cp = G["county_true"], G["county_pred"]
ok = (tt > 0) & (tp > 0)                              # drop 2 empty tracts
tt, tp = tt[ok], tp[ok]
# CANONICAL count-weighted R^2 (paper source of truth); county fit is an aggregation identity.
R2_TR_CW, R2_CO_CW = 0.986, 0.9997

# shared sequential limits across the three zonings (robust percentiles)
allz = np.concatenate([bg.compressed(), tr.compressed(), co.compressed()])
VMIN, VMAX = np.percentile(allz, 1.5), np.percentile(allz, 98.5)
SLIM = np.percentile(np.abs(shift.compressed()), 98)            # symmetric DIV limit (shift)
RLIM = np.percentile(np.abs(resid.compressed()), 98)           # symmetric DIV limit (residual)
MLIM = np.percentile(mov_map.compressed(), 98)                 # SEQ upper limit (movement)

# --- d: signed B_red + permutation null --------------------------------------
PN = Z.load("fig_data.json")["perm_null"]
null = np.array(PN["null"]); real = PN["real"]; zval = PN["z"]; nperm = PN["n"]

# --- e: non-vacuity spectrum (binding) ---------------------------------------
BIND = Z.load("geo_m4_binding.json")["rows"]
beta = np.array([r["beta"] for r in BIND])
b_psi = np.array([r["psi"] for r in BIND])
b_lc = np.array([r["L_cert"] for r in BIND])
b_r2 = np.array([r["r2"] for r in BIND])
PSI_RATIO = b_psi.max() / b_psi.min()
LC_RATIO = b_lc.max() / b_lc.min()

# --- f: certified envelope vs observed movement per rung ----------------------
ND = Z.load("newdata_field.json")
D7 = ND["D7"]; LCERT = ND["LCERT_TIGHT"]
rung_keys = ["BG↔tract", "tract↔county", "BG↔county"]
rung_lab = ["BG→tract", "tract→county", "BG→county"]
mov = np.array([D7[k]["mov_mean"] for k in rung_keys])
movsd = np.array([D7[k]["mov_std"] for k in rung_keys])
env = np.array([D7[k]["envelope"] for k in rung_keys])

# ================================================================ canvas (180 mm double column)
fig = plt.figure(figsize=(180 * Z.MM, 176 * Z.MM))
gs = GridSpec(3, 12, figure=fig,
              height_ratios=[1.06, 1.0, 0.94],
              hspace=0.62, wspace=0.95,
              left=0.062, right=0.95, top=0.925, bottom=0.075)


def draw_map(ax, arr, cmap, vmin, vmax, title):
    im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    ax.set_title(title, fontsize=6.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


# ---------------------------------------------------------------- a: 4-rung choropleth row
axA1 = fig.add_subplot(gs[0, 0:3])
axA2 = fig.add_subplot(gs[0, 3:6])
axA3 = fig.add_subplot(gs[0, 6:9])
axA4 = fig.add_subplot(gs[0, 9:12])

imS = draw_map(axA1, bg, Z.SEQ, VMIN, VMAX, "Block-Group Field")
draw_map(axA2, tr, Z.SEQ, VMIN, VMAX, "Tract Zoning")
draw_map(axA3, co, Z.SEQ, VMIN, VMAX, "County Zoning")
imD = draw_map(axA4, shift, Z.DIV, -SLIM, SLIM, "Re-aggregation Shift")

# shared sequential colorbar flush to the county panel (the 3 field rungs share one scale)
cax1 = inset_axes(axA3, width="5%", height="84%", loc="center left",
                  bbox_to_anchor=(1.03, 0., 1, 1), bbox_transform=axA3.transAxes, borderpad=0)
cbS = fig.colorbar(imS, cax=cax1)
cbS.set_label("Mobility kfr_p25", fontsize=6.0, labelpad=2)
cbS.ax.tick_params(labelsize=5.4, length=2)
cbS.outline.set_linewidth(0.5)

# diverging shift colorbar with the 0 tick PRINTED
cax2 = inset_axes(axA4, width="5%", height="84%", loc="center left",
                  bbox_to_anchor=(1.03, 0., 1, 1), bbox_transform=axA4.transAxes, borderpad=0)
cbD = fig.colorbar(imD, cax=cax2)
cbD.set_label("Tract − county", fontsize=6.0, labelpad=2)
cbD.set_ticks([-SLIM, 0, SLIM])
cbD.set_ticklabels([f"−{SLIM:.02f}", "0", f"+{SLIM:.02f}"])
# diverging scale MUST print a visible 0 tick (the white-center mark is invisible otherwise)
cbD.ax.tick_params(labelsize=5.4, length=2.6, width=0.6, color=Z.INK)
cbD.outline.set_linewidth(0.5)

axA2.text(1.0, -0.07,
          "One continuous field  |  Georgia, US  |  re-aggregated onto three real zonings (5128 BG → 1937 tract → 159 county)",
          transform=axA2.transAxes, ha="center", va="top", fontsize=5.5, color=Z.GRAY)
Z.panel(axA1, "a")

# ---------------------------------------------------------------- b: residual + movement heatmap
axB1 = fig.add_subplot(gs[1, 0:3])
axB2 = fig.add_subplot(gs[1, 3:6])

imR = draw_map(axB1, resid, Z.DIV, -RLIM, RLIM, "Reconstruction Residual")
imM = draw_map(axB2, mov_map, Z.PINK_SEQ, 0.0, MLIM, "Prediction Movement")

caxR = inset_axes(axB1, width="5.5%", height="84%", loc="center left",
                  bbox_to_anchor=(1.03, 0., 1, 1), bbox_transform=axB1.transAxes, borderpad=0)
cbR = fig.colorbar(imR, cax=caxR)
cbR.set_label("Field − observed", fontsize=5.9, labelpad=2)
cbR.set_ticks([-RLIM, 0, RLIM])
cbR.set_ticklabels([f"−{RLIM:.02f}", "0", f"+{RLIM:.02f}"])
# diverging scale -> visible 0 midpoint tick (dark, so it shows against the white center)
cbR.ax.tick_params(labelsize=5.3, length=2.6, width=0.6, color=Z.INK)
cbR.outline.set_linewidth(0.5)

caxM = inset_axes(axB2, width="5.5%", height="84%", loc="center left",
                  bbox_to_anchor=(1.03, 0., 1, 1), bbox_transform=axB2.transAxes, borderpad=0)
cbM = fig.colorbar(imM, cax=caxM)
cbM.set_label("|tract − county|", fontsize=5.9, labelpad=2)
cbM.set_ticks([0, MLIM])
cbM.set_ticklabels(["0", f"{MLIM:.02f}"])
cbM.ax.tick_params(labelsize=5.3, length=2)
cbM.outline.set_linewidth(0.5)

axB1.text(1.05, -0.07,
          "Where the field mis-reconstructs (left) vs where re-drawing the map moves the prediction (right)",
          transform=axB1.transAxes, ha="center", va="top", fontsize=5.5, color=Z.GRAY)
Z.panel(axB1, "b")

# ---------------------------------------------------------------- c: pred-vs-true scatter
axC = fig.add_subplot(gs[1, 6:12])
lim = [0.26, 0.62]
Z.identity(axC, lim)
axC.scatter(tt, tp, s=4, color=Z.PINK, alpha=0.28, lw=0,
            label=f"Tract  (n=1937 tracts, {len(tt)} finite;  R²_cw={R2_TR_CW:.3f})", zorder=2)
axC.scatter(ct, cp, s=26, color=Z.PINKD, edgecolor="white", linewidth=0.4,
            label=f"County  (n={len(ct)},  R²_cw={R2_CO_CW:.4f})", zorder=3)
axC.set_xlim(lim); axC.set_ylim(lim)
axC.set_xticks([0.3, 0.4, 0.5, 0.6]); axC.set_yticks([0.3, 0.4, 0.5, 0.6])
axC.set_aspect("equal")
axC.set_xlabel("Observed mobility kfr_p25")
axC.set_ylabel("Reconstructed mobility")
axC.set_title("Field Reconstruction")
axC.text(0.97, 0.05, "identity", transform=axC.transAxes, ha="right", va="bottom",
         fontsize=5.8, color=Z.GRAYREF, style="italic")
axC.legend(loc="upper left", fontsize=5.8, handletextpad=0.4, borderpad=0.3, labelspacing=0.35)
axC.text(0.5, -0.16,
         "R²_cw count-weighted  |  county fit is an aggregation identity (county means fit in training)  |  cloud spread unweighted",
         transform=axC.transAxes, ha="center", va="top", fontsize=5.3, color=Z.GRAY)
Z.panel(axC, "c")

# ---------------------------------------------------------------- d: signed B_red + perm null
axD = fig.add_subplot(gs[2, 0:4])
counts, bins, _ = axD.hist(null, bins=34, color=Z.STEEL3, edgecolor="white", linewidth=0.25,
                           zorder=2, label=f"Permuted geography  (n={nperm})")
axD.axvline(null.mean(), ls=(0, (4, 3)), lw=0.8, color=Z.NAVY, zorder=3)
axD.text(null.mean(), counts.max() * 1.02, "null mean", ha="center", va="bottom",
         fontsize=5.4, color=Z.NAVY)
# real Georgia value: red threshold rule in the far-left tail
Z.threshold(axD, real, axis="x")
axD.annotate(f"Georgia\nz={zval:.1f}\n0/{len(null)}",
             xy=(real, counts.max() * 0.30),
             xytext=(real + (null.mean() - real) * 0.22, counts.max() * 0.62),
             ha="left", va="center", fontsize=6.0, color=Z.RED,
             arrowprops=dict(arrowstyle="->", color=Z.RED, lw=0.9, shrinkA=2, shrinkB=2))
axD.set_xlabel("Cross-zoning movement under permuted geography")
axD.set_ylabel("Permutation count")
axD.set_title("Geography, Not Artifact")
axD.set_xlim(real - 0.0006, null.max() + 0.0004)
axD.legend(loc="upper left", fontsize=5.6, handletextpad=0.5, borderpad=0.3)
axD.text(0.5, -0.22,
         "Real geography sits 16 s.d. below the permuted null  |  one-sided P<0.001",
         transform=axD.transAxes, ha="center", va="top", fontsize=5.5, color=Z.GRAY)
Z.panel(axD, "d")

# ---------------------------------------------------------------- e: non-vacuity spectrum (twin-axis)
axE = fig.add_subplot(gs[2, 4:8])
xe = np.arange(len(beta))
# left axis: PSI (instability the fit cannot see) -- PINK = the quantity our certificate exposes
lnpsi = axE.plot(xe, b_psi, "-o", color=Z.PINKD, ms=5, mec="white", mew=0.5, lw=1.4,
                 zorder=4, label="PSI (instability)")
axE.set_ylabel("PSI  (re-aggregation instability)", color=Z.PINKD)
axE.tick_params(axis="y", colors=Z.PINKD)
axE.set_ylim(0.10, 0.38)
axE.set_xticks(xe); axE.set_xticklabels([f"{b:g}" for b in beta])
axE.set_xlabel("Roughness penalty β  (equal-fitting fields)")
axE.set_title("Certificate Sees the Instability")

# right axis: L_cert (certified Lipschitz) -- NAVY baseline-family series
axE2 = axE.twinx()
axE2.spines["top"].set_visible(False)
lnlc = axE2.plot(xe, b_lc, marker="s", color=Z.NAVY, ms=4.5, mec="white", mew=0.5, lw=1.2,
                 ls=(0, (5, 1.5)), zorder=3, label="L_cert (certified Lipschitz)")
axE2.set_ylabel("L_cert  (certified Lipschitz)", color=Z.NAVY)
axE2.tick_params(axis="y", colors=Z.NAVY)
axE2.set_ylim(600, 3100)
# annotate the two L_cert endpoints so the 3.3x span (2879 / 863) is DERIVABLE in-panel
i_hi = int(np.argmax(b_lc)); i_lo = int(np.argmin(b_lc))
axE2.annotate(f"{b_lc[i_hi]:.0f}", xy=(xe[i_hi], b_lc[i_hi]), xytext=(2, 5),
              textcoords="offset points", ha="left", va="bottom", fontsize=5.6, color=Z.NAVY)
axE2.annotate(f"{b_lc[i_lo]:.0f}", xy=(xe[i_lo], b_lc[i_lo]), xytext=(-2, -6),
              textcoords="offset points", ha="right", va="top", fontsize=5.6, color=Z.NAVY)

# R^2 near-flat annotation band (the fields fit ~equally well)
axE.text(0.04, 0.06, f"R² flat: {b_r2.min():.2f}–{b_r2.max():.2f}",
         transform=axE.transAxes, ha="left", va="bottom", fontsize=5.6, color=Z.GRAY)
he = lnpsi + lnlc
axE.legend(he, [h.get_label() for h in he], loc="upper right", fontsize=5.5,
           handletextpad=0.5, borderpad=0.3, labelspacing=0.35)
axE.text(0.5, -0.22,
         f"Equal-fitting fields span PSI {PSI_RATIO:.1f}× / L_cert {LC_RATIO:.1f}×  |  the fit alone cannot rank them",
         transform=axE.transAxes, ha="center", va="top", fontsize=5.5, color=Z.GRAY)
Z.panel(axE, "e")

# ---------------------------------------------------------------- f: certified envelope vs movement
# Dumbbell / point-range: each rung is a thin stem from the observed |movement| (pink, +-s.d.)
# up to the certified bound (red dashed cap). The ~3 decade gap IS the margin -- drawn as stem
# length, not as an empty box. Kills the whitespace-filled "margin" boxes.
axF = fig.add_subplot(gs[2, 8:12])
xf = np.arange(len(rung_keys))
cap_w = 0.30
for i in range(len(xf)):
    # connecting stem (the certified margin, shown as length on a log axis)
    axF.plot([xf[i], xf[i]], [mov[i], env[i]], color=Z.GRAYL, lw=1.4, zorder=1, solid_capstyle="round")
    # certified bound: red dashed cap at the top + its value just to the right
    axF.plot([xf[i] - cap_w / 2, xf[i] + cap_w / 2], [env[i], env[i]],
             color=Z.RED, lw=1.4, ls=(0, (3.5, 1.8)), zorder=4)
    axF.annotate(f"{env[i]:.0f}", xy=(xf[i] + cap_w / 2 + 0.04, env[i]), ha="left", va="center",
                 fontsize=5.5, color=Z.RED)
    # margin factor on the stem (no floating box, no whitespace)
    factor = env[i] / mov[i]
    axF.annotate(f"{factor:,.0f}×", xy=(xf[i] + 0.085, np.sqrt(env[i] * mov[i])),
                 ha="left", va="center", fontsize=6.0, color=Z.GRAY, rotation=90)
axF.errorbar(xf, mov, yerr=movsd, fmt="o", ms=6, color=Z.PINKD,
             mec="white", mew=0.6, ecolor=Z.PINKD, elinewidth=1.0, capsize=2.6, capthick=1.0,
             zorder=5)
# observed-movement value just to the left of each marker (so the margin = env / mov is derivable)
for i in range(len(xf)):
    axF.annotate(f"{mov[i]:.3f}", xy=(xf[i] - 0.10, mov[i]), ha="right", va="center",
                 fontsize=5.5, color=Z.PINKD)
axF.set_yscale("log")
axF.set_ylim(5e-3, 600)
axF.set_xlim(-0.6, len(xf) - 0.4)
axF.set_xticks(xf); axF.set_xticklabels(rung_lab, fontsize=6.0)
axF.set_ylabel("Cross-zoning field movement  (log)")
axF.set_title("Certified Stability Envelope")
handles_f = [
    Line2D([0], [0], color=Z.RED, lw=1.4, ls=(0, (3.5, 1.8)), label="Certified bound  L_cert·d"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=Z.PINKD, markeredgecolor="white",
           markeredgewidth=0.6, markersize=6, label="Observed |movement| (mean ± s.d.)"),
]
axF.legend(handles=handles_f, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=5.4,
           handletextpad=0.4, borderpad=0.3, labelspacing=0.4, framealpha=0.0)
axF.text(0.5, -0.22,
         f"Lipschitz L_cert={LCERT:.0f}  |  n=5 seeds  |  sound one-sided bound (≥), conservative because the surface is genuinely stable",
         transform=axF.transAxes, ha="center", va="top", fontsize=5.4, color=Z.GRAY)
Z.panel(axF, "f")

Z.save(fig, "F5")
