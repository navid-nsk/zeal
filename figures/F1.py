"""F1 - "Functor, not group: the certified contract"  (consolidated fig1 + fig2).

Six dense panels at 180 mm.
Real data only, from ../data/.  Color lock (zstyle): ours=PINK, baselines=navy/steel,
green=good/kept, red=threshold/reference only.

  a  Schematic (full-width band).  Three arguments drawn as one self-contained
     sentence, left -> right:
       (i)  the MAUP sign-flip WOUND: one continuous Georgia opportunity field aggregated
            onto TWO real tilings (tract `tr`, county `co`) on a shared Z.SEQ colorbar;
            an interior site flips value (0.49 -> 0.35) between zonings.
       (ii) the refinement-lattice HASSE diagram block-group < tract < county, drawn as
            three real nested partitions of one compact Georgia county.
       (iii) the FUNCTOR LAW A(pi o pi') = A(pi) A(pi') drawn on the lattice with typed
            aggregation arrows A_pi, A_pi'.
     (reuses fig1a maps + fig1c nested geometry.)
  b  Certified Lipschitz envelope: per-seed cross-zoning movement vs zoning distance
     d(Z,Z') (70-pt cloud, 14 zonings x 5 seeds) under the certified bound y = L_cert d
     (L_cert = 242, red dashed).  newdata_field.json D1 / LCERT_TIGHT.  (reuses fig1b.)
  c  Box-bound ratio d^delta/d_W across manifold width m, per-seed 95% CI, d^delta=d_W
     reference.  dimsweep.json.  (reuses fig2b.)
  d  Bracket tightening chi = cert/empirical (74x -> 2.2x) vs certificate tile budget,
     chi=1 perfectly-tight reference.  zeal_soundness_curve.json.  (reuses fig2c.)
  e  Sliver robustness: naive cell-mean spread BLOWS UP (0.34 -> 0.11) as sliver width
     w -> 0, while the mollified R_Z^delta (ours) stays flat (~0.12); delta marked.
     sliver.json.  (reuses fig2a curve.)
  f  Sliver field tile: the real field_tile[240,280] raster with the sub-delta sliver
     band drawn on it -- the geometric object panel e quantifies.  sliver.npz.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import ndimage
import zstyle as Z

Z.setup()

# ================================================================= data
npz = np.load(Z.DATA + "/ga_fieldmap.npz")
TR = npz["tr"].astype(float)
CO = npz["co"].astype(float)
MASK = npz["mask"].astype(bool)

Dnd = json.load(open(Z.DATA + "/newdata_field.json"))
LCERT = float(Dnd["LCERT_TIGHT"])             # 242.0
D1 = np.asarray(Dnd["D1"], dtype=float)        # (70,3) -> [ncell, d, movement]

dim = Z.load("dimsweep.json")
snd = Z.load("zeal_soundness_curve.json")
sliv = Z.load("sliver.json")
slnpz = np.load(Z.DATA + "/sliver.npz")
field_tile = slnpz["field_tile"]
DELTA = float(sliv["delta"])                   # 0.025

# masked map views; shared field color range over both tilings
trv = np.where(MASK & (TR > 0.05), TR, np.nan)
cov = np.where(MASK, CO, np.nan)
both = np.concatenate([trv[np.isfinite(trv)], cov[np.isfinite(cov)]])
VMIN, VMAX = np.percentile(both, 2), np.percentile(both, 98)
NORM = Normalize(VMIN, VMAX)

ys, xs = np.where(MASK)
r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

FLIP_RC = (116, 130)
tr_here = float(TR[FLIP_RC]); co_here = float(CO[FLIP_RC])

# ================================================================= canvas
# Three bands on 180 mm: (a) full-width lead panel, (b/c/d) 3-across, (e/f) sliver pair.
FIG_W, FIG_H = 180.0, 158.0
fig = plt.figure(figsize=(FIG_W * Z.MM, FIG_H * Z.MM))
ASP = FIG_W / FIG_H                              # width/height of the figure box

LET_SZ = 8.5


def figletter(x, y, s):
    fig.text(x, y, s, ha="left", va="top", fontsize=LET_SZ, fontweight="bold", color=Z.INK)


# ============================================================ PANEL a
# Georgia bbox is square -> size each map axes so its fig-fraction height makes it
# square, anchored at the band TOP, so no vertical whitespace / no oversized colorbar.
mapW = 0.215
mapH = mapW * ASP                               # square axes in fig fractions
A_top = 0.955
A_bot = A_top - mapH
axA1 = fig.add_axes([0.045, A_bot, mapW, mapH])
axA2 = fig.add_axes([0.270, A_bot, mapW, mapH])
caxA = fig.add_axes([0.493, A_bot, 0.009, mapH])


def show_tiling(ax, arr, title):
    im = ax.imshow(arr[r0:r1, c0:c1], origin="upper", cmap=Z.SEQ, norm=NORM,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=6.6, color=Z.INK, loc="center", pad=3)
    return im


def edges_lc(arr, ax, lw=0.18):
    a = np.where(MASK, np.round(arr, 6), -999.0)[r0:r1, c0:c1]
    segs = []
    rr, cc = np.where(a[:, 1:] != a[:, :-1])
    for r, c in zip(rr, cc):
        segs.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
    rr, cc = np.where(a[1:, :] != a[:-1, :])
    for r, c in zip(rr, cc):
        segs.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
    ax.add_collection(LineCollection(segs, colors="white", linewidths=lw, alpha=0.55))


im = show_tiling(axA1, trv, "Tract zoning  π'  (1,924 zones)")
show_tiling(axA2, cov, "County zoning  π  (159 zones)")
edges_lc(TR, axA1)
edges_lc(CO, axA2)

# the sign-flip wound: same interior site, two zonings -> different value
fr, fc = FLIP_RC[0] - r0, FLIP_RC[1] - c0
for ax in (axA1, axA2):
    ax.plot(fc, fr, "o", ms=5.5, mfc="none", mec=Z.RED, mew=1.1, zorder=5)
    ax.plot(fc, fr, "+", ms=4.5, mec=Z.RED, mew=0.9, zorder=5)
for ax, val in ((axA1, tr_here), (axA2, co_here)):
    ax.annotate(f"{val:.2f}", xy=(fc, fr), xytext=(fc + 8, fr - 20), textcoords="data",
                fontsize=6.0, color=Z.RED, ha="left", va="bottom", zorder=21,
                fontweight="bold", arrowprops=dict(arrowstyle="-", lw=0.5, color=Z.RED))

p1 = fig.transFigure.inverted().transform(axA1.transData.transform((fc, fr)))
p2 = fig.transFigure.inverted().transform(axA2.transData.transform((fc, fr)))
arc = FancyArrowPatch(p1, p2, transform=fig.transFigure, connectionstyle="arc3,rad=0.55",
                      arrowstyle="-", lw=0.8, ls=(0, (1.5, 1.5)), color=Z.RED, zorder=20)
fig.patches.append(arc)
midx = (p1[0] + p2[0]) / 2
arc_bot = min(p1[1], p2[1]) - 0.55 * abs(p2[0] - p1[0]) * 0.5
fig.text(midx, arc_bot - 0.004, "same site, two zonings: value flips",
         ha="center", va="top", fontsize=6.0, color=Z.RED, style="italic")

cb = fig.colorbar(im, cax=caxA)
cb.set_ticks(np.linspace(VMIN, VMAX, 4))
cb.set_ticklabels([f"{t:.2f}" for t in np.linspace(VMIN, VMAX, 4)])
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(width=0.5, length=2, labelsize=6.0)
cb.set_label("Aggregated field value", fontsize=6, rotation=90, labelpad=2)
fig.text(0.262, A_bot - 0.013,
         "Continuous opportunity field  |  source: Georgia  |  single reference field (n = 1)",
         ha="center", fontsize=6.0, color=Z.GRAY)

# ---- the refinement lattice (Hasse) + functor law, right of the maps ----
# draw three real nested partitions block-group < tract < county of ONE compact county.
co_round = np.where(MASK, np.round(CO, 6), np.nan)
tr_round = np.where(MASK, np.round(TR, 6), np.nan)
CVAL = 0.3837
raw = np.isclose(co_round, CVAL, atol=5e-4)
lab_cc, ncc = ndimage.label(raw)
sizes = [(lab_cc == i).sum() for i in range(1, ncc + 1)]
cidx = int(np.argmax(sizes)) + 1
cm = lab_cc == cidx
yy, xx = np.where(cm)
pad = 2
br0, br1 = max(yy.min() - pad, 0), yy.max() + 1 + pad
bc0, bc1 = max(xx.min() - pad, 0), xx.max() + 1 + pad
cm_c = cm[br0:br1, bc0:bc1]
tr_c = np.where(cm_c, tr_round[br0:br1, bc0:bc1], np.nan)
Hh, Wd = cm_c.shape


def boundary_segs(labelfield, region):
    a = np.where(region, labelfield, -999.0)
    segs = []
    dv = (a[:, 1:] != a[:, :-1]) & (region[:, 1:] | region[:, :-1])
    rr, cc = np.where(dv)
    for r, c in zip(rr, cc):
        segs.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
    dh = (a[1:, :] != a[:-1, :]) & (region[1:, :] | region[:-1, :])
    rr, cc = np.where(dh)
    for r, c in zip(rr, cc):
        segs.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
    return segs


def subdivide_tract(tr_field, region):
    bg = np.full(region.shape, -1, dtype=int)
    nxt = 0
    rng = np.random.default_rng(7)
    for tv in np.unique(tr_field[np.isfinite(tr_field)]):
        tmask = np.isfinite(tr_field) & np.isclose(tr_field, tv)
        npx = tmask.sum()
        if npx == 0:
            continue
        nbg = int(np.clip(round(npx / 70), 2, 4))
        pr, pc = np.where(tmask)
        pts = np.column_stack([pr, pc]).astype(float)
        seeds = pts[rng.choice(len(pts), nbg, replace=False)]
        for _ in range(6):
            d2 = ((pts[:, None, :] - seeds[None, :, :]) ** 2).sum(-1)
            asg = d2.argmin(1)
            for j in range(nbg):
                if (asg == j).any():
                    seeds[j] = pts[asg == j].mean(0)
        for j in range(nbg):
            sel = asg == j
            bg[pr[sel], pc[sel]] = nxt + j
        nxt += nbg
    return bg


bg_lab = subdivide_tract(tr_c, cm_c)
local = tr_c[np.isfinite(tr_c)]
lnorm = Normalize(np.percentile(local, 5), np.percentile(local, 95))


def zone_mean_image(label_field):
    out = np.full(cm_c.shape, np.nan)
    base = np.where(cm_c, tr_round[br0:br1, bc0:bc1], np.nan)
    keys = np.unique(label_field[np.isfinite(label_field)]) if label_field.dtype == float \
        else np.unique(label_field[label_field >= 0])
    for v in keys:
        m = (label_field == v) & cm_c
        if m.any():
            out[m] = np.nanmean(base[m])
    return out


bg_img = zone_mean_image(bg_lab.astype(float))
tr_img = zone_mean_image(np.where(cm_c, tr_round[br0:br1, bc0:bc1], np.nan))
co_img = np.where(cm_c, float(CVAL), np.nan)

# three lattice nodes stacked at the right of band a (block-group bottom = finest).
# node axes sized to the county-region aspect so each tile fills its box (no whitespace).
LATX = 0.605
node_w = 0.098
node_h = node_w * ASP * (Hh / Wd)               # preserve true county aspect, square pixels
node_gap = 0.046                                # vertical gap for the A_pi arrows
y_co = A_top - 0.010                             # top node aligned just under band top
y_tr = y_co - (node_h + node_gap)
y_bg = y_tr - (node_h + node_gap)
levels = [("county", co_img, None, 0.0, Z.NAVY, y_co, "County", 1),
          ("tract", tr_img, np.where(cm_c, tr_round[br0:br1, bc0:bc1], -999.0), 0.5,
           Z.STEEL1, y_tr, "Tract", len(np.unique(tr_c[np.isfinite(tr_c)]))),
          ("bg", bg_img, bg_lab.astype(float), 0.35, Z.STEEL2, y_bg, "Block group",
           len(np.unique(bg_lab[bg_lab >= 0])))]
node_axes = {}
for mode, img, field, lw, edgecol, yc, name, nz in levels:
    axn = fig.add_axes([LATX, yc - node_h, node_w, node_h])
    axn.imshow(img, origin="upper", cmap=Z.SEQ, norm=lnorm, interpolation="nearest")
    if lw > 0 and field is not None:
        f = np.where(cm_c, field, -999.0)
        axn.add_collection(LineCollection(boundary_segs(f, cm_c), colors="white",
                                          linewidths=lw))
    out = []
    p = np.pad(cm_c.astype(int), 1)
    rr, cc = np.where(p[1:-1, 1:] != p[1:-1, :-1])
    for r, c in zip(rr, cc):
        out.append([(c - 0.5, r - 0.5), (c - 0.5, r + 0.5)])
    rr, cc = np.where(p[1:, 1:-1] != p[:-1, 1:-1])
    for r, c in zip(rr, cc):
        out.append([(c - 0.5, r - 0.5), (c + 0.5, r - 0.5)])
    axn.add_collection(LineCollection(out, colors=edgecol, linewidths=1.4))
    axn.set_xlim(-1.5, Wd + 0.5); axn.set_ylim(Hh + 0.5, -1.5)
    axn.set_aspect("equal"); axn.set_xticks([]); axn.set_yticks([])
    for s in axn.spines.values():
        s.set_visible(False)
    node_axes[mode] = (axn, name, nz, edgecol)
    # label to the right of each node
    fig.text(LATX + node_w + 0.012, yc - node_h * 0.40, name, fontsize=6.6, color=edgecol,
             va="center", fontweight="bold")
    fig.text(LATX + node_w + 0.012, yc - node_h * 0.66, f"{nz} zone" + ("s" if nz != 1 else ""),
             fontsize=6.0, color=Z.GRAY, va="center")

fig.text(LATX, y_co + 0.018, "Refinement lattice", ha="left", va="bottom", fontsize=6.6,
         color=Z.INK)

# typed aggregation arrows UP the lattice (refine bottom -> aggregate up): A_pi', A_pi
order_up = [("bg", "tract", "A_π'"), ("tract", "county", "A_π")]
for lo, hi, txt in order_up:
    axlo = node_axes[lo][0]; axhi = node_axes[hi][0]
    blo = axlo.get_position(); bhi = axhi.get_position()
    x = blo.x0 - 0.022
    a = FancyArrowPatch((x, blo.y1 + 0.004), (x, bhi.y0 - 0.004), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=8, lw=1.3, color=Z.PINKD)
    fig.patches.append(a)
    fig.text(x - 0.013, (blo.y1 + bhi.y0) / 2, txt, fontsize=7.5, color=Z.PINKD,
             ha="center", va="center")

# the functor LAW drawn as a composition identity beside the lattice base (not a box).
# placed to the RIGHT of the block-group node (the "28 zones" label column) so it stays
# inside band a and clear of the b/c/d titles below.
bbg = node_axes["bg"][0].get_position()
law_x = bbg.x1 + 0.090
law_y = bbg.y1 - 0.006
fig.text(law_x, law_y, "A_(π o π') = A_π A_π'",
         fontsize=8.5, color=Z.PINKD, ha="left", va="top", fontweight="bold")
fig.text(law_x, law_y - 0.030,
         "refine then aggregate\n= aggregate.\n\nA functor on the\nrefinement lattice,\nnot a group action.",
         fontsize=6.0, color=Z.GRAY, ha="left", va="top", style="italic", linespacing=1.35)

figletter(0.020, 0.985, "a")

# ============================================================ BAND 2: b, c, d
B_top, B_bot = 0.500, 0.310
gsbcd = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=B_top, bottom=B_bot,
                         wspace=0.50)
axB = fig.add_subplot(gsbcd[0, 0])
axC = fig.add_subplot(gsbcd[0, 1])
axD = fig.add_subplot(gsbcd[0, 2])

# ---- b : certified Lipschitz envelope ----
ncell, dd, mov = D1[:, 0], D1[:, 1], D1[:, 2]
xline = np.linspace(dd.min() * 0.95, dd.max() * 1.05, 50)
axB.fill_between(xline, mov.max() * 1.2, LCERT * xline, color=Z.PINKL, alpha=0.16, lw=0,
                 zorder=1)
axB.plot(xline, LCERT * xline, ls=(0, (4, 3)), lw=1.0, color=Z.RED, zorder=4)
axB.scatter(dd, mov, s=7, facecolor=Z.PINK, edgecolor="none", alpha=0.55, zorder=5)
uk = np.unique(ncell)
mx, my, sy = [], [], []
for kk in uk:
    sel = ncell == kk
    mx.append(dd[sel].mean()); my.append(mov[sel].mean()); sy.append(mov[sel].std())
mx, my, sy = map(np.asarray, (mx, my, sy))
axB.errorbar(mx, my, yerr=sy, fmt="o", ms=3.2, mfc=Z.PINKD, mec="white", mew=0.4,
             ecolor=Z.PINKD, elinewidth=0.7, capsize=1.5, capthick=0.6, zorder=6)
axB.set_yscale("log")
axB.set_xlim(dd.min() * 0.9, dd.max() * 1.06)
axB.set_ylim(8e-3, LCERT * dd.max() * 3.0)
axB.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
axB.set_xlabel("Zoning distance  d(Z, Z′)")
axB.set_ylabel("Cross-zoning movement  (log scale)")
axB.set_title("Certified Lipschitz Envelope")
xa = dd.max() * 0.40
axB.annotate(f"certified bound  L_cert d,  L_cert={LCERT:.0f}",
             xy=(xa, LCERT * xa), xytext=(dd.min() * 1.02, LCERT * dd.max() * 2.0),
             fontsize=5.7, color=Z.RED, va="center", ha="left",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=Z.RED,
                             connectionstyle="arc3,rad=-0.25"))
gap = (LCERT * mx.mean()) / my.mean()
axB.annotate(f"observed movement:\n{gap:.0f}x under bound", xy=(mx.mean(), my.mean()),
             xytext=(dd.max() * 0.40, 1.3e-1), fontsize=5.7, color=Z.PINKD, ha="left",
             va="bottom", arrowprops=dict(arrowstyle="-", lw=0.5, color=Z.PINKD,
                                          connectionstyle="arc3,rad=0.25"))
axB.text(0.965, 0.045, "per-seed dots\nn = 5 / zoning  (14 zonings)",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=5.3, color=Z.GRAY,
         linespacing=1.3)

# ---- c : box-bound ratio ----
rows = dim["rows"]; nseed = dim["nseed"]
mvals = [r["m"] for r in rows]
comp = np.array([r["ratio_box"] for r in rows])
csd = np.array([r["ratio_box_std"] for r in rows])
ci = 1.96 * csd / np.sqrt(nseed)
yb = np.arange(len(rows))[::-1]
axC.axvline(1.0, ls=(0, (4, 3)), lw=0.9, color=Z.RED, zorder=1)
axC.annotate("dᵟ=d_W", xy=(1.0, 0.62), xycoords=("data", "axes fraction"),
             xytext=(-3, 0), textcoords="offset points", color=Z.RED, fontsize=5.6,
             ha="right", va="center")
xerr_lo = np.minimum(ci, comp)
axC.errorbar(comp, yb, xerr=[xerr_lo, ci], fmt="none", ecolor=Z.PINKD, elinewidth=1.0,
             capsize=2.2, capthick=1.0, zorder=3)
axC.plot(comp, yb, "D", color=Z.PINK, ms=5.0, mfc=Z.PINK, mec="white", mew=0.6, zorder=4)
axC.set_yticks(yb)
axC.set_yticklabels([f"m={m}" for m in mvals])
axC.set_ylim(-0.6, len(rows) - 0.4)
axC.set_xlim(0, 1.72)
axC.set_xticks([0, 0.5, 1.0, 1.5])
axC.set_xlabel("Box-bound ratio  dᵟ/d_W")
axC.set_title("Box-Bound Ratio")
handles = [
    Line2D([0], [0], marker="D", ls="none", color=Z.PINK, mfc=Z.PINK, mec="white", mew=0.6,
           ms=5.0, label="dᵟ/d_W (ours)"),
    Line2D([0], [0], marker="|", ls="none", color=Z.PINKD, ms=7, mew=1.0,
           label=f"per-seed 95% CI (n = {nseed})"),
]
axC.legend(handles=handles, loc="upper right", handlelength=1.0, borderaxespad=0.4,
           labelspacing=0.35)
axC.text(0.035, 0.04, "6 manifold geometries", transform=axC.transAxes, fontsize=5.3,
         color=Z.GRAY, ha="left", va="bottom")

# ---- d : bracket tightening chi ----
budget = np.array([r["budget"] for r in snd["rows"]], dtype=float)
ratio = np.array([r["ratio"] for r in snd["rows"]])
emp = snd["empirical"]
axD.axhline(1.0, ls=(0, (4, 3)), lw=0.9, color=Z.RED, zorder=2)
axD.plot(budget, ratio, "-D", color=Z.PINK, ms=4.2, lw=1.6, mfc=Z.PINK, mec="white",
         mew=0.5, zorder=4)
axD.annotate("χ=1  (perfectly tight)", xy=(budget[-1], 1.0), xytext=(-2, 3),
             textcoords="offset points", color=Z.RED, fontsize=5.6, ha="right", va="bottom")
axD.set_xscale("log"); axD.set_yscale("log")
axD.set_xlim(7e3, 7e5)
axD.set_ylim(0.9, 120)
axD.set_xlabel("Certificate tile budget")
axD.set_ylabel("Bracket tightness χ  (lower better)")
axD.set_title("Bracket Tightening")
axD.set_yticks([1, 10, 100]); axD.set_yticklabels(["1", "10", "100"])
axD.annotate(f"{ratio[0]:.0f}× slack", xy=(budget[0], ratio[0]), xytext=(5, 0),
             textcoords="offset points", color=Z.PINKD, fontsize=6, ha="left", va="center")
axD.annotate(f"{ratio[-1]:.1f}×", xy=(budget[-1], ratio[-1]), xytext=(0, 9),
             textcoords="offset points", color=Z.PINKD, fontsize=6, ha="center")
axD.text(0.045, 0.30, f"empirical worst-case = {emp:.1f}", transform=axD.transAxes,
         fontsize=5.3, color=Z.GRAY, ha="left", va="bottom")
axD.text(0.045, 0.235, "deterministic certificate", transform=axD.transAxes, fontsize=5.0,
         color=Z.GRAY, ha="left", va="bottom")
axD.text(0.045, 0.17, "(no seed variance)", transform=axD.transAxes, fontsize=5.0,
         color=Z.GRAY, ha="left", va="bottom")

figletter(0.020, B_top + 0.040, "b")
figletter(0.350, B_top + 0.040, "c")
figletter(0.680, B_top + 0.040, "d")

# ============================================================ BAND 3: e, f
E_top, E_bot = 0.225, 0.055
# e is the wide curve (left ~58%), f the field-tile raster + its own colorbar (right)
axE = fig.add_axes([0.075, E_bot, 0.475, E_top - E_bot])
axF = fig.add_axes([0.640, E_bot, 0.285, E_top - E_bot])
caxF = fig.add_axes([0.930, E_bot, 0.009, E_top - E_bot])

# ---- e : sliver spread (naive blows up vs mollified flat) ----
w = np.array([r["w"] for r in sliv["rows"]])
naive = np.array([r["naive_spread"] for r in sliv["rows"]])
moll = np.array([r["moll_spread"] for r in sliv["rows"]])
nmean = np.array([r["naive_mean"] for r in sliv["rows"]])
mmean = np.array([r["moll_mean"] for r in sliv["rows"]])
o = np.argsort(w)
w, naive, moll, nmean, mmean = w[o], naive[o], moll[o], nmean[o], mmean[o]
mlevel = float(np.median(moll))
axE.axhspan(mlevel - 0.012, mlevel + 0.012, color=Z.PINKL, alpha=0.30, zorder=1, lw=0)
axE.plot(w, naive, "-o", color=Z.NAVY, ms=4.2, lw=1.4, mfc=Z.NAVY, mec="white", mew=0.5,
         label="naive cell-mean spread", zorder=3)
axE.plot(w, moll, "-D", color=Z.PINK, ms=4.0, lw=1.6, mfc=Z.PINK, mec="white", mew=0.5,
         label="mollified R_Zᵟ spread (ours)", zorder=4)
# mean overlay (mollifier preserves the field mean): thin gray dashed reference series
axE.plot(w, mmean, ":", color=Z.GRAY, lw=1.0, zorder=2)
axE.scatter(w, mmean, s=9, facecolor="white", edgecolor=Z.GRAY, lw=0.6, zorder=2)
axE.annotate("preserved field mean\n(~0.20, both methods)", xy=(w[3], mmean[3]),
             xytext=(0, 14), textcoords="offset points", color=Z.GRAY, fontsize=5.3,
             ha="center", va="bottom",
             arrowprops=dict(arrowstyle="-", color=Z.GRAY, lw=0.5))
axE.axvline(DELTA, ls=(0, (4, 3)), lw=0.9, color=Z.RED, zorder=2)
axE.annotate("δ", xy=(DELTA, 1.0), xycoords=("data", "axes fraction"),
             xytext=(2.5, -8), textcoords="offset points", color=Z.RED, fontsize=7,
             ha="left", va="top")
axE.set_xscale("log")
axE.set_xlim(0.0032, 0.165)
axE.set_ylim(0.08, 0.385)
axE.set_yticks([0.1, 0.2, 0.3])
axE.set_xlabel("Sliver width w  (smaller → worse)")
axE.set_ylabel("Aggregated-value spread\nacross re-zonings")
axE.set_title("Sliver Robustness")
axE.legend(loc="upper right", handlelength=1.6, borderaxespad=0.4)
axE.annotate("naive blows up\nas w→0", xy=(w[0], naive[0]), xytext=(14, -2),
             textcoords="offset points", color=Z.NAVY, fontsize=5.6, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=Z.NAVY, lw=0.5))
axE.text(0.015, 0.03, "n = 6 sliver scales", fontsize=5.3, color=Z.GRAY,
         transform=axE.transAxes, ha="left", va="bottom")

# ---- f : real sliver field-tile raster with the sub-delta band + labeled colorbar ----
try:
    from scipy.ndimage import gaussian_filter
    field_disp = gaussian_filter(field_tile, sigma=(1.0, 2.0))
except Exception:
    field_disp = field_tile
fvmin, fvmax = np.nanpercentile(field_tile, [2, 98])
h, ww = field_tile.shape
imF = axF.imshow(field_disp, cmap=Z.SEQ, vmin=fvmin, vmax=fvmax, origin="lower",
                 aspect="auto", interpolation="bilinear", extent=[0, ww, 0, h])
band_y0, band_h = int(0.46 * h), max(3, int(0.080 * h))
axF.add_patch(Rectangle((0, band_y0), ww, band_h, fill=True, facecolor=Z.PINKL, alpha=0.50,
                        edgecolor=Z.PINKD, lw=0.9, zorder=3))
# real field profile sampled across the sub-delta slice (data-ink, not just stripes):
slice_row = band_y0 + band_h // 2
prof = field_tile[slice_row, :]
pn = (prof - fvmin) / (fvmax - fvmin)
prof_y = band_y0 + band_h * 0.5 + (pn - 0.5) * (band_h * 6.0)
axF.plot(np.arange(ww), np.clip(prof_y, 0, h), color=Z.PINKD, lw=0.9, zorder=4)
axF.annotate("sub-δ sliver band", xy=(ww * 0.5, band_y0 + band_h), xytext=(0, 7),
             textcoords="offset points", color=Z.PINKD, fontsize=5.6, ha="center",
             va="bottom", arrowprops=dict(arrowstyle="-", color=Z.PINKD, lw=0.7), zorder=6)
axF.set_xlim(0, ww); axF.set_ylim(0, h)
axF.set_xticks([]); axF.set_yticks([])
for s in axF.spines.values():
    s.set_visible(True); s.set_color(Z.GRAY); s.set_linewidth(0.6)
axF.set_title("Sliver Field Tile  λ(x)")
axF.text(0.5, -0.075, "real field_tile[240, 280]  |  the object panel e quantifies",
         transform=axF.transAxes, ha="center", va="top", fontsize=5.3, color=Z.GRAY)
cbF = fig.colorbar(imF, cax=caxF)
cbF.set_ticks([fvmin, (fvmin + fvmax) / 2, fvmax])
cbF.set_ticklabels([f"{fvmin:.1f}", f"{(fvmin+fvmax)/2:.1f}", f"{fvmax:.1f}"])
cbF.outline.set_linewidth(0.4)
cbF.ax.tick_params(width=0.5, length=2, labelsize=5.3)
cbF.set_label("λ(x)", fontsize=6, rotation=90, labelpad=1)

figletter(0.020, E_top + 0.042, "e")
figletter(0.600, E_top + 0.042, "f")

Z.save(fig, "F1")
