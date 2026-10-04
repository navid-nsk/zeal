"""ED2 - Extended Data Fig. 2: class audits behind the rank and hot-spot certificates (plot layer only; data from data/ED2.json)."""
import os, json
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.colors import to_rgb, to_hex, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import agram as A
import zeal_style as Z

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data", "ED2.json"), encoding="utf-8"))

# === LAYOUT (all geometry, sizes and formatting constants; mm unless stated)
H = 115.0
VM, VM_PTS, VM_GAP, VM_DY = 1.5, ((0.0, 0.05), (0.35, -0.38), (1.0, 0.5)), 0.8, 0.25
MS_PT, MEW, BARW, CAT_PAD = 3.0, 0.5, 0.92, 0.55
KEY_GAP, LEG_PAD = 1.6, 0.2
TK_PAD, TK_LEN, TK_W = 1.4, 2.4, 0.6
RGB = 3
FMT2 = "{:.2f}"
B1_Y, B1_KEY1, B1_KEY2 = 3.0, 8.6, 12.4
B1_AX_Y, B1_AX_H = 18.0, 31.0
B1_TOK_Y = 56.0
# a
A_X = 1.0
A_AX1 = (9.0, B1_AX_Y, 29.0, B1_AX_H)
A_AX2 = (47.0, B1_AX_Y, 22.0, B1_AX_H)
A_BW = 0.2
A_YL, A_YT = (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0]
A_BOXW, A_CAPW, A_MED_DY = 0.42, 0.22, 0.025
A_PLAB_DX = 0.6
# b
B_X = 79.0
B_AX = (98.5, 21.0, 17.5, 24.0)
B_VERD_Y1, B_VERD_Y2 = 8.6, 12.4
B_BH, B_GAP = 0.66, 0.55
B_KEY_Y = 16.4
B_XL, B_XT = (0, 1250), [0, 500, 1000]
B_YPAD = 0.65
# c
C_X = 121.0
C_MAP = (123.5, 9.5, 39.5)            # x, y_top, side
C_KEY = (165.5, 11.0, 2.6, 5.4)       # x, y_top, swatch width, step
C_CNT_DY = 2.5
C_LW_CTY = 0.25
# band 2
B2_Y, B2_HDR, B2_HDR2 = 62.5, 68.5, 72.0
E_KEY_X = 116.0
B2_AX_Y, B2_AX_H = 75.5, 24.5
B2_TOK_Y = 111.6
# d
D_X = 1.0
D_HM = (22.5, B2_AX_Y, 42.0, B2_AX_H)   # heatmap box
D_CB = (66.0, B2_AX_Y, 2.4, B2_AX_H)    # colour bar
D_AX2 = (89.0, B2_AX_Y, 20.5, B2_AX_H)
D_CB_T = [4, 6, 8]
D_CB_N = 128
D_BOX_LW, D_SEP_LW = 0.9, 1.4
D_HDR_DY, D_TICK_DY, D_XLAB_DY = 1.6, 2.4, 5.6
D_LAB_DX, D_CB_TK, D_CB_TX = 1.0, 0.8, 1.2
D_CB_TITLE_DX, ROT90 = 6.9, 90
D_HW_YL, D_HW_YT = (0, 0.06), [0, 0.02, 0.04, 0.06]
# e
E_X = 114.0
E_AX1 = (123.0, B2_AX_Y, 18.0, B2_AX_H)
E_AX2 = (158.0, B2_AX_Y, 21.0, B2_AX_H)
E_YL1, E_YT1 = (0, 0.1), [0, 0.05, 0.1]
E_YL1R, E_YT1R = (0, 70), [0, 35, 70]
E_YL2, E_YT2, E_YTOP = (0, 0.8), [0, 0.2, 0.4, 0.6, 0.8], 0.8
E_BW = 0.25
E_TICK_MS = 2.2
# === END LAYOUT

NC = Z.NOISE; NO = Z.NOISE_ORDER
fig, cv = A.canvas(h_mm=H)
fmtg = mtick.FuncFormatter(lambda v, _: A.fnum(v, "{:g}"))


def vmark(x, y_top, c=A.GREEND):
    """verdict tick mark (drawn: Arial has no check-mark glyph), centred on the text row at y_top"""
    y = cv.Y(y_top)
    cv.plot([x + VM * p[0] for p in VM_PTS], [y + VM * p[1] for p in VM_PTS], color=c, lw=A.LW_DATA, solid_capstyle="round", solid_joinstyle="round", zorder=8)


def verdict(x, y_top, s, mark=True):
    A.T(cv, x, cv.Y(y_top), s, A.F_KEY, va="center_baseline", c=A.TXT)
    if mark:
        vmark(x + A.mmw(s, A.F_KEY) + VM_GAP, y_top - VM_DY)


def radlabels(ax, R):
    ax.set_xticks(np.arange(len(R))); ax.set_xticklabels([A.fnum(r, "{:g}") for r in R])


# =========================================================================================== a ==
a = D["a"]; R = a["radii_km"]; xr = np.arange(len(R))
A.head(cv, A_X, B1_Y, "a", "Feasible population by radius")
SER = [("contained", "anchor contained", dict(fc=A.GRAYL, ec="none")), ("mass_feasible", "mass ≥ ½ window", dict(fc=A.GREY, ec="none")),
       ("class_feasible", "≥2 units", dict(fc="none", ec=A.INK)), ("two_distinct", "≥2 distinct cells", dict(fc=A.INK, ec="none"))]
ax = A.pax(fig, cv, *A_AX1, name="a1")
for i, (k, lab, st) in enumerate(SER):
    ax.bar(xr + (i - (len(SER) - 1) / 2) * A_BW, a[k], width=A_BW * BARW, fc=st["fc"], ec=st["ec"], lw=A.LW_THIN if st["ec"] != "none" else 0, zorder=2)
A.clean(ax, xl=(-CAT_PAD, len(R) - 1 + CAT_PAD), yl=A_YL, xlab="radius (km)", ylab="population share"); radlabels(ax, R); ax.set_yticks(A_YT)
A.key_row(cv, A_AX1[0], B1_KEY1, [("sw", SER[0][1], SER[0][2]), ("sw", SER[1][1], SER[1][2])], gap=KEY_GAP)
A.key_row(cv, A_AX1[0], B1_KEY2, [("sw", SER[2][1], dict(fc=A.WHITE, ec=A.INK, lw=A.LW_THIN)), ("sw", SER[3][1], SER[3][2])], gap=KEY_GAP)
ax = A.pax(fig, cv, *A_AX2, name="a2")
for i, q in enumerate(a["ratio_q"]):
    q05, q25, q50, q75, q95 = q
    ax.add_patch(Rectangle((i - A_BOXW / 2, q25), A_BOXW, q75 - q25, fc=A.GRAYL, ec="none", zorder=2))
    ax.plot([i - A_BOXW / 2, i + A_BOXW / 2], [q50, q50], color=A.INK, lw=A.LW_DATA, zorder=3)
    for lo, hi in ((q05, q25), (q75, q95)):
        ax.plot([i, i], [lo, hi], color=A.INK, lw=A.LW_THIN, zorder=2)
    for yy in (q05, q95):
        ax.plot([i - A_CAPW / 2, i + A_CAPW / 2], [yy, yy], color=A.INK, lw=A.LW_THIN, zorder=2)
    ax.text(i, q95 + A_MED_DY, A.N(f"ed2_ratio_med_{i}", q50, FMT2), ha="center", va="bottom", fontsize=A.F_VAL, color=A.INK)
A.refline(ax, a["p"], axis="y", c=A.BRICK)
A.T(cv, A_AX2[0] + A_AX2[2] + A_PLAB_DX, cv.Y(A_AX2[1] + A_AX2[3] * (1 - a["p"] / A_YL[1])), "p = ½", A.F_VAL, va="center_baseline", c=A.BRICK)
A.clean(ax, xl=(-CAT_PAD, len(R) - 1 + CAT_PAD), yl=A_YL, xlab="radius (km)", ylab="eligible / window mass"); radlabels(ax, R); ax.set_yticks(A_YT)
A.tok(cv, A_AX1[0], B1_TOK_Y, f"{A.N('ed2_tracts', a['n_tracts'], '{:d}')} tracts | box q25–q75, whiskers q05–q95, median printed")

# =========================================================================================== b ==
b = D["b"]
A.head(cv, B_X, B1_Y, "b", "Incompatible named pairs")
rk = ["7.5", "15.0"]
verdict(B_X, B_VERD_Y1, f"all-named class empty  {A.N('ed2_empty', sum(b[r]['all_named_empty'] for r in rk), '{:d}')}/{len(rk)}")
verdict(B_X, B_VERD_Y2, f"producer = checker  {A.N('ed2_agree', sum(b[r]['agree'] for r in rk), '{:d}')}/{len(rk)}")
A.key_row(cv, B_X, B_KEY_Y, [("sw", "fractional mass", dict(fc=A.GREY)), ("sw", "subset sums", dict(fc=A.INK))], gap=KEY_GAP)
ax = A.pax(fig, cv, *B_AX, name="b1")
ylab, yy = [], []
for g, r in enumerate(rk):
    for j, who in enumerate(("producer", "checker")):
        y = g * (2 + B_GAP) + j; v = b[r][who]
        ax.barh(y, v["fractional"], height=B_BH, color=A.GREY, lw=0, zorder=2)
        ax.barh(y, v["subset_sums"], left=v["fractional"], height=B_BH, color=A.INK, lw=0, zorder=2)
        ax.text(v["total"], y, " " + A.N(f"ed2_tot_{r}_{who}", v["total"], "{:d}"), ha="left", va="center", fontsize=A.F_VAL, color=A.TXT)
        ylab.append(f"{A.fnum(float(r), '{:g}')} km, {who}"); yy.append(y)
A.clean(ax, xl=B_XL, yl=(max(yy) + B_YPAD, -B_YPAD), xlab="ordered pairs")
ax.set_yticks(yy); ax.set_yticklabels(ylab); ax.set_xticks(B_XT); ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: A.fnum(int(v), "{:d}")))
A.tok(cv, B_X, B1_TOK_Y, f"{A.N('ed2_named', b['7.5']['n_named'], '{:d}')} named tracts | {A.N('ed2_pairs', b['7.5']['ordered_pairs'], '{:d}')} pairs")

# =========================================================================================== c ==
c = D["c"]
A.head(cv, C_X, B1_Y, "c", "Smallest feasible radius per tract")
rgbof = lambda h: np.array(to_rgb(h))
cols = [rgbof(A.PURPLE), (rgbof(A.PURPLE) + rgbof(A.LAVENDER)) / 2, rgbof(A.LAVENDER), rgbof(A.LAVL), rgbof(A.GRAYL)]
g = np.array([[-1 if ch == "." else int(ch) for ch in r_] for r_ in c["grid_rows"]]); rgb = np.ones(g.shape + (RGB,))
for i, col in enumerate(cols):
    rgb[g == i] = col
x0, y0, side = C_MAP
A.img(cv, rgb, x0, y0, side, side, kind="art", interp="nearest")
# county boundaries: pixel edges between different counties inside the mask (runs precomputed in the data layer), drawn white
px = side / g.shape[0]; yt0 = cv.Y(y0); segs = []
for i, s_, e_, horiz in c["county_edges"]:
    if horiz:
        segs.append([(x0 + s_ * px, yt0 - (i + 1) * px), (x0 + e_ * px, yt0 - (i + 1) * px)])
    else:
        segs.append([(x0 + (i + 1) * px, yt0 - s_ * px), (x0 + (i + 1) * px, yt0 - e_ * px)])
cv.add_collection(LineCollection(segs, colors=A.WHITE, linewidths=C_LW_CTY, zorder=4))
kx, ky, kw, ks = C_KEY
for i, cat in enumerate(c["categories"]):
    lab = cat if cat == "none" else A.fnum(float(cat.split()[0]), "{:g}") + " km"
    A.img(cv, np.array([[cols[i]]]), kx, ky + i * ks, kw, kw, kind="art", interp="nearest")
    A.T(cv, kx + kw + KEY_GAP / 2, cv.Y(ky + i * ks + kw / 2), lab, A.F_KEY, va="center_baseline")
    A.T(cv, kx + kw + KEY_GAP / 2, cv.Y(ky + i * ks + kw / 2 + C_CNT_DY), A.N(f"ed2_ntr_{i}", c["tract_counts"][i], "{:d}"), A.F_VAL, va="center_baseline", c=A.GREY)
A.tok(cv, C_MAP[0], B1_TOK_Y, f"{A.N('ed2_tracts', c['n_tracts'], '{:d}')} tracts | {A.N('ed2_cty', c['n_counties'], '{:d}')} counties | {c['raster_px'][0]}² raster")

# =========================================================================================== d ==
d = D["d"]; C = np.array(d["c"]); nl, nc = C.shape
A.head(cv, D_X, B2_Y, "d", "Per-law critical values and half-widths")
norm = Normalize(vmin=min(D_CB_T) - 1, vmax=C.max())
hx, hy, hw_, hh = D_HM; cw, ch = hw_ / nc, hh / nl
cells = Z.HEAT_SEQ(norm(C))[..., :RGB]
A.img(cv, cells, hx, hy, hw_, hh, kind="art", interp="nearest")
for i in range(nl):
    A.T(cv, hx - D_LAB_DX, cv.Y(hy + (i + 0.5) * ch), d["law_labels"][i], A.F_KEY, ha="right", va="center_baseline")
    for j in range(nc):
        A.T(cv, hx + (j + 0.5) * cw, cv.Y(hy + (i + 0.5) * ch), A.N(f"ed2_c_{i}_{j}", C[i, j], FMT2), A.F_VAL, ha="center", va="center_baseline", c=A.contrast_text(to_hex(cells[i, j])))
for j in range(nc):        # box the family critical value (the maximum over the four laws) in every column
    i = int(np.argmax(C[:, j]))
    cv.add_patch(Rectangle((hx + j * cw, cv.Y(hy + (i + 1) * ch)), cw, ch, fill=False, ec=A.INK, lw=D_BOX_LW, zorder=6))
for j, col in enumerate(d["columns"]):
    A.T(cv, hx + (j + 0.5) * cw, cv.Y(hy + hh + D_TICK_DY), A.fnum(col["r"], "{:g}"), A.F_TICK, ha="center", va="center_baseline", c=A.INK)
ne = sum(1 for col in d["columns"] if col["stat"] == "endpoint")
A.T(cv, hx + ne * cw / 2, cv.Y(hy - D_HDR_DY), "endpoint statistic", A.F_KEY, ha="center", va="center_baseline", c=A.INK)
A.T(cv, hx + (ne + (nc - ne) / 2) * cw, cv.Y(hy - D_HDR_DY), "pair statistic", A.F_KEY, ha="center", va="center_baseline", c=A.INK)
cv.plot([hx + ne * cw] * 2, [cv.Y(hy), cv.Y(hy + hh)], color=A.WHITE, lw=D_SEP_LW, zorder=5)
A.T(cv, hx + hw_ / 2, cv.Y(hy + hh + D_XLAB_DY), "radius (km)", A.F_AXLAB, ha="center", va="center_baseline", c=A.INK)
grad = Z.HEAT_SEQ(np.linspace(1, 0, D_CB_N))[:, None, :RGB]
cbx, cby, cbw, cbh = D_CB
A.img(cv, grad, cbx, cby, cbw, cbh, kind="art", interp="bilinear", frame=A.HAIR)
for t in D_CB_T:
    yt = cby + cbh * (1 - norm(t))
    cv.plot([cbx + cbw, cbx + cbw + D_CB_TK], [cv.Y(yt)] * 2, color=A.INK, lw=A.LW_AXIS, zorder=6)
    A.T(cv, cbx + cbw + D_CB_TX, cv.Y(yt), A.fnum(t, "{:g}"), A.F_TICK, va="center_baseline", c=A.INK)
A.T(cv, cbx + cbw + D_CB_TITLE_DX, cv.Y(cby + cbh / 2), "critical value", A.F_KEY, ha="center", va="center", c=A.INK, rot=ROT90)
ax = A.pax(fig, cv, *D_AX2, name="d2")
xr4 = np.arange(len(d["half_width_gaussian"])); hdl = []
for k, vals, mk, ls in (("gaussian", d["half_width_gaussian"], "s", Z.LS_SOLID), ("family", d["half_width_family"], "^", Z.LS_DASH)):
    ax.plot(xr4, vals, color=NC[k]["c"], ls=ls, lw=A.LW_DATA, marker=mk, ms=MS_PT, mfc=NC[k]["c"], mec=A.TEALM, mew=MEW, zorder=3)
    hdl.append(Line2D([], [], color=NC[k]["c"], ls=ls, lw=A.LW_DATA, marker=mk, ms=MS_PT, mfc=NC[k]["c"], mec=A.TEALM, mew=MEW))
ax.legend(hdl, [NC["gaussian"]["label"], NC["family"]["label"]], loc="upper right", frameon=False, fontsize=A.F_KEY, borderaxespad=LEG_PAD)
A.clean(ax, xl=(-CAT_PAD, len(xr4) - 1 + CAT_PAD), yl=D_HW_YL, xlab="radius (km)", ylab="median half-width (kfr)")
radlabels(ax, [col["r"] for col in d["columns"][:len(xr4)]]); ax.set_yticks(D_HW_YT); ax.yaxis.set_major_formatter(fmtg)
A.tok(cv, D_X + D_LAB_DX, B2_TOK_Y, f"B = {A.N('ed2_B', d['B'], '{:d}')} draws per law | Atlas s.e. | boxed = family value")

# =========================================================================================== e ==
e = D["e"]; Re = e["radii_km"]; xe = np.arange(len(Re))
A.head(cv, E_X, B2_Y, "e", "Distinct admissible focal cells")
ax = A.pax(fig, cv, *E_AX1, name="e1")
ax.bar(xe, e["single_cell_only"], width=BARW * CAT_PAD, color=A.GRAYL, lw=0, zorder=2)
A.clean(ax, xl=(-CAT_PAD, len(Re) - 1 + CAT_PAD), yl=E_YL1, xlab="radius (km)", ylab="single-cell share"); radlabels(ax, Re); ax.set_yticks(E_YT1); ax.yaxis.set_major_formatter(fmtg)
axr = ax.twinx(); axr.set_facecolor("none")
axr.plot(xe, e["log2_median"], color=A.INK, lw=A.LW_DATA, marker="o", ms=MS_PT, mfc=A.WHITE, mec=A.INK, mew=MEW * 2, zorder=3)
axr.set_ylim(*E_YL1R); axr.set_yticks(E_YT1R)
axr.tick_params(axis="y", labelsize=A.F_TICK, pad=TK_PAD, length=TK_LEN, width=TK_W, colors=A.INK, direction="out")
axr.tick_params(axis="x", bottom=False, labelbottom=False)
for t_ in axr.get_xticklabels():
    t_.set_visible(False)
for sp in ("top", "left", "bottom"):
    axr.spines[sp].set_visible(False)
axr.spines["right"].set_linewidth(A.LW_AXIS); axr.spines["right"].set_color(A.INK)
axr.set_ylabel("median log2 count", fontsize=A.F_AXLAB, color=A.INK, labelpad=TK_PAD)
A.key_row(cv, E_KEY_X, B2_HDR2, [("sw", "single cell only", dict(fc=A.GRAYL)), ("marker", "log2 count", dict(marker="o", fc=A.WHITE, ec=A.INK))], gap=KEY_GAP)
ax2 = A.pax(fig, cv, *E_AX2, name="e2")
for i, k in enumerate(NO):
    ax2.bar(xe + (i - 1) * E_BW, e[f"not_hot_{k}"], width=E_BW * BARW, color=NC[k]["c"], lw=0, zorder=2)
    ax2.plot(xe + (i - 1) * E_BW, e[f"not_hot_two_{k}"], ls="none", marker="_", ms=MS_PT * E_TICK_MS, mew=A.LW_DATA, color=A.INK, zorder=4)
A.clean(ax2, xl=(-CAT_PAD, len(Re) - 1 + CAT_PAD), yl=E_YL2, xlab="radius (km)", ylab="not-hot share"); radlabels(ax2, Re); ax2.set_yticks(E_YT2); ax2.yaxis.set_major_formatter(fmtg); ax2.spines["left"].set_bounds(0, E_YTOP)
A.key_row(cv, E_KEY_X, B2_HDR, [("sw", NC[k]["label"], dict(fc=NC[k]["c"])) for k in NO]
          + [("marker", "≥2 cells", dict(marker="_", fc=A.INK, ec=A.INK, lw=A.LW_DATA, ms=MS_PT * E_TICK_MS))], gap=KEY_GAP)
A.tok(cv, E_AX1[0], B2_TOK_Y, "audited window class | p = ½ | population shares")

A.save(fig, "ED2", script=os.path.abspath(__file__))
