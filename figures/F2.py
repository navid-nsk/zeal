"""F2 - Fig. 2: map-level certified brackets on three geographies (plot layer only; data from data/F2.json, data/F2_maps.npz)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agram as A
import zeal_style as Z
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from matplotlib.colors import LogNorm, to_rgb
import matplotlib.ticker as mticker

D = json.load(open(os.path.join(HERE, "data", "F2.json"), encoding="utf-8"))
MP = np.load(os.path.join(HERE, "data", "F2_maps.npz"))
ORDER = D["order"]; ST = D["settings"]

# === LAYOUT (every numeric literal of the figure lives here) =========================================================
H = 198.6
KEY_TOPX = 79.0; KEY_TOPY = 2.9          # figure-wide setting key (panels c, d, e)
X0 = 1.5                                   # panel-letter x, left column
XR = 91.0                                  # panel-letter x, right column
# band a: maps
A_Y = 1.5; A_LAB1 = 7.0; A_LAB2 = 10.1; A_MAPTOP = 13.4; A_MAPH = 35.8; A_MAPX0 = 2.2; A_GAP = 2.6
A_CB_Y = 51.4; A_CB_H = 1.7; A_CB_X = 64.0; A_CB_W = 84.0; A_CB_TICKS = [0.25, 0.5, 1, 2, 4]; A_CB_TL = 0.8; A_CB_TDY = 0.2
A_CB_LABDX = 2.0; A_CB_LABDY = 0.15; A_CB_N = 512
A_CMAP_LO = 0.12                           # lowest ramp position: the smallest width stays distinguishable from the white mask
A_NOPAIR_X = 152.5; A_TOK_Y = 56.4; A_SEG_LW = 0.22
# band b / c
B_Y = 60.6; B_KEY1 = 66.4; B_KEY2 = 69.5; B_KEYX = 4.5; B_AX = (33.0, 73.0, 50.0, 45.5); B_XL = (0.93, 2.47); B_XT = [1.0, 1.5, 2.0, 2.5]
B_ROWY = [3.0, 2.0, 1.0, 0.0]; B_YL = (-0.66, 3.5); B_SUB1 = 0.25; B_SUB2 = 0.45; B_BAR_LW = 4.6; B_MS = 4.2; B_MS2 = 3.4
B_UDY = 0.17; B_MCOL_X = 2.45; B_MHEAD_DY = 0.42; B_TOK_Y = 125.9
C_AX = (101.0, 73.0, 67.5, 45.5); C_YT = [1.1, 1.2, 1.4, 1.6, 1.8]; C_YL = (1.07, 1.84); C_XL = (-0.55, 3.62)
C_VPOS = -0.16; C_BPOS = 0.27; C_VW = 0.5; C_BW = 0.2; C_MED_DX = 0.0; C_VALPHA = 0.55; C_TICK_DX = 0.06
C_KEYX = 118.0; C_MIN_SEG = 11.0; C_YTB = [0, 25, 50, 75, 100]; C_TOK_Y = 125.9; C_MED_DY = 0.012
# band d / e
D_Y = 130.2; D_AX = (11.0, 136.4, 74.0, 50.3); D_XL = (-0.05, 2.05); D_XT = [0, 0.5, 1, 1.5, 2]; D_YL = (-3.0, 103.0)
D_YT = [0, 25, 50, 75, 100]; D_TAB_X = 1.12; D_TAB_Y0 = 43.0; D_TAB_DY = 9.0; D_TAB_HEAD_Y = 52.0; D_TAB_MK = 0.03; D_TAB_TX = 0.09
D_TOK_Y = 193.6
E_AX1 = (101.0, 136.4, 61.0, 28.0); E_AXM = (163.0, 136.4, 13.5, 28.0); E_AX3 = (101.0, 172.2, 74.5, 14.5); E_MXT = [0, 1000, 2000]; E_MXL = ['0', '1', '2']
E_LT = 1e-3; E_XL1 = (-1.6, 1.6); E_XT1 = [-1, -0.01, 0, 0.01, 1]; E_YL1 = (10 ** -10.5, 10 ** -4.6); E_YT1 = [1e-10, 1e-8, 1e-6]
E_YTL1 = ["1e−10", "1e−8", "1e−6"]; E_MS = 1.1; E_ALPHA = 0.5; E_THIN = 3
E_XL3 = (10.0, 1000.0); E_XT3 = [10, 30, 100, 300, 1000]; E_YL3 = (0.15, 15); E_YT3 = [0.3, 1, 3, 10]; E_KROW = 1000.0
E_PASS_XY = (0.03, 0.97); E_TOK_Y = 193.6; B_LABPAD = 81.0
KEY_SW = 1.8; KEY_GAP = 2.2; KEY_LINE = 4.2; KEY_TDX = 0.9; MAXN = 3; RGB = 3; TEN = 10.0; MED_HALF = 0.13; ROT = 90
# === END LAYOUT =======================================================================================================

fig, cv = A.canvas(A.W_CANVAS, H)
SET = Z.SETTING
NAME_MAP = {"georgia": "Georgia", "gm_q4": "Greater Manchester, qualification", "gm_bad": "Greater Manchester, bad health", "mx_rwi": "Mexico"}
ROW_LAB = {"georgia": "Georgia", "gm_q4": "Manchester, qualification", "gm_bad": "Manchester, bad health", "mx_rwi": "Mexico"}
CAT_LAB = {"georgia": "Georgia", "gm_q4": "Manchester,\nqualification", "gm_bad": "Manchester,\nbad health", "mx_rwi": "Mexico"}
PLURAL = {"tract": "tracts", "county": "counties", "LSOA": "LSOAs", "MSOA": "MSOAs", "municipality": "municipalities", "state": "states"}


def mm_text(x, y_top, s, size=A.F_KEY, ha="left", c=A.TXT, va="top"):
    return A.T(cv, x, cv.Y(y_top), s, size, ha=ha, va=va, c=c)


def key_items(x, y_top, items, size=A.F_KEY):
    """horizontal key without box: (kind, label, style); kind 'lm' line + marker, 'm' marker, 'bar' thick line, 'ref' dashed, 'sw' swatch."""
    y = cv.Y(y_top); xx = x
    for kind, lab, st in items:
        if kind == "lm":
            cv.plot([xx, xx + KEY_LINE], [y, y], color=st["c"], ls=st["ls"], lw=A.LW_DATA, zorder=6, solid_capstyle="butt")
            cv.plot([xx + KEY_LINE / 2], [y], marker=st["mk"], ms=B_MS2, mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN / 2, ls="none", zorder=7); w = KEY_LINE
        elif kind == "m":
            cv.plot([xx + KEY_SW / 2], [y], marker=st["mk"], ms=st.get("ms", B_MS2), mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN, ls="none", zorder=7); w = KEY_SW
        elif kind == "bar":
            cv.plot([xx, xx + KEY_LINE], [y, y], color=st["c"], lw=st["lw"], zorder=6, solid_capstyle="butt"); w = KEY_LINE
        elif kind == "ref":
            cv.plot([xx, xx + KEY_LINE], [y, y], color=A.GRAYREF, ls=A.LS_REF, lw=A.LW_REF, zorder=6); w = KEY_LINE
        else:
            cv.add_patch(A.Rectangle((xx, y - KEY_SW / 2), KEY_SW, KEY_SW, fc=st["fc"], ec=st.get("ec", "none"), lw=st.get("lw", 0), zorder=6)); w = KEY_SW
        A.T(cv, xx + w + KEY_TDX, y, lab, size, va="center_baseline"); xx += w + KEY_TDX + A.mmw(lab, size) + KEY_GAP
    return xx


def fixed_ticks(axis, vals, labels):
    axis.set_major_locator(mticker.FixedLocator(vals)); axis.set_major_formatter(mticker.FixedFormatter(labels)); axis.set_minor_locator(mticker.NullLocator())


# ===================================================================================================== a: maps ==
A.head(cv, X0, A_Y, "a", "Per-cell certified pair-range widths")
key_items(KEY_TOPX, KEY_TOPY, [("lm", ROW_LAB[f], SET[f]) for f in ORDER])
lo, hi = D["map_range"]; norm = LogNorm(vmin=lo, vmax=hi)
ramp = lambda v: Z.FIELD_SEQ(A_CMAP_LO + (1 - A_CMAP_LO) * np.clip(norm(v), 0, 1))
grey_rgb = np.array(to_rgb(A.GRAYL)); x = A_MAPX0
for f in ORDER:
    s = ST[f]; lab = MP[f"fine_{f}"]; Hh, Ww = lab.shape; w_mm = A_MAPH * Ww / Hh
    vals = np.full(int(lab.max()) + 2, np.nan)
    for k, v in s["width_by_fine"].items(): vals[int(k)] = v
    v_img = np.where(lab >= 0, vals[np.where(lab >= 0, lab, -1)], np.nan)
    rgb = np.ones(lab.shape + (RGB,)); inside = lab >= 0; paired = inside & np.isfinite(v_img)
    rgb[paired] = ramp(v_img[paired])[:, :RGB]; rgb[inside & ~paired] = grey_rgb
    A.img(cv, rgb, x, A_MAPTOP, w_mm, A_MAPH, kind="art", interp="nearest")
    sg = MP[f"segs_{f}"].astype(float); sx, sy = w_mm / Ww, A_MAPH / Hh; ytop = cv.Y(A_MAPTOP)
    x0s, y0s, x1s, y1s = sg.T
    segs = np.stack([np.c_[x + x0s * sx, ytop - y0s * sy], np.c_[x + x1s * sx, ytop - y1s * sy]], 1)
    verts = segs.reshape(-1, 2); codes = np.tile([MPath.MOVETO, MPath.LINETO], len(segs))     # one compound path per map
    cv.add_patch(PathPatch(MPath(verts, codes), fc="none", ec=A.HAIR, lw=A_SEG_LW, capstyle="butt", zorder=4))
    mm_text(x, A_LAB1, NAME_MAP[f], A.F_AXLAB)
    fu, cu = s["fine_unit"], s["coarse_unit"]
    mm_text(x, A_LAB2, f"{A.N(f'F2.nfine.{f}', s['n_fine_pairs'], '{:d}')} {PLURAL[fu]} → {A.N(f'F2.ncoarse.{f}', s['n_coarse_pairs'], '{:d}')} {PLURAL[cu]}", A.F_KEY)
    x += w_mm + A_GAP
grad = np.linspace(np.log10(lo), np.log10(hi), A_CB_N)
A.img(cv, ramp(np.power(TEN, grad))[None, :, :RGB], A_CB_X, A_CB_Y, A_CB_W, A_CB_H, kind="art", interp="nearest")
for t in A_CB_TICKS:
    xt = A_CB_X + A_CB_W * (np.log10(t) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    cv.plot([xt, xt], [cv.Y(A_CB_Y + A_CB_H), cv.Y(A_CB_Y + A_CB_H + A_CB_TL)], color=A.INK, lw=A.LW_AXIS, zorder=6)
    mm_text(xt, A_CB_Y + A_CB_H + A_CB_TL + A_CB_TDY, A.fnum(t), A.F_TICK, ha="center")
mm_text(A_CB_X - A_CB_LABDX, A_CB_Y - A_CB_LABDY, "certified range width (× reference)", A.F_AXLAB, ha="right")
key_items(A_NOPAIR_X, A_CB_Y + A_CB_H / 2, [("sw", "no nested pair", dict(fc=A.GRAYL))])
A.tok(cv, A_MAPX0, A_TOK_Y, "fine-cell values | lifted outer bound, m = 2 (Mexico 1)")

# ===================================================================================================== b: brackets ==
A.head(cv, X0, B_Y, "b", "Map-level brackets per setting")
bk = dict(glob=Z.BOUND["global"], sep=Z.BOUND["separable"], ceil=Z.BOUND["lifted"], pix=Z.BOUND["pixel"], vo=Z.BOUND["value_only"])
mk = lambda k: dict(mk=bk[k]["mk"], mfc=bk[k]["mfc"], mec=bk[k]["mec"])
key_items(B_KEYX, B_KEY1, [("m", "global witness", mk("glob")), ("m", "separable witness", mk("sep")), ("m", "lifted outer bound", mk("ceil")),
                           ("bar", "certified bracket", dict(c=Z.BRACKET_FILL, lw=B_BAR_LW))])
key_items(B_KEYX, B_KEY2, [("m", "pixel outer bound", mk("pix")), ("m", "value-only bound", mk("vo")), ("ref", "reference", {})])
ax = A.pax(fig, cv, *B_AX, name="b")
A.refline(ax, 1.0, axis="x")
for f, yr in zip(ORDER, B_ROWY):
    b = ST[f]["bracket"]; L, U = b["L_checker"], b["U_checker"]
    ax.plot([L, U], [yr, yr], color=Z.BRACKET_FILL, lw=B_BAR_LW, solid_capstyle="butt", zorder=2)
    for key, xv, yy in (("sep", b["S_separable_witness_x"], yr), ("glob", L, yr), ("ceil", U, yr), ("pix", b["S_PhiC_x"], yr - B_SUB1), ("vo", b["S_V_x"], yr - B_SUB2)):
        st = bk[key]; ax.plot([xv], [yy], marker=st["mk"], ms=B_MS if key in ("glob", "ceil") else B_MS2, mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN, ls="none", zorder=4)
    ax.text(L, yr + B_UDY, A.N(f"F2.L.{f}", L, "{:.4f}"), fontsize=A.F_VAL, ha="center", va="bottom", color=Z.BOUND["global"]["c"], zorder=5)
    ax.text(U, yr + B_UDY, A.N(f"F2.U.{f}", U, "{:.4f}"), fontsize=A.F_VAL, ha="center", va="bottom", color=Z.BOUND["lifted"]["c"], zorder=5)
    ax.text(B_MCOL_X, yr, A.N(f"F2.m.{f}", ST[f]["m"], "{:d}"), fontsize=A.F_VAL, ha="right", va="center", color=A.TXT)
ax.text(B_MCOL_X, B_ROWY[0] + B_MHEAD_DY, "$m$", fontsize=A.F_VAL, ha="right", va="bottom", color=A.TXT)
A.clean(ax, xl=B_XL, yl=B_YL, xlab="map-level movement (× reference)")
ax.set_xticks(B_XT); ax.set_yticks(B_ROWY); ax.set_yticklabels([ROW_LAB[f] for f in ORDER], ha="left"); ax.tick_params(axis="y", length=0, pad=B_LABPAD)
ax.spines["left"].set_visible(False)
A.tok(cv, X0, B_TOK_Y, "checker-recomputed endpoints | exact witnesses | midpoint reference = 1")

# ===================================================================================================== c: per pair ==
A.head(cv, XR, B_Y, "c", "Per-pair ratios and sign resolution")
MAPSIGN = {"invariant": "sign resolved", "sensitive": "both signs realizable", "open": "open"}   # sign of a pair difference over the enclosure class (a fixed ladder: the aggregation does not vary)
key_items(C_KEYX, B_KEY2, [("sw", MAPSIGN[k], dict(fc=Z.DECISION[k]["c"])) for k in Z.DECISION_ORDER if k != "witnessed"])   # map signs: invariant / sensitive / open
axv = A.pax(fig, cv, *C_AX, name="c")
pos = np.arange(len(ORDER))
for i, f in enumerate(ORDER):
    r = np.log10(np.asarray(ST[f]["ratio"])); st = SET[f]; xv = i + C_VPOS
    vp = axv.violinplot([r], positions=[xv], widths=C_VW, showextrema=False, showmedians=False)
    for bdy in vp["bodies"]:
        bdy.set_facecolor(st["c"]); bdy.set_edgecolor(st["mec"]); bdy.set_linewidth(A.LW_FRAME); bdy.set_alpha(C_VALPHA)
    med = ST[f]["ratio_median"]; q = np.log10(med)
    axv.plot([xv - MED_HALF, xv + MED_HALF], [q, q], color=A.INK, lw=A.LW_DATA, zorder=4, solid_capstyle="butt")
    axv.plot([xv, xv], [r.min(), r.max()], color=A.INK, lw=A.LW_FRAME, zorder=3)
    axv.text(xv + C_MED_DX, r.max() + C_MED_DY, A.N(f"F2.ratio_med.{f}", med, "{:.3f}"), fontsize=A.F_VAL, ha="center", va="bottom", color=A.TXT, zorder=5)
A.clean(axv, xl=C_XL, yl=tuple(np.log10(C_YL)), ylab="outer / witness width")
fixed_ticks(axv.yaxis, np.log10(C_YT), [A.fnum(t) for t in C_YT]); axv.set_xticks(pos + C_TICK_DX); axv.set_xticklabels([CAT_LAB[f] for f in ORDER])
axb = axv.twinx(); axb.set_facecolor("none"); axb.tick_params(axis="x", bottom=False, labelbottom=False)
for i, f in enumerate(ORDER):
    sg = ST[f]["sign"]; n = sum(sg.values()); bot = 0.0
    for k in [k_ for k_ in Z.DECISION_ORDER if k_ in sg]:      # the sign categories of this panel (no 'witnessed' class for map signs)
        hgt = 100.0 * sg[k] / n; col = Z.DECISION[k]["c"]
        axb.bar([i + C_BPOS], [hgt], bottom=[bot], width=C_BW, color=col, edgecolor=A.WHITE, linewidth=A.LW_FRAME, zorder=2)
        if hgt >= C_MIN_SEG:
            axb.text(i + C_BPOS, bot + hgt / 2, A.N(f"F2.sign.{k}.{f}", sg[k], "{:d}"), fontsize=A.F_VAL, ha="center", va="center", rotation=ROT, color=A.contrast_text(col), zorder=4)
        bot += hgt
A.clean(axb, xl=C_XL, yl=(0, 100)); axb.spines["right"].set_visible(True); axb.spines["right"].set_linewidth(A.LW_AXIS); axb.spines["right"].set_color(A.INK)
axb.spines["left"].set_visible(False); axb.spines["bottom"].set_visible(False)
axb.set_ylabel("nested pairs (%)", fontsize=A.F_AXLAB, color=A.INK, labelpad=A.LW_AXIS); fixed_ticks(axb.yaxis, C_YTB, [A.fnum(t) for t in C_YTB])
A.tok(cv, XR, C_TOK_Y, f"{A.N('F2.pairs_total', sum(ST[f]['n_pairs'] for f in ORDER), '{:d}')} nested pairs | per-pair outer / witness ratio | sign of pair difference")

# ===================================================================================================== d: global witness ==
A.head(cv, X0, D_Y, "d", "Global witness construction")
ax = A.pax(fig, cv, *D_AX, name="d")
A.refline(ax, 1.0, axis="x")
for f in ORDER:
    g = ST[f]["global_witness"]; st = SET[f]; xs, ys = np.asarray(g["x"]), np.asarray(g["y"])
    ax.plot(xs, ys, color=st["c"], ls=st["ls"], lw=A.LW_DATA, zorder=3)
    ends = np.isin(xs, D_XT[::2])
    ax.plot(xs[ends], ys[ends], ls="none", marker=st["mk"], ms=B_MS2, mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN / 2, zorder=4)
ax.text(D_TAB_X, D_TAB_HEAD_Y, "lattice top → global witness", fontsize=A.F_VAL, ha="left", va="center", color=A.TXT)
for i, f in enumerate(ORDER):
    g = ST[f]["global_witness"]; st = SET[f]; yy = D_TAB_Y0 - i * D_TAB_DY
    ax.plot([D_TAB_X + D_TAB_MK], [yy], marker=st["mk"], ms=B_MS2, mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN / 2, ls="none", zorder=5)
    ax.text(D_TAB_X + D_TAB_TX, yy, f"{A.N(f'F2.lt.{f}', g['S_lattice_top_x'], '{:.3f}')} → {A.N(f'F2.gw.{f}', g['S_global_witness_x'], '{:.3f}')}  (n = {A.N(f'F2.nacc.{f}', g['n_accepted'], '{:d}')})",
            fontsize=A.F_VAL, ha="left", va="center", color=A.TXT)
A.clean(ax, xl=D_XL, yl=D_YL, xlab="cellwise sweeps (accepted improvements, scaled per sweep)", ylab="cumulative cellwise gain (% of total)")
fixed_ticks(ax.xaxis, D_XT, [A.fnum(t) for t in D_XT]); ax.set_yticks(D_YT)
A.tok(cv, X0, D_TOK_Y, "one nodal field per setting | verified feasible steps | values × reference")

# ===================================================================================================== e: Lean checker ==
A.head(cv, XR, D_Y, "e", "Exact-rational certificates, verified checker")
ax = A.pax(fig, cv, *E_AX1, name="e_scatter")
for f in ORDER:
    le = ST[f]["lean"]; st = SET[f]
    ax.plot(le["exact"], le["slack"], ls="none", marker=st["mk"], ms=E_MS, mfc=st["mfc"], mec=st["mec"], mew=0, alpha=E_ALPHA, zorder=3)
ax.set_xscale("symlog", linthresh=E_LT); ax.set_yscale("log")
A.clean(ax, xl=E_XL1, yl=E_YL1, xlab="exact certificate bound (pair difference)", ylab="relative outward slack")
fixed_ticks(ax.xaxis, E_XT1, [A.fnum(t) for t in E_XT1]); fixed_ticks(ax.yaxis, E_YT1, E_YTL1)
tot = D["lean_total"]
A.inlab(ax, E_PASS_XY[0], E_PASS_XY[1], f"{A.N('F2.lean_pass', tot['n_pass'], '{:d}')} / {A.N('F2.lean_n', tot['n'], '{:d}')} PASS", c=A.GREEND)
axm = A.pax(fig, cv, *E_AXM, name="e_marg")
ed = np.asarray(D["slack_hist"]["log10_edges"]); ctr = np.power(TEN, (ed[:-1] + ed[1:]) / 2); left = np.zeros(len(ctr))
for f in ORDER:
    cnt = np.asarray(D["slack_hist"]["counts"][f], float)
    axm.barh(ctr, cnt, left=left, height=np.diff(np.power(TEN, ed)), color=SET[f]["c"], edgecolor=A.WHITE, linewidth=A.LW_FRAME / 2, zorder=2); left += cnt
axm.set_yscale("log"); A.clean(axm, yl=E_YL1, xlab="endpoints (k)"); fixed_ticks(axm.yaxis, E_YT1, ["" for _ in E_YT1]); fixed_ticks(axm.xaxis, E_MXT, E_MXL)
ax3 = A.pax(fig, cv, *E_AX3, name="e_time")
for f in ORDER:
    le = ST[f]["lean"]; st = SET[f]
    ax3.plot(np.asarray(le["rows"][::E_THIN]) / E_KROW, le["check_s"][::E_THIN], ls="none", marker=st["mk"], ms=E_MS, mfc=st["mfc"], mec=st["mec"], mew=0, alpha=E_ALPHA, zorder=3)
ax3.set_xscale("log"); ax3.set_yscale("log")
A.clean(ax3, xl=E_XL3, yl=E_YL3, xlab="constraint rows (thousands)", ylab="check (s)")
fixed_ticks(ax3.xaxis, E_XT3, [A.fnum(t) for t in E_XT3]); fixed_ticks(ax3.yaxis, E_YT3, [A.fnum(t) for t in E_YT3])
A.tok(cv, XR, E_TOK_Y, "both endpoint directions | compiled verified checker | timing: every third endpoint")

bad = A.save(fig, "F2", script=os.path.abspath(__file__))
