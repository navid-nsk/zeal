"""F3 - Fig. 3 (transport fusion and realization mechanics): plot layer only. Every drawn number comes from data/F3.json
(written by F3_prep.py); numeric literals live only in the LAYOUT block. Mathematical scripts are set as separate 6.0 pt
runs (mathtext scripts would fall below the 6.0 pt floor)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agram as A
import zeal_style as Z
from agram import TEAL, TEALM, TEALL, TEALLL, BRICK, INK, TXT, GREY, GRAYREF, HAIR, GRAYL, GRAYLL, WHITE
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from matplotlib.colors import to_hex

D = json.load(open(os.path.join(HERE, "data", "F3.json"), encoding="utf-8"))

# === LAYOUT (mm from the top-left corner; sizes in pt; every numeric literal of the figure lives here)
FIG_H = 112.0
LET_X, TITLE_DX, TOP = 0.8, 4.4, 1.0
SUP_DY, SUB_DY, TEN, ROT90 = 1.05, 0.75, 10, 90
KEY_SW, KEY_TX, KEY_LEN, KEY_SWH, KEY_GAP, KEY_MS = 2.2, 0.7, 3.5, 1.76, 2.6, 2.8
Z_IMG, Z_TXT = 3, 6
AX_Y, AX_H = 9.2, 30.4                      # band 1 plot boxes (shared top and bottom)
TOK1_Y = 48.6
# ---- a: heatmap
A_GX, A_CW, A_CH = 12.6, 10.8, 7.6
A_CB_DX, A_CB_W, A_CB_T, A_CB_LAB_DY = 1.4, 1.1, 0.8, 1.9
A_VMIN, A_VMAX = 0.5, 1.0; A_CB_TICKS = [0.5, 0.75, 1.0]; A_CB_N = 256
A_MED_DY, A_Q_DY = -1.2, 1.7; A_BEST_LW = 1.0; A_GRID_LW = 0.4
A_ROWLAB_DX, A_ROWT_X, A_COLLAB_DY, A_COLT_DY = 1.0, 3.4, 1.0, 4.6
# ---- b: boundary displacement
B_X0 = 54.6
BX, BW = 64.4, 41.6
B_YL = (0.045, 1.3); B_YT = [0.05, 0.1, 0.2, 0.5, 1.0]; B_OFF = [-0.07, -0.025, 0.025, 0.07]; B_XPAD = 0.35; B_MS = 3.0
B_KEY_X, B_KEY_Y = 80.0, [13.0, 15.8]
# ---- c: conditioned classes (two small multiples + tent inset)
C_X0 = 110.8
C1X, C2X, CW = 120.4, 151.0, 27.4
C_SUB_Y = 5.4
C_PAD = 0.35; C_LW_OUT, C_LW_IN, C_MS = 1.7, 0.7, 2.4
C_XT = [-2.5, 0.0, 2.5]; C_YT = [-4, -2, 0, 2, 4]
C_KEY_X, C_KEY_Y = 121.8, [11.0, 13.8, 16.6]
INS_X, INS_Y, INS_W, INS_H = 152.0, 12.2, 11.2, 5.8
INS_BAR, INS_LAB_DY, INS_UNIT, INS_YL = 0.62, 0.9, 1e-4, (1e-5, 3.2e-3)
C_XLAB_Y = 44.8
# ---- d: interpolation-completed bracket vs m
BAND2 = 55.0
DX, DW = 10.6, 80.6
D_KEY_Y, D_AX_Y, D_AX_H = 60.2, 63.0, 34.8
D_OFF = [-0.09, -0.03, 0.03, 0.09]; D_MS = 3.0; D_YL = (0.022, 2.4); D_YT = [0.03, 0.1, 0.3, 1.0]; D_XPAD = 0.4
D_STYLE_X, D_STYLE_Y = 58.0, [66.0, 68.8]
D_PRINT = (14.0, 92.0); D_PRINT_LH = 3.0
TOK2_Y = 108.4
# ---- e: saving identity audit
E_X0 = 98.6
EX, EW = 110.6, 67.6
E_XL = (0.04, 140.0); E_YL = (0.0012, 140.0); E_XT = [0.1, 1.0, 10.0, 100.0]; E_YT = [0.01, 0.1, 1.0, 10.0, 100.0]
E_MS_F, E_MS_O = 3.0, 2.0; PCT = 100
E_KEY_X, E_KEY_Y = 112.6, [66.0, 68.8, 71.6]
E_PRINT = (145.0, 87.6); E_PRINT_LH = 3.0
# === END LAYOUT

fig, cv = A.canvas(h_mm=FIG_H)
Yc = cv.Y


def txt(x, y, s, size=A.F_VAL, ha="left", va="top", c=TXT, **kw):
    return A.T(cv, x, Yc(y), s, size, ha=ha, va=va, c=c, **kw)


def mlabel(x, y, segs, size=A.F_VAL, ha="left", c=TXT):
    """label with manual scripts at baseline y_top: segs = [(text, 'b' | 'sub' | 'sup')]; scripts at the 6.0 pt floor."""
    sz = [size if k == "b" else A.F_MIN for _, k in segs]; ws = [A.mmw(t, s_) for (t, _), s_ in zip(segs, sz)]
    x0 = x - sum(ws) / 2 if ha == "center" else (x - sum(ws) if ha == "right" else x)
    for (t, k), s_, w in zip(segs, sz, ws):
        dy = 0 if k == "b" else (-SUP_DY if k == "sup" else SUB_DY)
        A.T(cv, x0, Yc(y + dy), t, s_, ha="left", va="baseline", c=c); x0 += w


def sci(v):
    """mantissa and exponent strings of v (for a manual superscript)"""
    ex = int(np.floor(np.log10(abs(v)))); return f"{v / TEN ** ex:.1f}", A.fnum(ex)


def key_item(x, y, kind, label, c, ls="-", lw=A.LW_DATA, marker=None, mfc=None, ms=KEY_MS, fc=None):
    yy = Yc(y)
    if kind == "sw":
        cv.add_patch(Rectangle((x, yy - KEY_SWH / 2), KEY_SW, KEY_SWH, fc=fc or c, ec="none", zorder=Z_TXT)); w = KEY_SW
    else:
        w = KEY_LEN
        if lw > 0:
            cv.plot([x, x + w], [yy, yy], color=c, ls=ls, lw=lw, zorder=Z_TXT, solid_capstyle="butt")
        if marker:
            cv.plot([x + w / 2], [yy], marker=marker, ms=ms, mfc=mfc or c, mec=c, mew=A.LW_THIN, ls="none", zorder=Z_TXT)
    A.T(cv, x + w + KEY_TX, yy, label, A.F_KEY, va="center_baseline", c=TXT)
    return x + w + KEY_TX + A.mmw(label, A.F_KEY)


def log_ticks(axis, ticks):
    axis.set_minor_locator(NullLocator()); axis.set_ticks(ticks); axis.set_ticklabels([A.fnum(v) for v in ticks])


# ===================================================================================== panel a ==
a = D["a"]; kv, kg = a["kv"], a["kg"]; cells = a["cells"]
A.head(cv, LET_X, TOP, "a", "Fused / value-only by tile budget", dx=TITLE_DX)
cmap = Z.FIELD_SEQ.reversed()
M = np.full((len(kv), len(kg)), np.nan)
for i, v in enumerate(kv):
    for j, g in enumerate(kg):
        if f"v{v}g{g}" in cells:
            M[i, j] = cells[f"v{v}g{g}"]["median"]
A.img(cv, M, A_GX, AX_Y, A_CW * len(kg), A_CH * len(kv), cmap=cmap, vmin=A_VMIN, vmax=A_VMAX, interp="nearest", z=Z_IMG)
for i in range(len(kv) + 1):
    cv.plot([A_GX, A_GX + A_CW * len(kg)], [Yc(AX_Y + i * A_CH)] * 2, color=HAIR, lw=A_GRID_LW, zorder=Z_IMG + 1)
for j in range(len(kg) + 1):
    cv.plot([A_GX + j * A_CW] * 2, [Yc(AX_Y), Yc(AX_Y + len(kv) * A_CH)], color=HAIR, lw=A_GRID_LW, zorder=Z_IMG + 1)
for i, v in enumerate(kv):
    yc = AX_Y + (i + 0.5) * A_CH
    txt(A_GX - A_ROWLAB_DX, yc, str(v), ha="right", va="center_baseline")
    for j, g in enumerate(kg):
        xc = A_GX + (j + 0.5) * A_CW; k = f"v{v}g{g}"
        if k not in cells:
            txt(xc, yc, "–", ha="center", va="center_baseline", c=GREY); continue
        cl = cells[k]; fc = to_hex(cmap((cl["median"] - A_VMIN) / (A_VMAX - A_VMIN))); tc = A.contrast_text(fc)
        txt(xc, yc + A_MED_DY, A.fnum(cl["median"], "{:.2f}"), A.F_AXLAB, ha="center", va="center_baseline", c=tc)
        txt(xc, yc + A_Q_DY, A.fnum(cl["q10"], "{:.2f}") + "–" + A.fnum(cl["q90"], "{:.2f}"), ha="center", va="center_baseline", c=tc)
        if k == a["best"]:
            cv.add_patch(Rectangle((A_GX + j * A_CW, Yc(AX_Y + (i + 1) * A_CH)), A_CW, A_CH, fill=False, ec=INK, lw=A_BEST_LW, zorder=Z_IMG + 2))
for j, g in enumerate(kg):
    txt(A_GX + (j + 0.5) * A_CW, AX_Y + len(kv) * A_CH + A_COLLAB_DY, str(g), ha="center")
txt(A_GX + A_CW * len(kg) / 2, AX_Y + len(kv) * A_CH + A_COLT_DY, "gradient-tile coarsening", A.F_AXLAB, ha="center", c=INK)
txt(A_ROWT_X, AX_Y + len(kv) * A_CH / 2, "value-tile coarsening", A.F_AXLAB, ha="center", va="center", c=INK, rot=ROT90)
cbx = A_GX + A_CW * len(kg) + A_CB_DX; cbh = A_CH * len(kv)
cbv = np.linspace(A_VMAX, A_VMIN, A_CB_N)[:, None]
A.img(cv, cbv, cbx, AX_Y, A_CB_W, cbh, cmap=cmap, vmin=A_VMIN, vmax=A_VMAX, interp="nearest", frame=HAIR, z=Z_IMG)
for t_ in A_CB_TICKS:
    yy = AX_Y + (A_VMAX - t_) / (A_VMAX - A_VMIN) * cbh
    cv.plot([cbx + A_CB_W, cbx + A_CB_W + A_CB_T], [Yc(yy)] * 2, color=INK, lw=A.LW_AXIS, zorder=Z_IMG)
    txt(cbx + A_CB_W + A_CB_T + A_CB_T, yy, A.fnum(t_), ha="left", va="center_baseline")
txt(cbx, AX_Y - A_CB_LAB_DY, "U/V", ha="left", va="bottom")
A.tok(cv, LET_X, TOK1_Y, f"{cells[a['best']]['pairs']} nested pairs | Georgia | median, q10–q90")

# ===================================================================================== panel b ==
b = D["b"]
A.head(cv, B_X0, TOP, "b", "Boundary displacement", dx=TITLE_DX)
axb = A.pax(fig, cv, BX, AX_Y, BW, AX_H, name="b_disp")
SER = [("georgia|remove", "georgia", "remove"), ("gm_q4|remove", "gm_q4", "remove"), ("georgia|swap", "georgia", "swap"), ("gm_q4|swap", "gm_q4", "swap")]
deltas = None
for i, (key, s, mode) in enumerate(SER):
    st = Z.SETTING[s]; ser = b["series"][key]; dl = np.array([ser[k]["delta"] for k in ser]); deltas = dl
    mu = np.array([ser[k]["mean"] for k in ser]); sd = np.array([ser[k]["sd"] for k in ser])
    ls_, mfc = (Z.LS_SOLID, st["c"]) if mode == "remove" else (Z.LS_DOT, WHITE)
    axb.errorbar(np.log2(dl) + B_OFF[i], mu, yerr=sd, color=st["c"], ls=ls_, lw=A.LW_DATA, marker=st["mk"], ms=B_MS, mfc=mfc, mec=st["c"], mew=A.LW_THIN, zorder=Z_IMG + i, **Z.ERRORBAR)
axb.set_yscale("log")
A.refline(axb, 1.0)
A.clean(axb, xl=(-B_XPAD, np.log2(deltas.max()) + B_XPAD), yl=B_YL, xlab="displacement δ (pixels)", ylab="lifted width / tube bound")
log_ticks(axb.yaxis, B_YT)
axb.set_xticks(np.log2(deltas)); axb.set_xticklabels([str(int(v)) for v in deltas])
A.T(axb, -B_XPAD, 1.0, " tube bound", c=GRAYREF, ha="left", va="bottom")
x = B_KEY_X
for s in ("georgia", "gm_q4"):
    st = Z.SETTING[s]; x = key_item(x, B_KEY_Y[0], "line", Z.SETTING_SHORT[s], st["c"], lw=0, marker=st["mk"], mfc=st["c"]) + KEY_GAP
x = B_KEY_X
x = key_item(x, B_KEY_Y[1], "line", "strip removed", INK, ls=Z.LS_SOLID, lw=A.LW_DATA) + KEY_GAP
x = key_item(x, B_KEY_Y[1], "line", "swapped", INK, ls=Z.LS_DOT, lw=A.LW_DATA, marker="o", mfc=WHITE)
nmin = min(ser_["n"] for ser in b["series"].values() for ser_ in ser.values()); nmax = max(ser_["n"] for ser in b["series"].values() for ser_ in ser.values())
A.tok(cv, B_X0, TOK1_Y, f"mean ± s.d. | {nmin}–{nmax} displaced cells per δ")

# ===================================================================================== panel c ==
c = D["c"]; rows = c["rows"]
A.head(cv, C_X0, TOP, "c", "Conditioned classes: validity", dx=TITLE_DX)
tr = np.array([r["truth"] for r in rows])
lows = [r[m]["L_cond_outer"] for r in rows for m in ("m1", "m2")]; highs = [r[m]["U_cond_outer"] for r in rows for m in ("m1", "m2")]
ylim_c = (min(lows) - C_PAD, max(highs) + C_PAD); xlim_c = (tr.min() - C_PAD, tr.max() + C_PAD)
st_o, st_i = Z.BOUND["outer"], Z.BOUND["inner"]
for j, (m, xx0) in enumerate((("m1", C1X), ("m2", C2X))):
    ax = A.pax(fig, cv, xx0, AX_Y, CW, AX_H, name=f"c_{m}")
    lim = (min(xlim_c[0], ylim_c[0]), max(xlim_c[1], ylim_c[1]))
    ax.plot(lim, lim, color=GRAYREF, lw=A.LW_REF, ls=Z.LS_REF, zorder=1)
    for r in rows:
        q = r[m]
        ax.plot([r["truth"]] * 2, [q["L_cond_outer"], q["U_cond_outer"]], color=TEALL, lw=C_LW_OUT, solid_capstyle="butt", zorder=2)
        ax.plot([r["truth"]] * 2, [q["L_cond_inner"], q["U_cond_inner"]], color=BRICK, lw=C_LW_IN, solid_capstyle="butt", zorder=3)
    ax.plot(tr, [r[m]["U_cond_outer"] for r in rows], ls="none", marker=st_o["mk"], ms=C_MS, mfc=st_o["c"], mec=st_o["c"], mew=0, zorder=4)
    ax.plot(tr, [r[m]["U_cond_inner"] for r in rows], ls="none", marker=st_i["mk"], ms=C_MS, mfc=WHITE, mec=st_i["c"], mew=A.LW_THIN, zorder=5)
    A.clean(ax, xl=xlim_c, yl=ylim_c, ylab="conditioned bound" if j == 0 else None)
    ax.set_xticks(C_XT); ax.set_yticks(C_YT)
    if j == 1:
        ax.set_yticklabels([])
    inside = sum(bool(r[m]["truth_inside_cond_outer"]) for r in rows)
    txt(xx0 + CW / 2, C_SUB_Y, f"m = {m[1:]} | truth inside {inside}/{len(rows)}", ha="center")
txt((C1X + C2X + CW) / 2, C_XLAB_Y, "truth (block mean)", A.F_AXLAB, ha="center", va="center_baseline", c=INK)
key_item(C_KEY_X, C_KEY_Y[0], "line", "outer range", TEALL, lw=C_LW_OUT, marker=st_o["mk"], mfc=st_o["c"], ms=C_MS)
key_item(C_KEY_X, C_KEY_Y[1], "line", "inner range", BRICK, lw=C_LW_IN, marker=st_i["mk"], mfc=WHITE, ms=C_MS)
key_item(C_KEY_X, C_KEY_Y[2], "line", "identity", GRAYREF, ls=Z.LS_REF, lw=A.LW_REF)
# tent inset (upper-left of the m = 2 multiple, an empty region): inner deficit by m as log bars, values printed (units of 1e-4)
axi = A.pax(fig, cv, INS_X, INS_Y, INS_W, INS_H, name="c_tent", inset_of="c_m2")
tm = np.array([r["m"] for r in c["tent"]]); td = np.array([r["inner_deficit"] for r in c["tent"]]); xi = np.arange(len(tm))
axi.bar(xi, td, width=INS_BAR, color=A.BRICKL, edgecolor=BRICK, lw=A.LW_FRAME, log=True, zorder=3)
A.clean(axi, xl=(-INS_BAR, len(tm) - 1 + INS_BAR), yl=INS_YL)
axi.yaxis.set_major_locator(NullLocator()); axi.yaxis.set_minor_locator(NullLocator())
axi.set_xticks(xi); axi.set_xticklabels([str(int(v)) for v in tm])
for xv, yv in zip(xi, td):
    A.T(axi, xv, yv, A.fnum(float(f"{yv / INS_UNIT:.2g}")), size=A.F_MIN, c=BRICK, ha="center", va="bottom")
mant, ex = sci(INS_UNIT)
mlabel(INS_X, INS_Y - INS_LAB_DY, [("tent (×10", "b"), (ex, "sup"), (")", "b")], ha="left")
A.tok(cv, C_X0, TOK1_Y, f"{c['n']} instances | {c['K_min']}–{c['K_max']} units | tent case inset")

# ===================================================================================== panel d ==
d = D["d"]; ms = np.array(d["ms"]); cms = np.array(d["cert_ms"])
A.head(cv, LET_X, BAND2, "d", "Interpolation-completed bracket versus m", dx=TITLE_DX)
x = DX
for s in Z.SETTING_ORDER:
    st = Z.SETTING[s]; x = key_item(x, D_KEY_Y, "line", Z.SETTING_SHORT[s], st["c"], lw=0, marker=st["mk"], mfc=st["mfc"]) + KEY_GAP
axd = A.pax(fig, cv, DX, D_AX_Y, DW, D_AX_H, name="d_bracket")
for i, s in enumerate(Z.SETTING_ORDER):
    st = Z.SETTING[s]; e_ = d["settings"][s]
    for kind, mm, ls_, mfc in (("pixel", ms, Z.LS_DASH, WHITE), ("cert", cms, Z.LS_SOLID, st["mfc"])):
        med = np.array([e_[kind][str(m)]["median"] for m in mm]); lo = np.array([e_[kind][str(m)]["min"] for m in mm]); hi = np.array([e_[kind][str(m)]["max"] for m in mm])
        axd.errorbar(np.log2(mm) + D_OFF[i], med, yerr=[med - lo, hi - med], color=st["c"], ls=ls_, lw=A.LW_THIN, marker=st["mk"], ms=D_MS, mfc=mfc, mec=st["mec"],
                     mew=A.LW_THIN, zorder=Z_IMG + i, **Z.ERRORBAR)
axd.set_yscale("log")
A.clean(axd, xl=(-D_XPAD, np.log2(ms.max()) + D_XPAD), yl=D_YL, xlab="nodal resolution m", ylab="excess over witness")
log_ticks(axd.yaxis, D_YT)
axd.set_xticks(np.log2(ms)); axd.set_xticklabels([str(int(v)) for v in ms])
key_item(D_STYLE_X, D_STYLE_Y[0], "line", "pixel outer Φ′", INK, ls=Z.LS_DASH, lw=A.LW_THIN, marker="o", mfc=WHITE)
key_item(D_STYLE_X, D_STYLE_Y[1], "line", "certified bracket", INK, ls=Z.LS_SOLID, lw=A.LW_THIN, marker="o", mfc=INK)
mant, ex = sci(d["exact1d"]["max_err_UPhi"])
txt(D_PRINT[0], D_PRINT[1], f"exact 1-D theorem, {d['exact1d']['instances']} instances", va="baseline")
mlabel(D_PRINT[0], D_PRINT[1] + D_PRINT_LH, [("max |LP − closed form| = " + mant + " × 10", "b"), (ex, "sup")])
nu = sorted({d["settings"][s]["unique"] for s in Z.SETTING_ORDER})
A.tok(cv, LET_X, TOK2_Y, f"{nu[0]}–{nu[-1]} unique real patches per setting | verified duals | median, min–max")

# ===================================================================================== panel e ==
e = D["e"]; foc = e["focus"]
A.head(cv, E_X0, BAND2, "e", "Saving identity audit", dx=TITLE_DX)
axe = A.pax(fig, cv, EX, D_AX_Y, EW, D_AX_H, name="e_identity")
lim = (min(E_XL[0], E_YL[0]), max(E_XL[1], E_YL[1]))
axe.plot(lim, lim, color=GRAYREF, lw=A.LW_REF, ls=Z.LS_REF, zorder=1)
for bud, blk in e["budgets"].items():
    if bud == foc:
        continue
    axe.plot(np.array(blk["mass"]) * PCT, np.array(blk["G"]) * PCT, ls="none", marker="o", ms=E_MS_O, mfc=WHITE, mec=GRAYL, mew=A.LW_THIN, zorder=2)
fb = e["budgets"][foc]
axe.plot(np.array(fb["mass"]) * PCT, np.array(fb["G"]) * PCT, ls="none", marker="o", ms=E_MS_F, mfc=TEAL, mec=WHITE, mew=A.LW_FRAME, zorder=4)
axe.set_xscale("log"); axe.set_yscale("log")
A.clean(axe, xl=E_XL, yl=E_YL, xlab="benefiting mass share (%)", ylab="normalized saving G (%)")
log_ticks(axe.xaxis, E_XT); log_ticks(axe.yaxis, E_YT)
n_other = sum(blk["n"] for bud, blk in e["budgets"].items() if bud != foc)
key_item(E_KEY_X, E_KEY_Y[0], "line", f"{foc} budget", TEAL, lw=0, marker="o", mfc=TEAL, ms=E_MS_F)
key_item(E_KEY_X, E_KEY_Y[1], "line", "other budgets", GRAYL, lw=0, marker="o", mfc=WHITE, ms=E_MS_O)
key_item(E_KEY_X, E_KEY_Y[2], "line", "G = share", GRAYREF, ls=Z.LS_REF, lw=A.LW_REF)
m1, e1 = sci(e["residual_max_focus"]); m2, e2 = sci(e["residual_max_all"])
txt(E_PRINT[0], E_PRINT[1], "max identity residual", va="baseline")
mlabel(E_PRINT[0], E_PRINT[1] + E_PRINT_LH, [(f"{m1} × 10", "b"), (e1, "sup"), (f"  ({foc})", "b")])
mlabel(E_PRINT[0], E_PRINT[1] + 2 * E_PRINT_LH, [(f"{m2} × 10", "b"), (e2, "sup"), ("  (all budgets)", "b")])
A.tok(cv, E_X0, TOK2_Y, f"Georgia nested pairs | {fb['n']} per budget | {len(e['budgets'])} budgets | exact identity")

A.save(fig, "F3", script=os.path.abspath(__file__))
