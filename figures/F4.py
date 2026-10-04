"""F4 - Fig. 4: zoning statistics - certified slope ranges, hot spots and ranks (plot layer only; data from data/F4.json)."""
import os, json
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import agram as A
import zeal_style as Z

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data", "F4.json"), encoding="utf-8"))

# === LAYOUT (all geometry, sizes and formatting constants; mm unless stated)
H = 190.0
VM, VM_PTS, VM_GAP, VM_DY = 1.5, ((0.0, 0.05), (0.35, -0.38), (1.0, 0.5)), 0.8, 0.25   # drawn verdict tick mark
EXP_DY, EXP_DX = 1.15, 0.15            # raised exponent of a power of ten (mm)
MS_PT, MS_SM, MS_STRIP, MEW = 3.2, 2.6, 1.5, 0.5
ALPHA_PT = 0.8
BARW = 0.92                            # bar fill fraction of its slot
FMT3, FMT4 = "{:.3f}", "{:.4f}"
CAT_PAD = 0.55
# band 1 -- a
A_X, A_Y = 1.0, 3.0
A_KEY_Y, A_VERD_Y, A_COLT_Y = 8.6, 12.8, 17.4
A_AX_Y, A_AX_H = 20.0, 85.0
A_AX1_X, A_AX_W, A_AX_GAP = 18.5, 30.5, 2.0
A_RAT_X, A_RAT_W = 83.0, 11.0
A_CELL_X, A_CELL_W = 97.0, 11.0
A_CKEY_X = 72.0
A_COLT_DY = 0.4
A_XLAB_Y, A_TOK_Y = 110.4, 114.0
A_ROW_P, A_GRP_G = 1.0, 0.55           # row pitch and window-group gap (data units)
A_YPAD = 0.75
A_HV, A_HW = 0.74, 0.40                # bar heights: verified, witnessed (data units)
A_XLIM = (-1.75, 0.62)
A_XT = [-1.5, -1.0, -0.5, 0.0, 0.5]
A_RXL, A_RXT = (0.98, 1.28), [1.0, 1.1, 1.2]
A_CXL, A_CXT = (3.0, 9.0), [4, 6, 8]
A_WLAB_DX, A_KLAB_DX = -0.53, -0.06    # axes-fraction offsets of the window and K label columns
A_VERD_GAP = 5.0
# band 1 -- b
B_X, B_Y = 112.0, 3.0
B_VERD_Y, B_VERD_DX = 8.6, 3.0
B_H1 = (120.0, 14.5, 24.0, 30.0)       # histogram box
B_H2 = (150.0, 14.5, 29.0, 30.0)       # strip box
B_BINS = (1.00, 1.26, 0.02)
B_HYL, B_HYT, B_HXT = (0, 16), [0, 5, 10, 15], [1.0, 1.1, 1.2]
B_MED_TOP = 0.78
B_TXT_X, B_TXT_Y1, B_TXT_Y2 = 0.98, 0.99, 0.88
B_SXL, B_SXT = (-18.5, -8.0), [-18, -15, -12, -9]
B_JIT, B_JMUL, B_LAB_DY = 0.24, 37, 0.12
B_MAX_DX = 0.45                        # log10 offset of the printed maximum
B_TOK_Y = 53.0
# band 1 -- c
C_X, C_Y = 112.0, 59.5
C_BOX = (122.0, 67.5, 34.0, 33.5)
C_LIM, C_T = (-1.4, 0.8), [-1.0, -0.5, 0.0, 0.5]
C_KEY_X, C_KEY_Y, C_KEY_DY = 158.5, 71.0, 4.2
C_VERD_Y = 86.0
C_TOK_Y = 114.0
# band 2
B2_Y = 120.5
B2_KEY_Y = 126.0
B2_TOK_Y = 185.6
D_X, D_AX = 1.0, (9.0, 132.0, 48.0, 44.0)
D_BW, D_FW = 0.22, 0.86                # bar width, feasible-outline width (category units)
D_YL, D_YT = (0, 1.06), [0, 0.25, 0.5, 0.75, 1.0]
D_HOT_DY = 0.03
D_BONF_MS = 1.8
E_X = 61.0
E_AX1 = (70.0, 132.0, 18.5, 44.0)
E_AX2 = (97.0, 132.0, 21.0, 44.0)
E_YL1, E_YT1 = (0, 0.4), [0, 0.1, 0.2, 0.3, 0.4]
E_BW, E_DMS = 0.25, 0.85
F_X = 121.0
F_AX1 = (129.5, 132.0, 49.0, 23.0)
F_AX2 = (150.0, 163.5, 28.5, 12.5)
F_YL, F_YT = (0.90, 1.0), [0.90, 0.95, 1.0]
F_CAP, F_GL, F_BELOW_DY = 0.18, 0.32, 0.004
F_HXL, F_HXT, F_HBH = (0, 0.35), [0, 0.1, 0.2, 0.3], 0.62
F_HYPAD = 0.6
B_SYL = (-0.45, 2.9)
TEN = 10.0
KEY_GAP = 1.6
LSP2, LEG_PAD = 1.12, 0.2
# === END LAYOUT

NC = Z.NOISE; NO = Z.NOISE_ORDER
fig, cv = A.canvas(h_mm=H)
intfmt = mtick.FuncFormatter(lambda v, _: A.fnum(v, "{:g}"))


def vmark(x, y_top, c=A.GREEND):
    """verdict tick mark (drawn: Arial has no check-mark glyph), centred on the text row at y_top"""
    y = cv.Y(y_top)
    cv.plot([x + VM * p[0] for p in VM_PTS], [y + VM * p[1] for p in VM_PTS], color=c, lw=A.LW_DATA, solid_capstyle="round", solid_joinstyle="round", zorder=8)


def verdict(x, y_top, s, mark=True):
    A.T(cv, x, cv.Y(y_top), s, A.F_KEY, va="center_baseline", c=A.TXT)
    w = A.mmw(s, A.F_KEY)
    if mark:
        vmark(x + w + VM_GAP, y_top - VM_DY)
    return x + w + (VM_GAP + VM if mark else 0)


def pow10(ax, x, y, mant, v, size=A.F_VAL, c=A.INK):
    """'mant x 10' with the exponent as a separate raised 6-pt text (mathtext superscripts fall below 6 pt)"""
    e = int(np.floor(np.log10(v))); m = v / TEN ** e
    s = f"{mant}{m:.1f}×10"
    t = ax.text(x, y, s, fontsize=size, color=c, ha="left", va="center", zorder=5)
    fig.canvas.draw(); bb = t.get_window_extent(); inv = ax.transData.inverted()
    (x1, _), = inv.transform([[bb.x1, bb.y0]])
    dx_pt = EXP_DX / A.PT; dy_pt = EXP_DY / A.PT
    ax.annotate(A.fnum(e, "{:d}"), xy=(x1, y), xycoords="data", xytext=(dx_pt, dy_pt), textcoords="offset points", fontsize=size, color=c, ha="left", va="center", zorder=5)
    return e, m


# =========================================================================================== a ==
a = D["a"]; cfg = a["configs"]
A.head(cv, A_X, A_Y, "a", "Certified slope ranges, 30-OA windows")
A.key_row(cv, A_AX1_X, A_KEY_Y, [("sw", "verified range", dict(fc=Z.BRACKET_FILL)), ("sw", "witnessed range", dict(fc=A.TEAL)),
                                 ("line", "zero", dict(fc=A.GRAYREF, ls=A.LS_REF, lw=A.LW_REF))])
A.key_row(cv, A_CKEY_X, A_KEY_Y, [("marker", "β = ∞", dict(marker="o", fc=A.WHITE, ec=A.INK, ms=MS_SM)),
                                  ("marker", "β = ½", dict(marker="o", fc=A.INK, ec=A.INK, ms=MS_SM))], gap=KEY_GAP)
nv, nw, nn = A.N("f4_signs_ver", a["both_signs_verified"], "{:d}"), A.N("f4_signs_wit", a["both_signs_witnessed"], "{:d}"), A.N("f4_ncfg", a["n_config"], "{:d}")
xe_ = verdict(A_AX1_X, A_VERD_Y, f"both signs in verified range  {nv}/{nn}")
verdict(xe_ + A_VERD_GAP, A_VERD_Y, f"both signs witnessed  {nw}/{nn}", mark=False)
wins = sorted({c["window"] for c in cfg}); Ks = a["K_set"]
ypos = {}
for wi, w in enumerate(wins):
    for ki, K in enumerate(Ks):
        ypos[(w, K)] = wi * (len(Ks) * A_ROW_P + A_GRP_G) + ki * A_ROW_P
ymax = max(ypos.values()); YL = (ymax + A_ROW_P * A_YPAD, -A_ROW_P * A_YPAD)
SEPS = [(ypos[(w, Ks[-1])] + ypos[(wins[i + 1], Ks[0])]) / 2 for i, w in enumerate(wins[:-1])]


def window_rules(ax):
    """thin dividers between the window groups (one per group boundary)"""
    for yy in SEPS:
        ax.axhline(yy, color=A.HAIR, lw=A.LW_FRAME, zorder=0.5)
axs = []
for j, bkey in enumerate(["inf", 0.5]):
    x0 = A_AX1_X + j * (A_AX_W + A_AX_GAP)
    ax = A.pax(fig, cv, x0, A_AX_Y, A_AX_W, A_AX_H, name=f"a{j + 1}")
    for c in cfg:
        if c["beta"] != bkey:
            continue
        y = ypos[(c["window"], c["K"])]
        ax.barh(y, c["verified"][1] - c["verified"][0], left=c["verified"][0], height=A_HV, color=Z.BRACKET_FILL, lw=0, zorder=2)
        ax.barh(y, c["witnessed"][1] - c["witnessed"][0], left=c["witnessed"][0], height=A_HW, color=A.TEAL, lw=0, zorder=3)
    A.refline(ax, 0.0, axis="x"); window_rules(ax)
    A.clean(ax, xl=A_XLIM, yl=YL)
    ax.set_xticks(A_XT); ax.xaxis.set_major_formatter(intfmt)
    ax.set_yticks([ypos[k] for k in sorted(ypos)])
    ax.set_yticklabels([str(k[1]) for k in sorted(ypos)] if j == 0 else [])
    if j == 1:
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    A.T(cv, x0 + A_AX_W / 2, cv.Y(A_COLT_Y), "β = ∞" if bkey == "inf" else "β = ½", A.F_KEY, ha="center", va="center_baseline", c=A.INK)
    axs.append(ax)
for w in wins:
    yc = np.mean([ypos[(w, K)] for K in Ks])
    axs[0].text(A_WLAB_DX, yc, f"window {wins.index(w) + 1}", transform=axs[0].get_yaxis_transform(), ha="left", va="center", fontsize=A.F_KEY, color=A.TXT)
axs[0].text(A_KLAB_DX, YL[1], "K", transform=axs[0].get_yaxis_transform(), ha="right", va="bottom", fontsize=A.F_KEY, color=A.INK)
# per-configuration verified / witnessed width and admissible-cell count of the exhaustive enumeration
for nm, xx, ww, key, xl, xt, lab in (("a3", A_RAT_X, A_RAT_W, "ratio", A_RXL, A_RXT, "verified /\nwitnessed"),
                                     ("a4", A_CELL_X, A_CELL_W, "n_cells", A_CXL, A_CXT, "cells,\nlog10")):
    ax = A.pax(fig, cv, xx, A_AX_Y, ww, A_AX_H, name=nm)
    for c in cfg:
        v = c[key] if key == "ratio" else np.log10(c[key])
        ax.plot([v], [ypos[(c["window"], c["K"])]], ls="none", marker="o", ms=MS_SM, mfc=A.WHITE if c["beta"] == "inf" else A.INK, mec=A.INK, mew=MEW, zorder=3)
    window_rules(ax); A.clean(ax, xl=xl, yl=YL); ax.set_xticks(xt); ax.xaxis.set_major_formatter(intfmt)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    if key == "ratio":
        A.refline(ax, 1.0, axis="x")
    A.T(cv, xx + ww / 2, cv.Y(A_AX_Y - A_COLT_DY), lab, A.F_KEY, ha="center", va="bottom", c=A.INK, linespacing=LSP2)
A.T(cv, A_AX1_X + A_AX_W + A_AX_GAP / 2, cv.Y(A_XLAB_Y), "zoning slope (bad-health on Level-4 share)", A.F_AXLAB, ha="center", va="center_baseline", c=A.INK)
A.tok(cv, A_AX1_X, A_TOK_Y, f"{A.N('f4_nwin', a['n_windows'], '{:d}')} windows | {A.N('f4_noa', a['n_oa'], '{:d}')} OAs | K = 3/5/8 | β = ∞/½")

# =========================================================================================== b ==
b = D["b"]
A.head(cv, B_X, B_Y, "b", "Verification layer")
verdict(B_H1[0], B_VERD_Y, f"independent checker {A.N('f4_chk_pass', b['checker_pass'], '{:d}')}/{A.N('f4_chk_n', b['checker_n'], '{:d}')}")
verdict(B_H2[0] + B_VERD_DX, B_VERD_Y, f"kernel-checked {A.N('f4_kern_pass', b['kernel_pass'], '{:d}')}/{A.N('f4_kern_n', b['kernel_n'], '{:d}')}")
ax = A.pax(fig, cv, *B_H1, name="b1")
bins = np.arange(*B_BINS)
ax.hist(b["ratio"], bins=bins, color=A.TEALM, lw=0, rwidth=BARW, zorder=2)
med = b["ratio_median"]
ax.axvline(med, ymax=B_MED_TOP, color=A.INK, lw=A.LW_REF, ls=A.LS_REF, zorder=3)
A.clean(ax, xl=(bins[0], bins[-1]), yl=B_HYL, xlab="verified / witnessed width", ylab="configurations")
ax.set_xticks(B_HXT); ax.set_yticks(B_HYT)
A.inlab(ax, B_TXT_X, B_TXT_Y1, f"median {A.N('f4_ratio_med', med, FMT3)}", ha="right")
A.inlab(ax, B_TXT_X, B_TXT_Y2, f"range {A.N('f4_ratio_min', b['ratio_min'], FMT3)}–{A.N('f4_ratio_max', b['ratio_max'], FMT3)}", ha="right")
ax = A.pax(fig, cv, *B_H2, name="b2")
rows = [("charged ε", b["eps_used"], A.TEAL), ("checker ε", b["eps_checker"], A.TEALM), ("max reduced cost", b["eps_exact"], A.INK)]
n = len(b["eps_exact"]); jit = np.linspace(-B_JIT, B_JIT, n)
for i, (lab, vals, col) in enumerate(rows):
    order = np.argsort(np.argsort(vals))
    ax.plot(np.log10(vals), i + jit[(order * B_JMUL) % n], ls="none", marker="o", ms=MS_STRIP, mfc=col, mec="none", alpha=ALPHA_PT, zorder=3)
    ax.text(B_SXL[0], i + B_JIT + B_LAB_DY, " " + lab, fontsize=A.F_VAL, color=A.TXT, ha="left", va="bottom", zorder=5)
A.refline(ax, np.log10(b["tol_exact"]), axis="x", c=A.BRICK)
A.clean(ax, xl=B_SXL, yl=B_SYL, xlab="ε, log10 (LP units)")
ax.set_xticks(B_SXT); ax.xaxis.set_major_formatter(intfmt); ax.set_yticks([]); ax.spines["left"].set_visible(False)
pow10(ax, np.log10(b["eps_exact_max"]) + B_MAX_DX, len(rows) - 1, "max ", b["eps_exact_max"])
A.N("f4_eps_max", b["eps_exact_max"], "{:.1e}")
A.tok(cv, B_H1[0], B_TOK_Y, f"{A.N('f4_ndir', b['n_directions'], '{:d}')} certificates | exhaustive enumeration | exact-rational charge")

# =========================================================================================== c ==
c = D["c"]
A.head(cv, C_X, C_Y, "c", "Exact small windows")
ax = A.pax(fig, cv, *C_BOX, name="c1")
ax.plot(C_LIM, C_LIM, color=A.GRAYREF, lw=A.LW_REF, ls=A.LS_REF, zorder=1)
BST = {"inf": dict(marker="o", mfc=A.WHITE, mec=A.INK, lab="β = ∞"), "0.5": dict(marker="s", mfc=A.GREY, mec=A.GREY, lab="β = ½"),
       "0.25": dict(marker="D", mfc=A.TEAL, mec=A.TEAL, lab="β = ¼")}
for bk, st in BST.items():
    P = [p for p in c["points"] if p["beta"] == bk]
    ax.plot([p["exact"] for p in P], [p["lp"] for p in P], ls="none", marker=st["marker"], ms=MS_SM, mfc=st["mfc"], mec=st["mec"], mew=MEW, alpha=ALPHA_PT, zorder=3)
A.clean(ax, xl=C_LIM, yl=C_LIM, xlab="exact slope endpoint", ylab="LP certificate endpoint")
ax.set_xticks(C_T); ax.set_yticks(C_T); ax.xaxis.set_major_formatter(intfmt); ax.yaxis.set_major_formatter(intfmt)
for i, (bk, st) in enumerate(BST.items()):
    A.key_row(cv, C_KEY_X, C_KEY_Y + i * C_KEY_DY, [("marker", st["lab"], dict(marker=st["marker"], fc=st["mfc"], ec=st["mec"], ms=MS_PT))])
A.key_row(cv, C_KEY_X, C_KEY_Y + len(BST) * C_KEY_DY, [("line", "identity", dict(fc=A.GRAYREF, ls=A.LS_REF, lw=A.LW_REF))])
for i, (s_, mk_) in enumerate([(f"contained {A.N('f4_c_cont', c['n_contain'], '{:d}')}/{A.N('f4_c_ncls', c['n_classes'], '{:d}')}", True),
                               (f"false signs {A.N('f4_c_fs', c['false_signs'], '{:d}')}", True), (f"endpoints {A.N('f4_c_npts', c['n_endpoints'], '{:d}')}", False)]):
    verdict(C_KEY_X, C_VERD_Y + (i + 1) * C_KEY_DY, s_, mark=mk_)
A.tok(cv, C_BOX[0], C_TOK_Y, f"{A.N('f4_c_nwin', c['n_windows'], '{:d}')} windows | {A.N('f4_c_noa', c['n_oa'], '{:d}')} OAs | K = 2/3/4 | exact enumeration")

# =========================================================================================== d ==
d = D["d"]; R = d["radii_km"]; xr = np.arange(len(R))
A.head(cv, D_X, B2_Y, "d", "Certified not-hot shares, Georgia")
ax = A.pax(fig, cv, *D_AX, name="d1")
ax.bar(xr, d["feasible"], width=D_FW, fc="none", ec=A.INK, lw=A.LW_THIN, zorder=1)
for i, k in enumerate(NO):
    ax.bar(xr + (i - 1) * D_BW, d[f"not_hot_{k}"], width=D_BW * BARW, color=NC[k]["c"], lw=0, zorder=2)
ax.plot(xr + D_BW, d["bonferroni_not_hot"], ls="none", marker="_", ms=MS_PT * D_BONF_MS, mew=A.LW_DATA, color=A.INK, zorder=4)
for i in range(len(R)):
    v = d["hot_family"][i]
    ax.text(xr[i], d["feasible"][i] + D_HOT_DY, f"hot {A.N(f'f4_hot_{i}', v, '{:g}' if v == 0 else FMT4)}", ha="center", va="bottom", fontsize=A.F_VAL, color=A.TXT, zorder=5)
A.clean(ax, xl=(-CAT_PAD, len(R) - 1 + CAT_PAD), yl=D_YL, xlab="window radius (km)", ylab="population share")
ax.set_xticks(xr); ax.set_xticklabels([A.fnum(r, "{:g}") for r in R]); ax.set_yticks(D_YT)
A.key_row(cv, D_AX[0], B2_KEY_Y, [("sw", NC[k]["label"], dict(fc=NC[k]["c"])) for k in NO], gap=KEY_GAP)
hk = [Rectangle((0, 0), 1, 1, fc="none", ec=A.INK, lw=A.LW_THIN), Line2D([], [], ls="none", marker="_", ms=MS_PT * D_BONF_MS, mew=A.LW_DATA, color=A.INK)]
ax.legend(hk, ["feasible population", "Bonferroni, family"], loc="upper left", frameon=False, fontsize=A.F_KEY, borderaxespad=LEG_PAD)
A.tok(cv, D_AX[0], B2_TOK_Y, f"{A.N('f4_tracts', d['n_tracts'], '{:d}')} tracts | Atlas s.e. | mass share p = ½")

# =========================================================================================== e ==
e = D["e"]; Re = e["radii_km"]; xe = np.arange(len(Re))
A.head(cv, E_X, B2_Y, "e", "Pair order and rank sets")
ax1 = A.pax(fig, cv, *E_AX1, name="e1"); ax2 = A.pax(fig, cv, *E_AX2, name="e2")
for i, k in enumerate(NO):
    ax1.bar(xe + (i - 1) * E_BW, e[f"pair_order_{k}"], width=E_BW * BARW, color=NC[k]["c"], lw=0, zorder=2)
    ax2.bar(xe + (i - 1) * E_BW, e[f"not_top20_{k}"], width=E_BW * BARW, color=NC[k]["c"], lw=0, zorder=2)
    ax2.plot(xe + (i - 1) * E_BW, e[f"rank_width_median_{k}"], ls="none", marker="D", ms=MS_PT * E_DMS, mfc=A.WHITE, mec=A.BRICK, mew=A.LW_THIN, zorder=4)
A.clean(ax1, xl=(-CAT_PAD, len(Re) - 1 + CAT_PAD), yl=E_YL1, xlab="radius (km)", ylab="certified share")
A.clean(ax2, xl=(-CAT_PAD, len(Re) - 1 + CAT_PAD), yl=D_YL, xlab="radius (km)")
for ax, yt in ((ax1, E_YT1), (ax2, D_YT)):
    ax.set_xticks(xe); ax.set_xticklabels([A.fnum(r, "{:g}") for r in Re]); ax.set_yticks(yt)
A.inlab(ax1, 0.5, 1.0, "pair order", ha="center", va="bottom", c=A.INK)
A.inlab(ax2, 0.5, 1.0, "not in top 20 %", ha="center", va="bottom", c=A.INK)
A.key_row(cv, E_AX2[0], B2_KEY_Y, [("marker", "median rank-set width", dict(marker="D", fc=A.WHITE, ec=A.BRICK, ms=MS_PT * E_DMS))])
A.tok(cv, E_AX1[0], B2_TOK_Y, f"{A.N('f4_named', e['n_named'][0], '{:d}')} named tracts | witness-anchored universe {A.N('f4_univ75', e['universe_size'][0], '{:d}')}/{A.N('f4_univ15', e['universe_size'][1], '{:d}')}")

# =========================================================================================== f ==
f = D["f"]
A.head(cv, F_X, B2_Y, "f", "Calibration of the envelope")
ax = A.pax(fig, cv, *F_AX1, name="f1")
items = [("joint", "rank", f["joint_rank"], "o"), ("joint", "binomial", f["joint_binomial"], "s"),
         ("cond.", "rank", f["cond_rank"], "o"), ("cond.", "binomial", f["cond_binomial"], "s")]
for i, (grp, lab, it, mk) in enumerate(items):
    if grp == "joint":
        y = it["coverage"]; lo, hi = it["cp"]; g = it["guarantee"]; ls_ = "-"; caps = [lo, hi]
    else:
        y = it["median"]; lo, hi = it["min"], it["median"]; g = it["level"]; ls_ = Z.LS_DOT; caps = [lo]
    ax.plot([i, i], [lo, hi], color=A.TEALM, lw=A.LW_THIN, ls=ls_, zorder=2)
    for yy in caps:
        ax.plot([i - F_CAP / 2, i + F_CAP / 2], [yy, yy], color=A.TEALM, lw=A.LW_THIN, zorder=2)
    ax.plot([i], [y], marker=mk, ms=MS_PT, mfc=A.TEALM, mec=A.TEALM, ls="none", zorder=3)
    ax.plot([i - F_GL, i + F_GL], [g, g], color=A.BRICK, lw=A.LW_REF, ls=A.LS_REF, zorder=1.5)
    if grp == "cond.":
        ax.text(i, it["min"] - F_BELOW_DY, f"{A.fnum(100 * it['share_below'], '{:g}')} % below", ha="center", va="top", fontsize=A.F_VAL, color=A.TXT)
A.clean(ax, xl=(-0.5, len(items) - 0.5), yl=F_YL, ylab="coverage")
ax.set_xticks(range(len(items))); ax.set_xticklabels([f"{g_}\n{l_}" for g_, l_, _, _ in items]); ax.set_yticks(F_YT)
A.key_row(cv, F_AX1[0], B2_KEY_Y, [("line", "guarantee", dict(fc=A.BRICK, ls=A.LS_REF, lw=A.LW_REF)), ("line", "CP 95 %", dict(fc=A.TEALM, lw=A.LW_THIN)),
                                   ("line", "minimum", dict(fc=A.TEALM, lw=A.LW_THIN, ls=Z.LS_DOT))], gap=KEY_GAP)
ax = A.pax(fig, cv, *F_AX2, name="f2")
hw = f["half_width"]; labs = [("tailored rank", hw["tailored"], A.TEALM), ("Bonferroni, Gaussian", hw["bonf_gauss"], A.GRAYL), ("Bonferroni, family", hw["bonf_family"], A.GREY)]
for i, (lab, v, col) in enumerate(labs):
    ax.barh(i, v, height=F_HBH, color=col, lw=0, zorder=2)
    ax.text(v, i, " " + A.N(f"f4_hw_{i}", v, FMT4), va="center", ha="left", fontsize=A.F_VAL, color=A.TXT)
A.clean(ax, xl=F_HXL, yl=(len(labs) - 1 + F_HYPAD, -F_HYPAD), xlab="median half-width (kfr)")
ax.set_yticks(range(len(labs))); ax.set_yticklabels([l_ for l_, _, _ in labs]); ax.set_xticks(F_HXT); ax.xaxis.set_major_formatter(intfmt)
A.tok(cv, F_AX1[0], B2_TOK_Y, f"{A.N('f4_queries', f['queries'], '{:d}')} queries | {A.fnum(f['radius_km'], '{:g}')} km | B = {A.N('f4_B', f['B'], '{:d}')} | T = {f['T']}")

A.save(fig, "F4", script=os.path.abspath(__file__))
