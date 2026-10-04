"""ED1 - Extended Data Fig. 1: machine-checked development and checker cost (plot layer only; data from data/ED1.json)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agram as A
import zeal_style as Z
import matplotlib.ticker as mticker

D = json.load(open(os.path.join(HERE, "data", "ED1.json"), encoding="utf-8"))

# === LAYOUT (every numeric literal of the figure lives here) =========================================================
H = 114.6
X0 = 1.5; XR = 91.0
KEY_SW = 1.8; KEY_GAP = 2.0; KEY_TDX = 0.9; MS = 3.2; MS_E = 2.6
# band 1
Y1 = 1.5; HEAD1A = 7.4; HEAD1B = 10.0; TOP1 = 12.0
AX_A = (22.5, TOP1, 46.0, 46.5); A_XL = (0, 2000); A_XT = [0, 500, 1000, 1500, 2000]; A_BH = 0.66; A_COLS = [75.0, 85.0]
B_ROW = 2.3; B_LABX = 125.5; B_COLX = [127.0, 139.5, 152.0]; B_COLW = 12.0; B_GAPW = 0.35; B_NAMEX = 170.0; B_DIV = 0.6; B_TOTDY = 0.5
TOK1 = 68.0
# band 2
Y2 = 71.6; KEY2 = 76.8; AX2_Y = 80.2; AX2_H = 25.0; TOK2 = 111.4
C_AX1 = (11.0, 41.0); C_AX2 = (60.0, 27.5); C_XL1 = (10, 1000); C_XT1 = [10, 30, 100, 300, 1000]; C_YL1 = (0.2, 40); C_YT1 = [0.3, 1, 3, 10, 30]; C_KROW = 1000.0
C_THIN = 3; C_MS = 1.1; C_ALPHA = 0.5; C_XT2 = [1, 3, 10, 30, 100]; C_YT2 = [0, 600, 1200, 1800]
D_AX1 = (101.0, 36.0); D_AX2 = (149.0, 29.0); D_YL1 = (0.3, 25); D_YT1 = [0.5, 1, 2, 5, 10, 20]; D_GAP = 1.5
D_XL2 = (1, 4000); D_XT2 = [1, 10, 100, 1000]; D_YL2 = (1e-15, 1e-7); D_YT2 = [1e-14, 1e-11, 1e-8]; D_YTL2 = ["1e−14", "1e−11", "1e−8"]
TEN = 10.0
# === END LAYOUT =======================================================================================================

fig, cv = A.canvas(A.W_CANVAS, H)
SET = Z.SETTING; ROW_LAB = {"georgia": "Georgia", "gm_q4": "Manchester, qualification", "gm_bad": "Manchester, bad health", "mx_rwi": "Mexico"}


def key_items(x, y_top, items, size=A.F_KEY):
    y = cv.Y(y_top); xx = x
    for kind, lab, st in items:
        if kind == "m":
            cv.plot([xx + KEY_SW / 2], [y], marker=st["mk"], ms=st.get("ms", MS), mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN / 2, ls="none", zorder=7); w = KEY_SW
        else:
            cv.add_patch(A.Rectangle((xx, y - KEY_SW / 2), KEY_SW, KEY_SW, fc=st["fc"], ec=st.get("ec", "none"), lw=st.get("lw", 0), zorder=6)); w = KEY_SW
        A.T(cv, xx + w + KEY_TDX, y, lab, size, va="center_baseline"); xx += w + KEY_TDX + A.mmw(lab, size) + KEY_GAP
    return xx


def fixed_ticks(axis, vals, labels=None):
    axis.set_major_locator(mticker.FixedLocator(vals)); axis.set_major_formatter(mticker.FixedFormatter(labels or [A.fnum(t) for t in vals])); axis.set_minor_locator(mticker.NullLocator())


# ===================================================================================================== a: module map ==
A.head(cv, X0, Y1, "a", "Lean modules and theorems")
a = D["a"]; mods = a["modules"]; ax = A.pax(fig, cv, *AX_A, name="a"); ys = np.arange(len(mods))[::-1]
for m, yv in zip(mods, ys):
    left = 0.0
    for fl in m["files"]:
        ax.barh(yv, fl["lines"], left=left, height=A_BH, color=A.TEAL, edgecolor=A.WHITE, linewidth=A.LW_FRAME, zorder=2); left += fl["lines"]
for x_, head in zip(A_COLS, ("theorems", "exported")):
    A.T(cv, x_, cv.Y(HEAD1B), head, A.F_KEY, ha="center", va="baseline")
for m, yv in zip(mods, ys):
    ycv = cv.Y(AX_A[1] + AX_A[3] * ((len(mods) - 1 - yv + 0.5) / len(mods)))
    for x_, key in zip(A_COLS, ("theorems", "exported")):
        A.T(cv, x_, ycv, A.fnum(m[key], "{:d}"), A.F_VAL, ha="center", va="center_baseline")
A.clean(ax, xl=A_XL, yl=(-0.5, len(mods) - 0.5), xlab="lines of Lean (segments = modules)")
fixed_ticks(ax.xaxis, A_XT); ax.set_yticks(ys); ax.set_yticklabels([m["name"] for m in mods]); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
A.tok(cv, X0, TOK1, f"{A.N('ED1.lines', a['lines_total'], '{:d}')} lines of Lean | {A.N('ED1.warn', a['build']['warnings'], '{:d}')} warnings | standard axioms only")

# ===================================================================================================== b: coverage grid ==
A.head(cv, XR, Y1, "b", "Statement coverage in Lean")
fams = D["b"]["families"]; LV = {2: Z.VERIFY["kernel"]["c"], 1: A.TEALL, 0: A.GRAYL}
for x_, (h1, h2) in zip(B_COLX + [B_NAMEX], (("machine-", "checked"), ("finite", "core"), ("not", "formalized"), ("Lean", "names"))):
    xc = x_ + (B_COLW / 2 if x_ != B_NAMEX else 0)
    A.T(cv, xc, cv.Y(HEAD1A), h1, A.F_KEY, ha="center", va="baseline"); A.T(cv, xc, cv.Y(HEAD1B), h2, A.F_KEY, ha="center", va="baseline")
y = TOP1
for i, fm in enumerate(fams):
    if i > 0 and fm["code"].startswith("V") and not fams[i - 1]["code"].startswith("V"):
        y += B_DIV
    yc = cv.Y(y + B_ROW / 2)
    A.T(cv, B_LABX, yc, fm["name"], A.F_VAL, ha="right", va="center_baseline")
    col = {2: 0, 1: 1, 0: 2}[fm["level"]]
    for j, x_ in enumerate(B_COLX):
        cv.add_patch(A.Rectangle((x_ + B_GAPW / 2, cv.Y(y + B_ROW) + B_GAPW / 2), B_COLW - B_GAPW, B_ROW - B_GAPW, fc=LV[fm["level"]] if j == col else A.GRAYLL, ec="none", zorder=2))
    A.T(cv, B_NAMEX, yc, A.fnum(fm["n_decl"], "{:d}") if fm["n_decl"] else "–", A.F_VAL, ha="center", va="center_baseline")
    y += B_ROW
for j, n in enumerate(D["b"]["n_level"]):
    A.T(cv, B_COLX[j] + B_COLW / 2, cv.Y(y + B_TOTDY), f"n = {A.N(f'ED1.nlev{j}', n, '{:d}')}", A.F_VAL, ha="center", va="top")
A.tok(cv, XR, TOK1, f"{A.N('ED1.nT', D['b']['n_T'], '{:d}')} theorem families + {len(fams) - D['b']['n_T']} checker rules | Lean 4 + Mathlib")

# ===================================================================================================== c: checker cost ==
A.head(cv, X0, Y2, "c", "Verified checker cost per endpoint")
C = D["c"]; key_items(C_AX1[0], KEY2, [("m", ROW_LAB[f], dict(mk=SET[f]["mk"], mfc=SET[f]["mfc"], mec=SET[f]["mec"])) for f in C["order"]])
ax = A.pax(fig, cv, C_AX1[0], AX2_Y, C_AX1[1], AX2_H, name="c_time")
for f in C["order"]:
    s = C["settings"][f]; st = SET[f]; tt = np.asarray(s["check_s"]) + np.asarray(s["export_s"])
    ax.plot((np.asarray(s["rows"]) / C_KROW)[::C_THIN], tt[::C_THIN], ls="none", marker=st["mk"], ms=C_MS, mfc=st["mfc"], mec=st["mec"], mew=0, alpha=C_ALPHA, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
A.clean(ax, xl=C_XL1, yl=C_YL1, xlab="constraint rows (thousands)", ylab="export + check (s)")
fixed_ticks(ax.xaxis, C_XT1); fixed_ticks(ax.yaxis, C_YT1)
ax2 = A.pax(fig, cv, C_AX2[0], AX2_Y, C_AX2[1], AX2_H, name="c_size")
ed = np.asarray(C["MB_log10_edges"]); lo_ = np.power(TEN, ed[:-1]); wd = np.diff(np.power(TEN, ed)); bot = np.zeros(len(lo_))
for f in C["order"]:
    cnt = np.asarray(C["MB_counts"][f], float)
    ax2.bar(lo_, cnt, bottom=bot, width=wd, align="edge", color=SET[f]["c"], edgecolor=A.WHITE, linewidth=A.LW_FRAME / 2, zorder=2); bot += cnt
ax2.set_xscale("log"); A.clean(ax2, xl=(np.power(TEN, ed[0]), np.power(TEN, ed[-1])), xlab="certificate size (MB)", ylab="endpoints")
fixed_ticks(ax2.xaxis, C_XT2); fixed_ticks(ax2.yaxis, C_YT2)
A.tok(cv, X0, TOK2, f"{A.N('ED1.n_end', C['n_total'], '{:d}')} endpoints, {A.N('ED1.n_pass', C['n_pass'], '{:d}')} PASS | timing: every third endpoint")

# ===================================================================================================== d: Poincare ==
A.head(cv, XR, Y2, "d", "Verified spectral Poincaré constants")
Dd = D["d"]; cells = Dd["cells"]
toys = [c_ for c_ in cells if c_["kind"] == "toy"]; cty = [c_ for c_ in cells if c_["kind"] == "county"]
key_items(D_AX1[0], KEY2, [("m", "numerical λ2", dict(mk="o", mfc=A.WHITE, mec=A.INK)), ("m", "verified bound t", dict(mk="^", mfc=A.TEAL, mec=A.TEAL)),
                           ("m", "residual bound e", dict(mk="o", mfc=A.BRICK, mec=A.BRICK)), ("m", "margin ρ", dict(mk="s", mfc=A.WHITE, mec=A.TEAL))])
ax = A.pax(fig, cv, D_AX1[0], AX2_Y, D_AX1[1], AX2_H, name="d_lam")
xt = np.arange(len(toys)); xc = len(toys) + D_GAP + np.arange(len(cty))
for xs, grp in ((xt, toys), (xc, cty)):
    ax.plot(xs, [c_["lam2"] for c_ in grp], ls="none", marker="o", ms=MS, mfc=A.WHITE, mec=A.INK, mew=A.LW_THIN, zorder=3)
    ax.plot(xs, [c_["t"] for c_ in grp], ls="none", marker="^", ms=MS_E, mfc=A.TEAL, mec=A.TEAL, mew=0, zorder=4)
ax.set_yscale("log")
A.clean(ax, xl=(-1, xc[-1] + 1), yl=D_YL1, ylab="λ2 and verified t")
fixed_ticks(ax.yaxis, D_YT1)
ax.set_xticks([float(np.mean(xt)), float(np.mean(xc))]); ax.set_xticklabels([f"{A.N('ED1.ntoy', len(toys), '{:d}')} toy shapes", f"{A.N('ED1.ncounty', len(cty), '{:d}')} Georgia counties"])
ax.tick_params(axis="x", length=0)
ax2 = A.pax(fig, cv, D_AX2[0], AX2_Y, D_AX2[1], AX2_H, name="d_res")
nn = np.array([c_["n"] for c_ in cells])
ax2.plot(nn, [c_["rho"] for c_ in cells], ls="none", marker="s", ms=MS, mfc=A.WHITE, mec=A.TEAL, mew=A.LW_THIN, zorder=3)
ax2.plot(nn, [c_["e"] for c_ in cells], ls="none", marker="o", ms=MS_E, mfc=A.BRICK, mec=A.BRICK, mew=0, zorder=4)
ax2.set_xscale("log"); ax2.set_yscale("log")
A.clean(ax2, xl=D_XL2, yl=D_YL2, xlab="pixels per cell", ylab="certificate scale")
fixed_ticks(ax2.xaxis, D_XT2); fixed_ticks(ax2.yaxis, D_YT2, D_YTL2)
A.tok(cv, XR, TOK2, f"{A.N('ED1.nstrict', Dd['n_strict'], '{:d}')} / {len(cells)} strict, ρ ≥ e | {len(Dd['skipped_not_side_connected'])} counties not side-connected")

bad = A.save(fig, "ED1", script=os.path.abspath(__file__))
