"""F6 - Fig. 6: aggregation-space robustness beyond geography (temporal pooling of a learned hourly risk).
Plot layer only: every drawn number comes from data/F6.json (written by F6_prep.py)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agram as A
import zeal_style as Z
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

D = json.load(open(os.path.join(HERE, "data", "F6.json"), encoding="utf-8"))

# === LAYOUT (geometry in mm from the top-left corner; band positions are stacked from these heights) ===
L = dict(
    H=221.0, MEW=0.4, Z_TOP=4, Z_MID=3, XL=1.5, XR=178.5, KEY_GAP=2.2, KEY_SW=1.8, KEY_LINE=1.8, KEY_PAD=0.9, KEY_DY=1.35, TOK_DY=6.3, TOK_H=2.9, BAND_GAP=2.4,
    LINE2=2.75,
    # a
    A_TOP=1.0, A_X0=27.0, A_W=151.5, A_SER_DY=5.6, A_SER_H=19.5, A_BR_DY=1.4, A_ROW=2.95, A_NROW=6, A_LAB_DX=1.4,
    A_YT=[0, 0.2, 0.4, 0.6, 0.8], A_YL=(0.0, 0.85), A_LW_NOM=0.55, A_LW_EDGE=0.3, A_BR_LW=0.7, A_CAP=0.3, A_BLOCK_H=0.46, A_SUB=0.3,
    A_POD_X=[5, 10, 15], A_POD_W=3.6, A_POD_H=[0.8, 0.6, 0.42], A_LW_THICK=1.6, A_LW_THIN=0.55, A_ARROW_MS=2.4,
    # b
    B_SUB_DY=6.0, B_R1_DY=9.2, B_R1_H=16.0, B_R2_GAP=7.6, B_R2_H=12.5, B_X0=[13.0, 72.5, 132.0], B_W=46.0,
    B_XL=(0.0085, 0.118), B_XT=[0.01, 0.02, 0.05, 0.1], B_XTL=["0.01", "0.02", "0.05", "0.1"],
    B_YL=(2.0e-5, 2.5), B_YT=[1e-4, 1e-2, 1], B_YTL=["0.0001", "0.01", "1"],
    B_KXL=(0.8, 170.0), B_KXT=[1, 4, 16, 64], B_KYL_PAD=1.04, B_KYT=[0, 1, 2], B_DODGE=[0.955, 1.0, 1.045], B_KDODGE=[0.93, 1.0, 1.075],
    B_MS=2.6, B_LW=0.8, B_ELW=0.6, B_CAPS=1.1, B_REFLAB_X=0.95, B_REFLAB_DY=0.07,
    # c / d
    CD_KEY_DY=4.6, CD_SUB_DY=7.3, CD_PLOT_DY=13.0, CD_H=20.0, D_HEAD_X=93.0,
    C_X0=[23.0, 45.5, 68.0], C_W=19.5, C_GROUP=[0.27, 0.0, -0.27], C_BAR_H=0.25, C_XL=(0.0, 1.0), C_XT=[0, 0.5, 1], C_YL=(-0.55, 3.5), C_CAPH=0.35,
    D_X0=[104.0, 142.5], D_W=34.5, D_XL=(-0.62, 1.5), D_YL=(0.8, 75.0), D_YT=[1, 2, 5, 10, 20, 50], D_YTL=["1", "2", "5", "10", "20", "50"],
    D_OFF0=-0.2, D_DOFF=0.05, D_MS=2.6, D_LW=0.75, D_ELW=0.5, D_VAL_X=(-0.27, 1.27), D_LABEL=("jul2014", "feat24", "S5"),
    # e / f
    EF_KEY_DY=4.6, EF_SUB_DY=7.3, EF_PLOT_DY=13.0, EF_H=39.0, F_HEAD_X=93.0,
    E_X0=[31.5, 51.9, 72.3], E_W=17.2, E_LAB_X=19.6, E_EPS_X=20.6, E_GLYPH_X=29.6, E_ROW=1.9, E_EPS_GAP=0.55, E_CLS_GAP=1.5,
    E_FUSED=(0.08, 0.98), E_VONLY=(1.12, 0.6), E_XT=[0, 50, 100], E_GLYPH_MS=2.4, E_KEY_BAR=3.4, E_KEY_THICK=1.15, E_KEY_THIN=0.55,
    F_X0=111.5, F_W=56.5, F_LAB_X=107.4, F_GLYPH_X=109.6, F_CNT_X=[171.4, 176.6], F_CELL=0.84, F_GROUP_GAP=0.55, F_XT=[15, 22, 29, 36, 42],
)
# === END LAYOUT

fig, cv = A.canvas(A.W_CANVAS, L["H"])
DS, MD = D["datasets"], D["models"]
WC = Z.WINDOW_CLASS
DSL2 = {"synthetic": ["synthetic", "feeders"], "jul2014": ["electricity,", "jul 2014"], "jan2014": ["electricity,", "jan 2014"]}
DEC = dict(Z.DECISION, witnessed=dict(c=A.BRICKL, label="witnessed sensitive"))
DEC_ORDER = ["invariant", "sensitive", "witnessed", "open"]
WHISK = {"mlp": A.INK, "feat": A.TEAL, "feat24": A.TEALM}
MEC = {"mlp": Z.MODEL["mlp"]["mec"], "feat": Z.MODEL["feat"]["mec"], "feat24": Z.MODEL["feat24"]["mec"]}


def key_width(items, size=A.F_KEY):
    """width (mm) of an agram.key_row with the same items (mirrors its spacing)"""
    w = 0.0
    for kind, lab, st in items:
        w += (L["KEY_SW"] * L["KEY_LINE"] if kind == "line" else L["KEY_SW"]) + L["KEY_PAD"] + A.mmw(lab, size) + L["KEY_GAP"]
    return w - L["KEY_GAP"]


def key_right(y, items, x_right=None):
    x_right = L["XR"] if x_right is None else x_right
    return A.key_row(cv, x_right - key_width(items), y, items, gap=L["KEY_GAP"], sw=L["KEY_SW"])


def fixaxis(ax, axis, ticks, labels):
    a_ = ax.xaxis if axis == "x" else ax.yaxis
    a_.set_major_locator(FixedLocator(ticks)); a_.set_major_formatter(FixedFormatter(labels)); a_.set_minor_locator(NullLocator())


def ymm(ax, frac):
    """canvas-mm y (from the top) of an axes-fraction height"""
    x0, yt, w, h = ax._ag_box
    return yt + (1 - frac) * h


def sub2(x, y_top, lines, ha="center"):
    for li, s in enumerate(lines):
        A.T(cv, x, cv.Y(y_top + li * L["LINE2"]), s, ha=ha, va="top", c=A.INK)


def glyph(x, y_mm, m):
    st = Z.MODEL[m]
    cv.plot([x], [cv.Y(y_mm)], marker=st["mk"], ms=L["E_GLYPH_MS"], mfc=st["mfc"], mec=MEC[m], mew=L["MEW"], ls="none", zorder=6)


# ================================================================================ a. window classes on one series
a = D["a"]; t = np.array(a["t"]); f = np.array(a["f"]); lo = np.array(a["lo"]); hi = np.array(a["hi"]); X = (a["hour_start"], a["hour_end"])
y0 = L["A_TOP"]
A.head(cv, L["XL"], y0, "a", "Six window classes on one series")
key_right(y0 + L["KEY_DY"], [("line", "nominal risk", dict(fc=A.INK, lw=L["A_LW_NOM"] * 2)),
                             ("sw", "CROWN value box", dict(fc=Z.BAND_OUTER, ec=A.TEALL, lw=L["A_LW_EDGE"]))])
ser_top = y0 + L["A_SER_DY"]
ax = A.pax(fig, cv, L["A_X0"], ser_top, L["A_W"], L["A_SER_H"], name="a_series")
ax.fill_between(t, lo, hi, color=Z.BAND_OUTER, lw=0, zorder=2)
for edge in (lo, hi):
    ax.plot(t, edge, color=A.TEALL, lw=L["A_LW_EDGE"], zorder=2.5)
ax.plot(t, f, color=A.INK, lw=L["A_LW_NOM"], zorder=3)
A.clean(ax, xl=X, yl=L["A_YL"], ylab="risk f(t)")
ax.set_yticks(L["A_YT"]); ax.set_xticks(a["xticks"]); ax.set_xticklabels([])
br_top = ser_top + L["A_SER_H"] + L["A_BR_DY"]; br_h = L["A_ROW"] * L["A_NROW"]
br = A.pax(fig, cv, L["A_X0"], br_top, L["A_W"], br_h, name="a_brackets")
A.clean(br, xl=X, yl=(0, L["A_NROW"]), xlab="date, July 2014 (day)")
br.set_xticks(a["xticks"]); br.set_xticklabels(a["xticklabels"]); br.set_yticks([]); br.spines["left"].set_visible(False)
C = a["classes"]; row_y = {k: L["A_NROW"] - 0.5 - i for i, k in enumerate(Z.WINDOW_ORDER)}


def bracket(x0, x1, y, lw=L["A_BR_LW"], c=A.INK, cap=L["A_CAP"]):
    br.plot([x0, x1], [y, y], color=c, lw=lw, solid_capstyle="butt", zorder=4)
    for xx in (x0, x1):
        br.plot([xx, xx], [y - cap, y + cap], color=c, lw=lw, zorder=4)


def block(x0, x1, y, fc, h=L["A_BLOCK_H"]):
    x0, x1 = max(x0, X[0]), min(x1, X[1])
    if x1 > x0:
        br.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, fc=fc, ec=A.WHITE, lw=A.LW_FRAME, zorder=4))


# global range: any two windows of 24-168 h anywhere in the six weeks (the class runs past both ends of the stretch)
y = row_y["S1a"]
br.annotate("", xy=(X[0], y), xytext=(X[1], y), arrowprops=dict(arrowstyle="<->", lw=L["A_BR_LW"], color=A.INK, shrinkA=0, shrinkB=0), zorder=4)
# look-back range: class extremes and the certified arg-max pair of look-back lengths, all ending at one report time
y = row_y["S1b"]; e = C["S1b"]["e"]
for i, Lb in enumerate(sorted(C["S1b"]["L"])):
    bracket(e - Lb, e, y + (1 - i) * L["A_SUB"])
# before/after sign: admissible boundary zone, the two blocks of one sampled choice, its boundary
y = row_y["S2"]; s2 = C["S2"]
br.add_patch(Rectangle((s2["tau_range"][0], y - 0.5), s2["tau_range"][1] - s2["tau_range"][0], 1.0, fc=A.GRAYLL, ec="none", zorder=3))
block(*s2["blocks"][0], y, A.GRAYL); block(*s2["blocks"][1], y, A.GREY)
br.plot([s2["tau"], s2["tau"]], [y - 0.5, y + 0.5], color=A.INK, lw=L["A_BR_LW"], zorder=5)
# binned trend: equal-count bins of one sampled choice (the 4-week span continues past the stretch)
y = row_y["S3"]
for i, (b0, b1) in enumerate(C["S3"]["bins"]):
    block(b0, b1, y, (A.GRAYL, A.GREY)[i % 2])
br.plot([X[1]], [y], marker=">", ms=L["A_ARROW_MS"], color=A.INK, zorder=5, clip_on=False)
# top-3 ranking: common look-back window (24-168 h) ending at a report time; three ranked clients
y = row_y["S4"]; e = C["S4"]["e"]; Lr = C["S4"]["L_range"]
br.plot([e - Lr[1], e], [y, y], color=A.INK, lw=L["A_LW_THIN"], zorder=4, solid_capstyle="butt")
br.plot([e - Lr[0], e], [y, y], color=A.INK, lw=L["A_LW_THICK"], zorder=4, solid_capstyle="butt")
for k in range(C["S4"]["top"]):                                    # podium glyph: ranks 1-3 at the report time
    hh = L["A_POD_H"][k]
    br.add_patch(Rectangle((e + L["A_POD_X"][k] - L["A_POD_W"] / 2, y - L["A_POD_H"][0] / 2), L["A_POD_W"], hh, fc=A.INK, ec="none", zorder=5))
# intra-day contrasts: the two adjacent intra-day windows of one sampled choice on each day of one week
y = row_y["S5"]
for p0, p1, p2 in C["S5"]["pairs"]:
    block(p0, p1, y, A.GRAYL); block(p1, p2, y, A.GREY)
for k in Z.WINDOW_ORDER:
    A.T(cv, L["A_X0"] - L["A_LAB_DX"], cv.Y(ymm(br, row_y[k] / L["A_NROW"])), WC[k], ha="right", va="center_baseline")
tk = br_top + br_h + L["TOK_DY"]
A.tok(cv, L["A_X0"], tk, f"client {a['client_id']} | 8–21 Jul 2014 | ε = {A.fnum(a['eps'])} sd | CROWN")

# ================================================================================ b. widths vs eps, kappa_k vs lag
b = D["b"]; EPS = np.array(b["eps"]); LAG = np.array(b["lags"])
y0 = tk + L["TOK_H"] + L["BAND_GAP"]
A.head(cv, L["XL"], y0, "b", "Enclosure tightness and coherence length")
VER = [("IBP", "IBP"), ("CROWN", "CROWN"), ("alpha-CROWN", "alpha-CROWN"), ("inner", "inner")]
items = [("marker", Z.VERIFIER[k]["name"], Z.key_marker(Z.VERIFIER[k])) for _, k in VER]
items += [("line", Z.DATASET[d_]["label"], dict(fc=A.INK, ls=Z.DATASET[d_]["ls"], lw=L["B_LW"])) for d_ in DS]
key_right(y0 + L["KEY_DY"], items)
KQ75 = max(max(b["kappa"][ds][m]["q75"]) for ds in DS for m in MD)
r1_top = y0 + L["B_R1_DY"]; r2_top = r1_top + L["B_R1_H"] + L["B_R2_GAP"]
for mi, m in enumerate(MD):
    x0 = L["B_X0"][mi]
    A.T(cv, x0, cv.Y(y0 + L["B_SUB_DY"]), Z.MODEL[m]["name"], size=A.F_AXLAB, va="top", c=A.INK)
    ax = A.pax(fig, cv, x0, r1_top, L["B_W"], L["B_R1_H"], name=f"b_w_{m}")
    ax.set_xscale("log"); ax.set_yscale("log")
    for di, ds in enumerate(DS):
        W = b["widths"][ds][m]
        for key, vk in VER:
            if key not in W:
                continue
            st = Z.VERIFIER[vk]; w = W[key]; xs = EPS * L["B_DODGE"][di]
            ax.plot(xs, w["median"], color=st["c"], ls=Z.DATASET[ds]["ls"], lw=L["B_LW"], marker=st["mk"], ms=L["B_MS"],
                    mfc=st["mfc"], mec=st["mec"], mew=L["MEW"], zorder=L["Z_TOP"] if key == "CROWN" else L["Z_MID"])
            if key == "CROWN":
                ax.errorbar(xs, w["median"], yerr=[np.array(w["median"]) - w["q25"], np.array(w["q75"]) - w["median"]],
                            fmt="none", ecolor=st["c"], elinewidth=L["B_ELW"], capsize=L["B_CAPS"], capthick=L["B_ELW"], zorder=4)
    A.clean(ax, xl=L["B_XL"], yl=L["B_YL"], xlab="ε (training sd)", ylab="value-box width" if mi == 0 else None)
    fixaxis(ax, "x", L["B_XT"], L["B_XTL"]); fixaxis(ax, "y", L["B_YT"], L["B_YTL"] if mi == 0 else [""] * len(L["B_YT"]))
    kx = A.pax(fig, cv, x0, r2_top, L["B_W"], L["B_R2_H"], name=f"b_k_{m}")
    kx.set_xscale("log", base=2)
    A.refline(kx, 1); A.refline(kx, 2)
    if mi == len(MD) - 1:
        for v, lab in ((1, "r = 1"), (2, "implied box")):
            A.T(kx, L["B_REFLAB_X"], v + L["B_REFLAB_DY"], lab, c=A.GRAYREF, va="bottom")
    for di, ds in enumerate(DS):
        k = b["kappa"][ds][m]; dst = Z.DATASET[ds]; cc = Z.VERIFIER["CROWN"]["c"]
        kx.errorbar(LAG * L["B_KDODGE"][di], k["median"], yerr=[np.array(k["median"]) - k["q25"], np.array(k["q75"]) - k["median"]],
                    color=cc, ls=dst["ls"], lw=L["B_LW"], marker=dst["mk"], ms=L["B_MS"], mfc=A.WHITE if di else cc,
                    mec=cc, mew=0.5, elinewidth=L["B_ELW"], capsize=L["B_CAPS"], capthick=L["B_ELW"], zorder=4)
    A.clean(kx, xl=L["B_KXL"], yl=(0, KQ75 * L["B_KYL_PAD"]), xlab="lag k (h)", ylab="r" if mi == 0 else None)
    fixaxis(kx, "x", L["B_KXT"], [str(v) for v in L["B_KXT"]]); fixaxis(kx, "y", L["B_KYT"], [str(v) for v in L["B_KYT"]] if mi == 0 else [""] * len(L["B_KYT"]))
tk = r2_top + L["B_R2_H"] + L["TOK_DY"]
A.tok(cv, L["B_X0"][0], tk, f"{b['n_series']} × {A.fnum(b['n_hours'], '{:d}')} outputs | median, IQR (CROWN) | r at ε = 0.01")

# ================================================================================ c. normalized saving G
c = D["c"]; CLS = c["classes"]
y0 = tk + L["TOK_H"] + L["BAND_GAP"]; p_top = y0 + L["CD_PLOT_DY"]
A.head(cv, L["XL"], y0, "c", "Normalized saving by window class")
A.key_row(cv, L["XL"], y0 + L["CD_KEY_DY"], [("sw", Z.MODEL[m]["name"], dict(fc=Z.MODEL[m]["c"])) for m in MD], gap=L["KEY_GAP"], sw=L["KEY_SW"])
for di, ds in enumerate(DS):
    x0 = L["C_X0"][di]
    sub2(x0 + L["C_W"] / 2, y0 + L["CD_SUB_DY"], DSL2[ds])
    ax = A.pax(fig, cv, x0, p_top, L["C_W"], L["CD_H"], name=f"c_{ds}")
    for ci, cl in enumerate(CLS):
        yc = len(CLS) - 1 - ci
        for mi, m in enumerate(MD):
            g = c["G"][ds][m][cl]; yy = yc + L["C_GROUP"][mi]
            ax.barh(yy, g["median"], height=L["C_BAR_H"], color=Z.MODEL[m]["c"], lw=0, zorder=3)
            ax.plot([g["median"], g["q90"]], [yy, yy], color=WHISK[m], lw=L["B_ELW"], zorder=4, solid_capstyle="butt")
            ax.plot([g["q90"], g["q90"]], [yy - L["C_BAR_H"] * L["C_CAPH"], yy + L["C_BAR_H"] * L["C_CAPH"]], color=WHISK[m], lw=L["B_ELW"], zorder=4)
    A.clean(ax, xl=L["C_XL"], yl=L["C_YL"], xlab="normalized saving G" if di == 1 else None)
    ax.set_xticks(L["C_XT"]); ax.set_xticklabels([A.fnum(v) for v in L["C_XT"]]); ax.set_yticks([])
    if di == 0:
        for ci, cl in enumerate(CLS):
            yc = len(CLS) - 1 - ci
            A.T(cv, x0 - L["A_LAB_DX"], cv.Y(p_top + (L["C_YL"][1] - yc) / (L["C_YL"][1] - L["C_YL"][0]) * L["CD_H"]), WC[cl], ha="right", va="center_baseline")
tk = p_top + L["CD_H"] + L["TOK_DY"]
A.tok(cv, L["XL"], tk, "CROWN, ε = 0.01 | dyadic lags | bar median, whisker q90")

# ================================================================================ d. excess over witness, value-only -> fused
d = D["d"]
A.head(cv, L["D_HEAD_X"], y0, "d", "Excess over witness, value-only to fused")
key_right(y0 + L["CD_KEY_DY"], [("line", Z.DATASET[ds]["label"], dict(fc=A.INK, ls=Z.DATASET[ds]["ls"], lw=L["D_LW"])) for ds in DS])
for k, cl in enumerate(d["classes"]):
    x0 = L["D_X0"][k]
    sub2(x0 + L["D_W"] / 2, y0 + L["CD_SUB_DY"] + L["LINE2"] / 2, [WC[cl]])
    ax = A.pax(fig, cv, x0, p_top, L["D_W"], L["CD_H"], name=f"d_{cl}")
    ax.set_yscale("log"); A.refline(ax, 1)
    o = L["D_OFF0"]
    for di, ds in enumerate(DS):
        for mi, m in enumerate(MD):
            ex = d["excess"][ds][m][cl]; st = Z.MODEL[m]; xs = np.array([0.0, 1.0]) + o
            ys = [ex["V"]["median"], ex["Z"]["median"]]
            ax.plot(xs, ys, color=st["c"], ls=Z.DATASET[ds]["ls"], lw=L["D_LW"], zorder=3)
            for xx, bt in zip(xs, ("V", "Z")):
                ax.plot([xx, xx], [ex[bt]["q10"], ex[bt]["q90"]], color=st["c"], lw=L["D_ELW"], zorder=2.5, solid_capstyle="butt")
            ax.plot(xs, ys, ls="none", marker=st["mk"], ms=L["D_MS"], mfc=st["mfc"], mec=MEC[m], mew=L["MEW"], zorder=4)
            if (ds, m, cl) == tuple(L["D_LABEL"]):
                A.T(ax, L["D_VAL_X"][0], ys[0], A.N("F6_excess_V_jul_dayblock_intraday", ys[0], "{:.1f}"), ha="right", va="center_baseline", c=A.INK)
                A.T(ax, L["D_VAL_X"][1], ys[1], A.N("F6_excess_Z_jul_dayblock_intraday", ys[1], "{:.1f}"), ha="left", va="center_baseline", c=A.INK)
            o += L["D_DOFF"]
    A.clean(ax, xl=L["D_XL"], yl=L["D_YL"], ylab="excess over witness" if k == 0 else None)
    fixaxis(ax, "y", L["D_YT"], L["D_YTL"] if k == 0 else [""] * len(L["D_YT"])); ax.set_xticks([0, 1]); ax.set_xticklabels(["value-only", "fused"])
nn = [d["excess"][ds][m][cl]["n"] for ds in DS for m in MD for cl in d["classes"]]
A.tok(cv, L["D_X0"][0], tk, f"CROWN, ε = 0.01 | median, q10–q90 | {min(nn)}–{max(nn)} choices")

# ================================================================================ e. decision certificates
e_ = D["e"]
y0 = tk + L["TOK_H"] + L["BAND_GAP"]; p_top = y0 + L["EF_PLOT_DY"]
A.head(cv, L["XL"], y0, "e", "Decision certificates, value-only vs fused")
kitems = [("sw", DEC[k]["label"], dict(fc=DEC[k]["c"])) for k in DEC_ORDER]
xk = A.key_row(cv, L["XL"], y0 + L["EF_KEY_DY"], kitems, gap=L["KEY_GAP"], sw=L["KEY_SW"]) + L["KEY_GAP"]
for lab, hh in (("fused", L["E_KEY_THICK"]), ("value-only", L["E_KEY_THIN"])):     # bar-thickness key (thick = fused, thin = value-only)
    cv.add_patch(Rectangle((xk, cv.Y(y0 + L["EF_KEY_DY"]) - hh / 2), L["E_KEY_BAR"], hh, fc=A.GREY, ec="none", zorder=6))
    A.T(cv, xk + L["E_KEY_BAR"] + L["KEY_PAD"], cv.Y(y0 + L["EF_KEY_DY"]), lab, va="center_baseline")
    xk += L["E_KEY_BAR"] + L["KEY_PAD"] + A.mmw(lab, A.F_KEY) + L["KEY_GAP"]
rows = []; yy = 0.0
for ci, cl in enumerate(e_["classes"]):
    for ei, ep in enumerate(e_["eps"]):
        for m in MD:
            rows.append((cl, ep, m, yy)); yy += L["E_ROW"]
        yy += L["E_EPS_GAP"] if ei == 0 else L["E_CLS_GAP"]
sc = L["EF_H"] / (yy - L["E_CLS_GAP"])
for di, ds in enumerate(DS):
    x0 = L["E_X0"][di]
    sub2(x0 + L["E_W"] / 2, y0 + L["EF_SUB_DY"], DSL2[ds])
    ax = A.pax(fig, cv, x0, p_top, L["E_W"], L["EF_H"], name=f"e_{ds}")
    for cl, ep, m, ry in rows:
        cnt = e_["counts"][ds][m][str(ep)][cl]
        for bt, (dy, hh) in (("Z", L["E_FUSED"]), ("V", L["E_VONLY"])):
            r = cnt[bt]; xx = 0.0
            for k in DEC_ORDER:
                wv = 100.0 * r[k] / r["n"]
                if wv > 0:
                    ax.add_patch(Rectangle((xx, (ry + dy) * sc), wv, hh * sc, fc=DEC[k]["c"], ec="none", zorder=3))
                xx += wv
    A.clean(ax, xl=(0, 100), yl=(L["EF_H"], 0), xlab="instances (%)" if di == 1 else None)
    ax.set_xticks(L["E_XT"]); ax.set_yticks([]); ax.spines["left"].set_visible(False)
for ci, cl in enumerate(e_["classes"]):
    rr = [r for r in rows if r[0] == cl]; ymid = (rr[0][3] + rr[-1][3] + L["E_ROW"]) / 2 * sc
    A.T(cv, L["E_LAB_X"], cv.Y(p_top + ymid), WC[cl], ha="right", va="center_baseline")
    for ep in e_["eps"]:
        r2 = [r for r in rr if r[1] == ep]; ym = (r2[0][3] + r2[-1][3] + L["E_ROW"]) / 2 * sc
        A.T(cv, L["E_EPS_X"], cv.Y(p_top + ym), f"ε {A.fnum(ep)}", va="center_baseline", c=A.INK)
for cl, ep, m, ry in rows:
    glyph(L["E_GLYPH_X"], p_top + (ry + L["E_ROW"] / 2) * sc, m)
tk = p_top + L["EF_H"] + L["TOK_DY"]
A.tok(cv, L["XL"], tk, "36 instances per class | 72 intra-day | CROWN, dyadic lags")

# ================================================================================ f. top-3 ranking across report times
fd = D["f"]
A.head(cv, L["F_HEAD_X"], y0, "f", "Top-3 ranking across report times")
fx = A.pax(fig, cv, L["F_X0"], p_top, L["F_W"], L["EF_H"], name="f_matrix")
days = [r["day"] for r in fd["S4"][DS[0]][MD[0]]["rows"]]; nC = len(days)
rowh = (L["EF_H"] - (len(DS) - 1) * L["F_GROUP_GAP"]) / (len(DS) * len(MD))
for di, ds in enumerate(DS):
    for mi, m in enumerate(MD):
        ry = di * (len(MD) * rowh + L["F_GROUP_GAP"]) + mi * rowh
        for ci, r in enumerate(fd["S4"][ds][m]["rows"]):
            fx.add_patch(Rectangle((ci + (1 - L["F_CELL"]) / 2, ry + rowh * (1 - L["F_CELL"]) / 2), L["F_CELL"], rowh * L["F_CELL"],
                                   fc=DEC[r["state"]]["c"], ec="none", zorder=3))
        cnt = fd["S4"][ds][m]
        for xk_, kk in zip(L["F_CNT_X"], ("invariant", "sensitive")):
            A.T(cv, xk_, cv.Y(p_top + ry + rowh / 2), A.fnum(cnt[f"n_{kk}"], "{:d}"), ha="right", va="center_baseline", c=DEC[kk]["c"])
        glyph(L["F_GLYPH_X"], p_top + ry + rowh / 2, m)
    ymid = di * (len(MD) * rowh + L["F_GROUP_GAP"]) + len(MD) * rowh / 2
    for li, s in enumerate(DSL2[ds]):
        A.T(cv, L["F_LAB_X"], cv.Y(p_top + ymid) + (0.5 - li) * L["LINE2"], s, ha="right", va="center_baseline", c=A.INK)
for xk_, kk in zip(L["F_CNT_X"], ("invariant", "sensitive")):
    cv.add_patch(Rectangle((xk_ - L["KEY_SW"], cv.Y(p_top) + L["KEY_PAD"]), L["KEY_SW"], L["KEY_SW"], fc=DEC[kk]["c"], ec="none", zorder=6))
A.clean(fx, xl=(0, nC), yl=(L["EF_H"], 0), xlab="report time (day of window)")
fx.set_xticks([days.index(v) + 0.5 for v in L["F_XT"]]); fx.set_xticklabels([str(v) for v in L["F_XT"]]); fx.set_yticks([]); fx.spines["left"].set_visible(False)
A.tok(cv, L["F_X0"], tk, "28 report times | top-3 of 12 clients | CROWN, ε = 0.01")

A.save(fig, "F6", script=os.path.abspath(__file__))
