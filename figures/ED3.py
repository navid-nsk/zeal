"""ED3 - Extended Data Fig. 3: temporal pooling - models, verifiers, search cost, exactness checks and the rigorous enclosure.
Plot layer only: every drawn number comes from data/ED3.json (written by ED3_prep.py)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agram as A
import zeal_style as Z
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

D = json.load(open(os.path.join(HERE, "data", "ED3.json"), encoding="utf-8"))

# === LAYOUT (geometry in mm from the top-left corner; style numbers) ===
L = dict(
    H=155.0, ROT=90, XL=1.5, XR=178.5, KEY_GAP=2.2, KEY_SW=1.8, KEY_LINE=1.8, KEY_PAD=0.9, TOK_DY=6.3, TOK_H=2.9, BAND_GAP=2.4, LINE2=2.75,
    MEW=0.4, MS=2.6, LW=0.8, ELW=0.6, CAPS=1.1, GLYPH_MS=2.4, TOPKEY=1.6, KEY_DY=4.4, HEAD_KEY_DY=1.35, Z_TOP=4,
    # band 1: a, b
    B1=4.6,
    A_LAB_X=15.6, A_GLYPH_X=17.6, A_X0=22.6, A_W=23.4, A_TOP_DY=9.0, A_XL=(0.70, 1.0), A_XT=[0.7, 0.8, 0.9, 1.0], A_RATE_X=53.5, A_GROUP_GAP=0.8,
    B_HEAD_X=57.0, B_X0=[71.0, 107.5, 144.0], B_W=34.5, B_SUB_DY=5.4, B_R1_DY=9.4, B_RH=12.6, B_RGAP=7.6,
    B_XL=(0.0085, 0.118), B_XT=[0.01, 0.02, 0.05, 0.1], B_XTL=["0.01", "0.02", "0.05", "0.1"], B_DODGE=[0.955, 1.0, 1.045],
    B_YL=[(0.85, 60.0), (0.85, 160.0)], B_YT=[1, 3, 10, 30, 100], B_YTL=["1", "3", "10", "30", "100"],
    # band 2: c, d
    C_LAB_X=15.2, C_GLYPH_X=17.4, C_X0=[19.6, 58.8], C_CW=4.4, C_TOP_DY=9.0, C_RH=3.0, C_GROUP_GAP=0.6, C_SUB_DY=5.4,
    C_BINS=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.1], C_CB_X=96.4, C_CB_W=1.6, C_CB_T=[0, 1, 2], C_BOX_LW=0.8, CB_TICK_DX=1.45,
    D_HEAD_X=106.0, D_X0=[117.0, 151.5], D_W=27.0, D_YL1=(0.0, 105.0), D_YT1=[0, 50, 100], D_YL2=(0.0, 47.0), D_YT2=[0, 20, 40],
    D_CAP_X=0.0094, D_CAP_DY=0.8,
    # band 3: e, f
    E_LAB_X=15.2, E_GLYPH_X=17.4, E_X0=19.6, E_CW=6.4, E_TOP_DY=9.6, E_CLO_X=48.6, E_CB_X=53.6, E_SCALE=1e13, E_BINS=[5.0, 7.0, 9.0, 11.0, 13.0],
    E_CB_T=[5, 9, 13],
    F_HEAD_X=63.0, F_VERD_X=66.6, F_X0=[73.0, 93.5, 114.0], F_W=18.5, F_SUB_DY=7.1, F_YL=(0.7, 1.015), F_YT=[0.7, 0.8, 0.9, 1.0],
    F_XL=(1e-9, 1.5), F_XT=[1e-8, 1e-5, 1e-2], F_XTL=["1e−8", "1e−5", "0.01"], F_PT=0.9, F_ALPHA=0.3, F_CLIP_MS=2.2,
    F_LOW_X=3.0, F_LOW_DY=0.012, F_VERD_GAP=1.4,
    F_H_X0=143.0, F_H_W=35.5, F_H_XL=(-24.0, -8.4), F_H_XT=[-21, -17, -13, -9], F_H_YL=(0.8, 3.0e5), F_H_YT=[1, 100, 10000],
    F_H_YTL=["1", "100", "10,000"], F_CHECK=(1.2, 1.0), F_CHECK_SHAPE=[(0.0, 0.0), (0.35, -0.45), (1.0, 0.55)], F_CHECK_LW=1.2,
    F_MARGIN_LAB_DX=0.35, F_MAX_LAB=(-23.5, 1.6e5),
)
# === END LAYOUT

fig, cv = A.canvas(A.W_CANVAS, L["H"])
DS, MD = D["datasets"], D["models"]
DSL2 = {"synthetic": ["synthetic", "feeders"], "jul2014": ["electricity,", "jul 2014"], "jan2014": ["electricity,", "jan 2014"]}
MEC = {m: Z.MODEL[m]["mec"] for m in MD}


def key_width(items, size=A.F_KEY):
    w = 0.0
    for kind, lab, st in items:
        w += (L["KEY_SW"] * L["KEY_LINE"] if kind == "line" else L["KEY_SW"]) + L["KEY_PAD"] + A.mmw(lab, size) + L["KEY_GAP"]
    return w - L["KEY_GAP"]


def fixaxis(ax, axis, ticks, labels):
    a_ = ax.xaxis if axis == "x" else ax.yaxis
    a_.set_major_locator(FixedLocator(ticks)); a_.set_major_formatter(FixedFormatter(labels)); a_.set_minor_locator(NullLocator())


def glyph(x, y_mm, m):
    st = Z.MODEL[m]
    cv.plot([x], [cv.Y(y_mm)], marker=st["mk"], ms=L["GLYPH_MS"], mfc=st["mfc"], mec=MEC[m], mew=L["MEW"], ls="none", zorder=6)


def row_layout(rh, gap):
    """y (mm, from the block top) of the 9 dataset x model rows, grouped by dataset"""
    return {(ds, m): di * (len(MD) * rh + gap) + mi * rh for di, ds in enumerate(DS) for mi, m in enumerate(MD)}


def row_labels(top, ry, rh, x_lab, x_glyph):
    for di, ds in enumerate(DS):
        ymid = top + ry[(ds, MD[0])] + len(MD) * rh / 2
        for li, s in enumerate(DSL2[ds]):
            A.T(cv, x_lab, cv.Y(ymid) + (0.5 - li) * L["LINE2"], s, ha="right", va="center_baseline", c=A.INK)
        for m in MD:
            glyph(x_glyph, top + ry[(ds, m)] + rh / 2, m)


def binned(v, bins, cols):
    i = int(np.searchsorted(bins, v, side="right")) - 1
    return cols[min(max(i, 0), len(cols) - 1)]


def stepbar(x, top, h, bins, cols, ticks, labels, title):
    """stepped colour bar (lock colours only) at canvas mm (x, top), bins mapped linearly onto the bar height"""
    lo_, hi_ = bins[0], bins[-1]
    yb = lambda v: top + (hi_ - v) / (hi_ - lo_) * h
    for k, cc in enumerate(cols):
        cv.add_patch(Rectangle((x, cv.Y(yb(bins[k]))), L["C_CB_W"], yb(bins[k]) - yb(bins[k + 1]), fc=cc, ec="none", zorder=5))
    cv.add_patch(Rectangle((x, cv.Y(top + h)), L["C_CB_W"], h, fill=False, ec=A.HAIR, lw=A.LW_FRAME, zorder=6))
    for v, s in zip(ticks, labels):
        cv.plot([x + L["C_CB_W"], x + L["C_CB_W"] + L["KEY_PAD"]], [cv.Y(yb(v))] * 2, color=A.INK, lw=A.LW_AXIS, zorder=6)
        A.T(cv, x + L["C_CB_W"] + L["KEY_PAD"] * L["CB_TICK_DX"], cv.Y(yb(v)), s, va="center_baseline")
    A.T(cv, x + L["C_CB_W"] / 2, cv.Y(top - L["KEY_PAD"]), title, ha="center", va="bottom", c=A.INK)


def checkmark(x, y_mm, c=A.GREEND):
    w, h = L["F_CHECK"]; y = cv.Y(y_mm)
    cv.plot([x + w * px for px, py in L["F_CHECK_SHAPE"]], [y + h * py for px, py in L["F_CHECK_SHAPE"]], color=c,
            lw=A.LW_DATA * L["F_CHECK_LW"], solid_capstyle="round", solid_joinstyle="round", zorder=6)


# ============================================================ figure-wide key: models (glyphs) and datasets (line styles)
top_items = [("marker", Z.MODEL[m]["name"], Z.key_marker(Z.MODEL[m])) for m in MD] + \
            [("line", Z.DATASET[ds]["label"], dict(fc=A.INK, ls=Z.DATASET[ds]["ls"], lw=L["LW"])) for ds in DS]
A.key_row(cv, L["XR"] - key_width(top_items), L["TOPKEY"], top_items, gap=L["KEY_GAP"], sw=L["KEY_SW"])

# ================================================================================ a. model quality
a = D["a"]; y0 = L["B1"]; top = y0 + L["A_TOP_DY"]
r1_top = y0 + L["B_R1_DY"]; bot = r1_top + 2 * L["B_RH"] + L["B_RGAP"]; hA = bot - top
A.head(cv, L["XL"], y0, "a", "Model quality on three datasets")
A.key_row(cv, L["XL"], y0 + L["KEY_DY"], [("marker", "evaluation", dict(marker="o", fc=A.INK, ec=A.INK)),
                                           ("marker", "early stopping", dict(marker="o", fc=A.WHITE, ec=A.INK, lw=L["ELW"]))], gap=L["KEY_GAP"], sw=L["KEY_SW"])
rh = (hA - (len(DS) - 1) * L["A_GROUP_GAP"]) / (len(DS) * len(MD)); ry = row_layout(rh, L["A_GROUP_GAP"])
ax = A.pax(fig, cv, L["A_X0"], top, L["A_W"], hA, name="a_auc")
for ds in DS:
    for m in MD:
        v = a["auc"][ds][m]; st = Z.MODEL[m]; yy = ry[(ds, m)] + rh / 2
        ax.plot([v["early_stop"], v["eval"]], [yy, yy], color=st["c"], lw=L["ELW"], zorder=3)
        ax.plot([v["eval"]], [yy], marker=st["mk"], ms=L["MS"], mfc=st["mfc"], mec=MEC[m], mew=L["MEW"], ls="none", zorder=4)
        ax.plot([v["early_stop"]], [yy], marker=st["mk"], ms=L["MS"], mfc=A.WHITE, mec=MEC[m], mew=L["ELW"], ls="none", zorder=4)
A.clean(ax, xl=L["A_XL"], yl=(hA, 0), xlab="AUC")
ax.set_xticks(L["A_XT"]); ax.set_xticklabels([A.fnum(v) for v in L["A_XT"]]); ax.set_yticks([])
row_labels(top, ry, rh, L["A_LAB_X"], L["A_GLYPH_X"])
A.T(cv, L["A_RATE_X"], cv.Y(top - L["KEY_PAD"]), "label rate", ha="right", va="bottom", c=A.INK)
for ds in DS:
    A.T(cv, L["A_RATE_X"], cv.Y(top + ry[(ds, MD[1])] + rh / 2), A.fnum(a["label_rate"][ds]["eval"], "{:.3f}"), ha="right", va="center_baseline")
A.tok(cv, L["XL"], bot + L["TOK_DY"], f"{a['n_chosen']} of {a['n_eligible']} eligible clients | point estimates")

# ================================================================================ b. verifier tightness vs PGD inner
b = D["b"]; EPS = np.array(b["eps"])
A.head(cv, L["B_HEAD_X"], y0, "b", "Verifier tightness against PGD inner")
VER = ["IBP", "CROWN", "alpha-CROWN"]
vitems = [("marker", Z.VERIFIER[k]["name"], Z.key_marker(Z.VERIFIER[k])) for k in VER]
A.key_row(cv, L["XR"] - key_width(vitems), y0 + L["HEAD_KEY_DY"], vitems, gap=L["KEY_GAP"], sw=L["KEY_SW"])
for mi, m in enumerate(MD):
    x0 = L["B_X0"][mi]
    A.T(cv, x0, cv.Y(y0 + L["B_SUB_DY"]), Z.MODEL[m]["name"], size=A.F_AXLAB, va="top", c=A.INK)
    for ri, (kind, ylab) in enumerate((("value", "value ÷ inner"), ("incr1", "lag-1 ÷ inner"))):
        ax = A.pax(fig, cv, x0, r1_top + ri * (L["B_RH"] + L["B_RGAP"]), L["B_W"], L["B_RH"], name=f"b_{kind}_{m}")
        ax.set_xscale("log"); ax.set_yscale("log"); A.refline(ax, 1)
        for di, ds in enumerate(DS):
            R = b["ratio"][ds][m]
            for meth in VER:
                if meth not in R:
                    continue
                st = Z.VERIFIER[meth]; w = R[meth][kind]; xs = EPS * L["B_DODGE"][di]
                ax.plot(xs, w["median"], color=st["c"], ls=Z.DATASET[ds]["ls"], lw=L["LW"], marker=st["mk"], ms=L["MS"], mfc=st["mfc"], mec=st["mec"], mew=L["MEW"], zorder=4)
                if meth == "CROWN":
                    ax.errorbar(xs, w["median"], yerr=[np.array(w["median"]) - w["q25"], np.array(w["q75"]) - w["median"]], fmt="none", ecolor=st["c"],
                                elinewidth=L["ELW"], capsize=L["CAPS"], capthick=L["ELW"], zorder=4)
        A.clean(ax, xl=L["B_XL"], yl=L["B_YL"][ri], xlab="ε (training sd)" if ri else None, ylab=ylab if mi == 0 else None)
        fixaxis(ax, "x", L["B_XT"], L["B_XTL"] if ri else [""] * len(L["B_XT"]))
        fixaxis(ax, "y", L["B_YT"], L["B_YTL"] if mi == 0 else [""] * len(L["B_YT"]))
A.tok(cv, L["B_X0"][0], bot + L["TOK_DY"], f"{b['n_series']} × {A.fnum(b['n_hours'], '{:d}')} outputs | median, IQR (CROWN) | dashed = PGD inner")

# ================================================================================ c. kappa heat maps
c = D["c"]; LAGS = c["lags"]
y0 = bot + L["TOK_DY"] + L["TOK_H"] + L["BAND_GAP"]; top = y0 + L["C_TOP_DY"]
A.head(cv, L["XL"], y0, "c", "Increment-to-value width ratio r")
hC = len(DS) * len(MD) * L["C_RH"] + (len(DS) - 1) * L["C_GROUP_GAP"]; ry = row_layout(L["C_RH"], L["C_GROUP_GAP"])
CCOL = [A.TEAL, A.TEALM, A.TEALL, A.TEALLL, A.BRICKL, A.BRICK]
for hi_, ep in enumerate(c["eps"]):
    x0 = L["C_X0"][hi_]; wC = L["C_CW"] * len(LAGS)
    A.T(cv, x0 + wC / 2, cv.Y(y0 + L["C_SUB_DY"]), f"ε = {A.fnum(ep)}", ha="center", va="top", c=A.INK, size=A.F_AXLAB)
    ax = A.pax(fig, cv, x0, top, wC, hC, name=f"c_{ep}")
    for ds in DS:
        for m in MD:
            ks = c["kappa"][ds][m][str(ep)]; yy = ry[(ds, m)]; first = next((i for i, v in enumerate(ks) if v >= 1.0), None)
            for i, v in enumerate(ks):
                fc = binned(v, L["C_BINS"], CCOL)
                ax.add_patch(Rectangle((i, yy), 1, L["C_RH"], fc=fc, ec=A.WHITE, lw=A.LW_FRAME, zorder=3))
                A.T(ax, i + 0.5, yy + L["C_RH"] / 2, f"{v:.2f}", ha="center", va="center_baseline", c=A.contrast_text(fc))
            if first is not None:
                ax.add_patch(Rectangle((first, yy), 1, L["C_RH"], fill=False, ec=A.INK, lw=L["C_BOX_LW"], zorder=5))
    A.clean(ax, xl=(0, len(LAGS)), yl=(hC, 0), xlab="lag k (h)")
    ax.set_xticks([i + 0.5 for i in range(len(LAGS))]); ax.set_xticklabels([str(k) for k in LAGS]); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
row_labels(top, ry, L["C_RH"], L["C_LAB_X"], L["C_GLYPH_X"])
stepbar(L["C_CB_X"], top, hC, L["C_BINS"], CCOL, L["C_CB_T"], [str(v) for v in L["C_CB_T"]], "r")
tkc = top + hC + L["TOK_DY"]
A.tok(cv, L["XL"], tkc, "CROWN | median over outputs | box = first r ≥ 1")

# ================================================================================ d. look-back search cost
d = D["d"]
A.head(cv, L["D_HEAD_X"], y0, "d", "Look-back search cost")
for k, (key, ylab, yl, yt) in enumerate((("settled", "settled cases (%)", L["D_YL1"], L["D_YT1"]), ("lp", "LPs per case", L["D_YL2"], L["D_YT2"]))):
    ax = A.pax(fig, cv, L["D_X0"][k], top, L["D_W"], hC, name=f"d_{key}"); ax.set_xscale("log")
    if key == "lp":
        A.refline(ax, d["lazy_cap"])
        A.T(ax, L["D_CAP_X"], d["lazy_cap"] + L["D_CAP_DY"], f"cap {d['lazy_cap']}", c=A.GRAYREF, va="bottom")
    for di, ds in enumerate(DS):
        for m in MD:
            s = d["search"][ds][m]; st = Z.MODEL[m]; xs = EPS * L["B_DODGE"][di]
            if key == "settled":
                ys = 100.0 * np.array(s["settled"]) / np.array(s["n"])
            else:
                ys = np.array(s["lp_median"])
                ax.errorbar(xs, ys, yerr=[ys - s["lp_q10"], np.array(s["lp_q90"]) - ys], fmt="none", ecolor=st["c"], elinewidth=L["ELW"], capsize=0, zorder=3)
            ax.plot(xs, ys, color=st["c"], ls=Z.DATASET[ds]["ls"], lw=L["LW"], marker=st["mk"], ms=L["MS"], mfc=st["mfc"], mec=MEC[m], mew=L["MEW"], zorder=4)
    A.clean(ax, xl=L["B_XL"], yl=yl, xlab="ε (training sd)", ylab=ylab)
    fixaxis(ax, "x", L["B_XT"], L["B_XTL"]); ax.set_yticks(yt)
sec = [a_ / b_ for ds in DS for m in MD for a_, b_ in zip(d["search"][ds][m]["lp_seconds"], d["search"][ds][m]["n_lp"])]
A.tok(cv, L["D_HEAD_X"], tkc, f"{d['search'][DS[0]][MD[0]]['n'][0]} cases per point | median, q10–q90 | {min(sec):.2f}–{max(sec):.2f} s per LP")

# ================================================================================ e. closure gain and exact 1-D check
e_ = D["e"]
y0 = tkc + L["TOK_H"] + L["BAND_GAP"]; top = y0 + L["E_TOP_DY"]
A.head(cv, L["XL"], y0, "e", "Closure gain and exact 1-D check")
ry = row_layout(L["C_RH"], L["C_GROUP_GAP"])
ECOL = [A.TEALLL, A.TEALL, A.TEALM, A.TEAL]
ax = A.pax(fig, cv, L["E_X0"], top, L["E_CW"] * len(e_["eps"]), hC, name="e_exact")
for ds in DS:
    for m in MD:
        yy = ry[(ds, m)]
        for i, v in enumerate(e_["exact1d"][ds][m]):
            vv = v * L["E_SCALE"]; fc = binned(vv, L["E_BINS"], ECOL)
            ax.add_patch(Rectangle((i, yy), 1, L["C_RH"], fc=fc, ec=A.WHITE, lw=A.LW_FRAME, zorder=3))
            A.T(ax, i + 0.5, yy + L["C_RH"] / 2, f"{vv:.1f}", ha="center", va="center_baseline", c=A.contrast_text(fc))
        g = max(e_["closure_gain_median_max"][ds][m], e_["closure_gain_max_all_outputs"][ds][m])
        A.T(cv, L["E_CLO_X"], cv.Y(top + yy + L["C_RH"] / 2), A.fnum(g), ha="center", va="center_baseline")
A.clean(ax, xl=(0, len(e_["eps"])), yl=(hC, 0), xlab="ε (training sd)")
ax.set_xticks([i + 0.5 for i in range(len(e_["eps"]))]); ax.set_xticklabels([A.fnum(v) for v in e_["eps"]]); ax.set_yticks([])
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
row_labels(top, ry, L["C_RH"], L["E_LAB_X"], L["E_GLYPH_X"])
for li, s in enumerate(["closure", "gain"]):
    A.T(cv, L["E_CLO_X"], cv.Y(top - L["KEY_PAD"] - (1 - li) * L["LINE2"]), s, ha="center", va="bottom", c=A.INK)
A.T(cv, L["E_X0"] + L["E_CW"] * len(e_["eps"]) / 2, cv.Y(top - L["KEY_PAD"]), "|LP − closed form|", ha="center", va="bottom", c=A.INK)
stepbar(L["E_CB_X"], top, hC, L["E_BINS"], ECOL, L["E_CB_T"], [str(v) for v in L["E_CB_T"]], "1e−13")
tke = top + hC + L["TOK_DY"]
A.tok(cv, L["XL"], tke, f"closure: {e_['n_configs']} configurations | residual: CROWN, lag-1 LP")

# ================================================================================ f. rigorous enclosure vs the verifier
f_ = D["f"]; T_ = f_["totals"]
A.head(cv, L["F_HEAD_X"], y0, "f", "Rigorous enclosure vs the verifier")
xv = L["F_VERD_X"]; yv = y0 + L["KEY_DY"]
for lab, num in (("rigorous enclosure", f"{A.N('rig_sound', T_['sound'], '{:d}')} / {A.N('rig_n', T_['n_boxes'], '{:d}')} boxes sound"),
                 ("verified checker", f"{A.N('lean_pass', T_['lean_pass'], '{:d}')} / {A.N('lean_n', T_['lean_n'], '{:d}')} PASS")):
    A.T(cv, xv, cv.Y(yv), lab, va="center_baseline", c=A.INK); xv += A.mmw(lab, A.F_VAL) + L["F_VERD_GAP"]
    A.T(cv, xv, cv.Y(yv), num, va="center_baseline", w="bold"); xv += A.mmw(num, A.F_VAL, w="bold") + L["F_VERD_GAP"]
    checkmark(xv, yv - L["F_CHECK"][1] * L["F_CHECK_SHAPE"][1][1] / 2); xv += L["F_CHECK"][0] + L["KEY_GAP"] * 2
ptop = y0 + L["E_TOP_DY"]
sub = f_["sub"]; wa = np.array(sub["verifier_width"]); wr = np.array(sub["rigorous_width"]); dsi = np.array(sub["dataset"])
for di, ds in enumerate(DS):
    x0 = L["F_X0"][di]
    A.T(cv, x0 + L["F_W"] / 2, cv.Y(y0 + L["F_SUB_DY"]), Z.DATASET[ds]["label"], ha="center", va="top", c=A.INK)
    ax = A.pax(fig, cv, x0, ptop, L["F_W"], hC, name=f"f_sc_{ds}"); ax.set_xscale("log")
    sel = dsi == di; r = wr[sel] / wa[sel]; x = wa[sel]; low = r < L["F_YL"][0]
    A.refline(ax, 1)
    ax.scatter(x[~low], r[~low], s=L["F_PT"], c=Z.VERIFIER["rigorous"]["c"], alpha=L["F_ALPHA"], lw=0, zorder=3)
    if low.any():
        ax.plot(x[low], np.full(low.sum(), L["F_YL"][0]), ls="none", marker="v", ms=L["F_CLIP_MS"], mfc=A.WHITE, mec=Z.VERIFIER["rigorous"]["c"],
                mew=L["MEW"], zorder=L["Z_TOP"], clip_on=False)
        A.T(ax, x[low].max() * L["F_LOW_X"], L["F_YL"][0] + L["F_LOW_DY"], f"{int(low.sum())} below {A.fnum(L['F_YL'][0])}", c=Z.VERIFIER["rigorous"]["c"], va="bottom")
    A.clean(ax, xl=L["F_XL"], yl=L["F_YL"], xlab="verifier box width" if di == 1 else None, ylab="rigorous ÷ verifier" if di == 0 else None)
    fixaxis(ax, "x", L["F_XT"], L["F_XTL"]); fixaxis(ax, "y", L["F_YT"], [A.fnum(v) for v in L["F_YT"]] if di == 0 else [""] * len(L["F_YT"]))
hx = A.pax(fig, cv, L["F_H_X0"], ptop, L["F_H_W"], hC, name="f_hist"); hx.set_yscale("log")
edges = np.array(f_["bins_log10"])
for ds in DS:
    hcount = np.array(f_["datasets"][ds]["overshoot_hist"], float)
    hx.stairs(np.where(hcount > 0, hcount, np.nan), edges, color=Z.VERIFIER["rigorous"]["c"], ls=Z.DATASET[ds]["ls"], lw=L["LW"], zorder=3)
mg = np.log10(f_["margin"])
hx.axvline(mg, color=Z.GUARANTEE_LINE["c"], ls=Z.GUARANTEE_LINE["ls"], lw=Z.GUARANTEE_LINE["lw"], zorder=2)
A.T(hx, mg - L["F_MARGIN_LAB_DX"], np.sqrt(L["F_H_YL"][0] * L["F_H_YL"][1]), "margin 1e−9", c=Z.GUARANTEE_LINE["c"], ha="center", va="bottom", rot=L["ROT"])
A.T(hx, L["F_MAX_LAB"][0], L["F_MAX_LAB"][1], f"max {T_['overshoot_max']:.1e}".replace("e-", "e−"), ha="left", va="top", c=A.INK)
A.clean(hx, xl=L["F_H_XL"], yl=L["F_H_YL"], xlab="beyond raw float box", ylab="boxes")
fixaxis(hx, "x", L["F_H_XT"], [f"1e−{-v}" for v in L["F_H_XT"]]); fixaxis(hx, "y", L["F_H_YT"], L["F_H_YTL"])
A.tok(cv, L["F_HEAD_X"], tke, f"IEEE-754 outward rounding | exact sigmoid | {A.fnum(f_['n_sub'], '{:d}')} of {A.fnum(T_['n_boxes'], '{:d}')} boxes")

A.save(fig, "ED3", script=os.path.abspath(__file__))
