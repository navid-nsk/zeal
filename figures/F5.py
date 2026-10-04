"""F5 - Fig. 5: identification and reconstruction, Greater Manchester (plot layer only; data from data/F5.json)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agram as A
import zeal_style as Z
import matplotlib.ticker as mticker

D = json.load(open(os.path.join(HERE, "data", "F5.json"), encoding="utf-8"))
P = D["panels"]; VARS = D["vars"]

# === LAYOUT (every numeric literal of the figure lives here) =========================================================
H = 111.0
X0 = 1.5; XR = 91.0
KEY_SW = 1.8; KEY_GAP = 2.2; KEY_LINE = 4.2; KEY_TDX = 0.9; MS = 3.4; MS_S = 2.8
# band 1
Y1 = 1.5; KEY1 = 6.6; KEY1B = 9.6; SUB1 = 11.0; AX1_Y = 14.0; AX1_H = 32.0; TOK1 = 52.6
A_AX = [(10.5, 35.0), (51.5, 35.0)]; A_XT = [1, 1.5, 2, 4]; A_XL = (0.9, 4.4); A_YL = (-0.17, 1.06); A_YT = [0, 0.25, 0.5, 0.75, 1]
A_EXCL_DX = 1.1; A_EXCL_Y = -0.1
B_AX = (124.0, 52.5); B_XL = (0, 0.9); B_XT = [0, 0.2, 0.4, 0.6, 0.8]; B_BH = 0.36; B_OFF = 0.19; B_YL = (-0.6, 7.6); B_DMS = 3.0
B_KAP_DX = 0.012; B_KEYX = 93.0; C_LAB_DX = 0.04; A_EXCL_DY = 0.1
# band 2
Y2 = 56.6; KEY2 = 61.7; SUB2 = 66.2; AX2_Y = 69.0; AX2_H = 31.0; TOK2 = 107.2
C_AX = [(10.5, 35.0), (51.5, 35.0)]; C_XMAX = 2.6; C_N = 300; C_LAB_DY = 0.12; C_K1_XY = (0.03, 0.97)
C_YL = {"q4": (-3.0, 4.6), "bad": (-0.042, 0.064)}; C_YT = {"q4": [-2, 0, 2, 4], "bad": [-0.04, -0.02, 0, 0.02, 0.04, 0.06]}
D_AX1 = (102.0, 36.0); D_AX2 = (149.0, 29.0); D_XL1 = (0.012, 1.3); D_XT1 = [0.02, 0.05, 0.1, 0.2, 0.5, 1]; D_YL1 = (0, 1.08); D_YT1 = [0, 0.25, 0.5, 0.75, 1]
D_BW = 0.38; D_BOFF = 0.2; D_SLX = 147.6; D_SLGAP = 1.6; D_YT2 = [0, 25, 50, 75, 100]; D_SLOPE_XY = (0.97, 0.16); D_SLOPE_DY = 0.11; D_PCT = 100.0
# === END LAYOUT =======================================================================================================

fig, cv = A.canvas(A.W_CANVAS, H)
VST = {"q4": Z.SETTING["gm_q4"], "bad": Z.SETTING["gm_bad"]}
VNAME = {"q4": "Level-4 qualification share", "bad": "bad-health share"}
VSUB = {"q4": "qualification", "bad": "bad health"}


def key_items(x, y_top, items, size=A.F_KEY):
    y = cv.Y(y_top); xx = x
    for kind, lab, st in items:
        if kind == "lm":
            cv.plot([xx, xx + KEY_LINE], [y, y], color=st["c"], ls=st["ls"], lw=A.LW_DATA, zorder=6, solid_capstyle="butt")
            if st.get("mk"): cv.plot([xx + KEY_LINE / 2], [y], marker=st["mk"], ms=MS, mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN, ls="none", zorder=7)
            w = KEY_LINE
        elif kind == "m":
            cv.plot([xx + KEY_SW / 2], [y], marker=st["mk"], ms=st.get("ms", MS), mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN, ls="none", zorder=7); w = KEY_SW
        elif kind == "ref":
            cv.plot([xx, xx + KEY_LINE], [y, y], color=st.get("c", A.GRAYREF), ls=A.LS_REF, lw=A.LW_REF, zorder=6); w = KEY_LINE
        else:
            cv.add_patch(A.Rectangle((xx, y - KEY_SW / 2), KEY_SW, KEY_SW, fc=st["fc"], ec=st.get("ec", "none"), lw=st.get("lw", 0), zorder=6)); w = KEY_SW
        A.T(cv, xx + w + KEY_TDX, y, lab, size, va="center_baseline"); xx += w + KEY_TDX + A.mmw(lab, size) + KEY_GAP
    return xx


def fixed_ticks(axis, vals, labels=None):
    axis.set_major_locator(mticker.FixedLocator(vals)); axis.set_major_formatter(mticker.FixedFormatter(labels or [A.fnum(t) for t in vals])); axis.set_minor_locator(mticker.NullLocator())


def subtitle(x, y_top, s, c=A.INK):
    return A.T(cv, x, cv.Y(y_top), s, A.F_AXLAB, va="baseline", c=c)


# ===================================================================================================== a ==
A.head(cv, X0, Y1, "a", "Identified-set widths of OA means")
CLS = {"lipschitz_graph": dict(c=A.INK, ls=Z.LS_SOLID, mk="o", mfc=A.INK, mec=A.INK, lab="graph Lipschitz"),
       "graphTV": dict(c=A.INK, ls=Z.LS_DASH, mk="s", mfc=A.INK, mec=A.INK, lab="graph total variation")}
key_items(A_AX[0][0], KEY1, [("lm", CLS[c]["lab"], CLS[c]) for c in CLS] + [("m", "mean", dict(mk="o", mfc=A.WHITE, mec=A.INK, ms=MS_S)), ("ref", "within-LSOA s.d.", {})])
for j, v in enumerate(VARS):
    x, w = A_AX[j]; ax = A.pax(fig, cv, x, AX1_Y, w, AX1_H, name=f"a_{v}"); st = VST[v]; d = P["a"][v]
    subtitle(x, SUB1, VSUB[v], c=st["c"] if st["c"] != A.YELLOW else A.INK)
    A.refline(ax, d["within_sd"])
    for c, cs in CLS.items():
        m = np.array([e["multiple"] for e in d[c]]); med = np.array([e["width_median"] for e in d[c]]); mean = np.array([e["width_mean"] for e in d[c]])
        ok = np.array([e["truth_compatible_share"] for e in d[c]]) > 0
        ax.plot(m, med, color=st["c"], ls=cs["ls"], lw=A.LW_DATA, zorder=3)
        ax.plot(m[ok], med[ok], ls="none", marker=cs["mk"], ms=MS, mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN / 2, zorder=4)
        ax.plot(m[~ok], med[~ok], ls="none", marker=cs["mk"], ms=MS, mfc=A.WHITE, mec=st["c"], mew=A.LW_THIN, zorder=4)
        ax.plot(m, mean, ls="none", marker=cs["mk"], ms=MS_S, mfc=A.WHITE, mec=st["c"], mew=A.LW_THIN / 2, zorder=5)
        for mm_, ok_ in zip(m, ok):
            if not ok_: ax.text(mm_ * A_EXCL_DX, A_EXCL_Y, "truth excluded", fontsize=A.F_VAL, ha="left", va="center", color=A.TXT)
    ax.set_xscale("log")
    A.clean(ax, xl=A_XL, yl=A_YL, xlab="budget (× data-implied minimum)", ylab="identified-set width" if j == 0 else None)
    fixed_ticks(ax.xaxis, A_XT); fixed_ticks(ax.yaxis, A_YT, None if j == 0 else ["" for _ in A_YT])
A.tok(cv, X0, TOK1, f"{A.N('F5.n_oa', D['n_oa'], '{:d}')} OAs | {P['a']['q4']['n_sampled']} sampled | median filled, mean open | LP endpoints")

# ===================================================================================================== b ==
A.head(cv, XR, Y1, "b", "Estimators against OA truth")
key_items(B_KEYX, KEY1, [("sw", VNAME[v], dict(fc=VST[v]["c"])) for v in VARS])
key_items(B_KEYX, KEY1B, [("m", "noisy-data mean", dict(mk="D", mfc=A.WHITE, mec=A.INK, ms=B_DMS)), ("ref", "painting", {})])
ax = A.pax(fig, cv, B_AX[0], AX1_Y, B_AX[1], AX1_H, name="b")
order = P["b"]["order"]; ys = np.arange(len(order))[::-1]
for j, v in enumerate(VARS):
    st = VST[v]; E = P["b"]["vars"][v]["est"]; off = B_OFF * (1 - 2 * j)
    ax.axvline(E["painting"]["R2"], color=st["c"], ls=A.LS_REF, lw=A.LW_REF, zorder=1)
    for k, yv in zip(order, ys):
        e = E[k]; ax.barh(yv + off, e["R2"], height=B_BH, color=st["c"], edgecolor="none", zorder=2)
        if "R2_noisy" in e:
            ax.errorbar(e["R2_noisy"], yv + off, xerr=2 * e["R2_noisy_se"], fmt="D", ms=B_DMS, mfc=A.WHITE, mec=A.INK, mew=A.LW_THIN / 2, ecolor=A.INK, **Z.ERRORBAR, zorder=4)
    ku = P["b"]["vars"][v]["kappa_up"]; e = E["field_shrunk_kappa_up"]
    ax.text(e["R2"] + B_KAP_DX, ys[-1] + off, A.N(f'F5.kup.{v}', ku, '{:.3f}'), fontsize=A.F_VAL, ha="left", va="center", color=A.TXT, zorder=5)
A.clean(ax, xl=B_XL, yl=B_YL, xlab="R² against OA truth")
fixed_ticks(ax.xaxis, B_XT); ax.set_yticks(ys); ax.set_yticklabels([P["b"]["labels"][k] for k in order]); ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
A.tok(cv, XR, TOK1, "OA truth | ±2 s.e. over 200 noisy replicates | exact LSOA means")

# ===================================================================================================== c ==
A.head(cv, X0, Y2, "c", "Shrinkage risk identity")
key_items(C_AX[0][0], KEY2, [("sw", "improvement region", dict(fc=A.TEALLL)), ("m", "level-up κ", dict(mk="D", mfc=A.INK, mec=A.INK)),
                             ("m", "oracle κ*", dict(mk="o", mfc=A.WHITE, mec=A.INK)), ("ref", "painting", {})])
for j, v in enumerate(VARS):
    x, w = C_AX[j]; ax = A.pax(fig, cv, x, AX2_Y, w, AX2_H, name=f"c_{v}"); st = VST[v]; d = P["c"][v]
    subtitle(x, SUB2, VSUB[v], c=st["c"])
    kmax = C_XMAX * d["kappa_star"]; k = np.linspace(0, kmax, C_N); rel = D_PCT * (-2 * k * d["rho"] * d["m"] + k * k * d["m"] ** 2)
    ax.axvspan(0, d["two_kappa_star"], color=A.TEALLL, lw=0, zorder=0)
    A.refline(ax, 0.0)
    ax.plot(k, rel, color=st["c"], ls=st["ls"], lw=A.LW_DATA, zorder=3)
    ax.plot([d["kappa_star"]], [d["rel_min_pct"]], ls="none", marker="o", ms=MS, mfc=A.WHITE, mec=A.INK, mew=A.LW_THIN, zorder=4)
    ax.plot([d["kappa_up"]], [d["rel_kappa_up_stored_pct"]], ls="none", marker="D", ms=MS, mfc=A.INK, mec=A.INK, mew=A.LW_THIN / 2, zorder=5)
    yl = C_YL[v]; dy = C_LAB_DY * (yl[1] - yl[0])
    ax.text(d["kappa_star"], d["rel_min_pct"] + dy, A.N(f"F5.kstar.{v}", d["kappa_star"], "{:.3f}"), fontsize=A.F_VAL, ha="center", va="bottom", color=A.TXT)
    ax.text(d["kappa_up"] + C_LAB_DX * kmax, d["rel_kappa_up_stored_pct"], A.N(f"F5.kup.{v}", d["kappa_up"], "{:.3f}"), fontsize=A.F_VAL, ha="left", va="center", color=A.TXT)
    A.inlab(ax, C_K1_XY[0], C_K1_XY[1], f"κ = 1: +{A.N(f'F5.k1.{v}', d['rel_kappa1_pct'], '{:.1f}')} %", ha="left")
    A.clean(ax, xl=(0, kmax), yl=yl, xlab="detail shrinkage κ", ylab="risk change vs painting (%)" if j == 0 else None)
    fixed_ticks(ax.yaxis, C_YT[v]); ax.xaxis.set_major_locator(mticker.MaxNLocator(len(A_XT)))
A.tok(cv, X0, TOK2, "quadratic identity, stored ν and ρ | level-up κ: stored result | OA truth")

# ===================================================================================================== d ==
A.head(cv, XR, Y2, "d", "Zoning effect vs scale effect")
key_items(D_AX1[0], KEY2, [("sw", "same-K re-zoning, q05–q95", dict(fc=A.GRAYLL, ec=A.GRAYL, lw=A.LW_FRAME)), ("sw", "moved term", dict(fc=A.GREY)),
                           ("sw", "staying term", dict(fc=A.WHITE, ec=A.GREY, lw=A.LW_THIN))])
ax = A.pax(fig, cv, D_AX1[0], AX2_Y, D_AX1[1], AX2_H, name="d_disp")
A.refline(ax, 1.0)
for v in VARS:
    st = VST[v]; d = P["d"][v]; R = {r["key"]: r for r in d["rows"]}
    sk = R["same_K"]; ax.axhspan(sk["lo"], sk["hi"], color=st["c"], alpha=Z.POOLED["iqr"]["alpha"] / 2, lw=0, zorder=1)
    ax.axhline(sk["value"], color=st["c"], lw=A.LW_THIN, zorder=1.5)
    dp = [r for r in d["rows"] if r["kind"] == "displacement"]
    ax.plot([r["moved_share"] for r in dp], [r["value"] for r in dp], color=st["c"], ls=st["ls"], lw=A.LW_DATA, marker=st["mk"], ms=MS, mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN / 2, zorder=3)
ax.set_xscale("log")
A.clean(ax, xl=D_XL1, yl=D_YL1, xlab="OAs moved (share)", ylab="movement / OA → LSOA scale")
fixed_ticks(ax.xaxis, D_XT1); fixed_ticks(ax.yaxis, D_YT1)
for j, v in enumerate(VARS):
    d = P["d"][v]
    A.inlab(ax, D_SLOPE_XY[0], D_SLOPE_XY[1] - j * D_SLOPE_DY, f"log–log slope {A.N(f'F5.slope_disp.{v}', d['loglog_slope_mov_vs_moved_share'], '{:.2f}')}", c=VST[v]["c"], ha="right")
ax2 = A.pax(fig, cv, D_AX2[0], AX2_Y, D_AX2[1], AX2_H, name="d_decomp")
qs = [r["q"] for r in P["d"]["q4"]["decomp"]]; xq = np.arange(len(qs))
for j, v in enumerate(VARS):
    st = VST[v]; dec = P["d"][v]["decomp"]; off = D_BOFF * (2 * j - 1)
    mv = np.array([r["moved_share"] for r in dec]) * D_PCT; sy = np.array([r["stay_share"] for r in dec]) * D_PCT
    ax2.bar(xq + off, mv, width=D_BW, color=st["c"], edgecolor="none", zorder=2)
    ax2.bar(xq + off, sy, bottom=mv, width=D_BW, color=A.WHITE, edgecolor=st["c"], linewidth=A.LW_THIN, zorder=2)
A.clean(ax2, yl=(0, D_PCT), xlab="move probability q", ylab="share of movement² (%)")
xs_ = D_SLX
A.T(cv, xs_, cv.Y(SUB2), 'slopes', A.F_VAL, va='baseline', c=A.TXT); xs_ += A.mmw('slopes', A.F_VAL) + D_SLGAP
for v in VARS:
    d = P['d'][v]; s_ = f"{A.N(f'F5.slm.{v}', d['slope_moved'], '{:.2f}')} | {A.N(f'F5.sls.{v}', d['slope_stay'], '{:.2f}')}"
    A.T(cv, xs_, cv.Y(SUB2), s_, A.F_VAL, va='baseline', c=VST[v]['c']); xs_ += A.mmw(s_, A.F_VAL) + D_SLGAP
ax2.set_xticks(xq); ax2.set_xticklabels([A.fnum(q) for q in qs]); fixed_ticks(ax2.yaxis, D_YT2)
nz = next(r for r in P['d']['q4']['rows'] if r['key'] == 'same_K')['n']
A.tok(cv, XR, TOK2, f"{A.N('F5.n_oa', D['n_oa'], '{:d}')} OAs | exact arithmetic | {A.N('F5.nrezone', nz, '{:d}')} re-zonings | medians of 10 replicates")

bad = A.save(fig, "F5", script=os.path.abspath(__file__))
