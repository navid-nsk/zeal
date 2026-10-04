"""F1 - Fig. 1 (the certified chain): plot layer only. Every drawn number comes from data/F1.json (written by F1_prep.py);
numeric literals live only in the LAYOUT block. Mathematical sub/superscripts are set as separate 6.0 pt runs (mathtext
scripts would fall below the 6.0 pt floor)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agram as A
import zeal_style as Z
from agram import TEAL, TEALM, TEALL, TEALLL, BRICK, INK, TXT, GREY, GRAYREF, HAIR, GRAYL, GRAYLL, WHITE
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import NullLocator

D = json.load(open(os.path.join(HERE, "data", "F1.json"), encoding="utf-8"))

# === LAYOUT (mm from the top-left corner; sizes in pt; every numeric literal of the figure lives here)
FIG_H = 225.0
LET_X, TITLE_DX = 0.8, 4.4
N1, N2, N3, N4, N5, N6, N7 = 0, 1, 2, 3, 4, 5, 6          # node indices, left -> right
Z_RECT, Z_POLY, Z_ROUTE, Z_BLOCK, Z_KEY = 3, 4, 5, 4, 6
SUP_DY, SUB_DY = 1.05, 0.75                               # manual script offsets (mm)
# ---- panel a: nodes, thumbnails, connectors, lane
A_Y = 1.0
NX0, NW, NG, NY, NH = 1.4, 22.9, 2.45, 9.4, 54.6
HEAD_PAD, HEAD_LH = 1.2, 2.75
ROWS_Y = [46.4, 49.3, 52.2, 55.1, 58.0, 60.9]             # shared text / key rows inside every node
KEY_SW, KEY_TX, KEY_LEN, KEY_SWH = 2.2, 0.7, 3.5, 1.76
TXT_UP, TXT_DN = 0.5, 0.6
HOURS_PER_DAY = 24
NODE_LW, NODE_R = 0.5, 1.2
BLK_Y, BLK_SH, BLK_HH, BLK_HL, BLK_IN = 28.0, 1.5, 3.0, 1.05, 0.22
MAP_W = 20.6; MAP_Y = 16.6; CB_DY, CB_H, CB_TICK_DY = 0.8, 0.9, 0.4; STRIP_LW = 1.0
STRIP_BOX = (1.0, 17.0, 20.9, 20.6); STRIP_PAD = 0.06; STRIP_KEY_Y = [40.2, 43.0]
MESH_W = 20.0; MESH_Y = 16.8; MESH_PIX = 3; MESH_LW_PIX, MESH_LW_SUB, NODE_MS = 0.6, 0.35, 1.5; MESH_LAB_Y = 39.4
BR_BOX = (1.2, 23.5, 20.5, 9.0); BR_XL = (0.93, 1.56); BR_LW = 4.2; BR_MS = 4.2; BR_LAB_DY = 2.6; BR_KEY_F = 0.8; BR_REF_H = 0.45
FIB_BOX = (1.2, 17.0, 20.5, 23.0); FIB_BAR_W = 0.86; FIB_YL = (0.0, 1.05)
LAW_BOX = (1.2, 17.0, 20.5, 24.0); LAW_LS = ["-", (0, (3.0, 1.2)), (0, (1.0, 1.0)), (0, (3.0, 1.0, 1.0, 1.0))]; LAW_LW, ENV_LW = 0.55, 1.1; LAW_PAD = 0.06
OA_CELL = 4.0; OA_Y = 17.2; OA_LW_ZONE = 0.8; ZONE_FILLS = [TEALLL, TEALL, TEALM]
SL_BOX = (1.2, 33.8, 20.5, 5.0); SL_XL = (-1.02, 0.18); SL_LW_LP, SL_LW_EX = 5.0, 2.4; SL_LAB_DY = 2.0
CERT_LW, CHK_LW, COND_LW = 0.6, 0.7, 0.6
LS_CHK, LS_COND = (0, (1.0, 1.2)), (0, (3.0, 1.6))
HEAD = "-|>,head_length=0.45,head_width=0.25"; HEAD_SCALE = 6.0
COND_Y = 6.6
LEG_Y, LEG_X1, LEG_GAP = 3.6, 178.6, 3.0
BUS_Y = 67.6; CERT_ROUTE_Y = 70.8; LANE_Y, LANE_H = 73.8, 11.6
OPT_X, OPT_W = 21.0, 22.0
DOC_CX, DOC_W, DOC_H, DOC_FOLD = 61.0, 8.4, 10.6, 2.2
CHK_X, CHK_W = 79.0, 26.0
KER_X, KER_W = 140.0, 26.0
LANE_TXT_Y = [87.9, 90.6]
LANE_LAB = (1.4, 77.8)
POLY_R, POLY_N, POLY_DX, OPT_MS_F = 2.6, 5, 0.6, 1.6
A_TOK_Y = 95.6
# ---- band 2: panels b, c
B_Y = 103.0
BX, BW = 9.6, 77.5
B_KEY1_Y, B_AX1_Y, B_AX1_H = 108.4, 110.4, 21.0
B_KEY2_Y, B_AX2_Y, B_AX2_H = 134.4, 136.4, 14.6
B_TOK_Y = 158.6
B_YL1 = (0.0, 1.0); B_YT1 = [0.0, 0.5, 1.0]; B_DAY_STEP = 2; B_MARK_EVERY = 24; B_MARK_OFF = 12; B_MS = 2.6
B_YL2 = (0.006, 0.3); B_YT2 = [0.01, 0.1]
C_X0 = 96.0
CX, CW = 106.4, 71.4
C_AX_Y, C_AX_H = 110.4, 38.6
C_YL = (-0.04, 1.06); C_YT = [0.0, 0.25, 0.5, 0.75, 1.0]; C_XPAD = 0.55; C_OFF1616 = 0.42; C_MS = 3.4
C_JIT, C_DOT_MS, C_DOT_A = 0.32, 1.6, 0.9
C_LAB_G = (-4.4, 0.115); C_LAB_UV = (-4.4, 0.905); C_LAB_1616 = (0.42, 0.47)
C_TOK_Y = 158.6
# ---- band 3: panels d, e
D_Y = 165.8
DX, DW = 9.6, 77.5
D_KEY_Y, D_AX_Y, D_AX_H = 171.4, 173.8, 34.6
D_TICK_DY, D_XLAB_DY = 3.4, 7.2
D_TOK_Y = 221.4
D_OFF = [-0.21, -0.07, 0.07, 0.21]; D_PAD = 0.02; D_MS = 3.0; D_REF_LAB = (-0.42, 1.004)
E_X0 = 96.0
E_RULE_Y, E_HEAD_DY, E_ROW0_Y, E_ROW_H = 180.2, 0.7, 181.0, 4.75
E_X1 = 178.8; E_SQ = 2.5; E_CNT_DX = 0.9; E_RULE_LW = 0.5
E_TOK_Y = 221.4
# === END LAYOUT

fig, cv = A.canvas(h_mm=FIG_H)
Yc = cv.Y
nan = float("nan")


def vec(v):
    return np.array([nan if x is None else x for x in v], float)


def sq(v):
    a = vec(v); n = int(round(np.sqrt(a.size))); return a.reshape(n, n)


def txt(x, y, s, size=A.F_VAL, ha="left", va="top", c=TXT, **kw):
    return A.T(cv, x, Yc(y), s, size, ha=ha, va=va, c=c, **kw)


def mlabel(x, y, segs, size=A.F_AXLAB, ha="center", c=TXT):
    """label with manual scripts at baseline y_top: segs = [(text, 'b' | 'sub' | 'sup')]; scripts at the 6.0 pt floor."""
    sz = [size if k == "b" else A.F_MIN for _, k in segs]; ws = [A.mmw(t, s_) for (t, _), s_ in zip(segs, sz)]
    x0 = x - sum(ws) / 2 if ha == "center" else x
    for (t, k), s_, w in zip(segs, sz, ws):
        dy = 0 if k == "b" else (-SUP_DY if k == "sup" else SUB_DY)
        A.T(cv, x0, Yc(y + dy), t, s_, ha="left", va="baseline", c=c); x0 += w


def rect(x, y, w, h, fc, ec="none", lw=0, z=Z_RECT):
    cv.add_patch(Rectangle((x, Yc(y + h)), w, h, fc=fc, ec=ec, lw=lw, zorder=z))


def rbox(x, y, w, h, fc=WHITE, ec=HAIR, lw=NODE_LW, r=NODE_R, z=1):
    cv.add_patch(FancyBboxPatch((x, Yc(y + h)), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def poly(pts, fc, ec="none", lw=0, z=Z_POLY, closed=True):
    cv.add_patch(Polygon([(px, Yc(py)) for px, py in pts], closed=closed, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="miter"))


CONN = {"cert": dict(c=INK, lw=CERT_LW, ls="-"), "check": dict(c=INK, lw=CHK_LW, ls=LS_CHK), "cond": dict(c=INK, lw=COND_LW, ls=LS_COND)}


def route(pts, kind, z=Z_ROUTE):
    """Manhattan polyline [(x, y_top), ...] with an arrowhead on the last segment."""
    st = CONN[kind]; X = [p[0] for p in pts]; Yb = [Yc(p[1]) for p in pts]
    if len(pts) > 2:
        cv.plot(X[:-1], Yb[:-1], color=st["c"], lw=st["lw"], ls=st["ls"], solid_capstyle="butt", zorder=z)
    cv.add_patch(FancyArrowPatch((X[-2], Yb[-2]), (X[-1], Yb[-1]), arrowstyle=HEAD, mutation_scale=HEAD_SCALE, color=st["c"], lw=st["lw"],
                                 ls=st["ls"], shrinkA=0, shrinkB=0, zorder=z))


def block_arrow(x0, x1, yc, z=Z_BLOCK):
    hx = x1 - BLK_HL
    poly([(x0, yc - BLK_SH / 2), (hx, yc - BLK_SH / 2), (hx, yc - BLK_HH / 2), (x1, yc), (hx, yc + BLK_HH / 2), (hx, yc + BLK_SH / 2), (x0, yc + BLK_SH / 2)], HAIR, z=z)


def node_x(k):
    return NX0 + k * (NW + NG)


def node_title(k, s):
    x0 = node_x(k) + HEAD_PAD; avail = NW - 2 * HEAD_PAD
    if A.mmw(s, A.F_AXLAB) <= avail:
        txt(x0, NY + HEAD_PAD, s, A.F_AXLAB, c=INK); return
    w = s.split(" "); best = None
    for i in range(1, len(w)):
        l1, l2 = " ".join(w[:i]), " ".join(w[i:]); m = max(A.mmw(l1, A.F_AXLAB), A.mmw(l2, A.F_AXLAB))
        if best is None or m < best[0]:
            best = (m, l1, l2)
    txt(x0, NY + HEAD_PAD, best[1], A.F_AXLAB, c=INK); txt(x0, NY + HEAD_PAD + HEAD_LH, best[2], A.F_AXLAB, c=INK)


def node_text(k, y, s, c=TXT):
    txt(node_x(k) + NW / 2, y, s, A.F_VAL, ha="center", va="center_baseline", c=c)


def key_item(x, y, kind, label, c, ls="-", lw=A.LW_DATA, marker=None, mfc=None, ms=B_MS, fc=None):
    """one key entry centred at (x, y_top); returns the end x."""
    yy = Yc(y)
    if kind == "sw":
        cv.add_patch(Rectangle((x, yy - KEY_SWH / 2), KEY_SW, KEY_SWH, fc=fc or c, ec="none", zorder=Z_KEY)); w = KEY_SW
    elif kind == "block":
        w = KEY_LEN; block_arrow(x, x + w, y, z=Z_KEY)
    elif kind == "arrow":
        w = KEY_LEN; st = CONN[c]
        cv.add_patch(FancyArrowPatch((x, yy), (x + w, yy), arrowstyle=HEAD, mutation_scale=HEAD_SCALE, color=st["c"], lw=st["lw"], ls=st["ls"], shrinkA=0, shrinkB=0, zorder=Z_KEY))
    else:
        w = KEY_LEN
        if lw > 0:
            cv.plot([x, x + w], [yy, yy], color=c, ls=ls, lw=lw, zorder=Z_KEY, solid_capstyle="butt")
        if marker:
            cv.plot([x + w / 2], [yy], marker=marker, ms=ms, mfc=mfc or c, mec=c, mew=A.LW_THIN / 2, ls="none", zorder=Z_KEY)
    A.T(cv, x + w + KEY_TX, yy, label, A.F_KEY, va="center_baseline", c=TXT)
    return x + w + KEY_TX + A.mmw(label, A.F_KEY)


def thumb(name, k, box):
    bx, by, bw, bh = box
    ax = A.pax(fig, cv, node_x(k) + bx, by, bw, bh, name=name); ax.set_axis_off()
    ax.xaxis.set_major_locator(NullLocator()); ax.yaxis.set_major_locator(NullLocator()); return ax


# ===================================================================================== panel a ==
a = D["a"]; R = ROWS_Y
A.head(cv, LET_X, A_Y, "a", "The certified chain", dx=TITLE_DX)
TITLES = ["Local enclosures", "Fused bound", "Realizable witness", "Map-level bracket", "Identification", "Declared-law inference", "Zoning statistics"]
for k, s in enumerate(TITLES):
    rbox(node_x(k), NY, NW, NH); node_title(k, s)
    if k < len(TITLES) - 1:
        block_arrow(node_x(k) + NW + BLK_IN, node_x(k + 1) - BLK_IN, BLK_Y)

# -- node 1: Georgia raster of value-box width, colour bar, strip location
a1 = a["a1"]; W1 = vec(a1["raster"]).reshape(a1["shape"]); mx = node_x(N1) + (NW - MAP_W) / 2
A.img(cv, W1, mx, MAP_Y, MAP_W, MAP_W, cmap=Z.FIELD_SEQ, vmin=a1["vmin"], vmax=a1["vmax"], interp="nearest")
cb = np.linspace(a1["vmin"], a1["vmax"], a1["shape"][1])[None, :]
cby = MAP_Y + MAP_W + CB_DY
A.img(cv, cb, mx, cby, MAP_W, CB_H, cmap=Z.FIELD_SEQ, vmin=a1["vmin"], vmax=a1["vmax"], interp="nearest", frame=HAIR)
ty = cby + CB_H + CB_TICK_DY
txt(mx, ty, A.fnum(a1["vmin"]), ha="left"); txt(mx + MAP_W, ty, A.N("a1_vmax", a1["vmax"], "{:.2f}"), ha="right")
txt(mx + MAP_W / 2, ty, "box width", ha="center")
s2 = a["a2"]; npx = a1["shape"][0]
yy = MAP_Y + (s2["display_row"] + 0.5) * MAP_W / npx; xa = mx + s2["display_col0"] * MAP_W / npx; xb = mx + (s2["display_col0"] + s2["length"]) * MAP_W / npx
cv.plot([xa, xb], [Yc(yy), Yc(yy)], color=INK, lw=STRIP_LW, solid_capstyle="butt", zorder=Z_KEY)
node_text(N1, R[0], A.N("pixels", a1["pixels"], "{:d}") + " pixels")
node_text(N1, R[1], "median width " + A.N("w_med", a1["width_median"], "{:.3f}"))
node_text(N1, R[2], A.N("tracts", a1["tracts"], "{:d}") + " tracts")

# -- node 2: one strip, value boxes (grey) and chain closure (teal)
ax = thumb("a_strip", N2, STRIP_BOX); n2 = len(s2["lo"]); t2 = np.arange(n2)
lo2, hi2, l2, u2 = vec(s2["lo"]), vec(s2["hi"]), vec(s2["l_star"]), vec(s2["u_star"])
ax.fill_between(t2, lo2, hi2, color=GRAYL, lw=0, step="mid", zorder=2)
ax.fill_between(t2, l2, u2, color=TEALL, lw=0, step="mid", zorder=3)
ax.plot(t2, u2, color=TEAL, lw=A.LW_FRAME, drawstyle="steps-mid", zorder=4); ax.plot(t2, l2, color=TEAL, lw=A.LW_FRAME, drawstyle="steps-mid", zorder=4)
pad = STRIP_PAD * (np.nanmax(hi2) - np.nanmin(lo2)); ax.set_xlim(-0.5, n2 - 0.5); ax.set_ylim(np.nanmin(lo2) - pad, np.nanmax(hi2) + pad)
key_item(node_x(N2) + HEAD_PAD, STRIP_KEY_Y[0], "sw", "value box", GRAYL)
key_item(node_x(N2) + HEAD_PAD, STRIP_KEY_Y[1], "sw", "lattice closure", TEALL)
node_text(N2, R[0], f"{s2['length']}-pixel strip")
node_text(N2, R[1], "closure / box " + A.N("strip_ratio", s2["ratio_strip"], "{:.2f}"))
node_text(N2, R[2], f"tile ratio {s2['k_v']} : {s2['k_g']}")

# -- node 3: lifted mesh X^(1): 3 x 3 pixels, nodes at h/2, the global nodal field shaded
s3 = a["a3"]; img3 = sq(s3["image"]); mx3 = node_x(N3) + (NW - MESH_W) / 2
A.img(cv, img3, mx3, MESH_Y, MESH_W, MESH_W, cmap=Z.FIELD_SEQ, vmin=s3["vmin"], vmax=s3["vmax"], interp="bilinear")
nd = sq(s3["nodal"]); nn = nd.shape[0]; step = MESH_W / (nn - 1)
for q in range(nn):
    major = q % (nn // MESH_PIX) == 0
    lw_, c_ = (MESH_LW_PIX, INK) if major else (MESH_LW_SUB, WHITE)
    cv.plot([mx3, mx3 + MESH_W], [Yc(MESH_Y + q * step)] * 2, color=c_, lw=lw_, zorder=Z_ROUTE, solid_capstyle="butt")
    cv.plot([mx3 + q * step] * 2, [Yc(MESH_Y), Yc(MESH_Y + MESH_W)], color=c_, lw=lw_, zorder=Z_ROUTE, solid_capstyle="butt")
gx, gy = np.meshgrid(np.arange(nn) * step + mx3, np.arange(nn) * step + MESH_Y)
cv.plot(gx.ravel(), [Yc(v) for v in gy.ravel()], ls="none", marker="o", ms=NODE_MS, mfc=INK, mec=INK, mew=0, zorder=Z_KEY)
txt(node_x(N3) + NW / 2, MESH_LAB_Y, "pixel h | nodes h/2", ha="center")
node_text(N3, R[0], "global nodal field")
node_text(N3, R[1], A.N("nodes", s3["nodes_total"], "{:d}") + " nodes")
node_text(N3, R[2], "exact-feasible" if a["a4"]["witness_feasible_exact"] else "tolerance-feasible")

# -- node 4: map-level bracket (checker values)
s4 = a["a4"]; ax = thumb("a_bracket", N4, BR_BOX)
ax.plot([s4["L"], s4["U"]], [0, 0], color=Z.BRACKET_FILL, lw=BR_LW, solid_capstyle="butt", zorder=2)
ax.plot([BR_XL[0], BR_XL[1]], [0, 0], color=HAIR, lw=A.LW_FRAME, zorder=1)
ax.plot([s4["reference"]] * 2, [-BR_REF_H, BR_REF_H], color=GRAYREF, lw=A.LW_REF, ls=Z.LS_REF, zorder=1)
st_w, st_c = Z.BOUND["global"], Z.BOUND["lifted"]
ax.plot([s4["L"]], [0], marker=st_w["mk"], ms=BR_MS, mfc=st_w["mfc"], mec=st_w["mec"], ls="none", zorder=4)
ax.plot([s4["U"]], [0], marker=st_c["mk"], ms=BR_MS, mfc=st_c["mfc"], mec=st_c["mec"], ls="none", zorder=4)
ax.set_xlim(*BR_XL); ax.set_ylim(-1, 1)
bx0 = node_x(N4) + BR_BOX[0]; sc = BR_BOX[2] / (BR_XL[1] - BR_XL[0]); ymid = BR_BOX[1] + BR_BOX[3] / 2
xm = lambda v: bx0 + (v - BR_XL[0]) * sc
txt(xm(s4["L"]), ymid - BR_LAB_DY, A.N("brL", s4["L"], "{:.4f}"), ha="center", va="bottom")
txt(xm(s4["U"]), ymid - BR_LAB_DY, A.N("brU", s4["U"], "{:.4f}"), ha="center", va="bottom")
txt(xm(s4["reference"]), ymid + BR_LAB_DY, A.fnum(s4["reference"]), ha="center", va="top")
k0x = node_x(N4) + HEAD_PAD
key_item(k0x, R[0], "line", "global witness", BRICK, lw=0, marker=st_w["mk"], mfc=st_w["mfc"], ms=BR_MS * BR_KEY_F)
key_item(k0x, R[1], "line", "lifted ceiling", TEAL, lw=0, marker=st_c["mk"], mfc=st_c["mfc"], ms=BR_MS * BR_KEY_F)
key_item(k0x, R[2], "line", "reference", GRAYREF, ls=Z.LS_REF, lw=A.LW_REF)
node_text(N4, R[3], "× reference movement")

# -- node 5: identification fiber glyph
s5 = a["a5"]; ax = thumb("a_fiber", N5, FIB_BOX); xs5 = vec(s5["x"])
for k, m_ in enumerate(s5["means"]):
    ax.add_patch(Rectangle((k + (1 - FIB_BAR_W) / 2, 0), FIB_BAR_W, m_, fc=GRAYLL, ec="none", zorder=1))
    ax.plot([k + (1 - FIB_BAR_W) / 2, k + (1 + FIB_BAR_W) / 2], [m_, m_], color=GREY, lw=A.LW_THIN, zorder=2, solid_capstyle="butt")
ax.plot(xs5, vec(s5["f1"]), color=INK, lw=A.LW_THIN, zorder=3); ax.plot(xs5, vec(s5["f2"]), color=TEAL, lw=A.LW_THIN, zorder=3)
ax.set_xlim(0, len(s5["means"])); ax.set_ylim(*FIB_YL)
k5 = node_x(N5) + HEAD_PAD
key_item(k5, R[0], "sw", "unit means y", GRAYL, fc=GRAYLL)
key_item(k5, R[1], "line", "field f", INK, lw=A.LW_THIN)
key_item(k5, R[2], "line", "field g", TEAL, lw=A.LW_THIN)

# -- node 6: four declared laws and the family envelope
s6 = a["a6"]; ax = thumb("a_laws", N6, LAW_BOX); zz = vec(s6["z"])
for (nm, dv), ls_ in zip(s6["dens"].items(), LAW_LS):
    ax.plot(zz, vec(dv), color=GREY, lw=LAW_LW, ls=ls_, zorder=2)
ax.plot(zz, vec(s6["envelope"]), color=TEAL, lw=ENV_LW, zorder=3)
ax.set_xlim(zz.min(), zz.max()); ax.set_ylim(0, np.nanmax(vec(s6["envelope"])) * (1 + LAW_PAD))
k6 = node_x(N6) + HEAD_PAD
for i, ((nm, lab), ls_) in enumerate(zip(s6["labels"].items(), LAW_LS)):
    key_item(k6, R[i], "line", lab, GREY, ls=ls_, lw=LAW_LW)
key_item(k6, R[len(LAW_LS)], "line", "family envelope", TEAL, lw=ENV_LW)

# -- node 7: 12-OA window glyph, one contiguous K = 3 partition, slope range
s7 = a["a7"]; zones = np.array(s7["glyph_zones"]); nr, nc = zones.shape; ox = node_x(N7) + (NW - nc * OA_CELL) / 2
for r_ in range(nr):
    for c_ in range(nc):
        rect(ox + c_ * OA_CELL, OA_Y + r_ * OA_CELL, OA_CELL, OA_CELL, ZONE_FILLS[zones[r_, c_]], ec=WHITE, lw=A.LW_THIN)
for r_ in range(nr):
    for c_ in range(nc):
        if c_ + 1 < nc and zones[r_, c_] != zones[r_, c_ + 1]:
            xx = ox + (c_ + 1) * OA_CELL; cv.plot([xx, xx], [Yc(OA_Y + r_ * OA_CELL), Yc(OA_Y + (r_ + 1) * OA_CELL)], color=INK, lw=OA_LW_ZONE, zorder=Z_ROUTE, solid_capstyle="projecting")
        if r_ + 1 < nr and zones[r_, c_] != zones[r_ + 1, c_]:
            yy_ = OA_Y + (r_ + 1) * OA_CELL; cv.plot([ox + c_ * OA_CELL, ox + (c_ + 1) * OA_CELL], [Yc(yy_), Yc(yy_)], color=INK, lw=OA_LW_ZONE, zorder=Z_ROUTE, solid_capstyle="projecting")
cv.add_patch(Rectangle((ox, Yc(OA_Y + nr * OA_CELL)), nc * OA_CELL, nr * OA_CELL, fill=False, ec=INK, lw=OA_LW_ZONE, zorder=Z_ROUTE))
ax = thumb("a_slope", N7, SL_BOX)
ax.plot(s7["slope_lp"], [0, 0], color=TEALL, lw=SL_LW_LP, solid_capstyle="butt", zorder=2)
ax.plot(s7["slope_exact"], [0, 0], color=TEAL, lw=SL_LW_EX, solid_capstyle="butt", zorder=3)
ax.plot([0, 0], [0, 1], color=GRAYREF, lw=A.LW_REF, ls=Z.LS_REF, zorder=1); ax.set_xlim(*SL_XL); ax.set_ylim(-1, 1)
sx0 = node_x(N7) + SL_BOX[0]; ssc = SL_BOX[2] / (SL_XL[1] - SL_XL[0]); sm = lambda v: sx0 + (v - SL_XL[0]) * ssc
sy = SL_BOX[1] + SL_BOX[3] / 2
txt(sm(s7["slope_exact"][0]), sy + SL_LAB_DY, A.N("slo", s7["slope_exact"][0], "{:.2f}"), ha="center")
txt(sm(s7["slope_exact"][1]), sy + SL_LAB_DY, A.N("shi", s7["slope_exact"][1], "{:.2f}"), ha="center")
txt(sm(0), sy - SL_LAB_DY, "0", ha="center", va="bottom", c=GREY)
k7 = node_x(N7) + HEAD_PAD
key_item(k7, R[0], "sw", "exact slope range", TEAL)
key_item(k7, R[1], "sw", "LP bound", TEALL)
node_text(N7, R[2], f"{s7['n_oa']} OAs | K = {s7['K']}")
node_text(N7, R[3], A.N("parts", s7["partitions"], "{:d}") + " partitions")

# -- connector legend (top right)
items = [("block", "forward data", None), ("arrow", "certificate", "cert"), ("arrow", "check", "check"), ("arrow", "conditioning", "cond")]
wid = sum(KEY_LEN + KEY_TX + A.mmw(lab, A.F_KEY) for _, lab, _ in items) + LEG_GAP * (len(items) - 1)
x = LEG_X1 - wid
for kind, lab, kk in items:
    x = key_item(x, LEG_Y, kind, lab, kk if kk else HAIR) + LEG_GAP

# -- conditioning: identification -> realizable witness (the observed means condition the class)
c3x, c5x = node_x(N3) + NW / 2, node_x(N5) + NW / 2
route([(c5x, NY), (c5x, COND_Y), (c3x, COND_Y), (c3x, NY)], "cond")

# -- certificate lane
lane = a["lane"]; nb = NY + NH
opt_cx = OPT_X + OPT_W / 2
drops = [node_x(N2) + NW / 2, node_x(N4) + NW / 2, node_x(N7) + NW / 2]
for xd in drops:
    cv.plot([xd, xd], [Yc(nb), Yc(BUS_Y)], color=INK, lw=CERT_LW, zorder=Z_ROUTE, solid_capstyle="butt")
cv.plot([max(drops), opt_cx], [Yc(BUS_Y), Yc(BUS_Y)], color=INK, lw=CERT_LW, zorder=Z_ROUTE, solid_capstyle="butt")
route([(opt_cx, BUS_Y), (opt_cx, LANE_Y)], "cert")
lm = LANE_Y + LANE_H / 2
rbox(OPT_X, LANE_Y, OPT_W, LANE_H, fc=Z.VERIFY["external"]["c"], ec="none")
pcx, pcy = OPT_X + POLY_R + HEAD_PAD + POLY_DX, lm
ang = np.linspace(0, 2 * np.pi, POLY_N + 1)[:-1] + np.pi / 2
pts = [(pcx + POLY_R * np.cos(t_), pcy - POLY_R * np.sin(t_)) for t_ in ang]
poly(pts, WHITE, ec=INK, lw=A.LW_THIN, z=Z_KEY)
cv.plot([pts[0][0]], [Yc(pts[0][1])], marker="o", ms=NODE_MS * OPT_MS_F, mfc=BRICK, mec=BRICK, ls="none", zorder=Z_KEY)
ox_t = pcx + POLY_R + HEAD_PAD
txt(ox_t, lm - HEAD_LH * TXT_UP, "optimizer", A.F_AXLAB, va="center_baseline", c=A.contrast_text(GRAYL))
txt(ox_t, lm + HEAD_LH * TXT_DN, "external", A.F_VAL, va="center_baseline", c=A.contrast_text(GRAYL))
dx0, dy0 = DOC_CX - DOC_W / 2, lm - DOC_H / 2
poly([(dx0, dy0), (dx0 + DOC_W - DOC_FOLD, dy0), (dx0 + DOC_W, dy0 + DOC_FOLD), (dx0 + DOC_W, dy0 + DOC_H), (dx0, dy0 + DOC_H)], Z.VERIFY["exact"]["c"], ec=INK, lw=A.LW_THIN, z=Z_KEY)
poly([(dx0 + DOC_W - DOC_FOLD, dy0), (dx0 + DOC_W - DOC_FOLD, dy0 + DOC_FOLD), (dx0 + DOC_W, dy0 + DOC_FOLD)], WHITE, ec=INK, lw=A.LW_THIN, z=Z_KEY, closed=True)
txt(DOC_CX, lm, "p/q", A.F_AXLAB, ha="center", va="center_baseline", c=INK, style="italic")
txt(DOC_CX, LANE_TXT_Y[0], "exact-rational", ha="center", va="center_baseline"); txt(DOC_CX, LANE_TXT_Y[1], "certificate", ha="center", va="center_baseline")
for (bx_, bw_, fc_, l1, l2) in ((CHK_X, CHK_W, Z.VERIFY["checker"]["c"], "verified checker", "compiled"), (KER_X, KER_W, Z.VERIFY["kernel"]["c"], "Lean kernel", "Lean 4 + Mathlib")):
    rbox(bx_, LANE_Y, bw_, LANE_H, fc=fc_, ec="none")
    txt(bx_ + bw_ / 2, lm - HEAD_LH * TXT_UP, l1, A.F_AXLAB, ha="center", va="center_baseline", c=A.contrast_text(fc_))
    txt(bx_ + bw_ / 2, lm + HEAD_LH * TXT_DN, l2, A.F_VAL, ha="center", va="center_baseline", c=A.contrast_text(fc_))
route([(OPT_X + OPT_W, lm), (dx0, lm)], "cert")
route([(dx0 + DOC_W, lm), (CHK_X, lm)], "cert")
route([(DOC_CX, dy0), (DOC_CX, CERT_ROUTE_Y), (KER_X + KER_W / 2, CERT_ROUTE_Y), (KER_X + KER_W / 2, LANE_Y)], "cert")
route([(KER_X, lm), (CHK_X + CHK_W, lm)], "check")
txt(CHK_X + CHK_W / 2, LANE_TXT_Y[0], A.N("map_ep", lane["map_pass"], "{:d}") + " map endpoints PASS", ha="center", va="center_baseline")
txt(CHK_X + CHK_W / 2, LANE_TXT_Y[1], A.N("pool", lane["pooling_pass"], "{:d}") + " pooling instances PASS", ha="center", va="center_baseline")
txt(KER_X + KER_W / 2, LANE_TXT_Y[0], A.N("charges", lane["pricing_charges"], "{:d}") + " pricing charges kernel-checked", ha="center", va="center_baseline")
txt(LANE_LAB[0], LANE_LAB[1], "certificate", A.F_AXLAB, c=INK); txt(LANE_LAB[0], LANE_LAB[1] + HEAD_LH, "lane", A.F_AXLAB, c=INK)
A.tok(cv, LET_X, A_TOK_Y, "Georgia, 512² raster | 12-OA window | real-array thumbnails | checker log counts")

# ===================================================================================== panel b ==
b = D["b"]; hours = b["hours"]; tb = np.arange(hours)
A.head(cv, LET_X, B_Y, "b", "A real one-dimensional instance", dx=TITLE_DX)
x = BX
x = key_item(x, B_KEY1_Y, "line", "nominal f(t)", INK, lw=A.LW_THIN) + LEG_GAP
x = key_item(x, B_KEY1_Y, "sw", "value box", Z.BAND_OUTER) + LEG_GAP
x = key_item(x, B_KEY1_Y, "line", "closure u*, l*", TEAL, lw=A.LW_FRAME) + LEG_GAP
ax1 = A.pax(fig, cv, BX, B_AX1_Y, BW, B_AX1_H, name="b_risk")
ax1.fill_between(tb, vec(b["lo"]), vec(b["hi"]), color=Z.BAND_OUTER, lw=0, zorder=1)
ax1.plot(tb, vec(b["u_star"]), color=TEAL, lw=A.LW_FRAME, zorder=2); ax1.plot(tb, vec(b["l_star"]), color=TEAL, lw=A.LW_FRAME, zorder=2)
ax1.plot(tb, vec(b["f"]), color=INK, lw=A.LW_THIN, zorder=3)
days = np.arange(0, hours, HOURS_PER_DAY * B_DAY_STEP)
A.clean(ax1, xl=(0, hours - 1), yl=B_YL1, ylab="risk f(t)"); ax1.set_yticks(B_YT1); ax1.set_xticks(days); ax1.set_xticklabels([])
WSER = [("width_box", "value box", TEAL, Z.LS_SOLID, "o", None, 0),
        ("width_closure", "closure", TEALL, Z.LS_DASH, "^", None, B_MARK_OFF),
        ("width_inner", "PGD inner", BRICK, Z.LS_SOLID, "D", None, B_MARK_OFF // 2),
        ("width_incr1", "lag-1 box", TEALM, Z.LS_DASHDOT, "s", WHITE, 0),
        ("width_incr1_implied", "implied lag-1", GREY, Z.LS_DOT, "v", WHITE, B_MARK_OFF)]
x = BX
for key, lab, c_, ls_, mk, mfc, off in WSER:
    x = key_item(x, B_KEY2_Y, "line", lab, c_, ls=ls_, lw=A.LW_THIN, marker=mk, mfc=mfc or c_) + LEG_GAP
ax2 = A.pax(fig, cv, BX, B_AX2_Y, BW, B_AX2_H, name="b_width")
for key, lab, c_, ls_, mk, mfc, off in WSER:
    v = vec(b[key])
    ax2.plot(np.arange(len(v)), v, color=c_, ls=ls_, lw=A.LW_THIN, marker=mk, ms=B_MS, mfc=mfc or c_, mec=c_, mew=A.LW_THIN / 2, markevery=(off, B_MARK_EVERY), zorder=3)
ax2.set_yscale("log")
A.clean(ax2, xl=(0, hours - 1), yl=B_YL2, ylab="width", xlab="date (July 2014)")
ax2.yaxis.set_minor_locator(NullLocator()); ax2.set_yticks(B_YT2); ax2.set_yticklabels([A.fnum(v) for v in B_YT2])
ax2.set_xticks(days); ax2.set_xticklabels([str(b["day0"] + int(d_) // HOURS_PER_DAY) for d_ in days])
A.tok(cv, LET_X, B_TOK_Y, f"client {b['client']} | jul 2014 | ε = {A.fnum(b['eps'])} sd | {b['method']}")

# ===================================================================================== panel c ==
c = D["c"]
A.head(cv, C_X0, B_Y, "c", "When increments pay", dx=TITLE_DX)
axc = A.pax(fig, cv, CX, C_AX_Y, CW, C_AX_H, name="c_regime")
xr = np.array(c["log2_ratio"]); buds = c["budgets"]
Gm = np.array([c["G"][k]["median"] for k in buds]); Gl = Gm - np.array([c["G"][k]["q25"] for k in buds]); Gh = np.array([c["G"][k]["q75"] for k in buds]) - Gm
Um = np.array([c["UV"][k]["median"] for k in buds]); Ul = Um - np.array([c["UV"][k]["q10"] for k in buds]); Uh = np.array([c["UV"][k]["q90"] for k in buds]) - Um
sG, sU = dict(c=TEAL, mk="o"), dict(c=TEALM, mk="s", ls=Z.LS_DASH)
A.refline(axc, 0)
for xi, k in zip(xr, buds):                                   # per-pair normalized savings (30 per budget), deterministic jitter
    gp = np.array(c["G_pairs"][k]); jit = (np.arange(len(gp)) / max(len(gp) - 1, 1) - 0.5) * 2 * C_JIT
    axc.plot(xi + jit, gp, ls="none", marker="o", ms=C_DOT_MS, mfc=TEALLL, mec=TEALL, mew=A.LW_FRAME, alpha=C_DOT_A, zorder=2)
axc.errorbar(xr, Um, yerr=[Ul, Uh], color=sU["c"], ls=sU["ls"], lw=A.LW_DATA, marker=sU["mk"], ms=C_MS, mfc=WHITE, mec=sU["c"], mew=A.LW_THIN, zorder=3, **Z.ERRORBAR)
axc.errorbar(xr, Gm, yerr=[Gl, Gh], color=sG["c"], ls=Z.LS_SOLID, lw=A.LW_DATA, marker=sG["mk"], ms=C_MS, mfc=sG["c"], mec=sG["c"], zorder=4, **Z.ERRORBAR)
g16, u16 = c["G_1616"], c["UV_1616"]
axc.errorbar([C_OFF1616], [g16["median"]], yerr=[[g16["median"] - g16["q25"]], [g16["q75"] - g16["median"]]], color=sG["c"], marker=sG["mk"], ms=C_MS, mfc=WHITE, mec=sG["c"], mew=A.LW_THIN, ls="none", zorder=4, **Z.ERRORBAR)
axc.errorbar([C_OFF1616], [u16["median"]], yerr=[[u16["median"] - u16["q10"]], [u16["q90"] - u16["median"]]], color=sU["c"], marker=sU["mk"], ms=C_MS, mfc=GRAYLL, mec=sU["c"], mew=A.LW_THIN, ls="none", zorder=4, **Z.ERRORBAR)
A.clean(axc, xl=(xr.min() - C_XPAD, xr.max() + C_XPAD), yl=C_YL, ylab="ratio (G, U/V)", xlab="value : gradient tile coarsening")
axc.set_xticks(xr); axc.set_xticklabels(buds); axc.set_yticks(C_YT)
A.T(axc, *C_LAB_G, "normalized saving G", c=sG["c"], ha="left", va="bottom")
A.T(axc, *C_LAB_UV, "fused/value-only U/V", c=sU["c"], ha="left", va="top")
A.T(axc, *C_LAB_1616, "hollow 16:16", c=TXT, ha="center", va="center_baseline")
A.tok(cv, C_X0, C_TOK_Y, f"Georgia nested pairs | G: {g16['n']} per budget, IQR | U/V: {u16['n']}, q10–q90")

# ===================================================================================== panel d ==
d = D["d"]; tags = d["tags"]
A.head(cv, LET_X, D_Y, "d", "Realization ladder on four settings", dx=TITLE_DX)
x = DX
for s in Z.SETTING_ORDER:
    st = Z.SETTING[s]; x = key_item(x, D_KEY_Y, "line", Z.SETTING_SHORT[s], st["c"], ls=st["ls"], lw=A.LW_THIN, marker=st["mk"], mfc=st["mfc"], ms=D_MS) + LEG_GAP
axd = A.pax(fig, cv, DX, D_AX_Y, DW, D_AX_H, name="d_ladder")
lo_all, hi_all = [], []
for i, s in enumerate(Z.SETTING_ORDER):
    st = Z.SETTING[s]; bt = d["settings"][s]["by_tag"]; xx = np.arange(len(tags)) + D_OFF[i]
    med = np.array([bt[t_]["median"] for t_ in tags]); mn = np.array([bt[t_]["min"] for t_ in tags]); mxv = np.array([bt[t_]["max"] for t_ in tags])
    lo_all.append(mn.min()); hi_all.append(mxv.max())
    axd.errorbar(xx, med, yerr=[med - mn, mxv - med], color=st["c"], ls=st["ls"], lw=A.LW_THIN, marker=st["mk"], ms=D_MS, mfc=st["mfc"], mec=st["mec"], mew=A.LW_THIN / 2, zorder=Z_RECT + i, **Z.ERRORBAR)
A.refline(axd, 1.0)
A.T(axd, *D_REF_LAB, "witness = 1", c=GRAYREF, ha="left", va="bottom")
xlim_d = (-0.5, len(tags) - 0.5)
A.clean(axd, xl=xlim_d, yl=(min(min(lo_all), 1.0) - D_PAD, max(hi_all) + D_PAD), ylab="outer / witness width")
axd.set_xticks(np.arange(len(tags))); axd.set_xticklabels([])
LADDER = [[("pixel Φ′", "b")], [("Q", "b"), ("4", "sub"), (" + ε", "b"), ("4", "sub")], [("X", "b"), ("(1)", "sup")], [("X", "b"), ("(2)", "sup")],
          [("X", "b"), ("L", "sub"), ("(2)", "sup")]]
for i, segs in enumerate(LADDER):
    xmm = DX + (i - xlim_d[0]) / (xlim_d[1] - xlim_d[0]) * DW
    mlabel(xmm, D_AX_Y + D_AX_H + D_TICK_DY, segs)
txt(DX + DW / 2, D_AX_Y + D_AX_H + D_XLAB_DY, "outer set, by refinement", A.F_AXLAB, ha="center", va="center_baseline", c=INK)
nu = sorted({d["settings"][s]["unique"] for s in Z.SETTING_ORDER})
A.tok(cv, LET_X, D_TOK_Y, f"{'/'.join(str(v) for v in nu)} unique patches per setting | verified solves | median, min–max")

# ===================================================================================== panel e ==
e = D["e"]
A.head(cv, E_X0, D_Y, "e", "Verification status by stage", dx=TITLE_DX)
grid = e["grid"]
lab_map = {"inference under declared law": "declared-law inference"}
labs = [lab_map.get(g["stage"], g["stage"]) for g in grid]
col0 = max(A.mmw(s, A.F_VAL) for s in labs) + HEAD_PAD
COLS = [("kernel", ["machine-", "checked", "theorem"]), ("checker", ["verified", "checker"]), ("exact", ["exact-", "rational", "certificate"]), ("external", ["external", "numerics"])]
cw = (E_X1 - E_X0 - col0) / len(COLS)
for j, (key, lines) in enumerate(COLS):
    cx_ = E_X0 + col0 + j * cw + HEAD_PAD
    for q, ln in enumerate(reversed(lines)):
        txt(cx_, E_RULE_Y - E_HEAD_DY - q * HEAD_LH, ln, ha="left", va="baseline")
cv.plot([E_X0, E_X1], [Yc(E_RULE_Y), Yc(E_RULE_Y)], color=HAIR, lw=E_RULE_LW, zorder=2)
for i, (g, lab) in enumerate(zip(grid, labs)):
    yc = E_ROW0_Y + (i + 0.5) * E_ROW_H
    if i % 2 == 1:
        rect(E_X0, E_ROW0_Y + i * E_ROW_H, E_X1 - E_X0, E_ROW_H, GRAYLL, z=1)
    txt(E_X0, yc, lab, ha="left", va="center_baseline")
    vals = [(g["machine_checked"], g.get("mc_count")), (g["verified_checker"], g["verified_checker"]), (g["exact_rational"], g["exact_rational"]), (g["external"], None)]
    for j, ((on, cnt), (key, _)) in enumerate(zip(vals, COLS)):
        cx_ = E_X0 + col0 + j * cw + HEAD_PAD
        on = on is not None and on is not False
        rect(cx_, yc - E_SQ / 2, E_SQ, E_SQ, Z.VERIFY[key]["c"] if on else WHITE, ec="none" if on else HAIR, lw=0 if on else A.LW_FRAME, z=Z_POLY)
        lab_c = None if (cnt is None or isinstance(cnt, (bool, str))) else A.fnum(int(cnt), "{:d}")
        if key == "exact" and g.get("exact_rational_2") is not None:
            lab_c = f"{lab_c} + {A.fnum(int(g['exact_rational_2']), '{:d}')}"
        if key == "kernel" and lab_c:
            lab_c = lab_c + " charges"
        if lab_c:
            txt(cx_ + E_SQ + E_CNT_DX, yc, lab_c, ha="left", va="center_baseline")
A.tok(cv, E_X0, E_TOK_Y, "Lean 4 + Mathlib | 0 admitted propositions | standard axioms only")

A.save(fig, "F1", script=os.path.abspath(__file__))
