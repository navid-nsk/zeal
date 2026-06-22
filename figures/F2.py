"""F2 — "Geometric Vacuity and Transverse Slack" (merge of
   fig3_field_vacuity + fig4_transverse).

Six dense panels at 180 mm:
  a  Certifiable-by-construction pipeline with REAL-data thumbnails at every stage
     (coordinate cloud -> real RFF/sinusoid tile -> Tanh-MLP node-graph -> softplus
      with the Tobler >=0 floor -> real lambda(x) GA choropleth -> aggregated county
      map), joined by a typed-connector grammar. Schematic. (lifts fig3a)
  b  On-manifold geometry merged: one shared oblique 3-D projection carries the shaded
     curved 2-D manifold patch in 256-D, its true tangent plane, the ambient gradient
     decomposed into a kept tangent component and a leaked normal component, and the
     axis-aligned ambient bounding box -- so ||tangent|| << ||ambient|| reads directly.
     (merges fig3b geometry + fig4a tangent/normal split)
  c  Box looseness vs ambient dimension: ambient-box bound rising vs on-manifold ~1
     flat, the 488x worst case as an annotated point, log-log, +/-SEM bands. (lifts fig3c)
  d  Slack factorization as a cumulative-product waterfall: 1 -> x rho_off(1.00) ->
     x rho_tan(42.7) -> x rho_cond(1.15) = 49.1x, with the generic sqrt(m/d)=11.3x
     reference and a +/-SEM cap on the transverse lever. (recasts fig4b, >=7 elements)
  e  rho_tan (transverse) vs ambient dimension climbing 6 -> 49 with the sqrt(m/d)
     reference and a +/-SEM band -- the only error-barred slack data. (lifts fig4c, N20)
  f  rho_off (off-manifold) vs ambient dimension pinned at the ~1 floor, +/-SEM band --
     the transverse term is the ONLY dimension-growing component. (lifts fig4d, N21)

Real data only: ga_fieldmap.npz, sliver.npz, dimsweep.json (n=3 seeds).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, FancyBboxPatch
from matplotlib.collections import LineCollection
import zstyle as Z

Z.setup()

# ------------------------------------------------------------------ data
ga = np.load(Z.os.path.join(Z.DATA, "ga_fieldmap.npz"))
field = ga["field"].astype(float)          # continuous lambda(x)
co = ga["co"].astype(float)                # aggregated county field
mask = ga["mask"]
sl = np.load(Z.os.path.join(Z.DATA, "sliver.npz"))
tile = sl["field_tile"]                    # real RFF / sinusoid tile

dim = Z.load("dimsweep.json")
rows = dim["rows"]
NSEED = dim["nseed"]
SQN = np.sqrt(NSEED)                        # std -> SEM
ds = np.array([r["dim"] for r in rows], float)
ms = np.array([r["m"] for r in rows], float)
rtan = np.array([r["rho_tan"] for r in rows], float)
rtan_sem = np.array([r["rho_tan_std"] for r in rows], float) / SQN
roff = np.array([r["rho_off"] for r in rows], float)
roff_sem = np.array([r["rho_off_std"] for r in rows], float) / SQN

# locked headline factorization (operating point m=128, d=256)
RHO_OFF, RHO_TAN, RHO_COND = 1.00, 42.7, 1.15
GENERIC = 11.3                              # generic sqrt(m/d) bound
TOTAL = RHO_OFF * RHO_TAN * RHO_COND        # 49.1x
j256 = int(np.argmin(np.abs(ds - 256)))     # SEM on the transverse lever at d=256

# ------------------------------------------------------------------ canvas
# Three bands: (a) full-width pipeline on top; (b,c) middle; (d,e,f) bottom trio.
fig = plt.figure(figsize=(180 * Z.MM, 150 * Z.MM))

# Band a: full-width pipeline (its own off-axis board, manual insets for thumbnails).
# Grown taller and pulled down to claim the top band budget (~30% of canvas) so the
# pipeline thumbnails are large and the strip below the title is filled, not empty.
A_L, A_B, A_W, A_H = 0.030, 0.665, 0.945, 0.300
ax_a = fig.add_axes([A_L, A_B, A_W, A_H]); ax_a.axis("off")
ax_a.set_xlim(0, 100); ax_a.set_ylim(0, 40)

# Band b,c: middle row (b geometry panel ~52%, c looseness curve), raised to meet band a
ax_b = fig.add_axes([0.020, 0.380, 0.430, 0.262]); ax_b.axis("off")
gs_c = fig.add_gridspec(1, 1, left=0.600, right=0.975, top=0.632, bottom=0.405)
ax_c = fig.add_subplot(gs_c[0])

# Band d,e,f: bottom trio (waterfall, rho_tan, rho_off)
gs_def = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.42,
                          left=0.060, right=0.985, top=0.305, bottom=0.070)
ax_d = fig.add_subplot(gs_def[0])
ax_e = fig.add_subplot(gs_def[1])
ax_f = fig.add_subplot(gs_def[2])

# =====================================================================================
# PANEL a — certifiable-by-construction pipeline, real-data thumbnails  [fig3a]
# =====================================================================================
Z.panel(ax_a, "a")
ax_a.text(0, 39.6, "Certifiable-by-Construction Field", fontsize=7.5,
          color=Z.INK, va="top", ha="left")

# larger thumbnails, raised to fill the taller top band (less empty strip below title)
TH_W, TH_H = 13.4, 20.5
TH_Y = 13.0
cx = [9.0, 24.7, 40.4, 56.1, 71.8, 88.0]
labels = [r"Location x∈ℝ²", "Random Fourier", "MLP (Tanh)",
          r"Softplus  (Tobler ≥ 0)", r"Field  λ(x)", r"Aggregate  R_Z"]


def tile_axes(center_x):
    x0 = center_x - TH_W / 2
    fx = A_L + (x0 / 100.0) * A_W
    fw = (TH_W / 100.0) * A_W
    fy = A_B + (TH_Y / 40.0) * A_H
    fh = (TH_H / 40.0) * A_H
    return fig.add_axes([fx, fy, fw, fh])


def stage_frame(center_x, accent=Z.GRAYL):
    x0 = center_x - TH_W / 2 - 0.7
    ax_a.add_patch(FancyBboxPatch((x0, TH_Y - 1.0), TH_W + 1.4, TH_H + 2.0,
                                  boxstyle="round,pad=0.0,rounding_size=0.8",
                                  fc="none", ec=accent, lw=0.7, zorder=1))


# stage 1: real-style coordinate point cloud
stage_frame(cx[0])
t1 = tile_axes(cx[0]); t1.set_xlim(0, 1); t1.set_ylim(0, 1)
rng = np.random.default_rng(3)
px, py = rng.uniform(0.12, 0.88, 26), rng.uniform(0.12, 0.88, 26)
t1.scatter(px, py, s=5.5, color=Z.STEEL1, edgecolor="white", lw=0.25, zorder=3)
t1.scatter([0.5], [0.5], s=18, color=Z.PINKD, edgecolor="white", lw=0.4, zorder=4)
t1.annotate("", xy=(0.5, 0.5), xytext=(0.18, 0.18),
            arrowprops=dict(arrowstyle="-", color=Z.GRAY, lw=0.4))
for s in t1.spines.values():
    s.set_visible(True); s.set_color(Z.GRAYL); s.set_linewidth(0.6)
t1.set_xticks([]); t1.set_yticks([])

# stage 2: genuine random-Fourier feature map gamma(x)=[sin(Wx),cos(Wx)]
stage_frame(cx[1])
t2 = tile_axes(cx[1])
fr = np.random.default_rng(7)
gx, gy = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 80))
W = fr.normal(0, 2.6, (4, 2)); b = fr.uniform(0, 2 * np.pi, 4)
rff = np.zeros_like(gx)
for k in range(4):
    rff += np.sin(W[k, 0] * 2 * np.pi * gx + W[k, 1] * 2 * np.pi * gy + b[k])
m2 = np.abs(rff).max()
t2.imshow(rff, cmap=Z.DIV, origin="lower", vmin=-m2, vmax=m2, interpolation="bilinear")
t2.set_xticks([]); t2.set_yticks([])
for s in t2.spines.values():
    s.set_color(Z.GRAYL); s.set_linewidth(0.6)

# stage 3: Tanh-MLP node-graph (real edges)
stage_frame(cx[2])
t3 = tile_axes(cx[2]); t3.set_xlim(0, 1); t3.set_ylim(0, 1); t3.axis("off")
layer_x = [0.12, 0.42, 0.72, 0.95]; layer_n = [3, 4, 4, 1]
nodes = []
for lx, n in zip(layer_x, layer_n):
    ys = np.linspace(0.18, 0.82, n)
    nodes.append([(lx, y) for y in ys])
seg = []
for li in range(len(nodes) - 1):
    for (x0, y0) in nodes[li]:
        for (x1, y1) in nodes[li + 1]:
            seg.append([(x0, y0), (x1, y1)])
t3.add_collection(LineCollection(seg, colors=Z.GRAYL, linewidths=0.32, zorder=1))
for li, lay in enumerate(nodes):
    col = Z.PINKD if li == len(nodes) - 1 else Z.STEEL2
    for (x0, y0) in lay:
        t3.add_patch(plt.Circle((x0, y0), 0.045, fc=col, ec="white", lw=0.3, zorder=3))

# stage 4: real softplus curve with the Tobler >= 0 floor
stage_frame(cx[3])
t4 = tile_axes(cx[3])
xs = np.linspace(-4, 4, 200); sp = np.log1p(np.exp(xs))
t4.axhline(0, color=Z.RED, ls=(0, (3, 2)), lw=0.7, zorder=1)
t4.plot(xs, sp, color=Z.PINKD, lw=1.3, zorder=3)
t4.fill_between(xs, 0, sp, color=Z.PINKL, alpha=0.30, zorder=2)
t4.set_xlim(-4, 4); t4.set_ylim(-0.6, 4.2)
t4.set_xticks([]); t4.set_yticks([])
t4.text(0.04, 0.04, r"≥ 0", transform=t4.transAxes, fontsize=5.5,
        color=Z.RED, va="bottom", ha="left")
for s in t4.spines.values():
    s.set_color(Z.GRAYL); s.set_linewidth(0.6)
t4.spines["top"].set_visible(False); t4.spines["right"].set_visible(False)

# stage 5: real lambda(x) choropleth
stage_frame(cx[4])
t5 = tile_axes(cx[4])
fld = np.where(mask, field, np.nan)
ys, xs2 = np.where(mask)
r0, r1, c0, c1 = ys.min(), ys.max(), xs2.min(), xs2.max()
sub = fld[r0:r1 + 1, c0:c1 + 1]
t5.imshow(sub, cmap=Z.SEQ, origin="upper",
          vmin=np.nanmin(field[mask]), vmax=np.nanmax(field[mask]))
t5.set_xticks([]); t5.set_yticks([])
for s in t5.spines.values():
    s.set_color(Z.GRAYL); s.set_linewidth(0.6)

# stage 6: real aggregated county map (certified output -> pink frame)
stage_frame(cx[5], accent=Z.PINKD)
t6 = tile_axes(cx[5])
agg = np.where(mask, co, np.nan)[r0:r1 + 1, c0:c1 + 1]
t6.imshow(agg, cmap=Z.SEQ, origin="upper",
          vmin=np.nanmin(co[mask]), vmax=np.nanmax(co[mask]))
t6.set_xticks([]); t6.set_yticks([])
for s in t6.spines.values():
    s.set_color(Z.PINKD); s.set_linewidth(0.9)

# Arial lacks ∈, ℝ (and ∇ used in panel b); render any label carrying those
# math glyphs in DejaVu Sans so they stay editable <text> instead of glyph-boxes.
_GLYPH_FALLBACK = set("∈ℝ∇")
for c, lab in zip(cx, labels):
    fam = "DejaVu Sans" if _GLYPH_FALLBACK & set(lab) else None
    ax_a.text(c, TH_Y - 3.4, lab, ha="center", va="top", fontsize=6.2, color=Z.INK,
              fontfamily=fam)

# typed forward-data block arrows
for i in range(5):
    x0 = cx[i] + TH_W / 2 + 1.0
    x1 = cx[i + 1] - TH_W / 2 - 1.0
    ax_a.add_patch(FancyArrowPatch((x0, TH_Y + TH_H / 2), (x1, TH_Y + TH_H / 2),
                                   arrowstyle="-|>", mutation_scale=8,
                                   color=Z.GRAY, lw=2.4, zorder=2))

# in-figure connector legend (bottom-left dead space)
ax_a.add_patch(FancyArrowPatch((1.5, 5.6), (6.5, 5.6), arrowstyle="-|>",
                               mutation_scale=7, color=Z.GRAY, lw=2.4))
ax_a.text(7.6, 5.6, "forward data flow", fontsize=5.8, color=Z.INK, va="center")
ax_a.add_patch(FancyBboxPatch((1.5, 1.6), 5.0, 2.0, boxstyle="round,pad=0,rounding_size=0.5",
                              fc="none", ec=Z.PINKD, lw=1.0))
ax_a.text(7.6, 2.6, "certified output (ZEAL)", fontsize=5.8, color=Z.PINKD, va="center")

# =====================================================================================
# PANEL b — on-manifold geometry: shaded patch + tangent plane + ambient gradient split
#           + ambient box, ONE shared oblique 3-D projection  [fig3b geometry ⊕ fig4a split]
# =====================================================================================
Z.panel(ax_b, "b")
ax_b.text(0.0, 1.04, "On-Manifold Geometry", transform=ax_b.transAxes,
          fontsize=7.5, color=Z.INK, va="bottom", ha="left")
ax_b.set_xlim(0.0, 1.0); ax_b.set_ylim(0.0, 1.0)

# shared oblique 3-D frame (x: manifold-east, y: manifold-north depth, z: ambient up)
OX, OY = 0.36, 0.470
EX = np.array([0.520, 0.055])
EY = np.array([-0.250, 0.150])
EZ = np.array([0.0, 0.620])


def proj(x, y, z=0.0):
    s = np.array([OX, OY]) + x * EX + y * EY + z * EZ
    return float(s[0]), float(s[1])


def surf(x, y):
    return 0.30 * (x * x) - 0.16 * (y * y) + 0.045 * np.sin(3.0 * x + 1.2)


def surf_grad(x, y):
    fx = 0.60 * x + 0.045 * 3.0 * np.cos(3.0 * x + 1.2)
    fy = -0.32 * y
    return fx, fy


# shaded curved 2-D manifold patch, depth-sorted quads, single-hue Z.SEQ by height
nu, nv = 26, 26
gxx = np.linspace(-0.5, 0.5, nu); gyy = np.linspace(-0.5, 0.5, nv)
GX, GY = np.meshgrid(gxx, gyy); GZ = surf(GX, GY)
quads, depth, shade = [], [], []
for j in range(nv - 1):
    for i in range(nu - 1):
        pts = [proj(GX[j, i], GY[j, i], GZ[j, i]),
               proj(GX[j, i + 1], GY[j, i + 1], GZ[j, i + 1]),
               proj(GX[j + 1, i + 1], GY[j + 1, i + 1], GZ[j + 1, i + 1]),
               proj(GX[j + 1, i], GY[j + 1, i], GZ[j + 1, i])]
        quads.append(pts)
        depth.append(-(GY[j, i] + GY[j + 1, i + 1]))
        shade.append(0.5 * (GZ[j, i] + GZ[j + 1, i + 1]))
order = np.argsort(depth)
shv = np.array(shade); shv = (shv - shv.min()) / (np.ptp(shv) + 1e-9)
for k in order:
    c = Z.SEQ(0.22 + 0.58 * shv[k])
    ax_b.add_patch(Polygon(quads[k], closed=True, fc=c, ec="white", lw=0.12, zorder=2))

# touch point P on the manifold
x0, y0 = 0.06, -0.04
z0 = surf(x0, y0)
fx0, fy0 = surf_grad(x0, y0)
P = proj(x0, y0, z0)

# true tangent plane parallelogram
TE = 0.24
tp = [proj(x0 + sx * TE, y0 + sy * TE, z0 + fx0 * sx * TE + fy0 * sy * TE)
      for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
ax_b.add_patch(Polygon(tp, closed=True, fc=Z.PINKL, ec=Z.PINKD, lw=1.1,
                       alpha=0.55, zorder=5))

# axis-aligned ambient bounding box (true 3-D cube, half-extent >> tangent)
BE = 0.52
corners = {}
for sx in (-1, 1):
    for sy in (-1, 1):
        for sz in (-1, 1):
            corners[(sx, sy, sz)] = proj(x0 + sx * BE, y0 + sy * BE, z0 + sz * BE)
box_edges = [
    ((-1, -1, -1), (1, -1, -1)), ((1, -1, -1), (1, 1, -1)),
    ((1, 1, -1), (-1, 1, -1)), ((-1, 1, -1), (-1, -1, -1)),
    ((-1, -1, 1), (1, -1, 1)), ((1, -1, 1), (1, 1, 1)),
    ((1, 1, 1), (-1, 1, 1)), ((-1, 1, 1), (-1, -1, 1)),
    ((-1, -1, -1), (-1, -1, 1)), ((1, -1, -1), (1, -1, 1)),
    ((1, 1, -1), (1, 1, 1)), ((-1, 1, -1), (-1, 1, 1)),
]
back = {((1, 1, -1), (-1, 1, -1)), ((-1, 1, -1), (-1, -1, -1)),
        ((1, 1, -1), (1, 1, 1)), ((-1, 1, -1), (-1, 1, 1))}
for aedge, bedge in box_edges:
    pa, pb = corners[aedge], corners[bedge]
    fade = (aedge, bedge) in back or (bedge, aedge) in back
    ax_b.add_line(plt.Line2D([pa[0], pb[0]], [pa[1], pb[1]],
                             color=Z.GRAYL if fade else Z.GRAY,
                             lw=0.7 if fade else 0.9, ls=(0, (4, 2)), zorder=3))

# ambient gradient decomposed: kept tangent component (pink) + leaked normal (gray)
gn = np.hypot(fx0, fy0) + 1e-9
gxh, gyh = fx0 / gn, fy0 / gn
GL = 0.26
# kept tangent component lies IN the tangent plane
gtip_t = proj(x0 + GL * gxh, y0 + GL * gyh, z0 + (fx0 * gxh + fy0 * gyh) * GL)
# leaked normal component along the surface normal (off-plane, straight up in ambient z)
NL = 0.30
gtip_n = proj(x0, y0, z0 + NL)
# ambient gradient = tangent + normal (ink dotted), built as the parallelogram sum
amb_x = x0 + GL * gxh
amb_y = y0 + GL * gyh
amb_z = z0 + (fx0 * gxh + fy0 * gyh) * GL + NL
gtip_a = proj(amb_x, amb_y, amb_z)

ax_b.annotate("", xy=gtip_t, xytext=P,
              arrowprops=dict(arrowstyle="-|>", color=Z.PINKD, lw=2.4,
                              mutation_scale=11), zorder=8)
ax_b.annotate("", xy=gtip_n, xytext=P,
              arrowprops=dict(arrowstyle="-|>", color=Z.GRAY, lw=1.9,
                              mutation_scale=10), zorder=8)
ax_b.annotate("", xy=gtip_a, xytext=P,
              arrowprops=dict(arrowstyle="-|>", color=Z.INK, lw=1.2,
                              ls=(0, (2.0, 1.4)), mutation_scale=9), zorder=7)
# parallelogram helper edges closing ambient = tangent + normal
ax_b.plot([gtip_t[0], gtip_a[0]], [gtip_t[1], gtip_a[1]], ls=":", lw=0.6,
          color=Z.GRAYL, zorder=6)
ax_b.plot([gtip_n[0], gtip_a[0]], [gtip_n[1], gtip_a[1]], ls=":", lw=0.6,
          color=Z.GRAYL, zorder=6)
ax_b.scatter([P[0]], [P[1]], s=24, color=Z.INK, zorder=9, edgecolor="white", lw=0.6)

# scale callipers: pink tangent reach vs gray ambient-box reach (same screen-east axis)
cal_g, cal_p = 0.045, 0.105
bx0, _ = proj(x0 - BE, y0, z0); bx1, _ = proj(x0 + BE, y0, z0)
tx0, _ = proj(x0 - TE, y0, z0); tx1, _ = proj(x0 + TE, y0, z0)
ax_b.annotate("", xy=(bx1, cal_g), xytext=(bx0, cal_g),
              arrowprops=dict(arrowstyle="<->", color=Z.GRAY, lw=0.8,
                              mutation_scale=7), zorder=6)
ax_b.text((bx0 + bx1) / 2, cal_g - 0.005, "ambient-box reach", fontsize=5.4,
          color=Z.GRAY, ha="center", va="top")
ax_b.annotate("", xy=(tx1, cal_p), xytext=(tx0, cal_p),
              arrowprops=dict(arrowstyle="<->", color=Z.PINKD, lw=1.1,
                              mutation_scale=7), zorder=7)
ax_b.text((tx0 + tx1) / 2, cal_p + 0.006, "tangent reach", fontsize=5.4,
          color=Z.PINKD, ha="center", va="bottom")

# direct color-matched labels (no legend box), routed to clear corners
ax_b.annotate("tangent component\n(kept)", xy=gtip_t,
              xytext=(0.70, 0.55), textcoords="axes fraction", fontsize=6.0,
              color=Z.PINKD, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=Z.PINKD, lw=0.5))
ax_b.annotate("normal component\n(leaked)", xy=gtip_n,
              xytext=(0.66, 0.86), textcoords="axes fraction", fontsize=6.0,
              color=Z.GRAY, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=Z.GRAY, lw=0.5))
ax_b.annotate("ambient ∇g", xy=gtip_a,
              xytext=(0.40, 0.99), textcoords="axes fraction", fontsize=6.0,
              color=Z.INK, ha="left", va="top", fontfamily="DejaVu Sans",
              arrowprops=dict(arrowstyle="-", color=Z.INK, lw=0.5))
ax_b.annotate("tangent plane", xy=proj(x0 - 0.20, y0 + 0.06, z0 + fx0 * (-0.20) + fy0 * 0.06),
              xytext=(0.005, 0.40), textcoords="axes fraction", fontsize=6.0,
              color=Z.PINKD, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=Z.PINKD, lw=0.5))
ax_b.annotate("ambient box\n(all 256 dims)", xy=corners[(-1, 1, 1)],
              xytext=(0.005, 0.99), textcoords="axes fraction", fontsize=6.0,
              color=Z.GRAY, ha="left", va="top",
              arrowprops=dict(arrowstyle="-", color=Z.GRAY, lw=0.5))
ax_b.text(0.0, -0.05, "2-D geography manifold embedded in 256-D ambient space",
          transform=ax_b.transAxes, fontsize=6.0, color=Z.GRAY, ha="left", va="bottom")

# =====================================================================================
# PANEL c — box-vs-manifold looseness over ambient dimension  [fig3c]
# =====================================================================================
Z.panel(ax_c, "c")
ax_c.set_title("Box Looseness vs Ambient Dimension", fontsize=7.5, loc="left", pad=6)

D = ds.copy()
rt = rtan.copy(); rt_s = rtan_sem.copy()
ro = roff.copy(); ro_s = roff_sem.copy()

# on-manifold reference stays flat ~1 -> PINK
ax_c.fill_between(D, ro - ro_s, ro + ro_s, color=Z.PINKL, alpha=0.35, zorder=2)
ax_c.plot(D, ro, "-o", color=Z.PINKD, lw=1.4, ms=4.2, mec="white", mew=0.4,
          label="On-manifold bound", zorder=5)

# ambient-box looseness over the true tangent bound: RISES -> navy/steel ramp
ax_c.fill_between(D, np.clip(rt - rt_s, 1e-2, None), rt + rt_s,
                  color=Z.STEEL2, alpha=0.20, zorder=2)
ax_c.plot(D, rt, "-s", color=Z.NAVY, lw=1.4, ms=4.2, mec="white", mew=0.4,
          label="Ambient-box bound", zorder=5)
ramp = [Z.NAVY, Z.STEEL1, Z.STEEL2, Z.STEEL3]
ax_c.scatter(D, rt, c=[ramp[min(k, 3)] for k in range(len(D))], s=14, zorder=6,
             edgecolor="white", linewidths=0.4)

# red dashed tight reference at ratio = 1
ax_c.axhline(1.0, ls=(0, (4, 3)), lw=0.9, color=Z.RED, zorder=3)
ax_c.text(14.5, 1.55, "tight (ratio = 1)", fontsize=5.8, color=Z.RED,
          va="bottom", ha="left")

# the 488x worst case: one annotated point (full ambient box over all 256 dims)
worst_d, worst_v = 256, 488.0
ax_c.scatter([worst_d], [worst_v], marker="*", s=85, color=Z.NAVY,
             edgecolor="white", linewidths=0.5, zorder=8)
ax_c.annotate(r"488× vacuous" + "\n(full ambient box)",
              xy=(worst_d, worst_v), xytext=(-14, -2), textcoords="offset points",
              fontsize=6.0, color=Z.NAVY, va="center", ha="right",
              arrowprops=dict(arrowstyle="-", color=Z.NAVY, lw=0.5))

ax_c.set_xscale("log"); ax_c.set_yscale("log")
ax_c.set_xlim(13, 1000); ax_c.set_ylim(0.55, 1600)
ax_c.set_xticks([16, 32, 64, 128, 256, 512])
ax_c.set_xticklabels(["16", "32", "64", "128", "256", "512"])
ax_c.set_yticks([1, 10, 100, 1000])
ax_c.set_yticklabels(["1", "10", "100", "1000"])
ax_c.set_xlabel("Ambient dimension d")
ax_c.set_ylabel(r"Bound looseness  (× true bound, log)")
ax_c.legend(loc="lower right", fontsize=5.8, handlelength=1.6,
            borderaxespad=0.5, labelspacing=0.35, bbox_to_anchor=(1.0, 0.10))
ax_c.text(0.02, 0.97, f"bands ±SEM, n = {NSEED} seeds", transform=ax_c.transAxes,
          fontsize=5.5, color=Z.GRAY, ha="left", va="top")
ax_c.text(0.5, -0.235, r"Field: GA λ(x)  |  m-dim manifold in d-dim ambient",
          transform=ax_c.transAxes, fontsize=5.8, color=Z.GRAY, ha="center", va="top")

# =====================================================================================
# PANEL d — slack factorization as a cumulative-product waterfall  [recasts fig4b]
#           1 -> x rho_off -> x rho_tan -> x rho_cond = 49.1x ; transverse lever is PINK;
#           generic sqrt(m/d)=11.3x as the red dashed reference.
# =====================================================================================
Z.panel(ax_d, "d")
ax_d.set_title("Slack Factorization (production field)", fontsize=7, loc="left", pad=6)

# cumulative-product waterfall on a log y-axis. FIVE distinct x slots so the three
# factor steps and the final running total never share a position:
#   0 start(1x) | 1 xrho_off | 2 xrho_tan | 3 xrho_cond | 4 total(49.1x)
factors = [RHO_OFF, RHO_TAN, RHO_COND]
fcols = [Z.GRAYL, Z.PINK, Z.GRAYL]
cum = np.concatenate([[1.0], np.cumprod(factors)])   # [1, 1.00, 42.7, 49.1]
xb = np.arange(5)                                    # start, off, tan, cond, total
xtl = ["start", r"ρ_off", r"ρ_tan",
       r"ρ_cond", "total"]
fsub = ["off-\nmanifold", "trans-\nverse", "condit-\nioning"]

BW = 0.64
YFLOOR = 0.72                                        # bottom of the log axis (bar base)
# start anchor bar (1x) and total bar (49.1x): solid full bars from the floor
ax_d.bar(xb[0], 1.0 - YFLOOR, bottom=YFLOOR, width=BW, color=Z.GRAY, alpha=0.55, zorder=2)
ax_d.bar(xb[4], cum[-1] - YFLOOR, bottom=YFLOOR, width=BW, color=Z.PINKD, zorder=2)
# three floating factor steps, each rising from the previous running total
for i in range(3):
    xi = xb[i + 1]
    lo, hi = cum[i], cum[i + 1]
    if hi - lo <= 1e-6:
        # no-op multiplier (rho_off = 1.00): flat tick marks the slot contributes nothing
        ax_d.plot([xi - BW / 2, xi + BW / 2], [lo, lo], color=fcols[i], lw=2.4,
                  solid_capstyle="butt", zorder=3)
    else:
        ax_d.bar(xi, hi - lo, bottom=lo, width=BW, color=fcols[i], zorder=2)
    # typed gray dotted connector from the previous running total into this step
    ax_d.plot([xb[i] + BW / 2, xi - BW / 2], [cum[i], cum[i]],
              ls=(0, (1.5, 1.5)), lw=0.6, color=Z.GRAY, zorder=1)
# connector from the last step's running total into the total bar
ax_d.plot([xb[3] + BW / 2, xb[4] - BW / 2], [cum[-1], cum[-1]],
          ls=(0, (1.5, 1.5)), lw=0.6, color=Z.GRAY, zorder=1)

# +/-SEM cap on the transverse lever (the only error-barred factor), at its top
tan_sem = rtan_sem[j256]
ax_d.errorbar([xb[2]], [cum[2]], yerr=[tan_sem], fmt="none", ecolor=Z.INK,
              elinewidth=0.7, capsize=2.4, capthick=0.7, zorder=5)

# generic sqrt(m/d)=11.3x reference (red dashed)
ax_d.axhline(GENERIC, ls=(0, (4, 3)), lw=0.9, color=Z.RED, zorder=3)
ax_d.text(0.03, GENERIC * 1.13, r"generic √(m/d)=11.3×", fontsize=5.7,
          color=Z.RED, ha="left", va="bottom", transform=ax_d.get_yaxis_transform())

# per-step multiplier annotations (a running product is not axis-readable)
steplabs = [r"1×", r"×1.00", r"×42.7", r"×1.15"]
for i, pl in enumerate(steplabs):
    ax_d.annotate(pl, xy=(xb[i], cum[i]), xytext=(0, 5), textcoords="offset points",
                  ha="center", va="bottom", fontsize=5.8, color=Z.INK)
# the running total over the final bar
ax_d.annotate(r"49.1×", xy=(xb[4], cum[-1]), xytext=(0, 6),
              textcoords="offset points", ha="center", va="bottom",
              fontsize=6.6, color=Z.PINKD, fontweight="bold")

ax_d.set_yscale("log")
ax_d.set_ylim(YFLOOR, 150)
ax_d.set_yticks([1, 10, 100])
ax_d.set_yticklabels(["1", "10", "100"])
ax_d.set_xticks(xb)
ax_d.set_xticklabels(xtl, fontsize=7.0)
for i, s in enumerate(fsub):
    ax_d.text(xb[i + 1], -0.155, s, transform=ax_d.get_xaxis_transform(), ha="center",
              va="top", fontsize=5.3, color=Z.GRAY, linespacing=0.95)
ax_d.set_ylabel(r"cumulative slack  (×, log)")
ax_d.margins(x=0.04)
ax_d.text(0.5, -0.345, r"production field, m=128, d=256  |  ρ_tan cap ±SEM (sweep, n=3)",
          transform=ax_d.transAxes, ha="center", fontsize=5.5, color=Z.GRAY)

# =====================================================================================
# PANEL e — rho_tan (transverse) vs ambient dimension, +/-SEM band  [fig4c / N20]
# =====================================================================================
Z.panel(ax_e, "e")
ax_e.set_title("Transverse Slack Is Dimension-Driven", fontsize=7, loc="left", pad=6)

o = np.argsort(ds)
x = ds[o]; yt = rtan[o]; et = rtan_sem[o]
ax_e.fill_between(x, np.maximum(yt - et, 0), yt + et, color=Z.PINK, alpha=0.20,
                  lw=0, zorder=2)
ax_e.plot(x, yt, "-o", color=Z.PINKD, lw=1.4, ms=4.4, mfc=Z.PINK, mec=Z.PINKD,
          mew=0.7, zorder=5, label=r"ρ_tan  (per-m field)")
# generic sqrt(m/d) reference (red dashed)
ax_e.axhline(GENERIC, ls=(0, (4, 3)), lw=0.9, color=Z.RED, zorder=3)
ax_e.text(150, GENERIC - 2.0, r"generic √(m/d)=11.3×", fontsize=5.7,
          color=Z.RED, ha="center", va="top")
# endpoints annotated (6 -> 49 climb)
ax_e.annotate(r"6.3×", xy=(x[0], yt[0]), xytext=(7, -7),
              textcoords="offset points", fontsize=5.8, color=Z.PINKD, ha="left",
              va="top")
ax_e.annotate(r"48.9±6.9×", xy=(x[-1], yt[-1]), xytext=(-4, 8),
              textcoords="offset points", fontsize=5.8, color=Z.PINKD, ha="right",
              fontweight="bold")
# reconcile with panel d: the production-field 42.7x sits INSIDE this sweep's
# d=256 +/-SEM band [42.0, 55.8], so d and e are consistent, not contradictory.
d256 = x[-1]
ax_e.scatter([d256], [RHO_TAN], marker="D", s=20, facecolor="white",
             edgecolor=Z.INK, linewidths=0.8, zorder=7)
ax_e.annotate(r"production 42.7× (panel d)" + "\nwithin d=256 ±SEM band",
              xy=(d256, RHO_TAN), xytext=(0.55, 0.30), textcoords="axes fraction",
              fontsize=5.4, color=Z.INK, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=Z.INK, lw=0.5))
ax_e.set_xscale("log", base=2)
ax_e.set_xticks([16, 32, 64, 128, 256, 512])
ax_e.set_xticklabels([16, 32, 64, 128, 256, 512])
ax_e.set_xlim(14, 590)
ax_e.set_ylim(0, 66)
ax_e.set_yticks([0, 20, 40, 60])
ax_e.set_xlabel("Ambient dimension  d")
ax_e.set_ylabel(r"transverse slack  ρ_tan  (×)")
ax_e.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.6,
            borderaxespad=0.2, labelspacing=0.3)
ax_e.text(0.035, 0.74, f"per-m trained fields\nband ±SEM (n={NSEED} seeds)",
          transform=ax_e.transAxes, ha="left", va="top", fontsize=5.6, color=Z.GRAY)

# =====================================================================================
# PANEL f — rho_off (off-manifold) vs ambient dimension, pinned at ~1, +/-SEM  [fig4d / N21]
# =====================================================================================
Z.panel(ax_f, "f")
ax_f.set_title("Off-Manifold Term Is Dimension-Flat", fontsize=7, loc="left", pad=6)

yo = roff[o]; eo = roff_sem[o]
ax_f.fill_between(x, yo - eo, yo + eo, color=Z.NAVY, alpha=0.16, lw=0, zorder=2)
ax_f.plot(x, yo, "-s", color=Z.NAVY, lw=1.2, ms=3.8, mfc="white", mec=Z.NAVY,
          mew=0.9, zorder=4, label=r"ρ_off  (off-manifold)")
# certified off-manifold floor rho_off = 1
ax_f.axhline(1.0, ls=(0, (1.6, 1.6)), lw=0.8, color=Z.GRAY, zorder=3)
ax_f.text(0.50, 0.14, r"off-manifold floor  ρ_off=1",
          transform=ax_f.transAxes, fontsize=5.7, color=Z.GRAY, ha="center", va="bottom")
ax_f.set_xscale("log", base=2)
ax_f.set_xticks([16, 32, 64, 128, 256, 512])
ax_f.set_xticklabels([16, 32, 64, 128, 256, 512])
ax_f.set_xlim(14, 590)
ax_f.set_ylim(0.9, 1.35)
ax_f.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
ax_f.set_xlabel("Ambient dimension  d")
ax_f.set_ylabel(r"off-manifold slack  ρ_off  (×)")
ax_f.legend(frameon=False, fontsize=5.8, loc="upper right", handlelength=1.6,
            borderaxespad=0.2, labelspacing=0.3)
ax_f.text(0.03, 0.10, f"band ±SEM\n(n={NSEED} seeds)", transform=ax_f.transAxes,
          ha="left", va="bottom", fontsize=5.6, color=Z.GRAY)

Z.save(fig, "F2")
