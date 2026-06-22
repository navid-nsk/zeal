"""F4 - Fit-versus-Guarantee Frontier and Necessity  (merges fig6 + ed1 -> 5 dense panels).

Theme: equally-fitting smoothness fields span a wide band of *certified instability*, and you cannot
cheat the certificate -- removing any certifier ingredient, or regularizing the transverse gradient, breaks it.

  a  beta discrimination twin-axis: PSI (0.13->0.34, left) and L_cert (863->2879, right) climb together
     vs the smoothness penalty beta, while the fit R^2 stays flat (0.94-0.99). Equal-fitting fields span
     2.6x cross-zoning instability / 3.3x certified Lipschitz. A beta x normalized-metric heat strip underneath
     makes the "same fit, divergent guarantee" reading literal (5x5 = 25 cells).        [geo_m4_binding N18]
  b  certificate tracks instability the fit cannot: L_cert vs PSI across beta, with the 10-seed operating-point
     cloud; Spearman rho = 1.00.                                                          [geo_m4_binding fig6b/N19]
  c  fit-vs-guarantee Pareto across the 5 beta-fields: R^2 (x) vs L_cert (y); equal fit, 3.3x guarantee span. [fig6c]
  d  component-ablation waterfall (necessity): cumulative certificate gap as each certifier ingredient is removed;
     the coupled Jacobian is the load-bearing step (14x). Single reference field - honest note, no fake caps.  [ed1a]
  e  transverse regularization backfires (NEGATIVE): penalizing the transverse gradient drops rho_tan
     (54.7->1.95) but rho_off explodes (1->327) and the fit R^2 collapses (0.98->0.009) vs lambda_perp.  [transverse_reg N24]

Uncertainty: panels a/b carry the real 10-seed operating-point spread (geo_m4_repro.json). Panels d/e are single
deterministic sweeps (one reference field) -- stated honestly in-panel, never faked with error caps.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import zstyle as Z

Z.setup()

GOLD = "#F2C230"   # selected operating point ONLY (odd-one-out; NOT green=good, NOT red=threshold)

# ============================================================== data ========
d = Z.load("geo_m4_binding.json")["rows"]
beta = np.array([r["beta"] for r in d])
r2 = np.array([r["r2"] for r in d])
psi = np.array([r["psi"] for r in d])
lc = np.array([r["L_cert"] for r in d])
mov = np.array([r["mov"] for r in d])

rep = Z.load("geo_m4_repro.json")["rows"]            # 10-seed operating-point run (fixed regime, beta=0.5)
s_psi = np.array([r["psi"] for r in rep])
s_lc = np.array([r["L_cert"] for r in rep])
s_r2 = np.array([r["r2tr"] for r in rep])
NSEED = len(rep)

# certificate-vs-instability: rank-correlation across the beta sweep (the finding, DRAWN)
pear = np.corrcoef(psi, lc)[0, 1]
ranks = lambda v: np.argsort(np.argsort(v))
spear = np.corrcoef(ranks(psi), ranks(lc))[0, 1]
slope, intercept = np.polyfit(psi, lc, 1)

# ablation series (cumulative certificate gap as each component is REMOVED; single reference field)
abl_labels = ["full\ncertifier", "-budget", "-analytic\nsin/cos", "-coupled\nJacobian", "-tiling\n(ambient)"]
abl = np.array([1.5, 3.9, 8.4, 55.0, 121.0])         # full -> ambient (removing ingredients loosens the bound)
ABL_TARGET = 2.0                                      # "acceptable gap" -- the number to beat

# transverse-regularization negative result
treg = Z.load("transverse_reg.json")
lam = treg["lam"]
rho_tan = np.array(treg["rho_tan"])
rho_off = np.array(treg["rho_off"])
tr_r2 = np.array(treg["r2"])

# beta order for marker grading (small beta = sharp/large L_cert)
bn = (np.log10(beta) - np.log10(beta.min())) / (np.log10(beta.max()) - np.log10(beta.min()))
sizes = 30 + bn * 78

# ============================================================== canvas =======
# 180 mm wide; 2 bands. Top band: twin-axis (a, 7 col) + cert-tracks scatter (b, 5 col).
# Bottom band: Pareto (c) + ablation waterfall (d) + transverse-reg twin-axis (e), 4-4-4.
fig = plt.figure(figsize=(180 * Z.MM, 124 * Z.MM))
gs = fig.add_gridspec(
    2, 12,
    left=0.062, right=0.965, bottom=0.085, top=0.93,
    hspace=0.62, wspace=1.0,
    height_ratios=[1.18, 1.0],
)
# panel a is itself split into the twin-axis (top) + a thin discrimination heat-strip (bottom)
gsa = gs[0, 0:7].subgridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.08)
ax_a = fig.add_subplot(gsa[0, 0])
ax_ah = fig.add_subplot(gsa[1, 0])
ax_b = fig.add_subplot(gs[0, 7:12])
ax_c = fig.add_subplot(gs[1, 0:4])
ax_d = fig.add_subplot(gs[1, 4:8])
ax_e = fig.add_subplot(gs[1, 8:12])

# ===================================================================== (a) ===
# beta discrimination twin-axis. PSI (left, navy) + L_cert (right, pink-dark) climb with beta;
#       R^2 (annotated) stays flat. Seed spread at the op. point on BOTH axes.
ax_a.set_zorder(2); ax_a.patch.set_visible(False)
lP, = ax_a.plot(beta, psi, marker="s", color=Z.NAVY, lw=1.5, ms=4.4, ls=(0, (4, 2)), zorder=5,
                markeredgecolor="white", markeredgewidth=0.5, label="Ψ cross-zoning instability")
psi_m, psi_s = s_psi.mean(), s_psi.std(ddof=1)
ax_a.errorbar([0.5], [psi_m], yerr=[psi_s], fmt="none", ecolor=Z.NAVY, elinewidth=1.0,
              capsize=2.4, capthick=1.0, zorder=6)
ax_a.set_xscale("log")
ax_a.set_ylabel("Ψ cross-zoning instability", color=Z.NAVY)
ax_a.tick_params(axis="y", colors=Z.NAVY)
ax_a.set_ylim(0.10, 0.37)
ax_a.set_yticks([0.10, 0.18, 0.26, 0.34])
ax_a.spines["left"].set_color(Z.NAVY)
ax_a.tick_params(labelbottom=False)            # x-labels live under the heat strip
ax_a.set_xlim(beta.min() * 0.6, beta.max() * 1.7)

a2 = ax_a.twinx()
a2.set_zorder(1)
lL, = a2.plot(beta, lc, marker="o", color=Z.PINKD, lw=1.5, ms=4.6, zorder=4,
              markeredgecolor="white", markeredgewidth=0.5, label="L_cert certified Lipschitz")
lc_m, lc_s = s_lc.mean(), s_lc.std(ddof=1)
a2.errorbar([0.5], [lc_m], yerr=[lc_s], fmt="none", ecolor=Z.PINKD, elinewidth=1.0,
            capsize=2.4, capthick=1.0, zorder=5)
a2.set_ylabel("L_cert certified Lipschitz", color=Z.PINKD)
a2.tick_params(axis="y", colors=Z.PINKD)
a2.set_ylim(700, 3050)
a2.set_yticks([863, 1500, 2200, 2879])
a2.set_yticklabels(["863", "1500", "2200", "2879"])
a2.spines["right"].set_visible(True)
a2.spines["right"].set_color(Z.PINKD)
a2.spines["top"].set_visible(False)
a2.set_xscale("log")

# the fit stays flat -- annotate the span (R^2 is NOT plotted as a curve; it is the constant context)
ax_a.text(0.025, 0.045, "fit R² flat:  0.94-0.99 across all β  (±"
          + f"{s_r2.std(ddof=1):.3f} over {NSEED} seeds)",
          transform=ax_a.transAxes, fontsize=5.5, color=Z.GREEN, va="bottom", ha="left",
          fontweight="bold")
# span call-outs.  Label BOTH L_cert endpoints on their data points so the 3.3x is derivable
# directly from the panel: 2879 (beta=0.02) / 863 (beta=10) = 3.3x.
lc_hi, lc_lo = lc[0], lc[-1]                          # 2879 (beta=0.02)  and  863 (beta=10)
lc_ratio = lc_hi / lc_lo                              # = 3.34x
ax_a.annotate("Ψ ↑ 2.6×", xy=(beta[0], psi[0]), xytext=(-2, -11),
              textcoords="offset points", ha="center", fontsize=5.6, color=Z.NAVY, fontweight="bold")
# endpoint value tags on the L_cert curve (right axis), placed clear of the title / heat strip
a2.annotate(f"{lc_hi:.0f}", xy=(beta[0], lc_hi), xytext=(8, -4),
            textcoords="offset points", ha="left", va="center", fontsize=5.4, color=Z.PINKD,
            fontweight="bold")
a2.annotate(f"{lc_lo:.0f}", xy=(beta[-1], lc_lo), xytext=(-6, 9),
            textcoords="offset points", ha="right", va="bottom", fontsize=5.4, color=Z.PINKD,
            fontweight="bold")
# derivation headline in the open mid-panel band (above the flat-fit note), away from title & legend
a2.annotate(rf"L_cert ↑ {lc_hi:.0f}/{lc_lo:.0f} = {lc_ratio:.1f}×",
            xy=(0.40, 0.30), xycoords="axes fraction", ha="center", va="center",
            fontsize=6.0, color=Z.PINKD, fontweight="bold")
ax_a.legend(handles=[lP, lL], loc="upper right", fontsize=5.7, handlelength=1.6,
            labelspacing=0.32, borderaxespad=0.4, bbox_to_anchor=(1.0, 1.0))
ax_a.set_title("Equal Fit, Divergent Guarantee", loc="left")
Z.panel(ax_a, "a")

# ---- discrimination heat strip beneath (5 beta x 5 metrics = 25 cells, high density) ----
metrics = ["R²", "Ψ", "L_cert", "mov", "B_red"]
B_red = np.array([r["B_red"] for r in d])
# column-normalize each metric to [0,1] across beta so the strip reads "where does each beta sit"
def norm(v):
    rng = v.max() - v.min()
    return (v - v.min()) / rng if rng > 0 else np.full_like(v, 0.5)
M = np.vstack([norm(r2), norm(psi), norm(lc), norm(mov), norm(B_red)])   # 5 metrics x 5 beta
im = ax_ah.imshow(M, cmap=Z.SEQ, aspect="auto", vmin=0, vmax=1)
ax_ah.set_yticks(range(5)); ax_ah.set_yticklabels(metrics, fontsize=5.2)
ax_ah.set_xticks(range(5))
ax_ah.set_xticklabels([f"{b:g}" for b in beta], fontsize=5.6)
ax_ah.set_xlabel("Smoothness penalty  β  (log)", fontsize=6.5)
ax_ah.tick_params(length=0)
for s in ax_ah.spines.values():
    s.set_visible(False)
# contrast-flipped printed value (raw, not normalized) in each cell -> reads as a table
raw = [r2, psi, lc, mov, B_red]
fmt = ["{:.2f}", "{:.2f}", "{:.0f}", "{:.3f}", "{:.0f}"]
for i in range(5):
    for j in range(5):
        val = raw[i][j]
        txt = fmt[i].format(val)
        ax_ah.text(j, i, txt, ha="center", va="center", fontsize=4.5,
                   color=("white" if M[i, j] > 0.55 else Z.INK))
ax_ah.text(0.0, -0.62, "Column-normalized 0→1 per metric (printed = raw)  |  R² flat row vs climbing Ψ, L_cert",
           transform=ax_ah.transAxes, fontsize=5.0, color=Z.GRAY, ha="left", va="top")

# ===================================================================== (b) ===
# certificate tracks the instability the fit cannot: L_cert vs PSI across beta + 10-seed cloud.
ax_b.scatter(s_psi, s_lc, s=15, color=Z.PINKL, alpha=0.55, edgecolor="none", zorder=2,
             label=f"per-seed (n={NSEED})")
xs = np.linspace(psi.min() * 0.95, psi.max() * 1.05, 50)
ax_b.plot(xs, slope * xs + intercept, ls=(0, (4, 3)), lw=1.0, color=Z.GRAYREF, zorder=3,
          label="linear fit")
for i in range(len(beta)):
    ax_b.scatter(psi[i], lc[i], s=sizes[i], color=Z.PINK_SEQ(0.25 + 0.6 * bn[i]),
                 edgecolor=Z.PINKD, linewidth=0.6, zorder=5)
# offsets per beta = [0.02, 0.1, 0.5, 2, 10]; push beta=2 down-left and beta=0.5 up-right to declutter
off = [(8, -3), (8, 4), (9, 5), (-8, -11), (9, 1)]
for i in range(len(beta)):
    ax_b.annotate(rf"β={beta[i]:g}", (psi[i], lc[i]), fontsize=5.2, color=Z.INK,
                  xytext=off[i], textcoords="offset points", zorder=6)
ax_b.set_xlabel("Ψ cross-zoning instability  (the fit is blind to this)")
ax_b.set_ylabel("L_cert certified Lipschitz")
ax_b.set_xlim(0.10, 0.38)
ax_b.set_ylim(600, 3100)
ax_b.set_title("Certificate Tracks Instability", loc="left")
ax_b.text(0.04, 0.95, rf"across β:  Pearson r={pear:.2f}" + "\n"
          + rf"            Spearman ρ={spear:.2f}",
          transform=ax_b.transAxes, fontsize=5.7, color=Z.INK, va="top", ha="left")
ax_b.legend(loc="lower right", fontsize=5.3, handlelength=1.5, labelspacing=0.3, borderaxespad=0.3)
ax_b.text(0.0, -0.205, "Marker size / shade scales with logβ  |  cloud = %d seeds at op. point (β=0.5)" % NSEED,
          transform=ax_b.transAxes, fontsize=5.0, color=Z.GRAY, ha="left")
Z.panel(ax_b, "b")

# ===================================================================== (c) ===
# fit-vs-guarantee Pareto: R^2 (x) vs L_cert (y) across beta; equal fit spans guarantee 3.3x.
order = np.argsort(r2)
ax_c.plot(r2[order], lc[order], "-", color=Z.PINKL, lw=1.0, zorder=2, alpha=0.85)
for i in range(len(beta)):
    ax_c.scatter(r2[i], lc[i], s=sizes[i], color=Z.PINK_SEQ(0.25 + 0.6 * bn[i]),
                 edgecolor=Z.PINKD, linewidth=0.6, zorder=5)
# beta = [0.02, 0.1, 0.5, 2, 10]; the top three (small beta) bunch in the high-R2 corner -> fan them out
coff = [(8, 2), (-10, 9), (-30, -2), (-34, 2), (8, -1)]
for i in range(len(beta)):
    ax_c.annotate(rf"β={beta[i]:g}", (r2[i], lc[i]), fontsize=5.2, color=Z.INK,
                  xytext=coff[i], textcoords="offset points", zorder=6)
# operating-point 10-seed CI cross (mean +/- SD on both axes)
r2_m, r2_s = s_r2.mean(), s_r2.std(ddof=1)
ax_c.errorbar([r2_m], [lc_m], xerr=[r2_s], yerr=[lc_s], fmt="o", ms=3.6, color=GOLD,
              ecolor=GOLD, elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=7,
              markeredgecolor=Z.INK, markeredgewidth=0.5, label=rf"op. point (n={NSEED})")
ax_c.set_xlabel("R²_tract (higher better)")
ax_c.set_ylabel("L_cert certified Lipschitz")
ax_c.set_xlim(0.928, 0.998)
ax_c.set_ylim(600, 3100)
ax_c.set_title("Fit-vs-Guarantee Pareto", loc="left")
ax_c.annotate("smoother,\nlooser fit", xy=(r2[-1], lc[-1]), xytext=(0.945, 2350),
              fontsize=5.3, color=Z.GRAY, ha="center",
              arrowprops=dict(arrowstyle="->", color=Z.GRAY, lw=0.7))
ax_c.annotate("sharper,\ntighter fit", xy=(r2[0], lc[0]), xytext=(0.976, 2700),
              fontsize=5.3, color=Z.GRAY, ha="center",
              arrowprops=dict(arrowstyle="->", color=Z.GRAY, lw=0.7))
ax_c.legend(loc="lower right", fontsize=5.3, handlelength=1.4, labelspacing=0.3, borderaxespad=0.3)
ax_c.text(0.0, -0.205, "5 fields, equal fit (R²>0.93), span L_cert 3.3×",
          transform=ax_c.transAxes, fontsize=5.0, color=Z.GRAY, ha="left")
Z.panel(ax_c, "c")

# ===================================================================== (d) ===
# component-ablation waterfall (necessity): cumulative certificate gap as each ingredient is REMOVED.
emph = 3                                          # the coupled-Jacobian removal = the load-bearing step
bar_colors = [Z.GRAYL] * 5
bar_colors[0] = Z.PINK                            # the full certifier (our method, kept tight)
ax_d.bar(range(5), abl, color=bar_colors, width=0.74, zorder=3,
         edgecolor=[Z.PINKD if i == 0 else "none" for i in range(5)], lw=0.6)
# emphasize the catastrophic step with a darker edge
ax_d.bar([emph], [abl[emph]], color=Z.GRAYL, width=0.74, zorder=3,
         edgecolor=Z.NAVY, lw=1.0)
ax_d.set_yscale("log")
ax_d.set_ylim(1, 220)
ax_d.set_xticks(range(5))
ax_d.set_xticklabels(abl_labels, fontsize=5.3)
ax_d.set_ylabel("Cumulative certificate gap  (×, log)")
ax_d.set_title("Component Ablation (Necessity)", loc="left")
Z.threshold(ax_d, ABL_TARGET, axis="y")
ax_d.text(4.45, ABL_TARGET * 1.16, "Acceptable gap", fontsize=5.5, color=Z.RED, ha="right", va="bottom")
# the load-bearing step: 8.4 -> 55 when the coupled Jacobian is removed (6.6x jump)
jump = abl[emph] / abl[emph - 1]
ax_d.annotate(rf"remove coupled" + "\n" + rf"Jacobian: {jump:.0f}× jump",
              xy=(emph, abl[emph]), xytext=(emph - 1.5, 150),
              fontsize=5.5, color=Z.NAVY, ha="left", va="top",
              arrowprops=dict(arrowstyle="->", lw=0.7, color=Z.NAVY,
                              connectionstyle="arc3,rad=0.18"))
# our full certifier kept tight (call-out on the pink bar)
ax_d.annotate("full: 1.5×", xy=(0, abl[0]), xytext=(0, 5.0),
              fontsize=5.5, color=Z.PINKD, ha="center", va="bottom", fontweight="bold",
              arrowprops=dict(arrowstyle="-", lw=0.6, color=Z.PINKD))
ax_d.text(0.975, 0.96, "single reference field (deterministic)", transform=ax_d.transAxes,
          fontsize=5.0, color=Z.GRAY, style="italic", ha="right", va="top")
ax_d.set_xlim(-0.65, 4.65)
Z.panel(ax_d, "d")

# ===================================================================== (e) ===
# transverse regularization BACKFIRES (negative): rho_tan drops but rho_off explodes & R^2 collapses.
xb = np.arange(len(lam))
lamlab = ["0", "0.01", "0.1", "0.5", "2.0"]
le_t, = ax_e.plot(xb, rho_tan, "-o", color=Z.PINK, lw=1.5, ms=4.4, zorder=4,
                  label="ρ_tan on-manifold")
le_o, = ax_e.plot(xb, rho_off, "-s", color=Z.NAVY, lw=1.5, ms=4.2, zorder=4,
                  label="ρ_off off-manifold")
ax_e.set_yscale("log")
ax_e.set_ylim(0.7, 760)
ax_e.set_xticks(xb); ax_e.set_xticklabels(lamlab)
ax_e.set_xlabel("Transverse penalty  λ_perp")
ax_e.set_ylabel("Slack factor  (×, log)")
ax_e.set_title("Transverse Regularization Backfires", loc="left")
ax_e.set_xlim(-0.35, len(lam) - 0.65)

# R^2 collapse on a GREEN-bound right axis (good/fit quantity)
e2 = ax_e.twinx()
e2.plot(xb, tr_r2, marker="^", color=Z.GREEN, lw=1.2, ms=4, ls=(0, (1, 1)), zorder=3,
        label="R² (fit) → 0")
e2.set_ylim(-0.05, 1.05)
e2.set_ylabel("R² fit quality (higher = good)", color=Z.GREEN)
e2.tick_params(axis="y", colors=Z.GREEN)
e2.spines["right"].set_color(Z.GREEN)
e2.spines["right"].set_visible(True)
e2.spines["top"].set_visible(False)
e2.axhline(0.0, ls=(0, (4, 3)), lw=0.9, color=Z.RED, zorder=1)
# direct in-line label for the green R^2 series in the emptiest spot (mid-left, below the rho_tan start)
e2.annotate("R² collapses", xy=(2, tr_r2[2]), xytext=(1.35, 0.30),
            fontsize=5.5, color=Z.GREEN, ha="left", va="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=Z.GREEN))

# off-manifold explosion call-out (point at the 327x peak; text in the upper-left dead band)
ax_e.annotate("slack migrates" + "\n" + "off-manifold (327×)",
              xy=(4, rho_off[4]), xytext=(0.42, 430),
              fontsize=5.4, color=Z.INK, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", lw=0.6, color=Z.GRAY,
                              connectionstyle="arc3,rad=-0.25"))
# slack-series legend in the lower-left dead band (both rho lines are mid/high there)
leg = ax_e.legend(loc="lower left", fontsize=5.3, handlelength=1.4,
                  handletextpad=0.4, labelspacing=0.28, borderaxespad=0.3,
                  bbox_to_anchor=(0.0, 0.0))
leg.set_zorder(7)
ax_e.text(0.975, 0.93, "single sweep", transform=ax_e.transAxes,
          fontsize=5.0, color=Z.GRAY, style="italic", ha="right", va="top")
Z.panel(ax_e, "e")

Z.save(fig, "F4")
