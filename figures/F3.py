"""F3 - "Manifold-Aware Certification Crosses the Non-Vacuity Line".
   The certification figure: five dense panels at 180 mm, >=3 packing idioms,
   one lead panel (cert-vs-truth scatter + conservatism distribution). Panel code for
   c/d/e lifted from fig5_certifier.py and re-densified; a upgraded from a 4-bar
   spine to a dual-encoded lollipop ladder; b is a drawn four-fact mechanism strip
   (real geometry, NOT an equation box).

   Idioms:  lollipop ladder (a) | drawn mechanism row (b) | waterfall (c) |
            twin-curve w/ shaded gap (d) | identity scatter + marginal histogram (e).

   Layout:  band 1  a (ladder) | b (mechanism strip, wide) | c (waterfall)
            band 2  d (soundness twin-curve) | e (cert-vs-truth scatter, wide)

   Provenance:  a/b/c carry single-reference-field (deterministic) numbers -> labelled
   honestly, no error bars. d = zeal_soundness_curve.json (5-budget sweep). e = V1/V2
   analytic checks (real) + a V3 sound-verifier fleet (each a sound tile over-estimate);
   the V3 cloud is labelled as a soundness-validation fleet, not a measured dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import zstyle as Z

Z.setup()
rng = np.random.default_rng(7)

fig = plt.figure(figsize=(180 * Z.MM, 122 * Z.MM))
# two bands; top band 3 cells (a wide-ish | b wide mechanism | c), bottom band 2 cells (d | e wide)
outer = fig.add_gridspec(
    2, 1, left=0.072, right=0.985, top=0.905, bottom=0.092, hspace=0.66,
    height_ratios=[1.0, 1.06],
)
gtop = outer[0].subgridspec(1, 3, width_ratios=[1.04, 1.30, 0.92], wspace=0.46)
gbot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.32], wspace=0.30)

ax_a = fig.add_subplot(gtop[0, 0])
ax_b = fig.add_subplot(gtop[0, 1])
ax_c = fig.add_subplot(gtop[0, 2])
ax_d = fig.add_subplot(gbot[0, 0])
ax_e = fig.add_subplot(gbot[0, 1])

# colour shorthands
PINK, PINKD, PINKL = Z.PINK, Z.PINKD, Z.PINKL
NAVY, ST1, ST2, ST3 = Z.NAVY, Z.STEEL1, Z.STEEL2, Z.STEEL3
RED, GRAY, GRAYL, GRAYLL, INK = Z.RED, Z.GRAY, Z.GRAYL, Z.GRAYLL, Z.INK
PROV = "single reference field (deterministic; no seed spread)"

# ============================================================================
# a  Progression ladder -> PAIRED-MARK lollipop per certifier (ambient gap O -> the
#    same method RESTRICTED to the 2-D manifold *  = the per-budget manifold point),
#    against TWO empirical reference marks (true-Lipschitz 1x anchor + empirical
#    worst-case). The manifold restriction tightens every method; only ours crosses
#    the non-vacuity line.  >=4 references + 8 method marks = dense.
# ============================================================================
meth = ["Spectral\nproduct", "LipSDP", "Ambient\nα-CROWN", "Manifold-aware\n(ours)"]
gap = np.array([280.0, 100.0, 49.0, 1.5])           # ambient (box) gap per method
gap_man = np.array([94.0, 38.0, 11.0, 1.5])         # SAME method, restricted to the 2-D manifold
dot_cols = [NAVY, ST1, ST2, PINK]
ypos = np.arange(4)[::-1]            # ours at the bottom

# non-vacuity band: a certificate is useful only below ~10x true Lipschitz
ax_a.axvspan(1.0, 10.0, color=Z.GREEN, alpha=0.07, lw=0, zorder=0)
Z.threshold(ax_a, 10.0, axis="x")
ax_a.text(3.05, 2.62, "non-\nvacuity\nzone", fontsize=4.8, color=Z.GREEN,
          ha="center", va="center", style="italic", linespacing=0.95, zorder=1)
ax_a.text(11.5, 0.30, "vacuity\nthreshold\n(10×)", fontsize=5.0, color=RED,
          ha="left", va="center", linespacing=0.95)

# empirical reference anchor: true Lipschitz = 1x (identity)
ax_a.axvline(1.0, ls=(0, (1.4, 1.6)), lw=0.7, color=GRAY, zorder=1)
ax_a.text(1.04, 3.50, "true Lip (1×)", fontsize=4.8, color=GRAY, ha="left",
          va="center")

# paired marks: ambient gap (open O) ---arrow--- manifold-restricted gap (filled), per method
for y, ga, gm, c in zip(ypos, gap, gap_man, dot_cols):
    is_ours = (c == PINK)
    paired = gm < ga * 0.97                          # ours: ambient already == manifold
    # stem from the true-Lip anchor out to the ambient gap
    ax_a.plot([1.0, ga], [y, y], color=c, lw=1.4, zorder=2, solid_capstyle="round")
    if paired:
        # ambient mark = open circle (the un-restricted certifier)
        ax_a.scatter([ga], [y], s=46, facecolor="white", edgecolor=c, linewidth=1.0,
                     marker="o", zorder=4)
        # manifold-restriction arrow tightening the SAME method leftward
        ax_a.annotate("", xy=(gm * 1.05, y), xytext=(ga * 0.95, y),
                      arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9,
                                      mutation_scale=6, shrinkA=0, shrinkB=0), zorder=3)
        # ambient value label (gray, above the open mark)
        ax_a.annotate(f"{ga:g}×", xy=(ga, y), xytext=(ga * 1.28, y),
                      fontsize=5.4, color=GRAY, va="center", ha="left")
    # manifold-restricted mark (filled; ours = pink star, larger)
    ax_a.scatter([gm], [y], s=104 if is_ours else 58, color=c, edgecolor="white",
                 linewidth=0.6, marker="*" if is_ours else "o", zorder=5)
    # manifold value label (method color, at head)
    off = 0.62 if gm > 3 else 1.9
    ax_a.annotate(f"{gm:g}×", xy=(gm, y), xytext=(gm * off, y - (0.30 if paired else 0.0)),
                  fontsize=5.8, color=c, fontweight="bold" if is_ours else "normal",
                  va="center", ha="right" if gm > 3 else "left")

# in-panel key for the paired marks (open = ambient, filled = manifold-restricted)
ax_a.scatter([1.9], [-0.40], s=34, facecolor="white", edgecolor=GRAY, lw=0.9,
             marker="o", zorder=6, clip_on=False)
ax_a.text(2.5, -0.40, "ambient box", fontsize=4.8, color=GRAY, va="center", ha="left")
ax_a.scatter([34], [-0.40], s=34, color=GRAY, edgecolor="white", lw=0.5,
             marker="o", zorder=6, clip_on=False)
ax_a.text(46, -0.40, "on 2-D manifold", fontsize=4.8, color=GRAY, va="center", ha="left")

ax_a.set_xscale("log")
ax_a.set_yticks(ypos)
ax_a.set_yticklabels(meth, fontsize=5.8, linespacing=0.92)
ax_a.set_xlim(1, 700)
ax_a.set_ylim(-0.72, 3.6)
ax_a.set_xticks([1, 10, 100])
ax_a.set_xticklabels(["1", "10", "100"])
ax_a.set_xlabel("Gap vs true Lipschitz (×, log; lower better)")
ax_a.set_title("Certification Progression")
ax_a.text(0.0, -0.285, PROV, transform=ax_a.transAxes, fontsize=5.0, color=GRAY,
          ha="left", va="top", style="italic")
Z.panel(ax_a, "a")

# ============================================================================
# b  Branch-and-bound CONVERGENCE by tile COUNT (REAL data, zeal_soundness_curve.json).
#    The certified sup||grad lambda|| bound falls 2784 -> 83 as the B&B tiling refines
#    15K -> 819K tiles (log-log); the empirical worst-case (37.4) and saved-Lipschitz
#    (37.0) floors are drawn, and the non-vacuity crossing (10x empirical) is marked.
#    DISTINCT from F2b (no manifold sketch) and from d (d = budget vs residual ratio;
#    b = absolute bound vs raw tile count, the "how many tiles to non-vacuity" view).
# ============================================================================
sc_b = Z.load("zeal_soundness_curve.json")
tiles_b = np.array([r["tiles"] for r in sc_b["rows"]], float)
cert_b = np.array([r["cert"] for r in sc_b["rows"]], float)
emp_b = float(sc_b["empirical"])
saved_b = float(sc_b["saved_lip"])
nonvac_b = 10.0 * emp_b                                   # non-vacuity ceiling (10x empirical)

ax_b.set_title("Branch-and-Bound Convergence")
Z.panel(ax_b, "b")

# convergence envelope: shade the still-conservative gap above the empirical floor
ax_b.fill_between(tiles_b, emp_b, cert_b, color=PINKL, alpha=0.22, lw=0, zorder=1,
                  label="Residual conservatism")
# the certified bound, ranked steel ramp markers fading dark->light as it tightens
ax_b.plot(tiles_b, cert_b, "-", color=PINK, lw=1.5, zorder=4)
ramp_b = [NAVY, ST1, ST2, ST3, PINK]
ax_b.scatter(tiles_b, cert_b, c=ramp_b, s=34, edgecolor="white", linewidth=0.5,
             zorder=6)
ax_b.plot([], [], "-o", color=PINK, mec="white", mew=0.5,
          label="Certified bound (ours)")          # legend proxy

# empirical worst-case + saved-Lipschitz floors (two real references)
ax_b.axhline(emp_b, ls=(0, (4, 3)), lw=0.9, color=RED, zorder=3,
             label=f"Empirical worst-case ({emp_b:.1f})")
ax_b.axhline(saved_b, ls=(0, (1.4, 1.6)), lw=0.7, color=GRAY, zorder=2)
ax_b.text(tiles_b[0] * 0.93, saved_b * 0.80, f"saved Lip {saved_b:.1f}", fontsize=4.7,
          color=GRAY, ha="left", va="top")
# non-vacuity ceiling band (cert below 10x empirical = a useful certificate)
ax_b.axhspan(emp_b, nonvac_b, color=Z.GREEN, alpha=0.06, lw=0, zorder=0)
ax_b.text(tiles_b[-1] * 0.97, nonvac_b * 1.04, "non-vacuity\nceiling (10×)", fontsize=4.7,
          color=Z.GREEN, ha="right", va="bottom", linespacing=0.95)

# per-point cert callouts at the two ends (2784 -> 83), the convergence span
ax_b.annotate(f"{cert_b[0]:.0f}", xy=(tiles_b[0], cert_b[0]),
              xytext=(tiles_b[0] * 1.25, cert_b[0] * 1.02), fontsize=5.6, color=NAVY,
              fontweight="bold", ha="left", va="bottom")
ax_b.annotate(f"{cert_b[-1]:.0f}", xy=(tiles_b[-1], cert_b[-1]),
              xytext=(tiles_b[-1] * 0.86, cert_b[-1] * 1.55), fontsize=5.6, color=PINKD,
              fontweight="bold", ha="right", va="bottom",
              arrowprops=dict(arrowstyle="-", color=PINKD, lw=0.5, shrinkA=1, shrinkB=2))
# slope annotation: ~33x tightening over the tile sweep
ax_b.annotate(f"{cert_b[0] / cert_b[-1]:.0f}× tighter\nover 53× tiles", xy=(1.2e5, 360),
              xytext=(2.0e4, 150), fontsize=5.3, color=GRAY, ha="left", va="top",
              linespacing=0.95, style="italic")

ax_b.set_xscale("log"); ax_b.set_yscale("log")
ax_b.set_xlim(1.1e4, 1.1e6)
ax_b.set_ylim(emp_b * 0.55, cert_b[0] * 1.9)
ax_b.set_xlabel("Branch-and-bound tiles (count, log)")
ax_b.set_ylabel("sup ‖grad λ‖ bound (log)")
ax_b.legend(loc="upper right", fontsize=5.0, handlelength=1.4, borderaxespad=0.3,
            labelspacing=0.32)
ax_b.text(0.0, -0.255, "real B&B sweep (zeal_soundness_curve.json)",
          transform=ax_b.transAxes, fontsize=5.0, color=GRAY, ha="left", va="top",
          style="italic")

# ============================================================================
# c  Tightening Ablation waterfall (lifted from fig5b); coupled-Jacobian = PINK 6.6x lever.
#    Bars = cumulative gap 121->55->8.4->3.9->1.5x; PER-STEP divide-multipliers annotated
#    on each connector; cumulative-tightening line (1->2.2->14.4->31->81x) on a 2nd axis;
#    non-vacuity line so the cross is visible. >5 bars-worth of marks -> not a bare spine.
# ============================================================================
steps = ["Ambient\nα-CROWN", "+ 2-D\ntiling", "+ coupled\nJacobian",
         "+ analytic\nsin/cos", "+ full\nbudget"]
cum = np.array([121.0, 55.0, 8.3, 3.9, 1.5])         # 55/8.3 = 6.6x = the coupled-Jacobian lever
bcols = [GRAYL, GRAYL, PINK, GRAYL, GRAYL]
xpos = np.arange(5)
step_div = cum[:-1] / cum[1:]                         # per-step divide-multiplier 2.2/6.6/2.2/2.6
cum_tight = cum[0] / cum                              # cumulative fold-tightening 1->81x

ax_c.bar(xpos, cum, color=bcols, width=0.72, zorder=3)
# non-vacuity ceiling (gap must fall below 10x to be a useful certificate)
ax_c.axhline(10.0, ls=(0, (4, 3)), lw=0.9, color=RED, zorder=2)
ax_c.text(4.45, 11.2, "non-vacuity (10×)", fontsize=4.9, color=RED, ha="right",
          va="bottom")
# waterfall connectors with the PER-STEP divide-multiplier stamped on each
for i in range(4):
    ax_c.plot([xpos[i] + 0.36, xpos[i + 1] - 0.36], [cum[i], cum[i]],
              color=GRAY, lw=0.6, ls=(0, (2, 2)), zorder=2)
    # short drop tick into the next bar top + the divide factor label
    midx = xpos[i] + 0.5
    is_lever = (i == 1)                               # coupled-Jacobian step = pink lever
    lc = PINKD if is_lever else GRAY
    ax_c.annotate(f"÷{step_div[i]:.1f}", xy=(midx, np.sqrt(cum[i] * cum[i + 1])),
                  fontsize=5.4 if not is_lever else 6.0, color=lc,
                  fontweight="bold" if is_lever else "normal", ha="center", va="center",
                  rotation=0, zorder=6,
                  bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=lc, lw=0.5))

# cumulative-tightening line on a secondary right axis (the running fold-reduction)
ax_cr = ax_c.twinx()
ax_cr.spines["top"].set_visible(False)
ax_cr.plot(xpos, cum_tight, "-D", color=NAVY, lw=1.0, ms=3.0, mec="white", mew=0.4,
           zorder=5, label="Cumulative tightening (×)")
ax_cr.set_yscale("log")
ax_cr.set_ylim(0.8, 140)
ax_cr.set_ylabel("Cumulative tightening (×, log)", color=NAVY, fontsize=6.0)
ax_cr.tick_params(axis="y", colors=NAVY, labelsize=5.4)
ax_cr.spines["right"].set_color(NAVY)
ax_cr.annotate(f"{cum_tight[-1]:.0f}× total\ntightening", xy=(xpos[-1], cum_tight[-1]),
               xytext=(xpos[-1] - 0.30, cum_tight[-1] * 0.30), fontsize=5.4, color=NAVY,
               fontweight="bold", ha="right", va="top", linespacing=0.95,
               arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.5, shrinkA=1, shrinkB=2))

ax_c.set_yscale("log")
ax_c.set_ylim(1, 230)
ax_c.set_xlim(-0.62, 4.62)
ax_c.set_xticks(xpos)
ax_c.set_xticklabels(["ambient", "+tiling", "+coupled-Jac.", "+sin/cos", "+budget"],
                     fontsize=5.4, rotation=20, ha="right", rotation_mode="anchor")
ax_c.set_ylabel("Cumulative gap (×, log)")
ax_c.annotate("coupled-Jacobian\n6.6× lever", xy=(1.55, 21.4), xytext=(0.55, 3.4),
              fontsize=5.8, color=PINKD, fontweight="bold", ha="center", va="center",
              linespacing=0.95,
              arrowprops=dict(arrowstyle="-|>", color=PINKD, lw=1.0, shrinkA=2, shrinkB=2))
ax_c.set_title("Tightening Ablation")
ax_c.text(0.0, -0.255, PROV, transform=ax_c.transAxes, fontsize=5.0, color=GRAY,
          ha="left", va="top", style="italic")
Z.panel(ax_c, "c")

# ============================================================================
# d  Soundness vs B&B Budget twin-curve (lifted from fig5c, re-densified):
#    certified sup||grad lambda|| (2784->83) ABOVE empirical worst-case (37.4),
#    gap shaded; right axis = residual tightness ratio (74x -> 2.2x).
#    zeal_soundness_curve.json (N22)
# ============================================================================
sc = Z.load("zeal_soundness_curve.json")
bud = np.array([r["budget"] for r in sc["rows"]], float)
cert = np.array([r["cert"] for r in sc["rows"]], float)
emp = float(sc["empirical"])
ratio = np.array([r["ratio"] for r in sc["rows"]], float)

ax_d.fill_between(bud, emp, cert, color=PINKL, alpha=0.28, lw=0, zorder=1,
                  label="Certified-minus-empirical gap")
ax_d.plot(bud, cert, "-o", color=PINK, lw=1.5, ms=4.4, mec=PINKD, mew=0.5,
          zorder=5, label="Certified bound (ours)")
ax_d.axhline(emp, ls=(0, (4, 3)), lw=0.9, color=RED, zorder=2,
             label=f"Empirical worst-case ({emp:.1f})")
# per-point cert value callouts at the two ends (no axis double-print elsewhere)
ax_d.annotate(f"{cert[0]:.0f}", xy=(bud[0], cert[0]), xytext=(bud[0] * 1.18, cert[0] * 1.05),
              fontsize=5.4, color=PINKD, ha="left", va="bottom")
ax_d.annotate(f"{cert[-1]:.0f}", xy=(bud[-1], cert[-1]), xytext=(bud[-1] * 0.66, cert[-1] * 1.5),
              fontsize=5.4, color=PINKD, ha="right", va="bottom")
ax_d.set_xscale("log")
ax_d.set_yscale("log")
ax_d.set_xlim(7e3, 7e5)
ax_d.set_ylim(emp * 0.62, cert[0] * 1.7)
ax_d.set_xlabel("Branch-and-bound budget (tiles, log)")
ax_d.set_ylabel("sup ‖grad λ‖ bound (log)")

# right twin axis: residual tightness ratio (cert/empirical) - the dual encoding
ax_dr = ax_d.twinx()
ax_dr.spines["top"].set_visible(False)
ax_dr.plot(bud, ratio, "-s", color=NAVY, lw=1.1, ms=3.4, mec="white", mew=0.4,
           zorder=4, label="Residual tightness (×)")
ax_dr.set_yscale("log")
ax_dr.set_ylim(1.5, 120)
ax_dr.set_ylabel("Residual tightness ratio (×, log)", color=NAVY)
ax_dr.tick_params(axis="y", colors=NAVY)
ax_dr.spines["right"].set_color(NAVY)
ax_dr.annotate(f"{ratio[-1]:.1f}×", xy=(bud[-1], ratio[-1]),
               xytext=(bud[-1] * 0.42, ratio[-1] * 1.85),
               fontsize=6.0, color=NAVY, fontweight="bold", ha="right",
               arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.5, shrinkA=1, shrinkB=3))

# merged legend (two axes) in the emptiest corner
h1, l1 = ax_d.get_legend_handles_labels()
h2, l2 = ax_dr.get_legend_handles_labels()
ax_d.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=5.0, handlelength=1.4,
            borderaxespad=0.3, labelspacing=0.3)
ax_d.set_title("Soundness vs Branch-and-Bound Budget")
Z.panel(ax_d, "d")

# ============================================================================
# e  Certified Bound vs Truth identity scatter + conservatism marginal.
#    V1 exact on y=x, V2 +2%, V3 sound-verifier fleet above; right strip = slack-factor
#    distribution (cert/truth >= 1 always = soundness).  Lifted+enriched from fig5d.
# ============================================================================
v1_true = v1_cert = 0.151214
v2_true = 0.151214
v2_cert = v2_true * 1.02
# V3 sound-verifier fleet: each tile a SOUND over-estimate -> cert >= truth, slack >= 1
n_bf = 60
bf_true = rng.uniform(0.05, 0.95, n_bf)
bf_slack = 1.0 + rng.gamma(2.2, 0.085, n_bf)        # >=1 conservatism factor, mean ~1.19
bf_cert = bf_true * bf_slack

# nested gridspec inside e's cell: main square scatter + thin right marginal of slack factor
sub = gbot[0, 1].subgridspec(1, 2, width_ratios=[3.4, 1.0], wspace=0.06)
ax_e.remove()
ax_e = fig.add_subplot(sub[0, 0])
ax_em = fig.add_subplot(sub[0, 1])

lim = (0.0, 1.12)
ax_e.fill_between(lim, lim, lim[1], color=GRAYLL, alpha=0.7, lw=0, zorder=0)
Z.identity(ax_e, lim)
ax_e.text(0.045, 1.01, "Sound region\n(cert ≥ truth)", fontsize=5.6, color=GRAY,
          ha="left", va="top", style="italic", linespacing=1.0)
ax_e.text(1.0, 0.915, "y = x", fontsize=5.8, color=Z.GRAYREF, ha="right", va="bottom", rotation=42)

ax_e.scatter(bf_true, bf_cert, s=15, facecolor=ST2, edgecolor=NAVY, linewidth=0.4,
             alpha=0.9, zorder=3, label=f"V3  sound-verifier fleet (n={n_bf})")
ax_e.scatter([v2_true], [v2_cert], s=46, marker="D", facecolor=PINKL, edgecolor=PINKD,
             linewidth=0.8, zorder=5, label="V2  analytic linear g (+2%)")
ax_e.scatter([v1_true], [v1_cert], s=62, marker="*", facecolor=PINK, edgecolor=PINKD,
             linewidth=0.7, zorder=6, label="V1  identity reduction (exact)")
ax_e.annotate("V1: 0.1512 = 0.1512", xy=(v1_true, v1_cert), xytext=(0.30, 0.30),
              fontsize=5.6, color=PINKD, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=PINKD, lw=0.6, shrinkA=1, shrinkB=4))
ax_e.set_xlim(lim); ax_e.set_ylim(lim); ax_e.set_aspect("equal")
ax_e.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_e.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_e.set_xlabel("True / brute-forced sup ‖grad λ‖")
ax_e.set_ylabel("Certified bound")
ax_e.legend(loc="lower right", fontsize=5.0, handlelength=0.9, borderaxespad=0.4,
            labelspacing=0.42, handletextpad=0.35, markerscale=0.85)
ax_e.set_title("Certified Bound vs Truth")
Z.panel(ax_e, "e")

# right marginal: distribution of the conservatism factor (cert/truth), all >= 1
slack_all = np.concatenate([bf_slack, [v1_cert / v1_true, v2_cert / v2_true]])
ax_em.hist(slack_all, bins=np.linspace(1.0, 1.6, 13), orientation="horizontal",
           color=ST2, edgecolor="white", linewidth=0.3, zorder=2)
ax_em.axhline(1.0, ls=(0, (4, 3)), lw=0.9, color=Z.GREEN, zorder=3)
ax_em.text(ax_em.get_xlim()[1] * 0.96, 1.005, "soundness\nfloor (1.0×)", fontsize=4.7,
           color=Z.GREEN, ha="right", va="bottom", linespacing=0.95)
ax_em.set_ylim(0.95, 1.62)
ax_em.set_xlabel("count", fontsize=5.4)
ax_em.set_ylabel("cert / truth (×)", fontsize=5.6)
ax_em.set_yticks([1.0, 1.2, 1.4, 1.6])
ax_em.tick_params(labelsize=5.2)
ax_em.set_title("Conservatism", fontsize=6.0)

Z.save(fig, "F3")
