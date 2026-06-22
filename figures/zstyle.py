"""Shared style + locked palette for ALL ZEAL figures.
   Color lock: ours=PINK, baselines=navy/steel, green=good/kept,
   red=threshold-only, orange not used as data. Arial, despined, 600 dpi."""
import json, os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---- LOCKED PALETTE ----
PINK = "#D56A9A"; PINKD = "#C9447A"; PINKL = "#E8A0B8"            # ZEAL / our method (the ONE accent)
NAVY = "#1F3A5F"; STEEL1 = "#3B6CA8"; STEEL2 = "#5B9BD5"; STEEL3 = "#9DC3E6"   # baselines ramp (ranked)
BASE_RAMP = [NAVY, STEEL1, STEEL2, STEEL3]
GREEN = "#548235"                                                 # good / kept / in-sample ONLY
RED = "#C00000"                                                   # threshold / alert / reference ONLY
GRAYREF = "#9A9A9A"                                               # chance / identity diagonal (dashed)
GRAY = "#7A7A78"; GRAYL = "#C8C7C0"; GRAYLL = "#E6E6E6"
INK = "#1A1A1A"

SEQ = LinearSegmentedColormap.from_list("zeal_seq", ["#EAF2FB", "#9DC3E6", "#3B6CA8", "#1F3A5F"])   # single-hue field
DIV = LinearSegmentedColormap.from_list("zeal_div", ["#2166AC", "#F7F7F7", "#B2182B"])              # signed (blue-white-red)
PINK_SEQ = LinearSegmentedColormap.from_list("zeal_pink", ["#FBE4EC", "#E8A0B8", "#C9447A", "#8C1D4E"])


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        # --- Illustrator compatibility: embed TrueType (42), NOT Type-3 (AI can't render Type-3);
        #     keep SVG text as real <text>; don't merge overlapping rasters into one flat image.
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "image.composite_image": False,
        "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "regular", "axes.titlelocation": "left", "axes.titlepad": 5,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4, "lines.markersize": 4,
    })


MM = 1 / 25.4                                                     # mm -> inch for Nature widths (89 / 180 mm)


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def panel(ax, letter):
    """Bold lowercase black panel letter, top-left OUTSIDE corner, above the title."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-22, 12),
                textcoords="offset points", fontsize=8.5, fontweight="bold", color=INK, va="bottom", ha="left")


def threshold(ax, val=0.0, axis="y", label=None):
    """Red dashed threshold/reference line (the only sanctioned use of red)."""
    (ax.axhline if axis == "y" else ax.axvline)(val, ls=(0, (4, 3)), lw=0.9, color=RED, zorder=1)


def identity(ax, lim):
    ax.plot(lim, lim, ls=(0, (4, 3)), lw=0.8, color=GRAYREF, zorder=1)


def save(fig, stem):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, stem + ".png"))
    fig.savefig(os.path.join(FIGS, stem + ".pdf"))      # TrueType (fonttype 42) -> Illustrator-editable
    fig.savefig(os.path.join(FIGS, stem + ".svg"))      # cleanest for Illustrator: vector + editable text
    print(f"saved figures/{stem}.png + .pdf + .svg")
    plt.close(fig)
