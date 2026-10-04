"""agram - figure grammar for the ZEAL figures (the author's zgram/cgram/agram house grammar).

One millimetre canvas per figure (180 mm wide, <= 225 mm tall; origin bottom-left internally, boxes are
given as [x, y_top, w, h] from the top-left corner). Charts are inset axes placed by `pax`; text via `T`;
panel letter + title via `head`; one grey token line per panel via `tok`; numbers via `N` (registry);
`audit` reports text collisions and off-canvas text; `save` writes png (600 dpi) + pdf + svg + a QA
sidecar read by qa_check.py.

Palette lock = the author's house palette.
Arial everywhere, TrueType (fonttype 42), SVG text kept as <text>. Nothing below 6.0 pt (editor rule).
"""
import os, re, json, math
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
ASSETS = os.path.join(HERE, "assets")

# =============================================================== palette lock (author's house palette) ==
TEAL = "#2F5E5B"        # primary series / ZEAL / dark teal fills
TEALM = "#6C9894"       # mid teal (secondary series, muted)
TEALL = "#8FC8C4"       # light teal fills
TEALLL = "#D3EBE8"      # palest teal (bands, backgrounds of cards)
BRICK = "#934234"       # accent: highlights, second category
BRICKL = "#F1D8D3"      # pale brick fill
PURPLE = "#422E5E"      # third category (dark)
LAVENDER = "#A68EC3"    # light purple series
LAVL = "#DCD2E8"        # pale lavender fill
YELLOW = "#EFCD49"      # fourth category / warm accent (small areas)
GREEN = "#AFD794"       # light green (no-pruning, healthy-light)
GREEND = "#345025"      # dark green (healthy, good)
CREAM = "#FFF8E8"       # heatmap background cell
INK = "#1D313C"         # axes, primary text (the author's dark ink)
TXT = "#231F20"         # labels
GREY = "#7A7A78"        # token lines, provenance
GRAYREF = "#9A9A9A"     # reference lines (dashed), identity
HAIR = "#BEBEC0"        # frames, dividers, block arrows
GRAYL = "#C8C7C0"       # light structure
GRAYLL = "#EEEEEC"      # bands
WHITE = "#FFFFFF"
LOCK = {k: v for k, v in list(globals().items()) if k.isupper() and isinstance(v, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", v)}
LOCK_HEX = {v.lower() for v in LOCK.values()}
CAT = [TEAL, BRICK, PURPLE, YELLOW]                       # categorical order (max 4 + greys)
TEAL_SEQ = LinearSegmentedColormap.from_list("zeal_teal", [WHITE, TEALLL, TEALL, TEALM, TEAL])
HEAT_SEQ = LinearSegmentedColormap.from_list("zeal_heat", [CREAM, YELLOW, "#D99228", BRICK])   # heatmap ramp 
DIV = LinearSegmentedColormap.from_list("zeal_div", [BRICK, WHITE, TEAL])

# ======================================================================= sizes ==
F_LETTER, F_TITLE, F_AXLAB, F_TICK, F_KEY, F_VAL, F_TOKEN = 8.0, 7.0, 6.5, 6.0, 6.0, 6.0, 6.0
F_MIN, F_MAX = 6.0, 7.0
F_SCALE = (6.0, 6.5, 7.0, 8.0)
LW_AXIS, LW_DATA, LW_THIN, LW_REF, LW_FRAME = 0.6, 1.0, 0.7, 0.6, 0.4
LS_REF = (0, (3, 2))
PT = 0.352778           # mm per point
LH = 1.32
W_CANVAS, H_MAX = 180.0, 225.0
X_LEFT, X_RIGHT = 1.0, 179.0
STOP_VERBS = ("is", "are", "was", "were", "shows", "show", "indicates", "indicate", "suggests", "suggest", "demonstrates", "reveals")

_CUR = {"fig": None}
_GID = {}


def _gid(prefix):
    _GID[prefix] = _GID.get(prefix, 0) + 1
    return f"{prefix}{_GID[prefix]}"


def setup():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "figure.facecolor": WHITE, "savefig.facecolor": WHITE, "axes.facecolor": "none",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": F_TICK, "axes.titlesize": F_TITLE, "axes.labelsize": F_AXLAB,
        "xtick.labelsize": F_TICK, "ytick.labelsize": F_TICK, "legend.fontsize": F_KEY,
        "axes.linewidth": LW_AXIS, "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.pad": 1.4, "ytick.major.pad": 1.4,
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": LW_DATA, "lines.markersize": 3.6, "errorbar.capsize": 2.0,
        "axes.edgecolor": INK, "text.color": TXT, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": False, "legend.frameon": False, "legend.handlelength": 1.3, "legend.handletextpad": 0.5,
        "axes.titleweight": "regular", "axes.titlelocation": "left", "axes.titlepad": 4, "axes.labelpad": 1.4,
        "axes.unicode_minus": True, "axes.prop_cycle": mpl.cycler(color=[TEAL, BRICK, PURPLE, YELLOW, TEALM, LAVENDER]),
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial", "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "image.composite_image": False,
        "svg.hashsalt": "agram", "path.simplify": False,
    })


# ================================================================================ canvas ==
def canvas(w_mm=W_CANVAS, h_mm=120.0):
    """Figure + one mm axis covering the page (origin bottom-left). cv.Y(y_top) converts a top-left y."""
    assert abs(w_mm - W_CANVAS) < 1e-6 and h_mm <= H_MAX + 1e-6, (w_mm, h_mm)
    setup()
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
    cv = fig.add_axes([0, 0, 1, 1]); cv.set_xlim(0, w_mm); cv.set_ylim(0, h_mm); cv.set_axis_off(); cv.set_facecolor("none")
    cv._w_mm, cv._h_mm = w_mm, h_mm
    cv.Y = lambda y_top, _h=h_mm: _h - y_top
    fig._ag = dict(cv=cv, w=w_mm, h=h_mm, pax=[], notes=[], titles=[], nums={}, imgs=[], letters=[])
    _CUR["fig"] = fig; _GID.clear()
    return fig, cv


def _boxes_overlap(a, b, tol=0.05):
    return (min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]) > tol and min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]) > tol)


def pax(fig, cv, x, y_top, w, h, name=None, inset_of=None, **kw):
    """Plot box at [x, y_top, w, h] mm from the top-left corner; registered (no overlap with other boxes)."""
    R = fig._ag; name = name or f"pax{len(R['pax']) + 1}"
    assert 0.5 <= x and x + w <= R["w"] - 0.5 + 1e-6, f"pax {name!r} outside the canvas width: {(x, y_top, w, h)}"
    assert 0.0 <= y_top and y_top + h <= R["h"] + 1e-6, f"pax {name!r} outside the canvas height: {(x, y_top, w, h)}"
    for o in R["pax"]:
        if inset_of == o["name"] or o.get("inset_of") == name: continue
        assert not _boxes_overlap((x, y_top, w, h), (o["x"], o["y_top"], o["w"], o["h"])), f"pax {name!r} overlaps {o['name']!r}"
    R["pax"].append(dict(name=name, x=x, y_top=y_top, w=w, h=h, inset_of=inset_of))
    ax = fig.add_axes([x / R["w"], (R["h"] - y_top - h) / R["h"], w / R["w"], h / R["h"]], **kw)
    ax.set_facecolor("none"); ax._ag_name, ax._ag_box = name, (x, y_top, w, h)
    return ax


def ax_mm(ax):
    b = ax.get_position(); f = ax.figure
    return b.width * f.get_figwidth() * 25.4, b.height * f.get_figheight() * 25.4


# ================================================================================== text ==
def T(ax, x, y, s, size=F_VAL, ha="left", va="baseline", c=TXT, w="normal", style="normal", rot=0, z=7, gid=None, **kw):
    assert size >= F_MIN - 1e-6, f"text below {F_MIN} pt: {s!r} at {size}"
    t = ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=c, fontweight=w, fontstyle=style, rotation=rot,
                rotation_mode="anchor", zorder=z, **kw)
    if gid: t.set_gid(gid)
    return t


def _words(s):
    s = re.sub(r"\$[^$]*\$", "x", s)
    return [w for w in s.replace("|", " ").split() if w.strip()]


def _text_rules(s, what):
    low = f" {s.lower()} "
    for v in STOP_VERBS: assert f" {v} " not in low, f"{what} contains the verb {v!r}: {s!r}"
    assert not s.rstrip().endswith(".") or s.rstrip().endswith(("e.g.", "i.e.", "vs.", "no.", "approx.", "n.a.", "n.d.")), f"{what} ends with a period: {s!r}"


def letter(cv, x, y_top, s):
    assert re.fullmatch(r"[a-z]", s), s
    cv.figure._ag["letters"].append(s)
    return T(cv, x, cv.Y(y_top), s, F_LETTER, va="top", c="#000000", w="bold", z=9, gid=f"letter_{s}")


def title(cv, x, y_top, s, size=F_TITLE, c=INK):
    assert len(_words(s)) <= 6, f"title has {len(_words(s))} words (max 6): {s!r}"
    _text_rules(s, "title"); cv.figure._ag["titles"].append(s)
    return T(cv, x, cv.Y(y_top), s, size=size, c=c, va="top", gid=_gid("title_"))


def head(cv, x, y_top, s, title_text, dx=4.4):
    """Panel letter at (x, y_top) and its short title beside it (both va = top)."""
    letter(cv, x, y_top, s); return title(cv, x + dx, y_top, title_text)


def tok(cv, x, y_top, s, size=F_TOKEN, ha="left"):
    """The single grey token line of a panel: telegraphic, ' | ' separators, <= 12 words."""
    assert len(_words(s)) <= 12, f"token line has {len(_words(s))} words (max 12): {s!r}"
    _text_rules(s, "token"); cv.figure._ag["notes"].append(s)
    T(cv, x, cv.Y(y_top), s, size, ha=ha, va="top", c=GREY, gid=_gid("tok_"))
    return y_top + size * PT * LH


def mmw(s, size, w="normal", style="normal", fig=None):
    fig = fig or _CUR["fig"]; fp = FontProperties(family=mpl.rcParams["font.sans-serif"], size=size, weight=w, style=style)
    wpx, _, _ = fig.canvas.get_renderer().get_text_width_height_descent(s, fp, ismath="$" in s)
    return wpx / fig.dpi * 25.4


def fnum(v, fmt="{:g}", commas=True):
    if isinstance(v, str): return v
    s = fmt.format(v)
    if commas:
        m = re.match(r"^(-?)(\d+)(.*)$", s)
        if m and len(m.group(2)) > 3: s = m.group(1) + f"{int(m.group(2)):,}" + m.group(3)
    return s.replace("-", "−")


def N(key, v, fmt="{:g}", commas=True):
    """Register a printed number under a key (same key -> same string in every figure)."""
    s = fnum(v, fmt, commas); nums = _CUR["fig"]._ag["nums"]
    if key in nums: assert nums[key] == s, f"number key {key!r}: {nums[key]!r} vs {s!r}"
    nums[key] = s; return s


def key_row(cv, x, y_top, items, size=F_KEY, gap=2.2, sw=1.8, c=TXT):
    """One horizontal key row on the canvas: items = [(kind, label, style)], kind 'sw' (square swatch),
    'line' (short line), 'marker' (marker glyph). style = dict(fc=, ec=, lw=, marker=, ls=). Returns the end x."""
    y = cv.Y(y_top); xx = x
    for kind, lab, st in items:
        if kind == "sw":
            cv.add_patch(Rectangle((xx, y - sw / 2), sw, sw, fc=st.get("fc", TEAL), ec=st.get("ec", "none"), lw=st.get("lw", 0), zorder=6)); w = sw
        elif kind == "line":
            cv.plot([xx, xx + sw * 1.8], [y, y], color=st.get("fc", TEAL), lw=st.get("lw", LW_DATA), ls=st.get("ls", "-"), zorder=6, solid_capstyle="butt"); w = sw * 1.8
        else:
            cv.plot([xx + sw / 2], [y], marker=st.get("marker", "o"), ms=st.get("ms", 3.4), mfc=st.get("fc", TEAL), mec=st.get("ec", st.get("fc", TEAL)), mew=st.get("lw", 0.5), ls="none", zorder=6); w = sw
        T(cv, xx + w + 0.9, y, lab, size, va="center_baseline", c=c); xx += w + 0.9 + mmw(lab, size) + gap
    return xx


# ==================================================================== axes formatting ==
def clean(ax, xl=None, yl=None, xlab=None, ylab=None, lp=1.4, ts=F_TICK, ls=F_AXLAB):
    if xl is not None: ax.set_xlim(*xl)
    if yl is not None: ax.set_ylim(*yl)
    if xlab: ax.set_xlabel(xlab, labelpad=lp, fontsize=ls, color=INK)
    if ylab: ax.set_ylabel(ylab, labelpad=lp, fontsize=ls, color=INK)
    ax.tick_params(labelsize=ts, pad=1.4, length=2.4, width=0.6, colors=INK, direction="out")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(LW_AXIS); ax.spines[sp].set_color(INK)
    return ax


def refline(ax, v, axis="y", lab=None, c=GRAYREF, lw=LW_REF, z=1.5):
    ln = (ax.axhline if axis == "y" else ax.axvline)(v, ls=LS_REF, lw=lw, color=c, zorder=z); ln.set_gid(_gid("ref_"))
    if lab:
        assert len(_words(lab)) <= 4, lab
        tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData) if axis == "y" else mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        t = ax.text(1.0, v, lab, transform=tr, fontsize=F_VAL, color=c, ha="right", va="bottom", zorder=9) if axis == "y" else \
            ax.text(v, 1.0, lab, transform=tr, fontsize=F_VAL, color=c, ha="left", va="top", zorder=9)
        t.set_gid(_gid("ref_lab_"))
    return ln


def inlab(ax, x, y, s, c=TXT, size=F_VAL, ha="left", va="top", w="normal", rot=0):
    assert size >= F_MIN - 1e-6, (s, size)
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=size, color=c, ha=ha, va=va, zorder=9, fontweight=w, rotation=rot)


def contrast_text(hexfill):
    """Dark text on light fills, white on dark (guideline 2.9)."""
    r, g, b = (int(hexfill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return TXT if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else WHITE


def stage_box(ax, x, y, w, h, ec=HAIR, lw=0.5, r=1.4, z=1, fc="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def rule(cv, y_top, x0=X_LEFT, x1=X_RIGHT, c=GRAYLL, lw=0.8):
    cv.plot([x0, x1], [cv.Y(y_top)] * 2, color=c, lw=lw, zorder=0, solid_capstyle="butt")


def img(ax, arr, x, y_top, w, h=None, kind="art", frame=None, z=3, interp="antialiased", **kw):
    """Raster (art = the author's vector artwork rasterised, exempt from the colour lock) at mm (x, y_top)."""
    a = np.asarray(arr); cv = ax.figure._ag["cv"]; h = w * a.shape[0] / a.shape[1] if h is None else h
    y = cv.Y(y_top) - h; gid = _gid(f"img_{kind}_")
    im = ax.imshow(a, extent=(x, x + w, y, y + h), zorder=z, aspect="auto", interpolation=interp, **kw); im.set_gid(gid)
    if frame: ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=frame, lw=LW_FRAME, zorder=z + 1.5))
    ax.figure._ag["imgs"].append(dict(gid=gid, kind=kind, px=[int(a.shape[1]), int(a.shape[0])], mm=[round(w, 2), round(h, 2)], dpi=round(a.shape[1] / (w / 25.4), 1)))
    return h


# ================================================================================ audit ==
def _text_items(fig):
    """Every visible text on the figure (canvas texts, axes texts, axis labels, tick labels) as canvas-mm boxes."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); cv = fig._ag["cv"]; inv = cv.transData.inverted(); out = []
    def add(t, kind):
        s = t.get_text()
        if not s.strip() or not t.get_visible(): return
        bb = t.get_window_extent(renderer=r)
        if bb.width <= 0 or bb.height <= 0: return
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        out.append(dict(s=s, kind=kind, size=float(t.get_fontsize()), weight=str(t.get_fontweight()), gid=t.get_gid() or "", box=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))
    for ax in fig.axes:
        for t in ax.texts: add(t, "text")
        if ax is cv: continue
        add(ax.xaxis.label, "axlabel"); add(ax.yaxis.label, "axlabel"); add(ax.title, "text")
        for t in ax.get_xticklabels() + ax.get_yticklabels(): add(t, "tick")
        for t in ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True): add(t, "tick")
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts(): add(t, "text")
    for t in fig.texts: add(t, "text")
    return out


def audit(fig, tol=0.12):
    """Text-text overlaps anywhere on the figure and text beyond the canvas. Returns the list of problems."""
    items = _text_items(fig); bad = []; W, H = fig._ag["w"], fig._ag["h"]
    for i in range(len(items)):
        a = items[i]["box"]
        if a[0] < -0.2 or a[2] > W + 0.2 or a[1] < -0.2 or a[3] > H + 0.2: bad.append(f"OFF CANVAS {items[i]['s'][:30]!r}")
        for j in range(i + 1, len(items)):
            b = items[j]["box"]; ox, oy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
            if ox > tol and oy > tol: bad.append(f"OVERLAP {ox:.2f}x{oy:.2f} mm {items[i]['s'][:28]!r} / {items[j]['s'][:28]!r}")
    fig._ag["texts"] = items; fig._ag["audit"] = bad
    return bad


def save(fig, stem, script=None, preview_dpi=150):
    """png (600 dpi) + pdf + svg + preview png + QA sidecar data/<stem>.qa.json."""
    os.makedirs(FIGS, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    bad = audit(fig)
    for ext in ("png", "pdf", "svg"): fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"))
    fig.savefig(os.path.join(FIGS, f"{stem}_preview.png"), dpi=preview_dpi)
    R = fig._ag
    if script and os.path.isabs(script):                        # recorded relative to figures/ (no machine-specific path)
        script = os.path.relpath(script, HERE).replace("\\", "/")
    side = dict(stem=stem, script=script, canvas=dict(w=R["w"], h=R["h"]), audit=bad, pax=R["pax"], notes=R["notes"], titles=R["titles"], letters=R["letters"], nums=R["nums"], imgs=R["imgs"],
                texts=[dict(s=t["s"][:40], kind=t["kind"], size=t["size"], weight=t["weight"], gid=t["gid"]) for t in R["texts"]])
    json.dump(side, open(os.path.join(DATA, f"{stem}.qa.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"saved figures/{stem}.png/.pdf/.svg  ({R['w']:.0f} x {R['h']:.0f} mm; {len(R['texts'])} texts, {len(bad)} audit problems)")
    for b in bad[:12]: print("   !!", b)
    plt.close(fig)
    return bad
