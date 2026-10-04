"""zeal_style - paper-wide style constants for the ZEAL figures (one mapping per meaning; builders import, never edit).

Every colour is an agram LOCK colour. Run `python zeal_style.py` for the self-check.
Contents: LS_* dash patterns; BOUND (bound roles); SETTING (the four geographical ladders); NOISE (noise models);
DECISION (certified invariant / sensitive / open); VERIFY (verification status); MODEL (ML models); DATASET (ML datasets,
line styles); VERIFIER (IBP / CROWN / α-CROWN / inner); POOLED; ramps; NAME / BANNED; fmt helpers.
"""
import re
import agram as A
from agram import (TEAL, TEALM, TEALL, TEALLL, BRICK, BRICKL, PURPLE, LAVENDER, LAVL, YELLOW, GREEN, GREEND,
                   CREAM, INK, TXT, GREY, GRAYREF, HAIR, GRAYL, GRAYLL, WHITE)

LS_SOLID = "-"
LS_DASH = (0, (4.0, 1.4))
LS_DOT = (0, (1.1, 1.1))
LS_DASHDOT = (0, (3.0, 1.0, 1.0, 1.0))
LS_LONG = (0, (5.0, 1.5))
LS_SHORT = (0, (2.2, 1.0))
LS_REF = A.LS_REF


def S(c, mk, ls, name, mfc=None, mec=None):
    return dict(c=c, mk=mk, ls=ls, name=name, mfc=mfc or c, mec=mec or c)


def line_kw(st, lw=A.LW_DATA, ms=None, mew=None):
    kw = dict(color=st["c"], ls=st["ls"], lw=lw, marker=st["mk"], mfc=st["mfc"], mec=st["mec"])
    if ms is not None: kw["ms"] = ms
    kw["mew"] = mew if mew is not None else (A.LW_DATA if st["mk"] in ("x", "+") else A.LW_THIN / 2)
    return kw


def key_marker(st, ms=3.4):
    return dict(marker=st["mk"], fc=st["mfc"], ec=st["mec"], ms=ms, lw=A.LW_DATA if st["mk"] == "x" else 0.5)


def key_line(st, lw=A.LW_DATA):
    return dict(fc=st["c"], ls=st["ls"], lw=lw)


# ------------------------------------------------------------------ bound roles (the core semantic set) --
# ZEAL's fused / lifted outer bound = TEAL (our method); the realizable witness = BRICK (the floor); the value-only bound
# = GREY (hollow); the pixel outer bound = PURPLE (hollow); reference (box-midpoint movement) = GRAYREF dashed line;
# truth = INK. Bracket bar [witness, ceiling] = TEAL at alpha 0.35 (draw as TEALL fill).
BOUND = {
    "fused":      S(TEAL, "^", LS_SOLID, "fused bound"),
    "lifted":     S(TEAL, "^", LS_SOLID, "lifted outer bound"),
    "witness":    S(BRICK, "D", LS_SOLID, "realizable witness"),
    "global":     S(BRICK, "D", LS_SOLID, "global witness"),
    "separable":  S(BRICK, "d", LS_DASH, "separable witness", mfc=WHITE, mec=BRICK),
    "value_only": S(GREY, "o", LS_DOT, "value-only bound", mfc=WHITE, mec=GREY),
    "pixel":      S(PURPLE, "s", LS_DASHDOT, "pixel outer bound", mfc=WHITE, mec=PURPLE),
    "truth":      S(INK, "x", LS_SOLID, "truth"),
    "inner":      S(BRICK, "v", LS_SOLID, "inner witness"),
    "outer":      S(TEAL, "^", LS_SOLID, "outer bound"),
}
BRACKET_FILL = TEALL           # the certified bracket bar between witness and ceiling
BAND_OUTER = TEALLL            # value-box band in series panels
BAND_CLOSURE = TEALL           # lattice-closure band inside it
REF_LINE = dict(c=GRAYREF, ls=LS_REF, lw=A.LW_REF)
ZERO_LINE = dict(c=GRAYREF, ls=LS_REF, lw=A.LW_REF)
GUARANTEE_LINE = dict(c=BRICK, ls=LS_REF, lw=A.LW_REF)    # a target / guarantee level (0.950)

# ---------------------------------------------------------------- the four geographical ladders --
SETTING = {
    "georgia": S(TEAL, "o", LS_SOLID, "Georgia"),
    "gm_q4":   S(PURPLE, "s", LS_DASH, "Greater Manchester, qualification"),
    "gm_bad":  S(LAVENDER, "D", LS_DASHDOT, "Greater Manchester, bad health"),
    "mx_rwi":  S(YELLOW, "^", LS_DOT, "Mexico", mec=INK),
}
SETTING_ORDER = ["georgia", "gm_q4", "gm_bad", "mx_rwi"]
SETTING_SHORT = {"georgia": "Georgia", "gm_q4": "Manchester q4", "gm_bad": "Manchester bad", "mx_rwi": "Mexico"}
SETTING_LADDER = {"georgia": "tract → county", "gm_q4": "LSOA → MSOA", "gm_bad": "LSOA → MSOA", "mx_rwi": "municipality → state"}

# ----------------------------------------------------------------------------- noise models --
NOISE = {"noiseless": dict(c=TEAL, label="noiseless"), "gaussian": dict(c=TEALM, label="Gaussian"),
         "family": dict(c=TEALL, label="four-law family")}
NOISE_ORDER = ["noiseless", "gaussian", "family"]

# ------------------------------------------------------------------------------- decisions --
DECISION = {"invariant": dict(c=GREEND, label="certified invariant"), "sensitive": dict(c=BRICK, label="certified sensitive"),
            "witnessed": dict(c=BRICKL, label="witnessed sensitive"), "open": dict(c=GRAYL, label="open")}
# witnessed sensitive = nominal signs differ or a PGD perturbation flips one; an observation, never a certificate
DECISION_ORDER = ["invariant", "sensitive", "witnessed", "open"]

# ---------------------------------------------------------------------- verification status --
VERIFY = {"kernel": dict(c=TEAL, label="machine-checked"), "checker": dict(c=TEALM, label="verified checker"),
          "exact": dict(c=TEALL, label="exact-rational certificate"), "external": dict(c=GRAYL, label="external numerics"),
          "pass": dict(c=GREEND, label="PASS"), "fail": dict(c=BRICK, label="FAIL"), "none": dict(c=GRAYLL, label="not formalized")}

# ------------------------------------------------------------------------------ ML transfer --
MODEL = {"mlp": S(TEAL, "o", LS_SOLID, "raw-lag network"), "feat": S(TEALM, "s", LS_SOLID, "block-mean network"),
         "feat24": S(TEALL, "^", LS_SOLID, "day-block network", mec=TEALM)}
MODEL_ORDER = ["mlp", "feat", "feat24"]
DATASET = {"synthetic": dict(ls=LS_SOLID, label="synthetic feeders", mk="o"), "jul2014": dict(ls=LS_DASH, label="electricity, jul 2014", mk="s"),
           "jan2014": dict(ls=LS_DOT, label="electricity, jan 2014", mk="^")}
DATASET_ORDER = ["synthetic", "jul2014", "jan2014"]
VERIFIER = {"IBP": S(GRAYL, "v", LS_DOT, "IBP", mec=GREY), "CROWN": S(TEAL, "o", LS_SOLID, "CROWN"),
            "alpha-CROWN": S(TEALM, "s", LS_DASH, "α-CROWN"), "inner": S(BRICK, "D", LS_SOLID, "PGD inner"),
            "rigorous": S(PURPLE, "^", LS_DASHDOT, "rigorous enclosure")}
WINDOW_CLASS = {"S1a": "global range", "S1b": "look-back range", "S2": "before/after sign", "S3": "binned trend",
                "S4": "top-3 ranking", "S5": "intra-day contrasts"}
WINDOW_ORDER = ["S1a", "S1b", "S2", "S3", "S4", "S5"]

# --------------------------------------------------------------------------- pooled summaries --
POOLED = {"run": dict(c=TEALM, lw=0.3, alpha=0.32, ls=LS_SOLID, label="single run"),
          "iqr": dict(fc=TEALL, alpha=0.7, label="interquartile range"),
          "p05_95": dict(fc=TEALLL, alpha=1.0, label="q05–q95"),
          "median": dict(c=INK, lw=1.2, ls=(0, (1.2, 1.2)), label="median"),
          "mean": dict(c=TEAL, lw=1.3, ls=LS_SOLID, label="mean")}
ERRORBAR = dict(elinewidth=0.6, capsize=1.6, capthick=0.6)

# ------------------------------------------------------------------------------------ ramps --
HEAT_SEQ = A.HEAT_SEQ            # counts / shares (CREAM → YELLOW → BRICK)
FIELD_SEQ = A.TEAL_SEQ           # certified widths on maps, field values
SIGNED_SEQ = A.DIV               # signed quantities (print the 0 tick)
HEAT_ZERO = CREAM

# -------------------------------------------------------------------------- canonical names --
NAME = {"mlp": "raw-lag network", "feat": "block-mean network", "feat24": "day-block network",
        "V": "value-only", "ZD": "fused", "Z1": "fused (lag 1)", "CI": "certified invariant", "CS": "certified sensitive",
        "WSn": "witnessed sensitive", "WFp": "witnessed sensitive", "open": "open", "gaussian": "Gaussian", "family": "four-law family",
        "noiseless": "noiseless", "q4": "Level-4 qualification share", "bad": "bad-health share", "kfr": "Atlas kfr",
        "georgia": "Georgia", "gm_q4": "Greater Manchester, qualification", "gm_bad": "Greater Manchester, bad health", "mx_rwi": "Mexico"}
BANNED = ["v5", "v6", "v7", "v8", "v7.5", "erratum", "superseded", "old LP",
          "exp1", "exp2", "exp3", ".json", ".npz", ".md", ".py", "PP1", "PP2", "PP3", "PP4", "PP5", "PP6", "PP7", "PP8", "PP9",
          "S1a", "S1b", "S2 ", "S3 ", "S4 ", "S5 ", "B15", "B13", "B14", "B12", "B11", "B3 ", "B7 ", "C1 ", "C3 ", "C5 ", "A3 ", "A4 ", "A7 ",
          "mlp", "feat24", "feat ", "mx_rwi", "gm_q4", "gm_bad", "auto_LiRPA", "Theorem X", "T14", "T17", "T13", "formally verified pipeline",
          "all mathematics", "sorry"]


def name(s): return NAME.get(s, s)


def banned_in(text):
    return [b for b in BANNED if re.search(rf"(?<![\w-]){re.escape(b.strip())}(?![\w])", text)]


def fmt_count(v): return A.fnum(int(round(v)), "{:d}")


def fmt_ratio(v, nd=3): return f"{v:.{nd}f}"


def fmt_pct(v, nd=1): return f"{100 * v:.{nd}f}"


def text_on(hexfill): return A.contrast_text(hexfill)


def _selfcheck():
    lock = A.LOCK_HEX; bad = []
    def chk(hx, where):
        if hx.lower() not in lock: bad.append(f"{where}: {hx} not in agram.LOCK")
    for fam, d in (("BOUND", BOUND), ("SETTING", SETTING), ("MODEL", MODEL), ("VERIFIER", VERIFIER)):
        for k, st in d.items():
            for f in ("c", "mfc", "mec"): chk(st[f], f"{fam}[{k}].{f}")
        trip = [(st["c"], st["mfc"], st["mk"], str(st["ls"])) for st in d.values()]
        if fam != "BOUND" and len(set(trip)) != len(trip): bad.append(f"{fam}: duplicate (colour, face, marker, dash)")
        # BOUND holds aliases on purpose (fused = lifted = outer; witness = global = inner role); never draw two aliases in one panel
    for d, nm in ((NOISE, "NOISE"), (DECISION, "DECISION"), (VERIFY, "VERIFY")):
        for k, v in d.items(): chk(v["c"], f"{nm}[{k}]")
    for k, v in POOLED.items(): chk(v.get("c", v.get("fc")), f"POOLED[{k}]")
    for x in (BRACKET_FILL, BAND_OUTER, BAND_CLOSURE, HEAT_ZERO): chk(x, "misc")
    for s in NAME.values():
        if banned_in(s): bad.append(f"canonical name {s!r} contains a banned string")
    return bad


if __name__ == "__main__":
    problems = _selfcheck()
    print("zeal_style self-check:", "OK" if not problems else f"{len(problems)} problems")
    for p in problems: print("  !!", p)
    raise SystemExit(1 if problems else 0)
