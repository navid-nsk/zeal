"""qa_check - automatic QA for the ZEAL figures (port of the author's qa_check.py, reduced).

Usage: python qa_check.py [stem ...]   (default: every data/*.qa.json)
Items:
  1 audit: 0 text overlaps, 0 off-canvas strings (from the sidecar written by agram.save)
  2 text sizes: every string 6.0-7.0 pt on the scale (6.0, 6.5, 7.0); panel letters exactly 8.0 pt bold;
    SVG font sizes (incl. mathtext runs) >= 6.0 pt
  3 titles <= 6 words; one token line per panel letter; no finite verb; no trailing period
  4 colour lock: every fill/stroke hex in the SVG is in agram.LOCK_HEX (rasters of kind 'art' exempt)
  5 data-literal lint: outside the '# === LAYOUT' ... '# === END LAYOUT' block the plot script carries no
    non-trivial numeric literal (numbers come from data/<stem>.json)
  6 empty 5 mm tiles at 300 dpi <= 30 %
  7 fonts: pdffonts shows Arial TrueType embedded; SVG has <text> elements and no glyph outlines
  8 canvas: 180.0 mm wide, height <= 225 mm; registered plot boxes inside and non-overlapping
Writes qa_report.md; exits non-zero on failure.
"""
import os, re, sys, ast, json, glob, subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agram as A  # noqa: E402

DATA, FIGS = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*"); TRIVIAL = {0, 1, 2, -1, 0.5, 100}


def _svg_tree(path):
    root = ET.parse(path).getroot(); parent = {c: p for p in root.iter() for c in p}; return root, parent


def item1(side):
    return (len(side["audit"]) == 0, f"{len(side['audit'])} problems" + ("; " + "; ".join(side["audit"][:5]) if side["audit"] else ""))


def item2(side, svg):
    bad = []
    for t in side["texts"]:
        if t["gid"].startswith("letter_"):
            if not (abs(t["size"] - A.F_LETTER) < 1e-6 and t["weight"] in ("bold", "700")): bad.append(f"letter {t['s']!r} {t['size']} {t['weight']}")
        elif t["size"] < A.F_MIN - 1e-6: bad.append(f"{t['s'][:24]!r} {t['size']} pt")
        elif not any(abs(t["size"] - v) < 1e-3 for v in A.F_SCALE): bad.append(f"{t['s'][:24]!r} {t['size']} pt off the scale")
    if os.path.exists(svg):
        root, _ = _svg_tree(svg)
        for el in root.iter():
            if el.tag.endswith("}text") or el.tag.endswith("}tspan"):
                m = re.search(r"font-size:\s*([\d.]+)px", el.get("style") or "")
                if m and float(m.group(1)) < A.F_MIN - 1e-6: bad.append(f"svg {''.join(el.itertext())[:20]!r} {float(m.group(1)):g} pt")
    return (not bad, f"{len(side['texts'])} strings; " + ("; ".join(sorted(set(bad))[:6]) if bad else "all 6.0-7.0 pt, letters 8.0 bold"))


def item3(side):
    bad = [f"title {len(A._words(s))} words: {s!r}" for s in side["titles"] if len(A._words(s)) > 6]
    if len(side["notes"]) != len(side["letters"]): bad.append(f"{len(side['notes'])} token lines for {len(side['letters'])} panels")
    for t in side["texts"]:
        s = t["s"].strip(); low = f" {re.sub(r'[^a-z ]', ' ', s.lower())} "
        for v in A.STOP_VERBS:
            if f" {v} " in low: bad.append(f"verb {v!r}: {s!r}")
    return (not bad, f"{len(side['titles'])} titles, {len(side['notes'])} tokens, {len(side['letters'])} letters" + ("; " + "; ".join(bad[:6]) if bad else ""))


def _styles(el):
    out = {}
    for part in (el.get("style") or "").split(";"):
        if ":" in part: k, v = part.split(":", 1); out[k.strip()] = v.strip()
    for k in ("fill", "stroke"):
        if el.get(k) is not None: out[k] = el.get(k)
    return out


def item4(svg):
    root, parent = _svg_tree(svg); bad = set(); n = 0; n_img = 0
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "image":
            n_img += 1; ids = [el.get("id") or ""]; p = el
            while p in parent: p = parent[p]; ids.append(p.get("id") or "")
            if not any(i.startswith("img_art_") for i in ids): bad.add(f"raster outside kind 'art' ({[i for i in ids if i][:2]})")
            continue
        sty = _styles(el)
        for k, v in sty.items():
            if k not in ("fill", "stroke") or not v.startswith("#"): continue
            if sty.get(f"{k}-opacity", "1").strip() in ("0", "0.0") or sty.get("opacity", "1").strip() in ("0", "0.0"): continue
            hx = v.lower()
            if len(hx) == 4: hx = "#" + "".join(ch * 2 for ch in hx[1:])
            n += 1
            if hx not in A.LOCK_HEX: bad.add(f"{hx} ({tag} {k})")
    return (not bad, f"{n} colour uses, {n_img} rasters; " + ("not in lock: " + ", ".join(sorted(bad)[:8]) if bad else "all in lock"))


def _layout_span(src):
    lines = src.splitlines()
    a = next((i for i, l in enumerate(lines) if l.startswith("# === LAYOUT")), None)
    b = next((i for i, l in enumerate(lines) if l.startswith("# === END LAYOUT")), None)
    return (a + 1, b + 1) if a is not None and b is not None else (0, 0)


def item5(script):
    if script and not os.path.isabs(script): script = os.path.join(HERE, script)      # sidecars record the script relative to figures/
    if not script or not os.path.exists(script): return (False, "plot script not found")
    src = open(script, encoding="utf-8").read(); a, b = _layout_span(src)
    if not a: return (False, "no '# === LAYOUT' ... '# === END LAYOUT' block")
    tree = ast.parse(src); parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    doc = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    hits = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Constant) or id(n) in doc or isinstance(n.value, (bool, str)) or n.value is None: continue
        p = parents.get(n)
        if isinstance(p, ast.keyword) and p.arg in ("zorder", "z"): continue
        if isinstance(p, ast.Subscript) and p.slice is n: continue
        ln = getattr(n, "lineno", 0)
        if a <= ln <= b or not isinstance(n.value, (int, float)) or float(n.value) in TRIVIAL: continue
        hits.append(f"line {ln}: {n.value!r}")
    return (not hits, f"{len(hits)} non-trivial numeric literals outside LAYOUT" + ("; " + "; ".join(hits[:8]) if hits else ""))


def item6(png):
    im = Image.open(png).convert("L"); dpi = im.info.get("dpi", (600, 600))[0]; s = 300.0 / dpi
    im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR); a = np.asarray(im); tile = 5.0 / 25.4 * 300.0
    nx, ny = int(a.shape[1] // tile), int(a.shape[0] // tile); empty = 0
    for j in range(ny):
        for i in range(nx):
            if a[int(j * tile):int((j + 1) * tile), int(i * tile):int((i + 1) * tile)].min() >= 245: empty += 1
    pct = 100.0 * empty / max(1, nx * ny)
    return (pct <= 30.0, f"{pct:.1f}% empty 5 mm tiles (limit 30%)")


def item7(pdf, svg):
    bad = []
    try:
        out = subprocess.run(["pdffonts", pdf], capture_output=True, text=True, timeout=60).stdout
        rows = [l for l in out.splitlines()[2:] if l.strip()]
        for r in rows:
            if "Arial" not in r.split()[0] or "TrueType" not in r: bad.append(r.strip()[:50])
        fonts = ", ".join(sorted({r.split()[0].split("+")[-1] for r in rows}))
    except FileNotFoundError:
        fonts = "pdffonts not found"
    root, _ = _svg_tree(svg); n_text = sum(1 for el in root.iter() if el.tag.endswith("}text"))
    glyphs = sum(1 for el in root.iter() if el.tag.endswith("}path") and (el.get("id") or "").startswith(("ArialMT-", "DejaVu")))
    if n_text == 0 or glyphs: bad.append(f"SVG text {n_text}, glyph outlines {glyphs}")
    return (not bad, f"PDF fonts: {fonts}; SVG <text> {n_text}" + ("; " + "; ".join(bad[:4]) if bad else ""))


def item8(side):
    w, h = side["canvas"]["w"], side["canvas"]["h"]; bad = []
    if abs(w - A.W_CANVAS) > 1e-6: bad.append(f"width {w}")
    if h > A.H_MAX + 1e-6: bad.append(f"height {h}")
    P = side["pax"]
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            a, b = P[i], P[j]
            if a.get("inset_of") == b["name"] or b.get("inset_of") == a["name"]: continue
            if A._boxes_overlap((a["x"], a["y_top"], a["w"], a["h"]), (b["x"], b["y_top"], b["w"], b["h"])): bad.append(f"{a['name']} overlaps {b['name']}")
    return (not bad, f"canvas {w:.1f} x {h:.1f} mm; {len(P)} plot boxes" + ("; " + "; ".join(bad[:4]) if bad else ""))


def run(stems=None):
    sides = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(DATA, "*.qa.json")))]
    if stems: sides = [s for s in sides if s["stem"] in stems]
    lines = ["# QA report (alternative 5, automatic items)", ""]; all_ok = True
    for side in sides:
        stem = side["stem"]; base = os.path.join(FIGS, stem); svg, png, pdf = base + ".svg", base + ".png", base + ".pdf"
        res = [("1 audit", item1(side)), ("2 text sizes", item2(side, svg)), ("3 titles, tokens, verbs", item3(side)), ("4 colour lock", item4(svg)),
               ("5 data-literal lint", item5(side.get("script"))), ("6 empty tiles", item6(png)), ("7 fonts, SVG text", item7(pdf, svg)), ("8 canvas, boxes", item8(side))]
        lines += [f"## {stem}", "", "| Item | Result | Detail |", "|---|---|---|"]; print(f"QA {stem}")
        for name, (ok, detail) in res:
            all_ok &= ok; lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail.replace('|', '/')} |")
            print(f"  {'PASS' if ok else 'FAIL'}  {name:24s} {detail}".encode("ascii", "replace").decode())
        lines.append("")
    open(os.path.join(HERE, "qa_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"QA {'PASSED' if all_ok else 'FAILED'}"); return all_ok


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1:] or None) else 1)
