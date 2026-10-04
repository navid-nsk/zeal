"""check_b15.py -- independent checker for the exhaustive-enumeration pricing certificates (V3, Supplementary Note 8; Theorem S17).
For every (window, class, direction) artifact of pricing_verify.py (--mode verify, with stored final duals):
  1. COVERAGE. (a) Classes with <= --full_limit admissible cells: an INDEPENDENT enumeration by breadth-first closure over subsets
     (start from singletons, add any adjacent unit, deduplicate by bitmask in a Python set; mass pruning only by the exact inequality
     mu(S) > t_hi on Python Fractions of the float masses) -- a different algorithm with no shared code; its cell set must equal the
     ESU cell set (as sets of bitmasks, not only counts).  (b) Larger classes: the ESU enumerator is rerun on a randomly relabelled
     unit order (order-independence consistency test) and the count must agree; coverage rests on the ESU theorem (Wernicke 2006) and the
     monotone-mass pruning argument, which are checked as statements, not re-executed.
  2. RESIDUAL BOUND. For every enumerated cell the reduced cost r(C) = (Sx Sy - b Sx^2)/t - pi(C) - theta is recomputed from the stored
     duals in EXACT rational arithmetic (Fractions of the float data) for the small classes, and in float with the proved forward-error
     inflation for the large classes; the maximum is compared with the artifact's eps_exact + allowance (checker max must be <= eps_used).
  3. CHARGE. The charged-pricing bound of Theorem S17, b + max(0, (1'pi + K(theta + eps))/Dmin) is recomputed in exact rationals from the stored duals and the
     checker's eps; it must not exceed the artifact's ub_cert (which carries the outward allowance).
Usage: python check_b15.py --artifacts <dir with pricing_verify_n30.json> --out <dir> [--full_limit 1200000]
"""
try:
    from . import paths                     # imported as part of the package
except ImportError:
    import paths                            # flat import (this directory on sys.path)
import argparse, io, json, os, sys, time, numpy as np
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pricing_certified as PC, pricing_verify as PV
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--artifacts", required=True); p.add_argument("--out", required=True); p.add_argument("--full_limit", type=int, default=1200000); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
D = PC.load_data(); art = json.load(open(os.path.join(a.artifacts, "pricing_verify_n30.json"), encoding="utf-8")); rng = np.random.default_rng(3)


def bfs_enumerate(W, cls):
    """independent enumeration: closure of singletons under 'add an adjacent unit', exact mass pruning, dedup by bitmask"""
    WF = [Fraction(float(m)) for m in W.mu]; thi = Fraction(float(cls.thi)); tlo = Fraction(float(cls.tlo)); smax = cls.smax
    seen = set(); frontier = []
    for v in range(W.n):
        if WF[v] <= thi:
            m = 1 << v; seen.add(m); frontier.append((m, WF[v], 1))
    while frontier:
        nxt = []
        for m, mass, size in frontier:
            if size >= smax:
                continue
            members = [i for i in range(W.n) if (m >> i) & 1]; nb = set()
            for i in members:
                nb.update(W.adj[i])
            for u in nb:
                if (m >> u) & 1:
                    continue
                mu2 = mass + WF[u]
                if mu2 > thi:
                    continue
                m2 = m | (1 << u)
                if m2 in seen:
                    continue
                seen.add(m2); nxt.append((m2, mu2, size + 1))
        frontier = nxt
    # admissible = mass >= tlo as well
    adm = set()
    for m in seen:
        mass = sum((WF[i] for i in range(W.n) if (m >> i) & 1), Fraction(0))
        if mass >= tlo:
            adm.add(m)
    return adm


def esu_masks(W, cls):
    en = PV.Enumerator(W, cls); cnt = en.count(); masks = np.zeros(cnt, np.int64); rr = np.zeros(cnt)
    PV.esu_pass(en.nbr, W.mu, W.mux, W.muy, np.zeros(W.n), W.n, cls.smax, cls.tlo, cls.thi, 0.0, 0.0, True, masks, rr, np.zeros(0), np.zeros(0, np.int64))
    return masks


rows = []; t_all = time.time()
for rec in art:
    nodes = rec.get("nodes") or rec.get("upper", {}).get("nodes") or rec.get("lower", {}).get("nodes")
    if nodes is None:
        print("artifact without stored nodes: skip", rec["window"], rec["K"], rec["beta"]); continue
    W = PC.Window(nodes, D); cls = PC.ZClass(W, rec["K"], rec["beta"]); t0 = time.time()
    row = dict(window=rec["window"], K=rec["K"], beta=rec["beta"], n_cells_artifact=rec["n_cells"])
    masks = esu_masks(W, cls); row["n_cells_esu_rerun"] = int(len(masks))
    if rec["n_cells"] <= a.full_limit:
        adm = bfs_enumerate(W, cls); row["coverage"] = "independent BFS enumeration"; row["n_cells_independent"] = len(adm)
        row["sets_equal"] = bool(adm == set(int(m) for m in masks)); row["esu_minus_bfs"] = int(len(set(int(m) for m in masks) - adm)); row["bfs_minus_esu"] = int(len(adm - set(int(m) for m in masks)))
    else:
        perm = rng.permutation(W.n); W2 = PC.Window([nodes[i] for i in perm], D); cls2 = PC.ZClass(W2, rec["K"], rec["beta"])
        row["coverage"] = "ESU theorem + relabelled rerun"; row["n_cells_relabelled"] = int(PV.Enumerator(W2, cls2).count()); row["sets_equal"] = (row["n_cells_relabelled"] == len(masks))
    for d_ in ("upper", "lower"):
        r_ = rec[d_]; du = r_["duals"]; pi = np.array(du["pi"]); theta = du["theta"]; b = du["b"]; sg = 1 if d_ == "upper" else -1
        if rec["n_cells"] <= a.full_limit:        # exact rational maximum reduced cost over the checker's own cell set
            piF = [Fraction(v) for v in du["pi"]]; thF = Fraction(theta); bF = Fraction(b); muF = [Fraction(float(v)) for v in W.mu]; sxF = [Fraction(float(v)) for v in W.mux]; syF = [Fraction(float(v)) * sg for v in W.muy]
            best = None
            for m in adm:
                mem = [i for i in range(W.n) if (m >> i) & 1]; t = sum((muF[i] for i in mem), Fraction(0)); Sx = sum((sxF[i] for i in mem), Fraction(0)); Sy = sum((syF[i] for i in mem), Fraction(0)); P = sum((piF[i] for i in mem), Fraction(0))
                r = (Sx * Sy - bF * Sx * Sx) / t - P - thF
                if best is None or r > best:
                    best = r
            eps_chk = float(best); mode = "exact rational"
        else:                                      # float pass with the proved forward-error inflation (3n+6 eps relative on each sum/product/division term)
            en = PV.Enumerator(W, cls); cnt, rmax, _ = en.max_reduced(sg, pi, theta, b)
            n_ = W.n; EPS = np.finfo(float).eps; Smag = float(np.abs(W.mux).sum() * np.abs(W.muy).sum() + abs(b) * np.abs(W.mux).sum() ** 2) / float(W.mu.min())
            eps_chk = float(rmax + (3 * n_ + 6) * EPS * (Smag + np.abs(pi).sum() + abs(theta))); mode = "float + proved inflation"
        okeps = eps_chk <= r_["eps_used"] + 0.0
        s_ = sum((Fraction(v) for v in du["pi"]), Fraction(0)) + rec["K"] * (Fraction(theta) + Fraction(eps_chk)); ub_chk = Fraction(b) + max(Fraction(0), s_ / Fraction(float(cls.Dmin)))
        row[d_] = dict(eps_checker=eps_chk, eps_artifact=r_["eps_used"], eps_ok=bool(okeps), mode=mode, ub_checker=float(ub_chk), ub_artifact=r_["ub_cert"], ub_ok=bool(float(ub_chk) <= r_["ub_cert"] * (1 + 1e-12) + 1e-12))
    row["seconds"] = time.time() - t0; row["pass"] = bool(row["sets_equal"] and row["upper"]["eps_ok"] and row["upper"]["ub_ok"] and row["lower"]["eps_ok"] and row["lower"]["ub_ok"])
    rows.append(row); print(f"win {row['window']} K={row['K']} beta={row['beta']} cells {row['n_cells_artifact']}: coverage={row['coverage']} sets_equal={row['sets_equal']} eps_ok=({row['upper']['eps_ok']},{row['lower']['eps_ok']}) ub_ok=({row['upper']['ub_ok']},{row['lower']['ub_ok']}) PASS={row['pass']} {row['seconds']:.0f}s", flush=True)
    json.dump(dict(rows=rows, n_pass=sum(r["pass"] for r in rows), n=len(rows), seconds=time.time() - t_all), open(os.path.join(a.out, "check_b15.json"), "w"), indent=1, default=float)
print("PASS", sum(r["pass"] for r in rows), "/", len(rows), f"{time.time()-t_all:.0f}s")
