"""b13_outer_compare.py -- outer-bound comparison (Supplementary Theorems S6-S7, V1; results exp2/B13): on the same instances (immutable IDs), by
actual VERIFIED solves, compare the certified outer bounds
    Phi'  (pixel LP: tent increments + Lemma S6.1)           [transport_core]
    X(1), X_L(1), X(2), X_L(2)  (lifted sets, zeal/outer_x.py; lens K = 9 incl. apex)
    Ubar_{Q_{2m}} + eps_{2m}    (Theorem S6 at nodal resolution 2m, m = 1, 2; keys *_13a_*)
against the realizable witness (inner Q_{2m} LP value, verified feasible) for both signs; widths, gaps, decisions.
Instances: the A2 synthetic stream (40) and the A2 real patches (12 per ladder; unique fine cells counted).
Residual convention: a certificate is 'violated' only if outer < witness - 1e-9 (negative slack); Ax - b residuals are not violations.
Usage: python b13_outer_compare.py --out <dir> [--synth 40] [--skip_real] [--ms 1,2]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, hashlib, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import solve_max, lemma12, grid_poly, inner_q1, eps_m, EPS
from outer_x import Mesh
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--synth", type=int, default=40); p.add_argument("--skip_real", action="store_true")
p.add_argument("--ms", default="1,2"); p.add_argument("--seed", type=int, default=23); p.add_argument("--patch", type=int, default=16); p.add_argument("--real_pairs", type=int, default=12); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); MS = [int(x) for x in a.ms.split(",")]
import synth; synth.set_rng(rng); from synth import build, random_mask, random_d, SPEC


def instance_id(lo, hi, G1, G2, d, h, mask):
    hsh = hashlib.sha256()
    for A in (lo, hi, G1, G2, d, np.array([h]), mask.astype(np.uint8)):
        hsh.update(np.ascontiguousarray(A).tobytes())
    return hsh.hexdigest()[:16]


def verified_solve(ms, obj):
    """max obj.x over the Mesh system with a verified dual upper bound (V1); returns (value, x, verified_ub)."""
    A, b = ms.system(); lo_n, hi_n = ms.lo_n, ms.hi_n
    res = linprog(-obj, A_ub=A, b_ub=b, bounds=list(zip(lo_n, hi_n)), method="highs")
    if res.status != 0:
        return np.nan, None, np.nan
    y = np.maximum(-res.ineqlin.marginals, 0.0); r = obj - A.T @ y; zp = np.maximum(r, 0); zm = np.maximum(-r, 0)
    hi_t = np.where(np.isfinite(hi_n), hi_n, 0.0); lo_t = np.where(np.isfinite(lo_n), lo_n, 0.0)
    if np.any((zp > 1e-12) & ~np.isfinite(hi_n)) or np.any((zm > 1e-12) & ~np.isfinite(lo_n)):
        return -res.fun, res.x, np.inf                      # residual on an unbounded coordinate: no finite dual certificate
    terms = np.concatenate([b * y, hi_t * zp, -lo_t * zm]); ub = float(np.sum(terms)) + (len(terms) + len(obj) + A.shape[0]) * EPS * float(np.sum(np.abs(terms)) + np.sum(np.abs(A.T @ y)) + np.sum(np.abs(obj)))
    return -res.fun, res.x, ub


def run_instance(lo, hi, G1, G2, h, mask, d, label):
    out = dict(label=label, id=instance_id(lo, hi, G1, G2, d, h, mask), n_pix=int(mask.sum()))
    lo2, hi2, _ = lemma12(lo, hi, G1, G2, h); polyC, _ = grid_poly(lo2, hi2, G1, G2, h, mask)
    for sgn, nm in ((1, "U"), (-1, "L")):
        dd = sgn * d
        r = solve_max(polyC, dd[mask]); out[f"{nm}_PhiC"] = sgn * r["verified"]           # verified outer
        for m in MS:
            # inner Q_{2m} witness and its verified optimum bound; Theorem S6 bracket at nodal resolution 2m
            v, x, ub = inner_q1(lo, hi, G1, G2, h, dd, 2 * m, mask, verified=True); e = eps_m(d, G1, G2, h, 2 * m, mask)
            out[f"{nm}_Q{2*m}_witness"] = sgn * v; out[f"{nm}_13a_{2*m}"] = sgn * (ub + e)
            for lens in (False, True):
                ms = Mesh(lo, hi, G1, G2, h, mask, 2 * m)
                if lens:
                    ms.add_lens(G1, G2, 9)
                vX, xX, ubX = verified_solve(ms, sgn * ms.obj_centre(d)); tag = f"XL{m}" if lens else f"X{m}"
                out[f"{nm}_{tag}"] = sgn * ubX; out[f"{nm}_{tag}_primal"] = sgn * vX
                if xX is not None:
                    out[f"{nm}_{tag}_witness"] = sgn * float((sgn * ms.obj_q1(d)) @ xX)   # Theorem S7: realizable witness from the outer optimizer
                if not lens:
                    out[f"beta_{m}"] = ms.deviation_bound(d, G1, G2, lens=False); out[f"beta_L{m}"] = ms.deviation_bound(d, G1, G2, lens=True)
    mmax = max(MS); wit_U = max(out[f"U_Q{2*m}_witness"] for m in MS); wit_L = min(out[f"L_Q{2*m}_witness"] for m in MS)
    out["witness_width"] = wit_U - wit_L
    for tag in ["PhiC"] + [t for m in MS for t in (f"X{m}", f"XL{m}", f"13a_{2*m}")]:
        U, L = out[f"U_{tag}"], out[f"L_{tag}"]
        out[f"width_{tag}"] = U - L; out[f"ratio_{tag}"] = (U - L) / out["witness_width"] if out["witness_width"] > 1e-3 else float("nan")
        out[f"violation_{tag}"] = bool(U < wit_U - 1e-9 or L > wit_L + 1e-9)
        out[f"crosses0_{tag}"] = bool(L < 0 < U)
    out["crosses0_witness"] = bool(wit_L < 0 < wit_U)
    out["best_cert_width"] = min(out[f"width_{t}"] for t in ["PhiC"] + [t for m in MS for t in (f"X{m}", f"XL{m}", f"13a_{2*m}")] if np.isfinite(out[f"width_{t}"]))
    print(f"  {label} [{out['id']}] n={out['n_pix']} ratios: Phi' {out['ratio_PhiC']:.3f} X1 {out['ratio_X1']:.3f} XL1 {out['ratio_XL1']:.3f}" + (f" X2 {out['ratio_X2']:.3f} XL2 {out['ratio_XL2']:.3f}" if 2 in MS else "") + f" 13a_2 {out['ratio_13a_2']:.3f}" + (f" 13a_4 {out['ratio_13a_4']:.3f}" if 2 in MS else ""), flush=True)
    return out


def summarize(rows, path, key=None):
    tags = ["PhiC"] + [t for m in MS for t in (f"X{m}", f"XL{m}", f"13a_{2*m}")]
    uniq = {}
    for r in rows:
        k = r[key] if key else r["id"]; uniq.setdefault(k, r)
    S = dict(rows=len(rows), unique=len(uniq))
    for tag in tags:
        vals = [r[f"ratio_{tag}"] for r in uniq.values() if np.isfinite(r[f"ratio_{tag}"])]
        S[tag] = dict(ratio_median=float(np.median(vals)) if vals else None, ratio_max=float(np.max(vals)) if vals else None, violations=int(sum(r[f"violation_{tag}"] for r in rows)),
                      unresolved_decisions=int(sum(r[f"crosses0_{tag}"] and not r["crosses0_witness"] for r in rows)), width_median=float(np.median([r[f"width_{tag}"] for r in uniq.values()])))
    json.dump(dict(summary=S, rows=rows), open(path, "w"), indent=1, default=float); print(json.dumps(S, indent=1))


rows = []
for t in range(a.synth):
    n1, n2 = int(rng.integers(5, 11)), int(rng.integers(5, 11)); h = float(rng.choice([0.1, 0.25])); widen = float(rng.choice([0.0, 0.3, 1.5]))
    mk = random_mask(n1, n2, rng.choice(["full", "full", "blobs"]))
    if mk.sum() < 6:
        continue
    inst = build(n1, n2, h, widen, mk); kind = str(rng.choice(["sparse", "nested"])); d = random_d(inst, kind)
    if d is None or abs(d.sum()) > 1e-12:
        continue
    rows.append(run_instance(inst["lo"], inst["hi"], inst["G1"], inst["G2"], h, mk, d, f"synth{t}_{kind}"))
summarize(rows, os.path.join(a.out, "b13_synthetic.json"))
if not a.skip_real:
    for fieldname in ("georgia", "gm_q4", "gm_bad", "mx_rwi"):
        F = np.load(paths.enclosure(fieldname)); lo, hi = F["lo"].astype(float), F["hi"].astype(float)
        g1l, g1h, g2l, g2h = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
        ras = np.load(SPEC[fieldname]); mask = ras["mask"].astype(bool) & np.isfinite(lo); N = mask.shape[0]; h = 2.0 / (N - 1)
        w = np.where(mask, ras["weight"], 0.0).astype(float); fine = np.where(mask, ras["tract_idx"], -1); coarse = np.where(mask, ras["county_idx"], -1)
        lo, hi, g1l, g1h, g2l, g2h, mask, w, fine, coarse = (A.T.copy() for A in (lo, hi, g1l, g1h, g2l, g2h, mask, w, fine, coarse))
        G1 = np.stack([g1l, g1h], -1); G2 = np.stack([g2l, g2h], -1)
        rng2 = np.random.default_rng(a.seed + 7); rr = []; P = a.patch; fines = np.unique(fine[fine >= 0]); tries = 0; seen = set()
        while len(rr) < a.real_pairs and tries < 600:
            tries += 1; f_ = int(rng2.choice(fines)); sel = fine == f_; npx = sel.sum()
            if npx < 4 or npx > P * P // 3 or f_ in seen:
                continue
            ii, jj = np.nonzero(sel); ci, cj = int(ii.mean()), int(jj.mean()); i0, j0 = max(0, ci - P // 2), max(0, cj - P // 2)
            sl = (slice(i0, i0 + P), slice(j0, j0 + P)); b = int(np.bincount(coarse[sel]).argmax()); mk = (coarse[sl] == b) & mask[sl]
            if (fine[sl] == f_).sum() < npx or mk.sum() < 2 * npx:
                continue
            wb = w[sl] * mk; wa = wb * (fine[sl] == f_)
            if wa.sum() <= 0:
                continue
            seen.add(f_); d = wa / wa.sum() - wb / wb.sum()
            rr.append(run_instance(lo[sl], hi[sl], G1[sl], G2[sl], h, mk, d, f"{fieldname}_f{f_}_b{b}")); rr[-1]["fine"] = f_
        summarize(rr, os.path.join(a.out, f"b13_real_{fieldname}.json"), key="fine")
