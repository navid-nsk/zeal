"""cert_export.py -- export real production certificates as exact-rational certificate.json files for the Lean checker
(lean/zeal/scripts/CertGen.lean; requirements V1 and V3 of Supplementary Note 8; formats in Supplementary Table 3).

(1) C5 map endpoints (per pair, per direction): the lifted system X^(m) is rebuilt (same declared construction as check_c5.py), every
    coefficient is converted to an EXACT rational: node bounds lo/hi = Fraction(float box value); segment costs = Fraction(hs) * Fraction(g)
    (exact product of the declared grid spacing and the declared derivative-box endpoint, not the rounded float product); objective
    c = Fraction(float d_p / m^2 contribution); dual y = Fraction(stored float); r = c - A^T y exactly; rp/rm = positive/negative parts;
    bound = y^T b + sum(rp*hi - rm*lo) exactly.  Kind "lp" (rows x_head - x_tail <= b_up, x_tail - x_head <= b_lo).  Also records whether
    the producer's float endpoint is >= the exact bound (then the reported endpoint is certified).  Pairs are selected by --max_rows
    (the list-based Lean checker is quadratic; all endpoints go through the array-based checker in cert_run_all.py).
(2) B15 pricing charges (per configuration, per direction): kind "b1" = (pi, theta, b, eps, K, Dmin, bound) with
    bound = b + max(0, (sum pi + K (theta + eps)) / Dmin) exactly (the charged-pricing arithmetic of Theorem S17; the enumeration coverage of eps stays with
    check_b15.py).  All 96 are tiny and kernel-checkable.
Usage: python cert_export.py --out <dir> [--fields georgia,gm_q4,gm_bad,mx_rwi] [--max_rows 4000] [--per_field 5]
"""
try:
    from . import paths                     # imported as part of the package
except ImportError:
    import paths                            # flat import (this directory on sys.path)
import argparse, io, json, os, sys, time, numpy as np, scipy.sparse as sp
from fractions import Fraction
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--fields", default="georgia,gm_q4,gm_bad,mx_rwi"); p.add_argument("--max_rows", type=int, default=4000)
p.add_argument("--per_field", type=int, default=5); p.add_argument("--skip_c5", action="store_true"); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
EXP = paths.results("exp2")
src = open(os.path.join(HERE, "check_c5.py"), encoding="utf-8").read()
exec(src[src.index("def lifted_system("):src.index("def dual_bound_outward(")])      # same declared construction (lifted_system, outer_objective)
F_ = lambda x: Fraction(float(x))
def rs(q): return f"{q.numerator}/{q.denominator}" if q.denominator != 1 else str(q.numerator)
log = []

# ---------------------------------------------------------------- (2) B15 charges
b15 = json.load(open(os.path.join(EXP, "B15", "with_duals", "pricing_verify_n30.json"), encoding="utf-8"))
nb = 0
for rec in b15:
    for d_ in ("upper", "lower"):
        r = rec[d_]; du = r["duals"]; pi = [F_(v) for v in du["pi"]]; theta = F_(du["theta"]); b = F_(du["b"]); eps = F_(r["eps_used"]); K = int(rec["K"]); Dmin = Fraction(1, 4)
        c = sum(pi, Fraction(0)) + K * (theta + eps); bound = b + max(Fraction(0), c / Dmin)
        name = f"b1_w{rec['window']}_K{K}_beta{str(rec['beta']).replace('.', 'p')}_{d_}"
        cert = dict(kind="b1", name=name, K=K, Dmin=rs(Dmin), pi=[rs(v) for v in pi], theta=rs(theta), b=rs(b), eps=rs(eps), bound=rs(bound),
                    note="Theorem B1 charge; eps = verified max reduced cost over the exhaustive enumeration (check_b15.py); scaled slope units")
        json.dump(cert, open(os.path.join(a.out, name + ".json"), "w")); nb += 1
        log.append(dict(kind="b1", name=name, bound_exact=float(bound), artifact_ub=r["ub_cert"], artifact_ge_exact=bool(r["ub_cert"] >= float(bound))))
print(f"B15: {nb} b1 certificates written", flush=True)

# ---------------------------------------------------------------- (1) C5 endpoints (smallest pairs per ladder)
if not a.skip_c5:
    for field in a.fields.split(","):
        M = 2 if field == "mx_rwi" else 4; ART = os.path.join(EXP, "C5_duals"); art = json.load(open(os.path.join(ART, f"c5_{field}.json"), encoding="utf-8"))["pairs"]
        F = np.load(paths.enclosure(field)); lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
        g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
        ras = np.load(SPEC[field]); mask0 = ras["mask"].astype(bool) & np.isfinite(lo0); N = mask0.shape[0]; h = 2.0 / (N - 1)
        w0 = np.where(mask0, ras["weight"], 0.0).astype(float); fine0 = np.where(mask0, ras["tract_idx"], -1); coarse0 = np.where(mask0, ras["county_idx"], -1)
        lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask0, w0, fine0, coarse0 = (A.T.copy() for A in (lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask0, w0, fine0, coarse0))
        small = sorted([r for r in art if r["rows"] <= a.max_rows], key=lambda r: r["rows"])[: a.per_field]
        for r in small:
            i0, i1, j0, j1 = r["window"]; sl = (slice(i0, i1), slice(j0, j1)); mk = (coarse0 == r["coarse"])[sl]
            A, bvec, lo_n, hi_n, nid, (tail, head, glo, ghi, hs) = lifted_system(lo0[sl], hi0[sl], g1l0[sl], g1h0[sl], g2l0[sl], g2h0[sl], mk, M)
            R = len(tail); hsF = F_(hs); bF = [hsF * F_(ghi[k]) for k in range(R)] + [-(hsF * F_(glo[k])) for k in range(R)]
            wb = w0[sl] * mk; wa = wb * (fine0[sl] == r["fine"]); d2 = wa / wa.sum() - wb / wb.sum(); c = outer_objective(d2, mk, nid, M)
            du = np.load(os.path.join(ART, f"duals_{field}_M{M}", r["dual_file"]))
            loF = [F_(v) for v in lo_n]; hiF = [F_(v) for v in hi_n]
            for nm, sgn in (("U", 1), ("L", -1)):
                cF = [F_(sgn * v) for v in c]; y = np.zeros(A.shape[0]); y[du[f"{nm}_idx"]] = du[f"{nm}_val"]; yF = [F_(v) for v in y]
                # r = c - A^T y exactly: row k (k < R): +1 at head[k], -1 at tail[k]; row R+k: +1 at tail[k], -1 at head[k]
                rr = list(cF)
                for k in range(R):
                    if yF[k] != 0:
                        rr[head[k]] -= yF[k]; rr[tail[k]] += yF[k]
                    if yF[R + k] != 0:
                        rr[tail[k]] -= yF[R + k]; rr[head[k]] += yF[R + k]
                rp = [max(v, Fraction(0)) for v in rr]; rm = [max(-v, Fraction(0)) for v in rr]
                bound = sum((yF[k] * bF[k] for k in range(2 * R)), Fraction(0)) + sum((rp[j] * hiF[j] - rm[j] * loF[j] for j in range(len(rr))), Fraction(0))
                name = f"c5_{field}_M{M}_c{r['coarse']}_f{r['fine']}_{nm}"
                rows = [dict(coefs=[[int(head[k]), "1"], [int(tail[k]), "-1"]], rhs=rs(bF[k])) for k in range(R)] + [dict(coefs=[[int(tail[k]), "1"], [int(head[k]), "-1"]], rhs=rs(bF[R + k])) for k in range(R)]
                cert = dict(kind="lp", name=name, n=int(A.shape[1]), rows=rows, obj=[[j, rs(cF[j])] for j in range(len(cF)) if cF[j] != 0], lo=[rs(v) for v in loF], hi=[rs(v) for v in hiF],
                            cert=dict(y=[rs(v) for v in yF], rp=[rs(v) for v in rp], rm=[rs(v) for v in rm], bound=rs(bound)),
                            note=f"C5 {field} pair (coarse {r['coarse']}, fine {r['fine']}), {nm} endpoint of the X^({M//2}) outer; exact-rational coefficients of the declared numerical problem")
                json.dump(cert, open(os.path.join(a.out, name + ".json"), "w"))
                art_v = r[f"{nm}_X"]; ok = (art_v >= sgn * float(bound)) if sgn == 1 else (art_v <= sgn * float(bound))
                log.append(dict(kind="lp", name=name, field=field, n=int(A.shape[1]), rows=int(2 * R), bound_exact=float(sgn * bound), artifact=art_v, artifact_certified=bool(ok)))
                print(f"  {name}: n={A.shape[1]} rows={2*R} exact bound {sgn*float(bound):.6f} artifact {art_v:.6f} certified={ok}", flush=True)
json.dump(log, open(os.path.join(a.out, "export_log.json"), "w"), indent=1); print("done", len(log), "certificates")
