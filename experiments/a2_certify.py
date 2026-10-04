"""a2_certify.py -- (1) certify synth28/synth3 with the interpolation-completed Q1 bracket [L_m, Ubar_m + eps_m]
(Supplementary Theorem S6) and decide the candidate counterexample to the pixel-level allowance (keys R2_*; cf.
Supplementary Proposition S1); (2) verify the exact 1-D formulas (Theorem S8) against the 89 A2 1-D instances (same rng replay).
Usage: python a2_certify.py"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import io, json, os, sys, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import solve_max, lemma12, grid_poly, inner_q1, eps_m, exact_1d, r1_1d
OUT = paths.results("exp1", "A2"); rng = np.random.default_rng(23)
import synth; synth.set_rng(rng); from synth import build, random_mask, random_d
# ---- (2) replay the 1-D instances and check the exact formulas
rows = []
for N, h in ((12, 1.0), (18, 0.5), (30, 0.25)):
    for pat in ("peak", "double_peak", "alternating", "random", "nested"):
        for rep in range(6):
            gm = rng.uniform(-1, 0.2, N); gp = gm + rng.uniform(0.2, 1.5, N); d = np.zeros(N)
            if pat == "peak":
                k = rng.integers(1, N - 1); d[k - 1:k + 2] = [-1, 2, -1]
            elif pat == "double_peak":
                k1, k2 = sorted(rng.choice(np.arange(1, N - 1), 2, replace=False))
                if k2 - k1 < 3:
                    continue
                d[k1 - 1:k1 + 2] += [-1, 2, -1]; d[k2 - 1:k2 + 2] += [-0.5, 1, -0.5]
            elif pat == "alternating":
                d = np.array([(-1) ** k for k in range(N)], float); d -= d.mean()
            elif pat == "random":
                d = rng.normal(size=N); d -= d.mean()
            else:
                k0, k1 = sorted(rng.choice(np.arange(N), 2, replace=False)); d[k0:k1 + 1] = 1 / (k1 - k0 + 1); d -= 1 / N
            ex = exact_1d(gm, gp, d, h); lp = r1_1d(gm, gp, d, h, m_inner=32)
            rows.append(dict(pattern=pat, N=N, UF_formula=ex["UF"], inner32=lp["inner"], UPhi_formula=ex["UPhi"], outer_lp=lp["outer"], gap_formula=ex["gap_formula"],
                             err_UPhi=abs(ex["UPhi"] - lp["outer"]), UF_minus_inner32=ex["UF"] - lp["inner"], gap_formula_vs_lp=abs(ex["gap_formula"] - (lp["outer"] - lp["inner"]))))
S1 = dict(instances=len(rows), max_err_UPhi=float(max(r["err_UPhi"] for r in rows)), UF_minus_inner32_max=float(max(r["UF_minus_inner32"] for r in rows)),
          UF_minus_inner32_min=float(min(r["UF_minus_inner32"] for r in rows)), gap_formula_vs_lp_max=float(max(r["gap_formula_vs_lp"] for r in rows)),
          UF_minus_inner32_median=float(np.median([r["UF_minus_inner32"] for r in rows])))
print("1-D exact formulas:", json.dumps(S1)); json.dump(dict(summary=S1, rows=rows), open(os.path.join(OUT, "a2_exact1d.json"), "w"), indent=1, default=float)
# ---- (1) replay the synthetic 2-D stream; certify synth28 and synth3
cert = []
for t in range(40):
    n1, n2 = int(rng.integers(5, 11)), int(rng.integers(5, 11)); h = float(rng.choice([0.1, 0.25])); widen = float(rng.choice([0.0, 0.3, 1.5]))
    mk = random_mask(n1, n2, rng.choice(["full", "full", "blobs"]))
    if mk.sum() < 6:
        continue
    inst = build(n1, n2, h, widen, mk); kind = str(rng.choice(["sparse", "nested"])); d = random_d(inst, kind)
    if d is None or abs(d.sum()) > 1e-12:
        continue
    if t not in (28, 3):
        continue
    lo, hi, G1, G2 = inst["lo"], inst["hi"], inst["G1"], inst["G2"]
    lo2, hi2, _ = lemma12(lo, hi, G1, G2, h); polyC, _ = grid_poly(lo2, hi2, G1, G2, h, mk); UPhiC = solve_max(polyC, d[mk])
    r = dict(label=f"synth{t}", n_pix=int(mk.sum()), rectangular=bool(mk.all()), U_PhiC=UPhiC["value"], U_PhiC_verified=UPhiC["verified"], R2_bound=eps_m(d, G1, G2, h, 1, mk))
    for m in (8, 16, 32):
        v, x, ub = inner_q1(lo, hi, G1, G2, h, d, m, mk, verified=True); e = eps_m(d, G1, G2, h, m, mk)
        r[f"m{m}"] = dict(witness=v, inner_lp_verified_ub=ub, eps_m=e, UF_upper=ub + e)
        print(f"  synth{t} m={m}: witness {v:.6f}  inner-LP verified ub {ub:.6f}  eps_m {e:.6f}  => U^F <= {ub+e:.6f}", flush=True)
    best = min(r[f"m{m}"]["UF_upper"] for m in (8, 16, 32)); r["UF_upper_best"] = best; r["falsification_threshold"] = r["U_PhiC"] - r["R2_bound"]
    r["R2_falsified"] = bool(best < r["falsification_threshold"]); r["gap_lower_bound"] = r["U_PhiC"] - best; r["gap_lower_over_R2bound"] = r["gap_lower_bound"] / r["R2_bound"]
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("m")}, indent=1, default=float)); cert.append(r)
json.dump(cert, open(os.path.join(OUT, "a2_certify_r2.json"), "w"), indent=1, default=float)
