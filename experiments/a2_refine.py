"""a2_refine.py -- replay the A2 synthetic stream (same seed) and refine selected instances: inner Q1 at m up to 32 and
sub-pixel outer LP (with Lemma S6.1 at the sub-pixel scale) at m up to 8, to decide whether an apparent violation of the pixel-level allowance (keys R2_*)
survives convergence. Usage: python a2_refine.py --out <dir> --labels synth28,synth3 [--seed 23] [--synth 40]"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import solve_max, lemma12, grid_poly, inner_q1, outer_sub
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--labels", required=True); p.add_argument("--seed", type=int, default=23)
p.add_argument("--synth", type=int, default=40); p.add_argument("--m_in", default="8,16,32"); p.add_argument("--m_out", default="2,4,8"); a = p.parse_args()
rng = np.random.default_rng(a.seed); import synth; synth.set_rng(rng); from synth import build, random_mask, random_d
want = set(a.labels.split(",")); out = []
# replay the rng draws of a2_realization.run_r1 (it runs before run_synthetic and consumes the same stream)
for N, h in ((12, 1.0), (18, 0.5), (30, 0.25)):
    for pat in ("peak", "double_peak", "alternating", "random", "nested"):
        for rep in range(6):
            gm = rng.uniform(-1, 0.2, N); gp = gm + rng.uniform(0.2, 1.5, N)
            if pat == "peak":
                rng.integers(1, N - 1)
            elif pat == "double_peak":
                rng.choice(np.arange(1, N - 1), 2, replace=False)
            elif pat == "random":
                rng.normal(size=N)
            elif pat == "nested":
                rng.choice(np.arange(N), 2, replace=False)
for t in range(a.synth):
    n1, n2 = int(rng.integers(5, 11)), int(rng.integers(5, 11)); h = float(rng.choice([0.1, 0.25])); widen = float(rng.choice([0.0, 0.3, 1.5]))
    mk = random_mask(n1, n2, rng.choice(["full", "full", "blobs"]))
    if mk.sum() < 6:
        continue
    inst = build(n1, n2, h, widen, mk); kind = str(rng.choice(["sparse", "nested"])); d = random_d(inst, kind)
    if d is None or abs(d.sum()) > 1e-12:
        continue
    label = f"synth{t}"
    if label not in want:
        continue
    lo, hi, G1, G2 = inst["lo"], inst["hi"], inst["G1"], inst["G2"]; t0 = time.time()
    lo2, hi2, _ = lemma12(lo, hi, G1, G2, h); polyC, _ = grid_poly(lo2, hi2, G1, G2, h, mk)
    r = dict(label=label, n_pix=int(mk.sum()), h=h, widen=widen, kind=kind, U_PhiC=solve_max(polyC, d[mk])["value"], L_PhiC=-solve_max(polyC, -d[mk])["value"])
    w1 = G1[..., 1] - G1[..., 0]; w2 = G2[..., 1] - G2[..., 0]; r["R2_bound"] = float(h / 8 * np.sum(np.abs(d[mk]) * (w1[mk] + w2[mk])))
    for m in [int(x) for x in a.m_out.split(",")]:
        try:
            r[f"U_out{m}"] = outer_sub(lo, hi, G1, G2, h, d, m, True, mk)[0]; r[f"L_out{m}"] = -outer_sub(lo, hi, G1, G2, h, -d, m, True, mk)[0]
        except Exception as e:
            r[f"out{m}_error"] = str(e)
        print(f"  {label} outer m={m}: U={r.get(f'U_out{m}', np.nan):.6f} L={r.get(f'L_out{m}', np.nan):.6f} {time.time()-t0:.0f}s", flush=True)
    for m in [int(x) for x in a.m_in.split(",")]:
        try:
            r[f"U_in{m}"] = inner_q1(lo, hi, G1, G2, h, d, m, mk)[0]; r[f"L_in{m}"] = -inner_q1(lo, hi, G1, G2, h, -d, m, mk)[0]
        except Exception as e:
            r[f"in{m}_error"] = str(e)
        print(f"  {label} inner m={m}: U={r.get(f'U_in{m}', np.nan):.6f} L={r.get(f'L_in{m}', np.nan):.6f} {time.time()-t0:.0f}s", flush=True)
    mi = max(int(x) for x in a.m_in.split(",")); mo = max(int(x) for x in a.m_out.split(","))
    r["gap_PhiC_minus_inner"] = r["U_PhiC"] - r[f"U_in{mi}"]; r["R2_ratio_PhiC"] = r["gap_PhiC_minus_inner"] / r["R2_bound"]
    r["gap_outer_refined_minus_inner"] = r.get(f"U_out{mo}", np.nan) - r[f"U_in{mi}"]
    out.append(r); print(json.dumps(r, indent=1, default=float))
json.dump(out, open(os.path.join(a.out, "a2_refine.json"), "w"), indent=1, default=float)
