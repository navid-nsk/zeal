"""a2_realization.py -- realization bracket (Supplementary Theorems S6-S8, Lemma S6.1); results exp1/A2.

For each instance (synthetic sinusoidal patches with sound boxes; real patches cut from the four ladders' tile
certificates around a small fine cell) and each sign of the objective:
  outer  : U_Phi (pixel LP), U_Phi' (Lemma S6.1 tightened), sub-pixel outer LP m=2 with coupling
  inner  : U_in^{Q1}(m), m = 1, 2, 4, 8  (Theorem S6; witnesses are genuine Lipschitz fields of the box class)
  gaps   : g+ = U_out - U_in(m) (share units), relative gap when U_in exceeds a floor, decision-crossing flag
  R2     : pixel-level allowance test (U_Phi' - U_in(8)) / ((h/8) sum |d| (w1 + w2)) (keys R2_*) -- sufficient only, failure refutes nothing
  1-D slices (Theorem S8; file a2_r1_1d.json): outer vs inner(32) vs (h/8) sum |d| w on peak, alternating and random patterns (observations)
Usage: python a2_realization.py --out <dir> [--synth 40] [--real_pairs 12] [--patch 16] [--skip_real]
Outputs: <out>/a2_synthetic.json, <out>/a2_real_<field>.json, <out>/a2_r1_1d.json
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import solve_max, lemma12, grid_poly, inner_q1, outer_sub, r1_1d, eps_m

p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--synth", type=int, default=40)
p.add_argument("--real_pairs", type=int, default=12); p.add_argument("--patch", type=int, default=16); p.add_argument("--skip_real", action="store_true")
p.add_argument("--ms", default="1,2,4,8"); p.add_argument("--seed", type=int, default=23); p.add_argument("--only_r1", action="store_true"); p.add_argument("--only_real", action="store_true"); p.add_argument("--tag", default=""); p.add_argument("--verified", action="store_true"); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); MS = [int(x) for x in a.ms.split(",")]
import synth; synth.set_rng(rng); from synth import build, random_mask, random_d, SPEC
FLOOR = 1e-3   # share units: ratios reported only when the inner value exceeds this


def bracket(lo, hi, G1, G2, h, mask, d, label, t0):
    """one instance, both signs. Returns dict of endpoints and gaps."""
    out = dict(label=label, n_pix=int(mask.sum()))
    lo2, hi2, _ = lemma12(lo, hi, G1, G2, h)
    polyP, _ = grid_poly(lo, hi, G1, G2, h, mask); polyC, _ = grid_poly(lo2, hi2, G1, G2, h, mask)
    dv = d[mask]
    for sgn, name in ((1, "U"), (-1, "L")):
        dd = sgn * dv
        rP = solve_max(polyP, dd); rC = solve_max(polyC, dd)
        out[f"{name}_Phi"] = sgn * rP["value"]; out[f"{name}_Phi_ver"] = sgn * rP["verified"]; out[f"{name}_PhiC"] = sgn * rC["value"]; out[f"{name}_PhiC_ver"] = sgn * rC["verified"]
        try:
            out[f"{name}_sub2"] = sgn * outer_sub(lo, hi, G1, G2, h, sgn * d, 2, couple=True, mask=mask)[0]
        except Exception as e:
            out[f"{name}_sub2"] = float("nan"); out["sub2_error"] = str(e)
        for m in MS:
            try:
                if a.verified:
                    v, _, ub = inner_q1(lo, hi, G1, G2, h, sgn * d, m, mask, verified=True); out[f"{name}_in{m}"] = sgn * v; out[f"{name}_in{m}_lpver"] = sgn * ub; out[f"eps{m}"] = eps_m(d, G1, G2, h, m, mask)
                    out[f"{name}_cert{m}"] = (min(out[f"{name}_PhiC"], ub + out[f"eps{m}"]) if sgn > 0 else max(out[f"{name}_PhiC"], -ub - out[f"eps{m}"]))   # certified outer: min{Phi', Ubar_m + eps_m}
                else:
                    v, _ = inner_q1(lo, hi, G1, G2, h, sgn * d, m, mask); out[f"{name}_in{m}"] = sgn * v
            except Exception as e:
                out[f"{name}_in{m}"] = float("nan"); out[f"in{m}_error"] = str(e)
    mmax = max(MS)
    out["gap_plus"] = out["U_PhiC"] - out[f"U_in{mmax}"]; out["gap_minus"] = out[f"L_in{mmax}"] - out["L_PhiC"]
    if a.verified:
        out["width_certified"] = out[f"U_cert{mmax}"] - out[f"L_cert{mmax}"]; out["gap_plus_certified"] = out[f"U_cert{mmax}"] - out[f"U_in{mmax}"]; out["gap_minus_certified"] = out[f"L_in{mmax}"] - out[f"L_cert{mmax}"]
    out["width_outer"] = out["U_PhiC"] - out["L_PhiC"]; out["width_inner"] = out[f"U_in{mmax}"] - out[f"L_in{mmax}"]
    out["width_ratio"] = out["width_outer"] / out["width_inner"] if out["width_inner"] > FLOOR else float("nan")
    out["rel_gap_plus"] = out["gap_plus"] / abs(out[f"U_in{mmax}"]) if abs(out[f"U_in{mmax}"]) > FLOOR else float("nan")
    out["crosses_zero_outer_not_inner"] = bool((out["L_PhiC"] < 0 < out["U_PhiC"]) and not (out[f"L_in{mmax}"] < 0 < out[f"U_in{mmax}"]))
    w1 = G1[..., 1] - G1[..., 0]; w2 = G2[..., 1] - G2[..., 0]
    out["R2_bound"] = float(h / 8 * np.sum(np.abs(d[mask]) * (w1[mask] + w2[mask])))
    out["R2_ratio_plus"] = out["gap_plus"] / out["R2_bound"] if out["R2_bound"] > 0 else float("nan")
    out["R2_ratio_minus"] = out["gap_minus"] / out["R2_bound"] if out["R2_bound"] > 0 else float("nan")
    mono = all(out[f"U_in{MS[i]}"] <= out[f"U_in{MS[i+1]}"] + 1e-9 for i in range(len(MS) - 1)) and all(out[f"L_in{MS[i]}"] >= out[f"L_in{MS[i+1]}"] - 1e-9 for i in range(len(MS) - 1))
    order = out[f"U_in{mmax}"] <= out["U_PhiC"] + 1e-9 <= out["U_Phi"] + 2e-9 and out[f"L_in{mmax}"] >= out["L_PhiC"] - 1e-9 >= out["L_Phi"] - 2e-9
    out["monotone"] = bool(mono); out["ordered"] = bool(order)
    print(f"  {label}: n={out['n_pix']} U: Phi={out['U_Phi']:.5f} Phi'={out['U_PhiC']:.5f} sub2={out['U_sub2']:.5f} in={[round(out[f'U_in{m}'],5) for m in MS]}  g+={out['gap_plus']:.2e} "
          f"R2={out['R2_ratio_plus']:.2f}  widthratio={out['width_ratio']:.3f} mono={mono} ord={order} {time.time()-t0:.0f}s", flush=True)
    return out


def run_synthetic():
    rows = []; t0 = time.time()
    for t in range(a.synth):
        n1, n2 = int(rng.integers(5, 11)), int(rng.integers(5, 11)); h = float(rng.choice([0.1, 0.25])); widen = float(rng.choice([0.0, 0.3, 1.5]))
        mk = random_mask(n1, n2, rng.choice(["full", "full", "blobs"]))
        if mk.sum() < 6:
            continue
        inst = build(n1, n2, h, widen, mk); kind = str(rng.choice(["sparse", "nested"])); d = random_d(inst, kind)
        if d is None or abs(d.sum()) > 1e-12:
            continue
        rows.append(bracket(inst["lo"], inst["hi"], inst["G1"], inst["G2"], h, mk, d, f"synth{t}_{kind}_w{widen}", t0))
    summarize(rows, os.path.join(a.out, "a2_synthetic.json"))


def summarize(rows, path):
    mmax = max(MS); ok = [r for r in rows if np.isfinite(r["gap_plus"])]
    S = dict(instances=len(rows), monotone_all=all(r["monotone"] for r in rows), ordered_all=all(r["ordered"] for r in rows),
             gap_plus_median=float(np.median([r["gap_plus"] for r in ok])), gap_plus_max=float(np.max([r["gap_plus"] for r in ok])),
             gap_minus_median=float(np.median([r["gap_minus"] for r in ok])),
             width_ratio_median=float(np.nanmedian([r["width_ratio"] for r in ok])), width_ratio_max=float(np.nanmax([r["width_ratio"] for r in ok])),
             width_ratio_Phi_raw_median=float(np.nanmedian([(r["U_Phi"] - r["L_Phi"]) / r["width_inner"] for r in ok if r["width_inner"] > FLOOR])),
             rel_gap_plus_median=float(np.nanmedian([r["rel_gap_plus"] for r in ok])), rel_gap_plus_max=float(np.nanmax([r["rel_gap_plus"] for r in ok])),
             R2_ratio_max=float(np.nanmax([max(r["R2_ratio_plus"], r["R2_ratio_minus"]) for r in ok])), R2_violations=int(sum(max(r["R2_ratio_plus"], r["R2_ratio_minus"]) > 1 + 1e-9 for r in ok)),
             crosses_zero_outer_not_inner=int(sum(r["crosses_zero_outer_not_inner"] for r in ok)),
             sub2_over_PhiC_median=float(np.nanmedian([(r["U_sub2"] - r[f"U_in{mmax}"]) / r["gap_plus"] for r in ok if r["gap_plus"] > 1e-9])),
             inner_progress=[float(np.nanmedian([(r[f"U_in{m}"] - r[f"U_in{MS[0]}"]) / r["gap_plus"] for r in ok if r["gap_plus"] > 1e-9])) for m in MS])
    if a.verified:
        S["width_certified_over_inner_median"] = float(np.nanmedian([r["width_certified"] / r["width_inner"] for r in ok if r["width_inner"] > FLOOR])); S["width_certified_over_inner_max"] = float(np.nanmax([r["width_certified"] / r["width_inner"] for r in ok if r["width_inner"] > FLOOR]))
        S["gap_plus_certified_median"] = float(np.median([r["gap_plus_certified"] for r in ok])); S["crosses_zero_certified_not_inner"] = int(sum((r[f"L_cert{mmax}"] < 0 < r[f"U_cert{mmax}"]) and not (r[f"L_in{mmax}"] < 0 < r[f"U_in{mmax}"]) for r in ok))
    json.dump(dict(summary=S, rows=rows), open(path, "w"), indent=1, default=float); print(json.dumps(S, indent=1))


def run_real(fieldname):
    F = np.load(paths.enclosure(fieldname)); lo, hi = F["lo"].astype(float), F["hi"].astype(float)
    g1l, g1h, g2l, g2h = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
    ras = np.load(SPEC[fieldname]); mask = ras["mask"].astype(bool) & np.isfinite(lo); N = mask.shape[0]; h = 2.0 / (N - 1)
    w = np.where(mask, ras["weight"], 0.0).astype(float); fine = np.where(mask, ras["tract_idx"], -1); coarse = np.where(mask, ras["county_idx"], -1)
    lo, hi, g1l, g1h, g2l, g2h, mask, w, fine, coarse = (A.T.copy() for A in (lo, hi, g1l, g1h, g2l, g2h, mask, w, fine, coarse))
    G1 = np.stack([g1l, g1h], -1); G2 = np.stack([g2l, g2h], -1)
    rng2 = np.random.default_rng(a.seed + 7); rows = []; t0 = time.time(); P = a.patch
    fines = np.unique(fine[fine >= 0]); tries = 0
    while len(rows) < a.real_pairs and tries < 400:
        tries += 1; f_ = int(rng2.choice(fines)); sel = fine == f_; npx = sel.sum()
        if npx < 4 or npx > P * P // 3:
            continue
        ii, jj = np.nonzero(sel); ci, cj = int(ii.mean()), int(jj.mean()); i0, j0 = max(0, ci - P // 2), max(0, cj - P // 2)
        sl = (slice(i0, i0 + P), slice(j0, j0 + P)); b = int(np.bincount(coarse[sel]).argmax())
        mk = (coarse[sl] == b) & mask[sl]
        if not (fine[sl] == f_).all() and (fine[sl] == f_).sum() < npx:   # fine cell must lie inside the patch
            continue
        if mk.sum() < 2 * npx:
            continue
        wb = w[sl] * mk; wa = wb * (fine[sl] == f_)
        if wa.sum() <= 0:
            continue
        d = wa / wa.sum() - wb / wb.sum()
        rows.append(bracket(lo[sl], hi[sl], G1[sl], G2[sl], h, mk, d, f"{fieldname}_f{f_}_b{b}", t0))
    summarize(rows, os.path.join(a.out, f"a2_real_{fieldname}{a.tag}.json"))


def run_r1():
    rows = []
    for N, h in ((12, 1.0), (18, 0.5), (30, 0.25)):
        for pat in ("peak", "double_peak", "alternating", "random", "nested"):
            for rep in range(6):
                gm = rng.uniform(-1, 0.2, N); gp = gm + rng.uniform(0.2, 1.5, N)
                d = np.zeros(N)
                if pat == "peak":
                    k = rng.integers(1, N - 1); d[k - 1:k + 2] = [-1, 2, -1]
                elif pat == "double_peak":
                    k1, k2 = sorted(rng.choice(np.arange(1, N - 1), 2, replace=False))
                    if k2 - k1 < 3: continue
                    d[k1 - 1:k1 + 2] += [-1, 2, -1]; d[k2 - 1:k2 + 2] += [-0.5, 1, -0.5]
                elif pat == "alternating":
                    d = np.array([(-1) ** k for k in range(N)], float); d -= d.mean()
                elif pat == "random":
                    d = rng.normal(size=N); d -= d.mean()
                else:
                    k0, k1 = sorted(rng.choice(np.arange(N), 2, replace=False)); d[k0:k1 + 1] = 1 / (k1 - k0 + 1); d -= 1 / N
                r = r1_1d(gm, gp, d, h); r.update(N=N, h=h, pattern=pat, rep=rep)
                # the bound restricted to sign changes of d (observation, not a claim)
                sgn = np.sign(d); peaks = [p for p in range(N) if d[p] != 0 and ((p > 0 and sgn[p - 1] == -sgn[p]) or (p < N - 1 and sgn[p + 1] == -sgn[p]))]
                r["bound_signchanges"] = float(h / 8 * sum(abs(d[p]) * (gp[p] - gm[p]) for p in peaks)); r["ratio_signchanges"] = r["gap"] / r["bound_signchanges"] if r["bound_signchanges"] > 0 else float("nan")
                rows.append(r)
    S = dict(instances=len(rows), ratio_max=float(np.nanmax([r["ratio"] for r in rows])), violations=int(sum(r["ratio"] > 1 + 1e-7 for r in rows)),
             by_pattern={pat: dict(n=len([r for r in rows if r["pattern"] == pat]), ratio_median=float(np.nanmedian([r["ratio"] for r in rows if r["pattern"] == pat])), ratio_max=float(np.nanmax([r["ratio"] for r in rows if r["pattern"] == pat])),
                              gap_median=float(np.median([r["gap"] for r in rows if r["pattern"] == pat])), ratio_signchanges_median=float(np.nanmedian([r["ratio_signchanges"] for r in rows if r["pattern"] == pat])),
                              ratio_signchanges_max=float(np.nanmax([r["ratio_signchanges"] for r in rows if r["pattern"] == pat])),
                              ratio_kinks_median=float(np.nanmedian([r["ratio_kinks"] for r in rows if r["pattern"] == pat])), ratio_kinks_min=float(np.nanmin([r["ratio_kinks"] for r in rows if r["pattern"] == pat])),
                              ratio_kinks_max=float(np.nanmax([r["ratio_kinks"] for r in rows if r["pattern"] == pat])), n_kinks_median=float(np.median([len(r["kinks"]) for r in rows if r["pattern"] == pat]))) for pat in ("peak", "double_peak", "alternating", "random", "nested")})
    json.dump(dict(summary=S, rows=rows), open(os.path.join(a.out, "a2_r1_1d.json"), "w"), indent=1, default=float); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    run_r1()
    if a.only_r1:
        sys.exit(0)
    if not a.only_real:
        run_synthetic()
    else:
        # consume the synthetic rng stream so that the real-patch rng2 selection is unchanged (it uses its own seed) -- nothing to do
        pass
    if not a.skip_real:
        for fn in ("georgia", "gm_q4", "gm_bad", "mx_rwi"):
            run_real(fn)
