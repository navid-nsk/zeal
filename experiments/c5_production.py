"""c5_production.py -- map-level transport certificates (Supplementary Theorems S3-S5, S7, S10; V1-V2; results exp2/C5) on the four
ladders with the production outer: the lifted pure-network set X^(2) (Theorem S7, mesh h/4) with verified dual bounds, for
EVERY nested pair (fine cell a inside coarse cell b), both endpoints; the realizable witness from the outer optimizer (Theorem S7:
Q_4 field), the pixel LP Phi' and the value-only V for comparison; the map-level separable bounds (RMS over pairs) S_X, S_Phi', S_V
beside the observed movement of the midpoint reference; local gaps g+-, and decisions (sign of the pair difference) resolved by the
outer / by the witness. Coarse cells larger than --max_pix pixels are solved on a window cropped around each fine cell (the pair
becomes (a, b ∩ window), named as such). Usage: python c5_production.py --field georgia [--max_pix 3000] [--max_pairs 0] [--M 4] [--out <dir>]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import solve_max, lemma12, grid_poly, verified_solve_system
from outer_x import Mesh
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--max_pix", type=int, default=3000); p.add_argument("--max_pairs", type=int, default=0)
p.add_argument("--M", type=int, default=4); p.add_argument("--out", default=None); p.add_argument("--seed", type=int, default=5); a = p.parse_args()
OUT = a.out or paths.results("exp2", "C5"); os.makedirs(OUT, exist_ok=True)
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
ras = np.load(SPEC[a.field]); mask = ras["mask"].astype(bool) & np.isfinite(lo0); N = mask.shape[0]; h = 2.0 / (N - 1)
w = np.where(mask, ras["weight"], 0.0).astype(float); fine = np.where(mask, ras["tract_idx"], -1); coarse = np.where(mask, ras["county_idx"], -1)
lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, fine, coarse = (A.T.copy() for A in (lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, fine, coarse))
G1 = np.stack([g1l0, g1h0], -1); G2 = np.stack([g2l0, g2h0], -1); lam = (lo0 + hi0) / 2
old = json.load(open(paths.transport_lp(a.field)))["summary"]
rng = np.random.default_rng(a.seed); rows = []; t0 = time.time(); mu_tot = w.sum(); acc = dict(mov=0.0, X=0.0, P=0.0, V=0.0, Xw=0.0)
cells = list(np.unique(coarse[coarse >= 0])); rng.shuffle(cells); npairs = 0
for b in cells:
    b = int(b); sel = coarse == b; fines = [f_ for f_ in np.unique(fine[sel]) if f_ >= 0 and w[sel & (fine == f_)].sum() > 0 and (fine[sel] == f_).sum() < sel.sum()]
    if not fines:
        continue
    big = sel.sum() > a.max_pix
    if not big:
        i0, i1 = np.nonzero(sel.any(1))[0][[0, -1]]; j0, j1 = np.nonzero(sel.any(0))[0][[0, -1]]; sl = (slice(i0, i1 + 1), slice(j0, j1 + 1)); mk = sel[sl]
        ms = Mesh(lo0[sl], hi0[sl], G1[sl], G2[sl], h, mk, a.M); lo2, hi2, _ = lemma12(lo0[sl], hi0[sl], G1[sl], G2[sl], h); polyC, _ = grid_poly(lo2, hi2, G1[sl], G2[sl], h, mk)
        A_, b_ = ms.system()
    for f_ in fines:
        if big:   # crop a window around the fine cell
            fs = sel & (fine == f_); ii, jj = np.nonzero(fs); ci, cj = int(ii.mean()), int(jj.mean()); half = int(np.sqrt(a.max_pix) / 2)
            i0, i1 = max(0, ci - half), min(N - 1, ci + half); j0, j1 = max(0, cj - half), min(N - 1, cj + half)
            if not ((ii >= i0) & (ii <= i1) & (jj >= j0) & (jj <= j1)).all():
                continue
            sl = (slice(i0, i1 + 1), slice(j0, j1 + 1)); mk = sel[sl]
            ms = Mesh(lo0[sl], hi0[sl], G1[sl], G2[sl], h, mk, a.M); lo2, hi2, _ = lemma12(lo0[sl], hi0[sl], G1[sl], G2[sl], h); polyC, _ = grid_poly(lo2, hi2, G1[sl], G2[sl], h, mk); A_, b_ = ms.system()
        wb = w[sl] * mk; wa = wb * (fine[sl] == f_); d2 = wa / wa.sum() - wb / wb.sum(); d = d2[mk]; mu_a = wa.sum()
        r = dict(coarse=b, fine=int(f_), cropped=bool(big), n_pix=int(mk.sum()), mu_a=float(mu_a), real_mid=float(d @ lam[sl][mk]))
        oc, oq = ms.obj_centre(d2), ms.obj_q1(d2)
        for sgn, nm in ((1, "U"), (-1, "L")):
            vX, xX, ubX = verified_solve_system(A_, b_, ms.lo_n, ms.hi_n, sgn * oc); r[f"{nm}_X"] = sgn * ubX; r[f"{nm}_X_primal"] = sgn * vX
            r[f"{nm}_wit"] = sgn * float((sgn * oq) @ xX) if xX is not None else float("nan")
            rp = solve_max(polyC, sgn * d); r[f"{nm}_PhiC"] = sgn * rp["verified"]
            dp, dm = np.maximum(sgn * d, 0), np.maximum(-sgn * d, 0); r[f"{nm}_V"] = sgn * float(dp @ polyC.hi - dm @ polyC.lo)
        r["gap_plus"] = r["U_X"] - r["U_wit"]; r["gap_minus"] = r["L_wit"] - r["L_X"]; r["width_X"] = r["U_X"] - r["L_X"]; r["width_wit"] = r["U_wit"] - r["L_wit"]
        r["sign_outer"] = "pos" if r["L_X"] > 0 else "neg" if r["U_X"] < 0 else "both"; r["sign_wit"] = "pos" if r["L_wit"] > 0 else "neg" if r["U_wit"] < 0 else "both"
        UX = max(abs(r["U_X"]), abs(r["L_X"])); UP = max(abs(r["U_PhiC"]), abs(r["L_PhiC"])); UV = max(abs(r["U_V"]), abs(r["L_V"])); UW = max(abs(r["U_wit"]), abs(r["L_wit"]))
        acc["mov"] += mu_a * r["real_mid"] ** 2; acc["X"] += mu_a * UX ** 2; acc["P"] += mu_a * UP ** 2; acc["V"] += mu_a * UV ** 2; acc["Xw"] += mu_a * UW ** 2
        rows.append(r); npairs += 1
        if npairs % 100 == 0:
            print(f"  {a.field} {npairs} pairs ({time.time()-t0:.0f}s): S_X {np.sqrt(acc['X']/acc['mov']):.3f}x S_Phi' {np.sqrt(acc['P']/acc['mov']):.3f}x S_V {np.sqrt(acc['V']/acc['mov']):.3f}x S_wit {np.sqrt(acc['Xw']/acc['mov']):.3f}x", flush=True)
            json.dump(dict(field=a.field, partial=True, pairs=rows), open(os.path.join(OUT, f"c5_{a.field}.json"), "w"), indent=1, default=float)
    if a.max_pairs and npairs >= a.max_pairs:
        break
mov = np.sqrt(acc["mov"] / mu_tot)
S = dict(field=a.field, pairs=npairs, M=a.M, cropped_pairs=int(sum(r["cropped"] for r in rows)), movement_mid=float(mov), S_X_x=float(np.sqrt(acc["X"] / mu_tot) / mov), S_PhiC_x=float(np.sqrt(acc["P"] / mu_tot) / mov), S_V_x=float(np.sqrt(acc["V"] / mu_tot) / mov),
         S_witness_x=float(np.sqrt(acc["Xw"] / mu_tot) / mov), old_lp_x=old["bound_lp_x"], old_value_only_x=old["bound_value_only_x"], old_global_x=old.get("bound_global_x"),
         width_X_over_wit_median=float(np.nanmedian([r["width_X"] / r["width_wit"] for r in rows if r["width_wit"] > 1e-9])), gap_plus_median=float(np.nanmedian([r["gap_plus"] for r in rows])),
         sign_resolved_outer=int(sum(r["sign_outer"] != "both" for r in rows)), sign_resolved_wit=int(sum(r["sign_wit"] != "both" for r in rows)), verified_inf=int(sum(not np.isfinite(r["U_X"]) or not np.isfinite(r["L_X"]) for r in rows)), seconds=time.time() - t0)
json.dump(dict(summary=S, pairs=rows), open(os.path.join(OUT, f"c5_{a.field}.json"), "w"), indent=1, default=float); print(json.dumps(S, indent=1))
