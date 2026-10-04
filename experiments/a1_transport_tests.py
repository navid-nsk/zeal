"""a1_transport_tests.py -- unit tests of the transport representation (Supplementary Theorems S3-S5, Lemma S6.1, V1); results exp1/A1.

Part 1 (synthetic): random sinusoidal fields on n1 x n2 grids with sound value/gradient boxes (sampled extremes
padded by analytic Lipschitz bounds, widened 0-600 %), random masks (1-3 components), signed closures, three kinds
of d (random sparse; nested rectangle cells with random population; d with nonzero mass on a component).
Checks: (i) LP = closed OT; (ii) Gamma criterion = (U == V*); (iii) gradient-only finiteness = zero mass per
component; (iv) uniform boxes: U = L W1(l1 grid), ratio to L W1(l2) in [1, sqrt 2], 4- vs 8-neighbour;
(v) saving-identity residual (Theorem S5 (i)); (vi) Lemma S6.1 never excludes a realizable mean (true pixel means stay inside);
(vii) verified dual upper bound >= LP value for every instance.
Part 2 (real ladders): first_order_{georgia,gm_q4,gm_bad,mx_rwi}.npz, coarse cells with <= max_pix pixels,
random nested pairs: tent increments vs max-increments (as in fields/transport_lp.py), U with verified dual, Gamma
criterion, saving identity with psi = L1-projection of the box midpoints onto Phi, per-pair slack map summary.

Usage: python a1_transport_tests.py --out <dir> [--trials 200] [--pairs 40] [--max_pix 1500] [--skip_real]
Outputs: <out>/a1_synthetic.json, <out>/a1_real_<field>.json, <out>/a1.log (stdout).
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import (Poly, solve_max, closure, mcshane, closed_cost, ot_cost, value_face, gradient_only,
                            prop_b, project_to_phi, lemma12, grid_poly)
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--trials", type=int, default=200)
p.add_argument("--pairs", type=int, default=40); p.add_argument("--max_pix", type=int, default=1500); p.add_argument("--skip_real", action="store_true")
p.add_argument("--seed", type=int, default=11); p.add_argument("--skip_synth", action="store_true"); p.add_argument("--fields", default="georgia,gm_q4,gm_bad,mx_rwi"); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
import synth; synth.set_rng(rng); from synth import build, random_mask, random_d, SPEC


def l2_w1(d, h, mask):
    ii, jj = np.nonzero(mask); X = np.stack([ii, jj], 1) * h
    C = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    dm, dp = np.maximum(-d[mask], 0), np.maximum(d[mask], 0)
    return ot_cost(C, dm, dp)


def run_synthetic():
    log = []; res = dict(n=0, lp_vs_ot=[], crit_ok=0, crit_n=0, grad_ok=0, grad_n=0, propb_err=[], lemma12_ok=0, lemma12_n=0,
                        verified_ok=0, verified_gap=[], lw1=[], neg_closure=0, components=[], sizes=[])
    t0 = time.time()
    for t in range(a.trials):
        n1, n2 = int(rng.integers(4, 13)), int(rng.integers(4, 13)); h = float(rng.choice([0.1, 0.25, 0.5]))
        widen = float(rng.choice([0.0, 0.3, 1.5, 6.0])); mk = random_mask(n1, n2, rng.choice(["full", "blobs"]))
        if mk.sum() < 4:
            continue
        inst = build(n1, n2, h, widen, mk); neigh = int(rng.choice([4, 8]))
        poly, pix = grid_poly(inst["lo"], inst["hi"], inst["G1"], inst["G2"], h, mk, neigh=neigh)
        psi = inst["psi"][mk]
        assert poly.feasible(psi), "soundness of boxes/increments"
        kind = str(rng.choice(["sparse", "nested", "unbalanced"])); d2 = random_d(inst, kind)
        if d2 is None:
            continue
        d = d2[mk]
        r = solve_max(poly, d); U = r["value"]
        res["n"] += 1; res["sizes"].append(int(poly.n)); k, lab = poly.components(); res["components"].append(int(k))
        # (vii) verified dual
        res["verified_ok"] += int(r["verified"] >= U - 1e-12); res["verified_gap"].append(float(r["verified"] - U))
        # (i) closed OT
        D = closure(poly, "fw"); res["neg_closure"] += int(D.min() < 0)
        hs, ls = mcshane(poly, D); C = closed_cost(D, hs, ls)
        dp, dm = np.maximum(d, 0), np.maximum(-d, 0)
        # with unbalanced d the LP is max over Phi; the OT form (Theorem S3) is compared for balanced d; for unbalanced d the ground route carries the surplus:
        # add a ground node explicitly by comparing the LP with OT between d- and d+ plus the surplus routed from/to ground at cost hi*/-lo*.
        if abs(d.sum()) < 1e-12:
            Uot = ot_cost(C, dm, dp)
        else:
            Uot = np.nan
        if np.isfinite(Uot):
            res["lp_vs_ot"].append(float(abs(U - Uot) / max(1e-12, abs(U))))
        # (ii) value-face criterion
        Gam, Vstar, V = value_face(poly, D, hs, ls, d)
        if abs(d.sum()) < 1e-12:
            eq = abs(U - Vstar) < 1e-7 * max(1.0, abs(U)); res["crit_n"] += 1; res["crit_ok"] += int(eq == (Gam <= 1e-9))
        # (iii) gradient-only
        T, zero_mass = gradient_only(poly, d); res["grad_n"] += 1; res["grad_ok"] += int(np.isfinite(T) == zero_mass)
        # (v) saving identity (Theorem S5 (i)) with psi = true means
        if abs(d.sum()) < 1e-12:
            pb = prop_b(poly, d, psi); res["propb_err"].append(float(abs(U - pb["d_psi"] - pb["slack_transport"])))
        # (vi) Lemma S6.1 (value-gradient coupling)
        lo2, hi2, sh = lemma12(inst["lo"], inst["hi"], inst["G1"], inst["G2"], h)
        res["lemma12_n"] += 1; res["lemma12_ok"] += int(np.all(lo2[mk] <= psi + 1e-12) and np.all(hi2[mk] >= psi - 1e-12))
        log.append(f"t={t} n={poly.n} comp={k} neigh={neigh} kind={kind} widen={widen} U={U:.6g} OT={Uot:.6g} V*={Vstar:.6g} V={V:.6g} Gam={Gam:.3g} T={T:.6g} ver-U={r['verified']-U:.2e}")
        if t % 20 == 0:
            print(f"  synthetic {t}/{a.trials}  {time.time()-t0:.0f}s", flush=True)
    # (iv) uniform boxes: L W1 reductions, 4 vs 8 neighbours
    for t in range(12):
        n1, n2 = int(rng.integers(5, 10)), int(rng.integers(5, 10)); h = 0.25; L = float(rng.uniform(0.5, 2))
        mk = np.ones((n1, n2), bool); lo = np.full((n1, n2), -1e6); hi = np.full((n1, n2), 1e6)
        G1 = np.zeros((n1, n2, 2)); G1[..., 0] = -L; G1[..., 1] = L; G2 = G1.copy()
        inst = dict(mask=mk)
        d2 = random_d(inst, "sparse")
        if d2 is None:
            continue
        d = d2[mk]
        out = {}
        for neigh in (4, 8):
            poly, _ = grid_poly(lo, hi, G1, G2, h, mk, neigh=neigh); out[neigh] = solve_max(poly, d)["value"]
        poly4, _ = grid_poly(lo, hi, G1, G2, h, mk, neigh=4); D = closure(poly4, "fw")
        w1_l1 = ot_cost(D, np.maximum(-d, 0), np.maximum(d, 0))            # = L h * l1 graph distance transport
        w1_l2 = l2_w1(d2, h, mk)
        res["lw1"].append(dict(L=L, U4=out[4], U8=out[8], LW1_l1=w1_l1, LW1_l2=L * w1_l2, ratio_l1=out[4] / w1_l1, ratio_l2=out[4] / (L * w1_l2), ratio8_l2=out[8] / (L * w1_l2)))
    summ = dict(instances=res["n"], max_rel_lp_vs_ot=float(np.max(res["lp_vs_ot"])), n_lp_vs_ot=len(res["lp_vs_ot"]),
                criterion_agreement=f"{res['crit_ok']}/{res['crit_n']}", gradient_only_agreement=f"{res['grad_ok']}/{res['grad_n']}",
                propb_max_err=float(np.max(res["propb_err"])), lemma12_ok=f"{res['lemma12_ok']}/{res['lemma12_n']}",
                verified_ok=f"{res['verified_ok']}/{res['n']}", verified_gap_max=float(np.max(res["verified_gap"])), verified_gap_median=float(np.median(res["verified_gap"])),
                signed_closures=res["neg_closure"], multi_component=int(np.sum(np.array(res["components"]) > 1)), sizes=[int(np.min(res["sizes"])), int(np.max(res["sizes"]))],
                lw1=res["lw1"], lw1_ratio_l1_range=[float(min(x["ratio_l1"] for x in res["lw1"])), float(max(x["ratio_l1"] for x in res["lw1"]))],
                lw1_ratio_l2_range=[float(min(x["ratio_l2"] for x in res["lw1"])), float(max(x["ratio_l2"] for x in res["lw1"]))],
                lw1_ratio8_l2_range=[float(min(x["ratio8_l2"] for x in res["lw1"])), float(max(x["ratio8_l2"] for x in res["lw1"]))], seconds=time.time() - t0)
    json.dump(dict(summary=summ, log=log), open(os.path.join(a.out, "a1_synthetic.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))




def run_real(fieldname):
    F = np.load(paths.enclosure(fieldname)); lo, hi = F["lo"].astype(float), F["hi"].astype(float)
    g1l, g1h, g2l, g2h = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
    ras = np.load(SPEC[fieldname]); mask = ras["mask"].astype(bool) & np.isfinite(lo); N = mask.shape[0]; h = 2.0 / (N - 1)
    w = np.where(mask, ras["weight"], 0.0).astype(float); fine = np.where(mask, ras["tract_idx"], -1); coarse = np.where(mask, ras["county_idx"], -1)
    # arrays are [row, col] = [y, x]; transport_core uses axis0 = x. Transpose everything so that axis 0 = column index (x), g1 = d/dx.
    lo, hi, g1l, g1h, g2l, g2h, mask, w, fine, coarse = (A.T.copy() for A in (lo, hi, g1l, g1h, g2l, g2h, mask, w, fine, coarse))
    lam = (lo + hi) / 2
    G1 = np.stack([g1l, g1h], -1); G2 = np.stack([g2l, g2h], -1)
    sizes = {int(b): int((coarse == b).sum()) for b in np.unique(coarse[coarse >= 0])}
    cand = [b for b, s in sizes.items() if 30 <= s]
    rng2 = np.random.default_rng(a.seed + 1); rows = []; t0 = time.time(); cropped = 0
    while len(rows) < a.pairs and cand:
        b = int(rng2.choice(cand)); sel = coarse == b
        fines = [f_ for f_ in np.unique(fine[sel]) if f_ >= 0 and w[sel & (fine == f_)].sum() > 0 and (fine[sel] == f_).sum() >= 2]
        if not fines:
            cand.remove(b); continue
        f_ = int(rng2.choice(fines))
        if sizes[b] <= a.max_pix:
            i0, i1 = np.nonzero(sel.any(1))[0][[0, -1]]; j0, j1 = np.nonzero(sel.any(0))[0][[0, -1]]
        else:   # coarse cell larger than max_pix: crop a window around the fine cell (the pair becomes (a, b ∩ window), still a certified-class pair)
            fs = sel & (fine == f_); ii, jj = np.nonzero(fs); ci, cj = int(ii.mean()), int(jj.mean()); half = int(np.sqrt(a.max_pix) / 2)
            i0, i1 = max(0, ci - half), min(N - 1, ci + half); j0, j1 = max(0, cj - half), min(N - 1, cj + half)
            if not ((ii >= i0) & (ii <= i1) & (jj >= j0) & (jj <= j1)).all():
                continue
            cropped += 1
        sl = (slice(i0, i1 + 1), slice(j0, j1 + 1)); mk = sel[sl]
        if mk.sum() > a.max_pix * 1.3 or mk.sum() < 30:
            continue
        wb = w[sl] * mk; wa = wb * (fine[sl] == f_)
        if (fine[sl][mk] == f_).sum() == mk.sum():
            continue
        d2 = wa / wa.sum() - wb / wb.sum(); d = d2[mk]
        # tent increments, 4-neighbour, with and without Lemma S6.1; max-increments as in fields/transport_lp.py ('old_max')
        out = dict(coarse=b, fine=f_, n_pix=int(mk.sum()), n_fine=int((fine[sl][mk] == f_).sum()), mu_a=float(wa.sum() / wb.sum()))
        lo_c, hi_c, _ = lemma12(lo[sl], hi[sl], G1[sl], G2[sl], h)
        polyN, pix = grid_poly(lo[sl], hi[sl], G1[sl], G2[sl], h, mk, neigh=4)
        polyC, _ = grid_poly(lo_c, hi_c, G1[sl], G2[sl], h, mk, neigh=4)
        poly8, _ = grid_poly(lo_c, hi_c, G1[sl], G2[sl], h, mk, neigh=8)
        # max-increments: cp = h*max(g+_p, g+_q), cm = h*min(g-_p, g-_q)
        G1o = G1[sl].copy(); G2o = G2[sl].copy()
        polyO = Poly(polyN.lo, polyN.hi, polyN.ep, polyN.eq, polyN.cm.copy(), polyN.cp.copy())
        ii, jj = np.nonzero(mk); inv = {int(pix[i, j]): (i, j) for i, j in zip(ii, jj)}
        for e in range(polyO.E):
            (i, j), (i2, j2) = inv[int(polyO.ep[e])], inv[int(polyO.eq[e])]
            G = G1o if i2 != i else G2o
            polyO.cp[e] = h * max(G[i, j, 1], G[i2, j2, 1]); polyO.cm[e] = h * min(G[i, j, 0], G[i2, j2, 0])
        polyO = Poly(polyO.lo, polyO.hi, polyO.ep, polyO.eq, polyO.cm, polyO.cp)
        for name, poly in (("old_max", polyO), ("tent", polyN), ("tent_L12", polyC), ("tent_L12_8n", poly8)):
            rp = solve_max(poly, d); rm = solve_max(poly, -d)
            out[f"U_{name}"] = max(rp["value"], rm["value"]); out[f"Uver_{name}"] = max(rp["verified"], rm["verified"]); out[f"Upos_{name}"] = rp["value"]
        V = float(np.maximum(d, 0) @ polyN.hi - np.maximum(-d, 0) @ polyN.lo); out["V_raw"] = V
        out["real_mid"] = float(d @ lam[sl][mk])
        # Theorems S3-S5 on tent_L12: closure (johnson), criterion, OT, saving identity with psi = projection of midpoints
        try:
            D = closure(polyC); hs, ls = mcshane(polyC, D); C = closed_cost(D, hs, ls)
            Gam, Vstar, Vc = value_face(polyC, D, hs, ls, d); out.update(Gamma=Gam, Vstar=Vstar, V_L12=Vc)
            Uot = ot_cost(C, np.maximum(-d, 0), np.maximum(d, 0)); out["U_ot"] = Uot; out["lp_vs_ot_rel"] = float(abs(out["Upos_tent_L12"] - Uot) / max(1e-12, abs(Uot)))
            out["criterion_agrees"] = bool((abs(out["Upos_tent_L12"] - Vstar) < 1e-7 * max(1, abs(Vstar))) == (Gam <= 1e-9))
            psi = project_to_phi(polyC, lam[sl][mk]); out["psi_minus_mid_L1"] = float(np.abs(psi - lam[sl][mk]).mean())
            pb = prop_b(polyC, d, psi, D); out.update(d_psi=pb["d_psi"], slack_transport=pb["slack_transport"], slack_bound_tv=pb["bound_tv"], propb_residual=float(abs(out["Upos_tent_L12"] - pb["d_psi"] - pb["slack_transport"])))
            T, zm = gradient_only(polyC, d); out["T"] = float(T); out["zero_mass"] = zm
        except Exception as e:
            out["error"] = str(e)
        rows.append(out); print(f"  {fieldname} pair {len(rows)}: n={out['n_pix']} U old={out['U_old_max']:.4g} tent={out['U_tent']:.4g} L12={out['U_tent_L12']:.4g} 8n={out['U_tent_L12_8n']:.4g} V={V:.4g} real~{out['real_mid']:.4g} "
                         f"OT={out.get('U_ot', np.nan):.4g} slack={out.get('slack_transport', np.nan):.4g} {time.time()-t0:.0f}s", flush=True)
    def med(k): return float(np.median([r[k] for r in rows if k in r and np.isfinite(r[k])]))
    def ratio(k1, k2): return float(np.median([r[k1] / r[k2] for r in rows if r[k2] > 1e-12]))
    zero_obj = int(sum(abs(r["U_tent_L12"]) < 1e-12 for r in rows))
    summ = dict(field=fieldname, pairs=len(rows), median_U_old=med("U_old_max"), median_U_tent=med("U_tent"), median_U_tent_L12=med("U_tent_L12"), median_U_8n=med("U_tent_L12_8n"), median_V=med("V_raw"),
                median_ratio_tent_over_old=ratio("U_tent", "U_old_max"), median_ratio_L12_over_tent=ratio("U_tent_L12", "U_tent"), median_ratio_8n_over_L12=ratio("U_tent_L12_8n", "U_tent_L12"), zero_objective_pairs=zero_obj,
                abs_lp_vs_ot_max=float(np.nanmax([abs(r.get("Upos_tent_L12", np.nan) - r.get("U_ot", np.nan)) for r in rows])),
                max_lp_vs_ot_rel=float(np.nanmax([r.get("lp_vs_ot_rel", np.nan) for r in rows])), criterion_agree=int(sum(r.get("criterion_agrees", False) for r in rows)),
                propb_residual_max=float(np.nanmax([r.get("propb_residual", np.nan) for r in rows])), verified_ok=int(sum(r["Uver_tent_L12"] >= r["U_tent_L12"] - 1e-12 for r in rows)),
                share_slack_over_U=float(np.nanmedian([r["slack_transport"] / r["Upos_tent_L12"] for r in rows if "slack_transport" in r and r["Upos_tent_L12"] > 0])),
                errors=int(sum("error" in r for r in rows)), cropped_windows=cropped, seconds=time.time() - t0)
    json.dump(dict(summary=summ, pairs=rows), open(os.path.join(a.out, f"a1_real_{fieldname}.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    if not a.skip_synth:
        run_synthetic()
    if not a.skip_real:
        for fn in a.fields.split(","):
            run_real(fn)
