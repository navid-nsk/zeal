"""b11_saving_diag.py -- the exact transport-saving identity (Supplementary Theorem S5) on real nested pairs
(same pair selection as A1) at tile budgets k in {1, 4, 16}: V* - U = max_pi sum pi [A_p + B_q - D_psi]_+ with psi = the L1-projection
of the box midpoints (reference named). Reports benefiting mass share, mean saving per benefiting unit mass, value-slack scale and the
normalized saving (V* - U)/(V* - d.psi). Usage: python b11_saving_diag.py --field georgia [--pairs 30] [--ks 1,4,16]. Results exp1/B11."""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import solve_max, closure, lemma12, grid_poly, project_to_phi, saving_identity
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--pairs", type=int, default=30); p.add_argument("--ks", default="1,4,16"); p.add_argument("--kvkg", default=""); p.add_argument("--max_pix", type=int, default=1200); p.add_argument("--seed", type=int, default=11); a = p.parse_args()
OUT = paths.results("exp1", "B11"); os.makedirs(OUT, exist_ok=True)
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
ras = np.load(SPEC[a.field]); mask = ras["mask"].astype(bool) & np.isfinite(lo0); N = mask.shape[0]; h = 2.0 / (N - 1)
w = np.where(mask, ras["weight"], 0.0).astype(float); fine = np.where(mask, ras["tract_idx"], -1); coarse = np.where(mask, ras["county_idx"], -1)
lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, fine, coarse = (A.T.copy() for A in (lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, fine, coarse))
lam = (lo0 + hi0) / 2
def block_boxes(k, kg=None):
    kg = k if kg is None else kg
    def pool(A, fn, fill, k):
        B = np.where(mask, A, fill); n1, n2 = B.shape; m1, m2 = -(-n1 // k), -(-n2 // k); P = np.full((m1 * k, m2 * k), fill); P[:n1, :n2] = B; P = P.reshape(m1, k, m2, k)
        R = fn(fn(P, axis=3), axis=1); return np.repeat(np.repeat(R, k, axis=0), k, axis=1)[:n1, :n2]
    return (pool(lo0, np.min, np.inf, k), pool(hi0, np.max, -np.inf, k), np.stack([pool(g1l0, np.min, np.inf, kg), pool(g1h0, np.max, -np.inf, kg)], -1), np.stack([pool(g2l0, np.min, np.inf, kg), pool(g2h0, np.max, -np.inf, kg)], -1))
sizes = {int(b): int((coarse == b).sum()) for b in np.unique(coarse[coarse >= 0])}; cand = [b for b, s_ in sizes.items() if 30 <= s_ <= a.max_pix]
rng2 = np.random.default_rng(a.seed + 1); sel_pairs = []
while len(sel_pairs) < a.pairs and cand:
    b = int(rng2.choice(cand)); sel = coarse == b; fines = [f_ for f_ in np.unique(fine[sel]) if f_ >= 0 and w[sel & (fine == f_)].sum() > 0 and (fine[sel] == f_).sum() >= 2 and (fine[sel] == f_).sum() < sel.sum()]
    if not fines:
        cand.remove(b); continue
    sel_pairs.append((b, int(rng2.choice(fines))))
res = {}
budgets = [(int(x), int(x)) for x in a.ks.split(",")] if not a.kvkg else [(int(t.split(":")[0]), int(t.split(":")[1])) for t in a.kvkg.split(",")]
for (k, kg) in budgets:
    t0 = time.time(); lo, hi, G1, G2 = block_boxes(k, kg); lo_c, hi_c, _ = lemma12(lo, hi, G1, G2, h); rows = []
    for (b, f_) in sel_pairs:
        sel = coarse == b; i0, i1 = np.nonzero(sel.any(1))[0][[0, -1]]; j0, j1 = np.nonzero(sel.any(0))[0][[0, -1]]; sl = (slice(i0, i1 + 1), slice(j0, j1 + 1)); mk = sel[sl]
        poly, _ = grid_poly(lo_c[sl], hi_c[sl], G1[sl], G2[sl], h, mk); wb = w[sl] * mk; wa = wb * (fine[sl] == f_); d = (wa / wa.sum() - wb / wb.sum())[mk]
        psi = project_to_phi(poly, lam[sl][mk]); D = closure(poly)
        r = saving_identity(poly, d, psi, D); r.update(coarse=b, fine=f_, n_pix=int(mk.sum())); rows.append(r)
    S = dict(k=k, kg=kg, pairs=len(rows), identity_residual_max=float(max(r["identity_residual"] for r in rows)), normalized_saving_median=float(np.median([r["normalized_saving"] for r in rows])),
             U_over_Vstar_median=float(np.median([r["U"] / r["Vstar"] for r in rows if abs(r["Vstar"]) > 1e-12])), mass_benefiting_median=float(np.median([r["mass_benefiting_share"] for r in rows])),
             saving_per_benefiting_mass_median=float(np.median([r["mean_saving_per_benefiting_mass"] for r in rows])), value_slack_scale_median=float(np.median([r["value_slack_scale"] for r in rows])), seconds=time.time() - t0)
    res[f"{k}:{kg}"] = dict(summary=S, pairs=rows); print(f"{a.field} k_v={k} k_g={kg}: residual {S['identity_residual_max']:.1e} U/V* {S['U_over_Vstar_median']:.4f} normalized saving {S['normalized_saving_median']:.3f} benefiting mass {S['mass_benefiting_median']:.2f} saving/benefiting mass {S['saving_per_benefiting_mass_median']:.4f} value-slack scale {S['value_slack_scale_median']:.4f} {time.time()-t0:.0f}s", flush=True)
json.dump(res, open(os.path.join(OUT, f"b11_saving_{a.field}" + ("_kvkg" if a.kvkg else "") + ".json"), "w"), indent=1, default=float)
