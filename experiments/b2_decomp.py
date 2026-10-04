"""b2_decomp.py -- unit-reassignment decomposition audit (Supplementary Theorem S18 (iii); results exp2/B2): on the Manchester OA
boundary-displacement experiment (b2_zoning_vs_scale.py, 'displace' with move probability q), compute the EXACT two-term decomposition
  mov^2 = sum_{u in M} mu_u (m_{c(u)} - m'_{c'(u)})^2 + sum_{u notin M} mu_u (m_{c(u)} - m'_{c(u)})^2 = mu(M) Cbar_M^2 + R_stay
with matched cell labels (the displaced zoning keeps the LSOA labels), and the conservative remainder bound of Theorem S18 (iii)
  R_stay <= 4 (u - l)^2 / m0^2 * mu(M)^2   (f in [l, u], minimum cell mass m0 after reassignment, normalized population),
for q in {0.02, 0.05, 0.1, 0.2, 0.5} (10 replicates each): shares of the two terms, Cbar_M vs the within-LSOA s.d., and the
log-log slopes of each term vs mu(M). Usage: python b2_decomp.py --out <dir>."""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=9); p.add_argument("--reps", type=int, default=10); a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
D = paths.GM_DIR; u = pd.read_csv(D + "gm_units.csv"); adj = pd.read_csv(D + "gm_adjacency.csv").values; n = len(u)
E = adj[adj[:, 0] != adj[:, 1]]; nbrs = [[] for _ in range(n)]
for x_, y_ in E:
    nbrs[int(x_)].append(int(y_)); nbrs[int(y_)].append(int(x_))
A_ls = u["LSOA21CD_i"].values; K = A_ls.max() + 1
wm = lambda v, w, g, Kk: np.bincount(g, weights=w * v, minlength=Kk) / np.maximum(np.bincount(g, weights=w, minlength=Kk), 1e-12)


def displace(lab0, q):
    lab = lab0.copy(); moved = np.zeros(n, bool); counts = np.bincount(lab, minlength=lab.max() + 1)
    for v in rng.permutation(n):
        other = [lab[z] for z in nbrs[v] if lab[z] != lab[v]]
        if other and rng.random() < q and counts[lab[v]] >= 2:
            counts[lab[v]] -= 1; lab[v] = other[int(rng.integers(len(other)))]; counts[lab[v]] += 1; moved[v] = True
    return lab, moved


OUT = {}
for oc, wc in (("q4", "w_q4"), ("bad", "w_bad")):
    w = u[wc].values.astype(float); t = np.where(w > 0, u[oc].values.astype(float), 0.0); w = np.where(w > 0, w, 0.0); mu = w / w.sum()
    m0_lsoa = wm(t, w, A_ls, K); sd_within = float(np.sqrt((mu * (t - m0_lsoa[A_ls]) ** 2).sum())); l_, u_ = float(t.min()), float(t.max())
    rows = []
    for q in (0.02, 0.05, 0.1, 0.2, 0.5):
        for rep in range(a.reps):
            lab, M = displace(A_ls, q); m1 = wm(t, w, lab, K)
            moved_term = float((mu[M] * (m0_lsoa[A_ls[M]] - m1[lab[M]]) ** 2).sum()); stay_term = float((mu[~M] * (m0_lsoa[A_ls[~M]] - m1[lab[~M]]) ** 2).sum())
            muM = float(mu[M].sum()); mov2 = float((mu * (m0_lsoa[A_ls] - m1[lab]) ** 2).sum())
            m_min = float(np.bincount(lab, weights=mu, minlength=K)[np.bincount(lab, minlength=K) > 0].min())
            rows.append(dict(q=q, rep=rep, mu_M=muM, mov2=mov2, moved_term=moved_term, stay_term=stay_term, identity_residual=abs(mov2 - moved_term - stay_term), Cbar_M=float(np.sqrt(moved_term / muM)) if muM > 0 else 0.0,
                             R_stay_bound=4 * (u_ - l_) ** 2 / m_min ** 2 * muM ** 2, m_min=m_min))
    byq = {}
    for q in (0.02, 0.05, 0.1, 0.2, 0.5):
        rr = [r for r in rows if r["q"] == q]
        byq[str(q)] = dict(mu_M=float(np.median([r["mu_M"] for r in rr])), mov=float(np.median([np.sqrt(r["mov2"]) for r in rr])), moved_share_of_mov2=float(np.median([r["moved_term"] / r["mov2"] for r in rr])), stay_share_of_mov2=float(np.median([r["stay_term"] / r["mov2"] for r in rr])),
                           Cbar_M=float(np.median([r["Cbar_M"] for r in rr])), Cbar_over_within_sd=float(np.median([r["Cbar_M"] for r in rr]) / sd_within), R_stay_bound_over_stay=float(np.median([r["R_stay_bound"] / max(r["stay_term"], 1e-18) for r in rr])), identity_residual_max=float(max(r["identity_residual"] for r in rr)))
    xs = np.log([r["mu_M"] for r in rows]); S = dict(within_sd=sd_within, slope_mov2_vs_muM=float(np.polyfit(xs, np.log([r["mov2"] for r in rows]), 1)[0]), slope_moved_term=float(np.polyfit(xs, np.log([r["moved_term"] for r in rows]), 1)[0]),
                                                     slope_stay_term=float(np.polyfit(xs, np.log([max(r["stay_term"], 1e-18) for r in rows]), 1)[0]), by_q=byq)
    OUT[oc] = S; print(oc, json.dumps({k: v for k, v in S.items() if k != "by_q"}, indent=1)); [print("  q", k, {a_: round(b_, 4) for a_, b_ in v.items()}) for k, v in byq.items()]
json.dump(OUT, open(os.path.join(a.out, "b2_decomp.json"), "w"), indent=1, default=float)
