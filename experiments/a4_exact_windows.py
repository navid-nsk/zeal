"""a4_exact_windows.py -- exact small-window zoning benchmark (Definition S2, Theorems S16-S17; data tier); results exp1/A4.

Windows of n <= 12 Output Areas (Greater Manchester; outcomes X = q4 degree share, Y = bad-health share; population
weights). For each window we ENUMERATE every partition into exactly K contiguous cells (K = 2, 3, 4), optionally
population-balanced (|mu(a) - 1/K| <= beta / K), and compute the exact range over the class of:
  (a) the focal cell's mean (cell containing the seed OA);  (b) the pair difference theta_x - theta_y (common zoning);
  (c) the within-window rank of the seed OA's cell value among the K cells (top-1 indicator);
  (d) the slope beta(Z) = N/D and correlation of Y on X over the K cell means;
and compare with the outer bounds of the paper: the separable bracket (Theorem S16) built from all contiguous
cells containing x (resp. y) inside the window, with and without the fixed-K/balance restriction (singleton-completion
counterexamples appear as strict inequalities), and the linear-fractional slope LP (Theorem S17) over per-unit
brackets (Charnes-Cooper) with and without the oscillation cuts. All numbers exact (enumeration) or LP-verified.
Usage: python a4_exact_windows.py --out <dir> [--windows 20] [--n 12] [--seed 5]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, itertools, collections, numpy as np, pandas as pd
from scipy.optimize import linprog
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--windows", type=int, default=20); p.add_argument("--n", type=int, default=12)
p.add_argument("--seed", type=int, default=5); a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
D = paths.GM_DIR; u = pd.read_csv(D + "gm_units.csv"); adj = pd.read_csv(D + "gm_adjacency.csv").values
n_all = len(u); E = adj[adj[:, 0] != adj[:, 1]]; nbrs = collections.defaultdict(set)
for a_, b_ in E:
    nbrs[int(a_)].add(int(b_)); nbrs[int(b_)].add(int(a_))
X_all = u["q4"].values.astype(float); Y_all = u["bad"].values.astype(float); W_all = u["w_q4"].values.astype(float)


def grow_window(seed, n):
    sel = [seed]; frontier = set(nbrs[seed])
    while len(sel) < n and frontier:
        k = int(rng.choice(sorted(frontier))); sel.append(k); frontier |= nbrs[k]; frontier -= set(sel)
    return sel


def contiguous(block, adjm):
    block = list(block); seen = {block[0]}; st = [block[0]]; bs = set(block)
    while st:
        v = st.pop()
        for w_ in adjm[v]:
            if w_ in bs and w_ not in seen:
                seen.add(w_); st.append(w_)
    return len(seen) == len(block)


def set_partitions(n, K):
    """restricted growth strings with exactly K blocks"""
    rgs = [0] * n
    def rec(i, m):
        if i == n:
            if m == K:
                yield list(rgs)
            return
        for b in range(min(m + 1, K)):
            rgs[i] = b; yield from rec(i + 1, max(m, b + 1))
    yield from rec(1, 1) if n > 0 else iter(())


def cell_mean(idx, x, w): return float((w[idx] * x[idx]).sum() / w[idx].sum())


def stats(cells, x, y, w):
    mu = np.array([w[c].sum() for c in cells]); mu = mu / mu.sum(); mx = np.array([cell_mean(c, x, w) for c in cells]); my = np.array([cell_mean(c, y, w) for c in cells])
    xb, yb = (mu * mx).sum(), (mu * my).sum(); Dx = (mu * (mx - xb) ** 2).sum(); Dy = (mu * (my - yb) ** 2).sum(); N = (mu * (mx - xb) * (my - yb)).sum()
    return mx, my, (N / Dx if Dx > 1e-12 else np.nan), (N / np.sqrt(Dx * Dy) if Dx * Dy > 1e-24 else np.nan), Dx


def contiguous_cells_containing(x_loc, nodes, adjm):
    """all contiguous subsets of the window containing x_loc (window of <= 12: 2^11 subsets)"""
    others = [v for v in nodes if v != x_loc]; out = []
    for r in range(len(others) + 1):
        for comb in itertools.combinations(others, r):
            c = (x_loc,) + comb
            if contiguous(c, adjm):
                out.append(list(c))
    return out


def slope_lfp(nodes, x, y, w, cells_by_unit, Dmin, cuts):
    """Linear-fractional slope bound (Theorem S17): per-unit brackets for the coarse X value phi_u in [min, max over admissible cells containing u] (exact over
    enumerated contiguous cells), mean preservation, the declared denominator floor D >= Dmin, optional
    Popoviciu cut; range of beta = <phi, c_y>/<phi, c_x> via Charnes-Cooper (variables z = t*phi, t = 1/D).
    In the data tier Y is known, so N(Z) = sum_u w_u (y_u - ybar) phi_u, D(Z) = sum_u w_u (x_u - xbar) phi_u (Theorem S17)."""
    m = len(nodes); ww = w[nodes] / w[nodes].sum(); xx = x[nodes]; yy = y[nodes]
    xb, yb = (ww * xx).sum(), (ww * yy).sum(); cy = ww * (yy - yb); cx = ww * (xx - xb)
    lo = np.array([min(cell_mean(c, x, w) for c in cells_by_unit[v]) for v in nodes]); hi = np.array([max(cell_mean(c, x, w) for c in cells_by_unit[v]) for v in nodes])
    A_ub = []; b_ub = []
    for i in range(m):
        row = np.zeros(m + 1); row[i] = 1; row[m] = -hi[i]; A_ub.append(row); b_ub.append(0)
        row = np.zeros(m + 1); row[i] = -1; row[m] = lo[i]; A_ub.append(row); b_ub.append(0)
    A_eq = [np.r_[cx, 0.0], np.r_[ww, -xb]]; b_eq = [1.0, 0.0]
    floor = Dmin
    if cuts:   # Popoviciu denominator cut with O_u = oscillation of x over the union of all admissible cells containing u
        Ou = np.array([np.ptp(x[sorted(set(itertools.chain(*cells_by_unit[v])))]) for v in nodes]); Dpop = (ww * (xx - xb) ** 2).sum() - 0.25 * (ww * Ou ** 2).sum()
        floor = max(floor, Dpop)
    if floor > 0:
        A_ub.append(np.r_[np.zeros(m), 1.0]); b_ub.append(1.0 / floor)       # t = 1/D <= 1/floor
    out = {}
    for sgn, nm in ((-1, "max"), (1, "min")):
        r = linprog(sgn * np.r_[cy, 0.0], A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(None, None)] * m + [(0, None)], method="highs")
        out[nm] = sgn * r.fun if r.status == 0 else (np.inf if nm == "max" else -np.inf)
    return out["min"], out["max"], float(floor)


def slope_setpart(nodes, x, y, w, all_cells, K, Dmin, beta_bal=np.inf):
    """Set-partitioning LP relaxation (Theorem S17) over ALL contiguous cells of the window (columns enumerated exactly),
    sum_{C ni u} z_C = 1, z >= 0, optional sum_C z_C = K, denominator floor sum_C z_C mu_C (X_C - xbar)^2 >= Dmin.
    'exists Z with beta(Z) >= b' => max_Z sum_C v_b(C) >= 0, and the LP relaxation value bounds that max from above; bisection on b gives
    a certified upper bound on beta over the class (lower bound symmetric). Returns (beta_lo, beta_hi)."""
    m = len(nodes); idx = {v: i for i, v in enumerate(nodes)}; ww = w[nodes] / w[nodes].sum(); xb = (ww * x[nodes]).sum(); yb = (ww * y[nodes]).sum()
    Wtot = w[nodes].sum()
    if K is not None and np.isfinite(beta_bal):      # balanced class: only cells with mu_C within beta_bal/K of 1/K are admissible columns
        all_cells = [c for c in all_cells if abs(w[c].sum() / Wtot - 1 / K) <= beta_bal / K]
        if not all_cells:
            return np.nan, np.nan
    muC = np.array([w[c].sum() / Wtot for c in all_cells]); mxC = np.array([cell_mean(c, x, w) for c in all_cells]); myC = np.array([cell_mean(c, y, w) for c in all_cells])
    cover = np.zeros((m, len(all_cells)))
    for j, c in enumerate(all_cells):
        for v in c:
            cover[idx[v], j] = 1
    A_eq = [cover]; b_eq = [np.ones(m)]
    if K is not None:
        A_eq.append(np.ones((1, len(all_cells)))); b_eq.append(np.array([K]))
    A_eq = np.vstack(A_eq); b_eq = np.concatenate(b_eq)
    A_ub = -(muC * (mxC - xb) ** 2)[None, :]; b_ub = np.array([-Dmin])
    def V(b, upper):
        q = muC * (mxC - xb) * ((myC - yb) - b * (mxC - xb)) if upper else muC * (mxC - xb) * (b * (mxC - xb) - (myC - yb))
        r = linprog(-q, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        return -r.fun if r.status == 0 else (np.nan if r.status == 2 else np.inf)
    def bisect(upper):
        lo_, hi_ = -50.0, 50.0
        if np.isnan(V(0.0, upper)):
            return np.nan
        for _ in range(40):
            mid = (lo_ + hi_) / 2
            if V(mid, upper) >= 0:     # some (fractional) partition reaches beta >= mid (upper) / <= mid (lower)
                lo_, hi_ = (mid, hi_) if upper else (lo_, mid)
            else:
                lo_, hi_ = (lo_, mid) if upper else (mid, hi_)
        return hi_ if upper else lo_
    return bisect(False), bisect(True)


rows = []; t_all = time.time()
for wi in range(a.windows):
    seed = int(rng.integers(n_all)); nodes = grow_window(seed, a.n)
    if len(nodes) < a.n:
        continue
    adjm = {v: nbrs[v] & set(nodes) for v in nodes}; x_loc = seed; y_loc = int(rng.choice([v for v in nodes if v != seed and v not in nbrs[seed]] or [v for v in nodes if v != seed]))
    t0 = time.time(); cells_x = contiguous_cells_containing(x_loc, nodes, adjm); cells_y = contiguous_cells_containing(y_loc, nodes, adjm)
    cells_by_unit = {v: contiguous_cells_containing(v, nodes, adjm) for v in nodes}
    rec = dict(window=wi, seed=x_loc, y=y_loc, n=len(nodes), n_cells_x=len(cells_x), classes={})
    # separable bracket (Theorem S16) over ALL contiguous cells (window class, singleton completion valid):
    Lx, Ux = min(cell_mean(c, X_all, W_all) for c in cells_x), max(cell_mean(c, X_all, W_all) for c in cells_x)
    Ly, Uy = min(cell_mean(c, X_all, W_all) for c in cells_y), max(cell_mean(c, X_all, W_all) for c in cells_y)
    rec["bracket_any"] = dict(focal=[Lx, Ux], pair=[Lx - Uy, Ux - Ly])
    # slope LFP outer over the window class
    ww_ = W_all[nodes] / W_all[nodes].sum(); VarX = float((ww_ * (X_all[nodes] - (ww_ * X_all[nodes]).sum()) ** 2).sum()); DMIN = 0.25 * VarX   # declared denominator floor: coarse X variance >= 25 % of the fine window variance
    bmin, bmax, fl0 = slope_lfp(nodes, X_all, Y_all, W_all, cells_by_unit, DMIN, cuts=False); bminc, bmaxc, fl1 = slope_lfp(nodes, X_all, Y_all, W_all, cells_by_unit, DMIN, cuts=True)
    rec["slope_lfp"] = dict(no_cuts=[bmin, bmax], cuts=[bminc, bmaxc], Dmin_declared=DMIN, floor_with_cut=fl1, VarX_window=VarX)
    all_cells = []; seen = set()
    for v in nodes:
        for c in cells_by_unit[v]:
            key = tuple(sorted(c))
            if key not in seen:
                seen.add(key); all_cells.append(list(key))
    rec["n_cells_all"] = len(all_cells); rec["slope_setpart"] = {}
    for K in (None, 2, 3, 4):
        for bb in (np.inf, 0.5, 0.25):
            sp_lo, sp_hi = slope_setpart(nodes, X_all, Y_all, W_all, all_cells, K, DMIN, bb); rec["slope_setpart"][f"K{K}_beta{bb}"] = [sp_lo, sp_hi]
            if K is None:
                break
    for K in (2, 3, 4):
        for beta in (np.inf, 0.5, 0.25):
            cnt = 0; fo = [np.inf, -np.inf]; pr = [np.inf, -np.inf]; sl = [np.inf, -np.inf]; co = [np.inf, -np.inf]; top1 = [1, 0]; Dmin = np.inf
            ix = nodes.index(x_loc); iy = nodes.index(y_loc)
            for rgs in set_partitions(len(nodes), K):
                cells = [[nodes[i] for i in range(len(nodes)) if rgs[i] == b] for b in range(K)]
                if not all(contiguous(c, adjm) for c in cells):
                    continue
                mu = np.array([W_all[c].sum() for c in cells]); mu /= mu.sum()
                if np.isfinite(beta) and np.abs(mu - 1 / K).max() > beta / K:
                    continue
                cnt += 1; mx, my, b_, r_, Dx = stats(cells, X_all, Y_all, W_all)
                fx = mx[rgs[ix]]; fy = mx[rgs[iy]]; fo = [min(fo[0], fx), max(fo[1], fx)]; pr = [min(pr[0], fx - fy), max(pr[1], fx - fy)]
                if np.isfinite(b_) and Dx >= DMIN:          # slope range over the declared denominator-floor subclass
                    sl = [min(sl[0], b_), max(sl[1], b_)]; co = [min(co[0], r_), max(co[1], r_)]; Dmin = min(Dmin, Dx)
                is_top = int(fx >= mx.max() - 1e-12); top1 = [min(top1[0], is_top), max(top1[1], is_top)]
            if cnt == 0:
                rec["classes"][f"K{K}_beta{beta}"] = dict(count=0); continue
            rec["classes"][f"K{K}_beta{beta}"] = dict(count=cnt, focal=fo, pair=pr, slope=sl, corr=co, top1=top1, Dmin=float(Dmin),
                                                       bracket_gap_pair=[float(pr[0] - (Lx - Uy)), float((Ux - Ly) - pr[1])], bracket_exact_pair=bool(abs(pr[0] - (Lx - Uy)) < 1e-12 and abs(pr[1] - (Ux - Ly)) < 1e-12),
                                                       slope_outer_contains=bool(bminc - 1e-9 <= sl[0] and sl[1] <= bmaxc + 1e-9), slope_outer_gap=[float(sl[0] - bminc), float(bmaxc - sl[1])],
                                                       slope_sign_exact=("neg" if sl[1] < 0 else "pos" if sl[0] > 0 else "both"), slope_sign_outer=("neg" if bmaxc < 0 else "pos" if bminc > 0 else "both"),
                                                       setpart_K=rec["slope_setpart"][f"K{K}_beta{beta}"], setpart_contains=bool(np.isnan(rec["slope_setpart"][f"K{K}_beta{beta}"][0]) or (rec["slope_setpart"][f"K{K}_beta{beta}"][0] - 1e-6 <= sl[0] and sl[1] <= rec["slope_setpart"][f"K{K}_beta{beta}"][1] + 1e-6)) if np.isfinite(sl[0]) else None,
                                                       setpart_sign=("neg" if rec["slope_setpart"][f"K{K}_beta{beta}"][1] < 0 else "pos" if rec["slope_setpart"][f"K{K}_beta{beta}"][0] > 0 else "both") if not np.isnan(rec["slope_setpart"][f"K{K}_beta{beta}"][0]) else "infeasible")
    rec["seconds"] = time.time() - t0; rows.append(rec)
    c = rec["classes"]; print(f"window {wi}: n={len(nodes)} cells_x={len(cells_x)} bracket pair=[{Lx-Uy:.4f},{Ux-Ly:.4f}] K2 exact pair={c.get('K2_betainf',{}).get('pair')} K4b.25 pair={c.get('K4_beta0.25',{}).get('pair')} "
          f"slope LFP cuts=[{bminc:.3f},{bmaxc:.3f}] setpart K4={rec['slope_setpart']['K4_betainf']} K4 exact slope={c.get('K4_betainf',{}).get('slope')} {rec['seconds']:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.out, "a4_windows.json"), "w"), indent=1, default=float)
# summary
S = dict(windows=len(rows))
for key in ("K2_betainf", "K3_betainf", "K4_betainf", "K4_beta0.5", "K4_beta0.25"):
    cl = [r["classes"][key] for r in rows if key in r["classes"] and r["classes"][key].get("count", 0) > 0]
    if cl:
        S[key] = dict(windows=len(cl), count_median=float(np.median([c["count"] for c in cl])), bracket_exact_share=float(np.mean([c["bracket_exact_pair"] for c in cl])),
                      bracket_gap_median=float(np.median([max(c["bracket_gap_pair"]) for c in cl])), pair_width_exact_median=float(np.median([c["pair"][1] - c["pair"][0] for c in cl])),
                      bracket_width_median=float(np.median([(r["bracket_any"]["pair"][1] - r["bracket_any"]["pair"][0]) for r in rows if key in r["classes"] and r["classes"][key].get("count", 0) > 0])),
                      slope_outer_contains_share=float(np.mean([c["slope_outer_contains"] for c in cl])), slope_sign_exact_counts=dict(collections.Counter(c["slope_sign_exact"] for c in cl)),
                      slope_sign_outer_counts=dict(collections.Counter(c["slope_sign_outer"] for c in cl)), slope_width_exact_median=float(np.median([c["slope"][1] - c["slope"][0] for c in cl if np.isfinite(c["slope"][0])])),
                      setpart_contains_share=float(np.mean([c["setpart_contains"] for c in cl if c.get("setpart_contains") is not None])), setpart_sign_counts=dict(collections.Counter(c.get("setpart_sign") for c in cl)),
                      setpart_width_median=float(np.nanmedian([c["setpart_K"][1] - c["setpart_K"][0] for c in cl])), sign_agreement_exact_vs_setpart=float(np.mean([c["slope_sign_exact"] == c.get("setpart_sign") for c in cl if np.isfinite(c["slope"][0])])))
S["slope_lfp_width_median_cuts"] = float(np.median([r["slope_lfp"]["cuts"][1] - r["slope_lfp"]["cuts"][0] for r in rows])); S["slope_lfp_width_median_nocuts"] = float(np.median([r["slope_lfp"]["no_cuts"][1] - r["slope_lfp"]["no_cuts"][0] for r in rows]))
S["seconds"] = time.time() - t_all
json.dump(dict(summary=S, windows=rows), open(os.path.join(a.out, "a4_windows.json"), "w"), indent=1, default=float); print(json.dumps(S, indent=1, default=float))
