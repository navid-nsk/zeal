"""pricing_verify.py -- the verification layer for the slope certificates (Supplementary Theorem S17, V3):
exact pricing by exhaustive enumeration of the admissible cells, so that no floating-point MILP dual bound is trusted.

For a window W (n OAs) and class C(K, beta) the admissible cells are the connected unit sets C with |C| <= smax and
tlo <= mu(C) <= thi (ZClass of zeal/pricing_certified.py).  They are enumerated exactly once each by the
ESU algorithm (Wernicke 2006) with mass/size pruning (mass is additive and positive, so every superset of an over-heavy
cell is over-heavy).  At any dual vector (pi, theta, b) the exact maximum reduced cost
        eps_exact = max_{C admissible} [ (Sx Sy - b Sx^2)/t - pi(C) - theta ]
is computed in the same pass (incremental sums along the enumeration path; a conservative inclusion tolerance 1e-10 on the
mass bounds so that every cell that might be admissible is priced), and Theorem S17 (charged pricing) gives the verified slope bound
        beta(Z) <= b + max(0, (1'pi + K(theta + eps_exact + rounding allowance)) / Dmin).
Column generation: hill-climbing witness pool -> Charnes-Cooper master -> heuristic pricing (pool / local / grow of
pricing_certified.py) -> exact enumeration pricing when the heuristics find nothing (the top improving cells are added) ->
stop when the exact maximum reduced cost <= tol.  The certificate is evaluated at the final duals with the exact maximum.
Soundness check of the MILP pricing layer of pricing_certified.py: at the same final duals the HiGHS spatial branch-and-bound pricing bound is
recomputed and compared with eps_exact (sound iff eps_milp >= eps_exact - 1e-9).

Modes: --mode count  : enumeration counts and times per (window, class) only
       --mode verify : the full verified certificates (writes pricing_verify_n{n}.json)
Usage: python pricing_verify.py --mode count --n 30 --windows 8 --seed 11 --out <dir>
       python pricing_verify.py --mode verify --n 30 --windows 8 --seed 11 --out <dir> [--procs 8]
"""
import argparse, json, math, os, sys, time, heapq
import numpy as np
from numba import njit

import pricing_certified as PC

INF = np.inf
MASS_TOL = 1e-10          # conservative inclusion tolerance on the mass bounds (wider than the class's 1e-12)


@njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - 1; c += 1
    return c


@njit(cache=True)
def esu_pass(nbr, mu, mux, muy, pi, n, smax, tlo, thi, b, theta, store, out_masks, out_r, topk_r, topk_mask):
    """One ESU enumeration of all connected cells with |C| <= smax, mu(C) <= thi + tol.
    Returns (count of cells with mass also >= tlo - tol, max reduced cost over them).
    If store: writes masks and reduced costs of admissible cells into out_masks/out_r (must be pre-sized).
    topk arrays (sorted ascending by r) collect the best cells for column generation."""
    stack_sub = np.zeros(4 * n * (smax + 2), np.int64); stack_ext = np.zeros_like(stack_sub); stack_excl = np.zeros_like(stack_sub)
    stack_m = np.zeros(stack_sub.shape[0]); stack_sx = np.zeros_like(stack_m); stack_sy = np.zeros_like(stack_m); stack_p = np.zeros_like(stack_m)
    count = 0; rmax = -1e300; kk = topk_r.shape[0]
    tol_lo = tlo - MASS_TOL; tol_hi = thi + MASS_TOL
    for v in range(n):
        above = ~((np.int64(1) << np.int64(v + 1)) - np.int64(1))      # bits > v
        vbit = np.int64(1) << np.int64(v)
        m0 = mu[v]
        if m0 > tol_hi:
            continue
        sx0 = mux[v]; sy0 = muy[v]; p0 = pi[v]
        if m0 >= tol_lo:
            r = (sx0 * sy0 - b * sx0 * sx0) / m0 - p0 - theta
            count += 1
            if r > rmax:
                rmax = r
            if store:
                out_masks[count - 1] = vbit; out_r[count - 1] = r
            if kk > 0 and r > topk_r[0]:
                j = 0
                while j + 1 < kk and topk_r[j + 1] < r:
                    topk_r[j] = topk_r[j + 1]; topk_mask[j] = topk_mask[j + 1]; j += 1
                topk_r[j] = r; topk_mask[j] = vbit
        if smax <= 1:
            continue
        sp = 0
        stack_sub[sp] = vbit; stack_ext[sp] = nbr[v] & above; stack_excl[sp] = nbr[v] | vbit
        stack_m[sp] = m0; stack_sx[sp] = sx0; stack_sy[sp] = sy0; stack_p[sp] = p0; sp += 1
        while sp > 0:
            sp -= 1
            sub = stack_sub[sp]; ext = stack_ext[sp]; excl = stack_excl[sp]
            m = stack_m[sp]; sx = stack_sx[sp]; sy = stack_sy[sp]; p = stack_p[sp]
            if ext == 0:
                continue
            wbit = ext & (-ext); w = _popcount(wbit - 1)
            ext2 = ext & ~wbit
            # sibling continuation
            stack_sub[sp] = sub; stack_ext[sp] = ext2; stack_excl[sp] = excl
            stack_m[sp] = m; stack_sx[sp] = sx; stack_sy[sp] = sy; stack_p[sp] = p; sp += 1
            mc = m + mu[w]
            if mc > tol_hi:
                continue
            subc = sub | wbit; sxc = sx + mux[w]; syc = sy + muy[w]; pc = p + pi[w]
            size = _popcount(subc)
            if mc >= tol_lo:
                r = (sxc * syc - b * sxc * sxc) / mc - pc - theta
                count += 1
                if r > rmax:
                    rmax = r
                if store:
                    out_masks[count - 1] = subc; out_r[count - 1] = r
                if kk > 0 and r > topk_r[0]:
                    j = 0
                    while j + 1 < kk and topk_r[j + 1] < r:
                        topk_r[j] = topk_r[j + 1]; topk_mask[j] = topk_mask[j + 1]; j += 1
                    topk_r[j] = r; topk_mask[j] = subc
            if size < smax:
                new = nbr[w] & ~excl & above
                stack_sub[sp] = subc; stack_ext[sp] = ext2 | new; stack_excl[sp] = excl | nbr[w] | wbit
                stack_m[sp] = mc; stack_sx[sp] = sxc; stack_sy[sp] = syc; stack_p[sp] = pc; sp += 1
    return count, rmax


class Enumerator:
    def __init__(self, W, cls):
        self.W, self.cls = W, cls; n = W.n
        nbr = np.zeros(n, np.int64)
        for i in range(n):
            for j in W.adj[i]:
                nbr[i] |= np.int64(1) << np.int64(j)
        self.nbr = nbr; self.n = n
        self._dummy_m = np.zeros(1, np.int64); self._dummy_r = np.zeros(1)

    def max_reduced(self, sg, pi, theta, b, topk=0):
        """exact max reduced cost over all admissible cells; optionally the top-k cells (masks) with their r"""
        W = self.W; cls = self.cls
        tr = np.full(max(topk, 1), -1e300); tm = np.zeros(max(topk, 1), np.int64)
        cnt, rmax = esu_pass(self.nbr, W.mu, W.mux, sg * W.muy, np.ascontiguousarray(pi, dtype=np.float64), self.n, cls.smax, cls.tlo, cls.thi,
                             float(b), float(theta), False, self._dummy_m, self._dummy_r, tr if topk else np.zeros(0), tm if topk else np.zeros(0, np.int64))
        cells = []
        if topk:
            for r, mk in zip(tr, tm):
                if r > -1e299:
                    cells.append((float(r), [i for i in range(self.n) if (int(mk) >> i) & 1]))
        return int(cnt), float(rmax), cells

    def count(self):
        return self.max_reduced(1, np.zeros(self.n), 0.0, 0.0)[0]


def brute_count(W, cls):
    """independent check for small smax: all subsets of size <= smax by combinations"""
    import itertools
    c = 0
    for k in range(1, cls.smax + 1):
        for comb in itertools.combinations(range(W.n), k):
            m = np.array(comb)
            t = W.mu[m].sum()
            if t <= cls.thi + MASS_TOL and t >= cls.tlo - MASS_TOL and W.connected(m):
                c += 1
    return c


def r_allowance(W, pi, theta, b):
    """outward rounding allowance for the reduced-cost evaluation of any cell (relative 1e-12 of the magnitudes involved)"""
    S = np.abs(W.mux).sum() + np.abs(W.muy).sum() + 1.0
    return 1e-12 * (S * S * (1 + abs(b)) / max(W.mu.min(), 1e-12) + np.abs(pi).sum() + abs(theta) + 1.0)


def verified_certificate(W, cls, pi, theta, b, eps_exact):
    """Charged-pricing bound (Theorem S17) with outward rounding: UB = b + max(0, (1'pi + K(theta + eps)) / Dmin)."""
    eps = eps_exact + r_allowance(W, pi, theta, b)
    s = math.fsum(pi.tolist()) + cls.K * (theta + eps)
    slack = 1e-12 * (np.abs(pi).sum() + cls.K * abs(theta) + cls.K * abs(eps) + 1.0)
    return float(b + max(0.0, (s + slack) / cls.Dmin) + 1e-12 * (abs(b) + 1)), float(eps)


def run_cg_exact(W, cls, sg, pool, enum, opts, verbose=False):
    """column generation with exact enumeration pricing; returns the verified record (scaled units)"""
    t0 = time.time(); cols = list(range(len(pool.mem))); inrmp = set(cols); it = 0; n_exact = 0; mp = None
    rng_cg = np.random.default_rng(12345 + sg)
    while True:
        it += 1
        mp = PC.solve_master(pool, cols, sg, cls.K, cls.Dmin)
        if mp is None:
            return dict(status="master_infeasible")
        pi, theta, b = mp["pi"], mp["theta"], mp["b"]
        r = pool.reduced(sg, pi, theta, b); r[list(inrmp)] = -INF
        add = [int(j) for j in np.argsort(-r)[:60] if r[j] > opts["tol_add"]]
        stage = "pool"
        if len(add) < 10:
            stage = "local"; rall = pool.reduced(sg, pi, theta, b)
            for j in [int(j) for j in np.argsort(-rall)[:30]]:
                m, rv = PC.improve_cell(W, cls, sg, pi, theta, b, pool.mem[j])
                if rv > opts["tol_add"] and cls.admissible(W, m):
                    jj, _ = pool.add(m)
                    if jj not in inrmp:
                        add.append(jj)
        if len(add) < 10:
            stage = "grow"
            for rv, m in sorted(PC.grow_cells(W, cls, sg, pi, theta, b, range(W.n)), key=lambda z: -z[0])[:40]:
                if rv > opts["tol_add"]:
                    m2, rv2 = PC.improve_cell(W, cls, sg, pi, theta, b, m)
                    for mm, rr in ((m, rv), (m2, rv2)):
                        if rr > opts["tol_add"] and cls.admissible(W, mm):
                            jj, _ = pool.add(mm)
                            if jj not in inrmp:
                                add.append(jj)
        if not add:
            stage = "exact"; n_exact += 1
            cnt, rmax, cells = enum.max_reduced(sg, pi, theta, b, topk=opts["exact_add"])
            for rv, m in cells:
                if rv > opts["tol_exact"] and cls.admissible(W, np.array(m)):
                    jj, _ = pool.add(np.array(m))
                    if jj not in inrmp:
                        add.append(jj)
            if verbose:
                print(f"  [{cls.name} sg={sg}] it={it} exact pricing: cells={cnt} max_r={rmax:.3e} add={len(add)} t={time.time()-t0:.0f}s", file=sys.stderr, flush=True)
            if rmax <= opts["tol_exact"] or not add or it > opts["max_it"] or time.time() - t0 > opts["cg_time"]:
                break
        if verbose and stage != "exact":
            print(f"  [{cls.name} sg={sg}] it={it} stage={stage} add={len(add)} lp={mp['value']:.6f} cols={len(cols)} t={time.time()-t0:.0f}s", file=sys.stderr, flush=True)
        for jj in dict.fromkeys(add):
            cols.append(jj); inrmp.add(jj)
        if it > opts["max_it"] or time.time() - t0 > opts["cg_time"]:
            break
    # final verified certificate at the final duals (recomputed: exact enumeration, no columns added)
    cnt, eps_exact, _ = enum.max_reduced(sg, pi, theta, b)
    ub, eps_used = verified_certificate(W, cls, pi, theta, b, eps_exact)
    rec = dict(status="verified" if eps_exact <= opts["tol_exact"] else "verified_budget", iters=it, exact_calls=n_exact, cols=len(cols),
               pool=len(pool.mem), n_cells=int(cnt), lp_value=float(mp["value"]), b=float(b), theta=float(theta), pi_sum=float(math.fsum(pi.tolist())),
               eps_exact=float(eps_exact), eps_used=eps_used, ub_cert=ub, charge=float(ub - (b + max(0.0, (math.fsum(pi.tolist()) + cls.K * theta) / cls.Dmin))),
               secs=time.time() - t0)
    rec["_duals"] = (pi.copy(), float(theta), float(b)); rec["_cols"] = cols
    return rec


def task_verify(args):
    nodes, K, beta, seed, opts, wi = args
    D = PC.load_data(); W = PC.Window(nodes, D); cls = PC.ZClass(W, K, beta); rng = np.random.default_rng(seed)
    enum = Enumerator(W, cls); t0 = time.time(); n_cells = enum.count(); t_enum = time.time() - t0
    rec = dict(window=wi, n=W.n, K=K, beta=beta, smax=cls.smax, smin=cls.smin, tlo=cls.tlo, thi=cls.thi, n_cells=n_cells, enum_secs=t_enum, conv=W.sy / W.sx)
    model = PC.PricingModel(W, cls, conn=True, integral=True)
    for sg, nm in ((1, "upper"), (-1, "lower")):
        pool = PC.Pool(W); t1 = time.time()
        hc = PC.hill_climb(W, cls, sg, rng, restarts=opts.get("restarts", max(40, 4000 // W.n)), pool=pool)
        if hc[1] is None:
            rec[nm] = dict(status="no_witness"); continue
        ok, beta_hc = PC.verify_partition(W, cls, hc[1])
        cg = run_cg_exact(W, cls, sg, pool, enum, opts, verbose=opts.get("verbose", False))
        if cg.get("status") == "master_infeasible":
            rec[nm] = cg; continue
        pi, theta, b = cg.pop("_duals"); cols = cg.pop("_cols"); cg["duals"] = dict(pi=[float(v) for v in pi], theta=float(theta), b=float(b)); cg["nodes"] = [int(v) for v in nodes]
        ip = PC.master_ip_witness(W, cls, sg, pool, cols, hc[0], time_limit=opts.get("ip_time", 60.0))
        ok2, beta_ip = PC.verify_partition(W, cls, ip[1]) if ip is not None else (False, None)
        # soundness check of the MILP pricing layer at the same duals
        t2 = time.time()
        cp = PC.certified_pricing(model, sg, pi, theta, b, opts["eps_target"], opts["tol_add"], opts["milp_time"], find=False, mil_tl=opts["milp_tl"])
        cg.update(hc_beta=beta_hc if ok else None, ip_beta=beta_ip if ok2 else None, bound_orig=sg * cg["ub_cert"] * W.sy / W.sx,
                  lp_orig=sg * cg["lp_value"] * W.sy / W.sx, eps_milp=cp["eps"], milp_nodes=cp["nodes"], milp_open=cp.get("open", 0), milp_secs=time.time() - t2,
                  milp_sound=bool(cp["eps"] >= cg["eps_exact"] - 1e-9), milp_excess=float(cp["eps"] - cg["eps_exact"]), total_secs=time.time() - t1)
        rec[nm] = cg
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", default="count"); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=30); p.add_argument("--windows", type=int, default=8); p.add_argument("--seed", type=int, default=11)
    p.add_argument("--procs", type=int, default=8); p.add_argument("--only", type=int, default=-1); p.add_argument("--verbose", action="store_true")
    p.add_argument("--classes", default="3:inf,3:0.5,5:inf,5:0.5,8:inf,8:0.5"); p.add_argument("--cg_time", type=float, default=3600.0)
    p.add_argument("--milp_time", type=float, default=600.0); p.add_argument("--brute", action="store_true")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); D = PC.load_data()
    rng = np.random.default_rng(a.seed); wins = []
    while len(wins) < a.windows:
        s = int(rng.integers(D["n"])); w = PC.bfs_window(D, s, a.n)
        if w is not None:
            wins.append(w)
    cl = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in a.classes.split(",")]
    t_all = time.time()
    if a.mode == "count":
        rows = []
        for wi, w in enumerate(wins):
            W = PC.Window(w, D)
            for K, beta in cl:
                cls = PC.ZClass(W, K, beta); en = Enumerator(W, cls); t0 = time.time(); c = en.count(); dt = time.time() - t0
                row = dict(window=wi, n=W.n, K=K, beta=beta, smax=cls.smax, smin=cls.smin, n_cells=c, secs=dt)
                if a.brute and cls.smax <= 9:
                    row["brute"] = brute_count(W, cls); row["agree"] = (row["brute"] == c)
                rows.append(row); print(row, flush=True)
        json.dump(rows, open(os.path.join(a.out, f"enum_counts_n{a.n}.json"), "w"), indent=1, default=float)
    else:
        opts = dict(tol_add=1e-7, tol_exact=1e-9, exact_add=40, eps_target=2e-5, cg_time=a.cg_time, max_it=4000, milp_time=a.milp_time, milp_tl=30.0,
                    ip_time=60.0, verbose=a.verbose)
        tasks = [(w, K, beta, 7 + 100 * wi + K, opts, wi) for wi, w in enumerate(wins) for K, beta in cl]
        if a.only >= 0:
            tasks = tasks[a.only:a.only + 1]
        import multiprocessing as mpc
        rows = []
        with mpc.Pool(min(a.procs, len(tasks))) as P:
            for r in P.imap_unordered(task_verify, tasks):
                rows.append(r); u_, l_ = r.get("upper", {}), r.get("lower", {})
                print(f"win {r['window']} K={r['K']} beta={r['beta']} cells={r['n_cells']}: verified=[{l_.get('bound_orig')},{u_.get('bound_orig')}] "
                      f"eps_exact=({l_.get('eps_exact')},{u_.get('eps_exact')}) milp_sound=({l_.get('milp_sound')},{u_.get('milp_sound')}) "
                      f"wit=[{l_.get('ip_beta')},{u_.get('ip_beta')}] secs=({l_.get('total_secs')},{u_.get('total_secs')}) ({time.time()-t_all:.0f}s)", flush=True)
                json.dump(sorted(rows, key=lambda z: (z["window"], z["K"], z["beta"])), open(os.path.join(a.out, f"pricing_verify_n{a.n}.json"), "w"), indent=1, default=float)
    print(f"total {time.time()-t_all:.0f}s")
