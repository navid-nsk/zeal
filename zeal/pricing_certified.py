"""pricing_certified.py -- verified column generation for the coarse slope under contiguous re-zoning.

Mathematics: Supplementary Theorem S17 (slopes; set-partitioning programme; charged pricing).

Object.  A window W of n Output Areas (Greater Manchester; X = q4 degree share, Y = bad-health share, weights w_q4,
normalised to mu(W) = 1).  Class C(K, beta): partitions of W into exactly K contiguous cells (window-induced adjacency),
optionally population-balanced |mu_C - 1/K| <= beta/K; declared denominator floor D(Z) >= Dmin = 0.25 Var_W(X).
Internally X, Y are centred and scaled to unit window variance (slope_orig = slope_scaled * sd_Y / sd_X; Dmin = 0.25).

Certificate (charged pricing, Theorem S17).  Charnes-Cooper master over generated columns gives (pi, theta, b);
a certified pricing bound  eps >= max_{C admissible} [n_C - b d_C - pi(C) - theta]  over ALL admissible cells gives
        beta(Z) <= b + max(0, (1'pi + K (theta + eps)) / Dmin)     for every Z in the class with D(Z) >= Dmin.
Pricing bound (MILP layer): n_C - b d_C = S_x V / t = (A^2 - B^2)/t with A, B linear cell sums; spatial branch and bound
on the cell ratio A/t (secant over-estimator of A^2/t), tangent cuts for -B^2/t, and EXACT contiguity by a rooted
single-commodity-flow MILP (HiGHS); the bound is the max of the branch MILP dual bounds (+ declared safety margin).
Relaxation (i) (no contiguity, fractional) is the same branch problem as an LP with a Neumaier-Shcherbina safe bound.

Modes
  --mode a4  : the 20 windows of 12 OAs of a4_exact_windows.py (seed 5, replicated RNG); all contiguous cells enumerated:
               full-column LP vs verified CG; exact max reduced cost vs certified pricing bound (soundness, tightness).
  --mode big : BFS windows of --n OAs from random seeds; K in {3,5,8}, beta in {inf, 0.5}; certified slope interval,
               pricing bound at termination, hill-climbing and restricted-master-IP witnesses, times.
  --mode theory : skeleton check (max of relaxation (i) attained on the <=2-fractional skeleton; LP B&B upper bound agrees).
Usage: python pricing_certified.py --mode a4 --out <dir>
       python pricing_certified.py --mode theory --out <dir>
       python pricing_certified.py --mode big --n 30 --windows 8 --seed 11 --out <dir> [--procs 12]
       python pricing_certified.py --mode big --n 100 --windows 4 --seed 11 --split_dirs --procs 20 --cg_time 900 \
              --final_price_time 600 --ip_time 30 --max_it 4000 --out <dir>
Every reported bound is valid at any stopping point (Theorem S17): budget exhaustion only widens it by the charge K*eps/Dmin.
CPU only (numpy, scipy HiGHS).
"""
try:
    from . import paths                     # imported as part of the package
except ImportError:
    import paths                            # flat import (this directory on sys.path)
import argparse, collections, heapq, io, json, math, os, sys, time
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

DATA = paths.GM_DIR
INF = np.inf
SAFE = 1e-6            # declared safety margin added to every floating-point MILP dual bound (scaled objective units)


# ----------------------------------------------------------------------------------------------------------- data
def load_data():
    u = pd.read_csv(DATA + "gm_units.csv"); adj = pd.read_csv(DATA + "gm_adjacency.csv").values
    assert (u["OA_i"].values == np.arange(len(u))).all()
    E = adj[adj[:, 0] != adj[:, 1]]; nbrs = collections.defaultdict(set)
    for a_, b_ in E:
        nbrs[int(a_)].add(int(b_)); nbrs[int(b_)].add(int(a_))
    return dict(n=len(u), nbrs=nbrs, X=u["q4"].values.astype(float), Y=u["bad"].values.astype(float),
                W=u["w_q4"].values.astype(float), area=u["area_km2"].values.astype(float))


class Window:
    def __init__(self, nodes, D):
        self.nodes = list(nodes); n = self.n = len(nodes); idx = {v: i for i, v in enumerate(self.nodes)}
        self.adj = [sorted(idx[w] for w in D["nbrs"][v] if w in idx) for v in self.nodes]
        self.edges = [(i, j) for i in range(n) for j in self.adj[i] if i < j]
        w = D["W"][self.nodes]; self.mu = w / w.sum(); X = D["X"][self.nodes]; Y = D["Y"][self.nodes]
        self.Xraw, self.Yraw, self.wraw = X, Y, w
        self.xbar = float((self.mu * X).sum()); self.ybar = float((self.mu * Y).sum())
        self.sx = math.sqrt(float((self.mu * (X - self.xbar) ** 2).sum())); self.sy = math.sqrt(float((self.mu * (Y - self.ybar) ** 2).sum()))
        self.x = (X - self.xbar) / self.sx; self.y = (Y - self.ybar) / self.sy
        self.mux = self.mu * self.x; self.muy = self.mu * self.y
        A = sparse.lil_matrix((n, n))
        for i, j in self.edges:
            A[i, j] = 1; A[j, i] = 1
        self.A = A.tocsr()
        self.area = float(D["area"][self.nodes].sum()); self.req_km = math.sqrt(self.area / math.pi)

    def connected(self, members):
        ms = set(int(i) for i in members)
        if not ms:
            return False
        st = [next(iter(ms))]; seen = {st[0]}
        while st:
            v = st.pop()
            for w_ in self.adj[v]:
                if w_ in ms and w_ not in seen:
                    seen.add(w_); st.append(w_)
        return len(seen) == len(ms)

    def slope_orig(self, labels, K):
        """slope and D of a partition, recomputed from raw data (independent of the scaled pipeline)"""
        mu = self.wraw / self.wraw.sum(); m = np.array([mu[labels == k].sum() for k in range(K)])
        mx = np.array([(mu[labels == k] * self.Xraw[labels == k]).sum() / m[k] for k in range(K)])
        my = np.array([(mu[labels == k] * self.Yraw[labels == k]).sum() / m[k] for k in range(K)])
        xb = (m * mx).sum(); yb = (m * my).sum(); Dx = (m * (mx - xb) ** 2).sum(); N = (m * (mx - xb) * (my - yb)).sum()
        return N / Dx, Dx


class ZClass:
    def __init__(self, W, K, beta, size_cap=True):
        self.K = K; self.beta = beta
        if np.isfinite(beta):
            self.tlo, self.thi = (1 - beta) / K, (1 + beta) / K
        else:
            self.tlo, self.thi = float(W.mu.min()), 1.0
        self.smax = W.n - K + 1 if size_cap else W.n
        srt = np.sort(W.mu)                                # valid size bounds implied by the mass bounds
        self.smax = int(min(self.smax, np.searchsorted(np.cumsum(srt), self.thi + 1e-12, side="right")))
        self.smin = int(max(1, np.searchsorted(np.cumsum(srt[::-1]), self.tlo - 1e-12, side="left") + 1))
        self.Dmin = 0.25                                   # 0.25 * Var_W(X) in scaled units
        self.name = f"K{K}_beta{beta}"

    def admissible(self, W, members):
        t = W.mu[members].sum()
        return (len(members) <= self.smax and t >= self.tlo - 1e-12 and t <= self.thi + 1e-12 and W.connected(members))


# ----------------------------------------------------------------------------------------------------------- pool
class Pool:
    """all admissible cells ever seen; sums are unsigned (Y direction applied at evaluation)"""
    def __init__(self, W):
        self.W = W; self.key2i = {}; self.mem = []; self.t = []; self.sx = []; self.sy = []; self._M = None

    def add(self, members):
        m = np.array(sorted(int(i) for i in members)); key = 0
        for i in m:
            key |= 1 << int(i)
        j = self.key2i.get(key)
        if j is not None:
            return j, False
        j = len(self.mem); self.key2i[key] = j; self.mem.append(m)
        self.t.append(self.W.mu[m].sum()); self.sx.append(self.W.mux[m].sum()); self.sy.append(self.W.muy[m].sum()); self._M = None
        return j, True

    def M(self):
        if self._M is None or self._M.shape[0] != len(self.mem):
            ind = np.concatenate(self.mem); ptr = np.r_[0, np.cumsum([len(m) for m in self.mem])]
            self._M = sparse.csr_matrix((np.ones(len(ind)), ind, ptr), shape=(len(self.mem), self.W.n))
        return self._M

    def arrays(self, sg):
        t = np.array(self.t); sx = np.array(self.sx); sy = sg * np.array(self.sy)
        return t, sx, sy, sx * sy / t, sx * sx / t

    def reduced(self, sg, pi, theta, b):
        t, sx, sy, nC, dC = self.arrays(sg)
        return nC - b * dC - self.M() @ pi - theta


# ----------------------------------------------------------------------------------------------------------- master
def solve_master(pool, cols, sg, K, Dmin):
    """Charnes-Cooper LP: max n'y  s.t. By = s 1, 1'y = K s, d'y = 1, Dmin s <= 1, y, s >= 0.  Returns value, (pi, theta, b), y"""
    W = pool.W; n = W.n; cols = np.array(cols); t, sx, sy, nC, dC = pool.arrays(sg); m = len(cols)
    Mc = pool.M()[cols].T.tocsr()                                           # n x m
    top = sparse.hstack([Mc, sparse.csr_matrix(-np.ones((n, 1)))])
    rowK = sparse.csr_matrix(np.r_[np.ones(m), -K][None, :]); rowd = sparse.csr_matrix(np.r_[dC[cols], 0.0][None, :])
    Aeq = sparse.vstack([top, rowK, rowd]).tocsc(); beq = np.r_[np.zeros(n + 1), 1.0]
    Aub = sparse.csr_matrix(np.r_[np.zeros(m), Dmin][None, :]); bub = np.array([1.0])
    res = linprog(np.r_[-nC[cols], 0.0], A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    if res.status != 0:
        return None
    lam = res.eqlin.marginals; pi = -lam[:n]; theta = -lam[n]; b = -lam[n + 1]
    return dict(value=-res.fun, pi=pi, theta=float(theta), b=float(b), y=res.x[:m], s=res.x[m])


def certificate(pool, cols, sg, K, Dmin, pi, theta, b, eps_price):
    """Charged-pricing bound (Theorem S17): UB = b + max(0, (1'pi + K(theta+eps))/Dmin), eps = max(eps_price, violations of stored columns),
    with an outward float allowance."""
    r = pool.reduced(sg, pi, theta, b)[np.array(cols)]
    eps_cols = float(r.max()); eps = max(eps_price, eps_cols)
    s = math.fsum(pi.tolist()) + K * (theta + eps)
    slack = 1e-12 * (np.abs(pi).sum() + K * abs(theta) + K * abs(eps) + 1.0)        # rounding allowance (outward)
    ub = b + max(0.0, (s + slack) / Dmin) + 1e-12 * (abs(b) + 1)
    return float(ub), float(eps_cols), float(eps)


# ----------------------------------------------------------------------------------------------------------- heuristic pricing
def cell_r(W, sg, pi, theta, b, t, Sx, Sy, P):
    return (Sx * Sy - b * Sx * Sx) / t - P - theta


def improve_cell(W, cls, sg, pi, theta, b, members, max_steps=300):
    n = W.n; inC = np.zeros(n, bool); inC[list(members)] = True
    muy = sg * W.muy; t = W.mu[inC].sum(); Sx = W.mux[inC].sum(); Sy = muy[inC].sum(); P = pi[inC].sum(); size = int(inC.sum())
    cur = cell_r(W, sg, pi, theta, b, t, Sx, Sy, P)
    for _ in range(max_steps):
        best = cur + 1e-12; mv = None
        if size < cls.smax:
            fr = np.where((W.A @ inC.astype(float) > 0) & ~inC & (t + W.mu <= cls.thi + 1e-12))[0]
            if len(fr):
                rr = cell_r(W, sg, pi, theta, b, t + W.mu[fr], Sx + W.mux[fr], Sy + muy[fr], P + pi[fr]); k = int(np.argmax(rr))
                if rr[k] > best:
                    best = rr[k]; mv = ("add", fr[k])
        if size > 1:
            cand = np.where(inC & (t - W.mu >= cls.tlo - 1e-12))[0]
            if len(cand):
                rr = cell_r(W, sg, pi, theta, b, t - W.mu[cand], Sx - W.mux[cand], Sy - muy[cand], P - pi[cand])
                for k in np.argsort(-rr)[:8]:
                    if rr[k] <= best:
                        break
                    rest = np.where(inC)[0]; rest = rest[rest != cand[k]]
                    if W.connected(rest):
                        best = rr[k]; mv = ("rem", cand[k]); break
        if mv is None:
            break
        u = mv[1]; s_ = 1 if mv[0] == "add" else -1; inC[u] = (s_ == 1)
        t += s_ * W.mu[u]; Sx += s_ * W.mux[u]; Sy += s_ * muy[u]; P += s_ * pi[u]; size += s_; cur = best
    return np.where(inC)[0], cur


def grow_cells(W, cls, sg, pi, theta, b, seeds):
    """greedy best-first growth from each seed unit; returns the best admissible prefix per seed"""
    n = W.n; muy = sg * W.muy; out = []
    for s0 in seeds:
        inC = np.zeros(n, bool); inC[s0] = True; t = W.mu[s0]; Sx = W.mux[s0]; Sy = muy[s0]; P = pi[s0]; size = 1
        best = (cell_r(W, sg, pi, theta, b, t, Sx, Sy, P), [s0]) if t >= cls.tlo - 1e-12 else (-INF, None)
        order = [s0]
        while size < cls.smax:
            fr = np.where((W.A @ inC.astype(float) > 0) & ~inC & (t + W.mu <= cls.thi + 1e-12))[0]
            if not len(fr):
                break
            rr = cell_r(W, sg, pi, theta, b, t + W.mu[fr], Sx + W.mux[fr], Sy + muy[fr], P + pi[fr]); k = int(np.argmax(rr)); u = fr[k]
            inC[u] = True; t += W.mu[u]; Sx += W.mux[u]; Sy += muy[u]; P += pi[u]; size += 1; order.append(u)
            if t >= cls.tlo - 1e-12 and rr[k] > best[0]:
                best = (rr[k], list(order))
        if best[1] is not None:
            out.append(best)
    return out


# ----------------------------------------------------------------------------------------------------------- certified pricing
class PricingModel:
    """Branch problem for ratio interval [lo, hi] of rho_A = A/t:
        max  sum_u [(lo+hi) mu_u a_u - lo hi mu_u - pi_u] x_u + z - theta
        s.t. z <= -2 alpha B + alpha^2 t (alpha in cut set);  lo t <= A <= hi t;  tlo <= t <= thi;  1 <= |C| <= smax;
             (conn) rooted single-commodity flow: C connected.   x binary (integral) or [0,1] (relaxation (i))."""
    def __init__(self, W, cls, conn=True, integral=True):
        n = W.n; self.W, self.cls, self.conn, self.integral, self.n = W, cls, conn, integral, n
        arcs = [(i, j) for i, j in W.edges] + [(j, i) for i, j in W.edges] if conn else []
        self.na = len(arcs); self.nv = (2 * n + self.na + 1) if conn else (n + 1); self.iz = self.nv - 1
        R, Cc, V, lb, ub = [], [], [], [], []
        def row(ent, l, u):
            r = len(lb)
            for c, v in ent:
                R.append(r); Cc.append(c); V.append(v)
            lb.append(l); ub.append(u)
        row([(i, W.mu[i]) for i in range(n)], cls.tlo - 1e-12, cls.thi + 1e-12)
        row([(i, 1.0) for i in range(n)], cls.smin, cls.smax)
        S = cls.smax
        if conn and cls.smin >= 2:                                    # a selected unit of a cell with >= 2 units has a selected neighbour
            for u in range(n):
                row([(u, 1.0)] + [(v, -1.0) for v in W.adj[u]], -INF, 0)
        if conn:
            row([(n + i, 1.0) for i in range(n)], 1, 1)
            for u in range(n):
                row([(n + u, 1.0), (u, -1.0)], -INF, 0)
            for u in range(1, n):                                     # root = smallest selected index (symmetry breaking)
                row([(v, 1.0) for v in range(u)] + [(n + u, float(S))], -INF, S)
            ain = collections.defaultdict(list); aout = collections.defaultdict(list)
            for a, (i, j) in enumerate(arcs):
                aout[i].append(a); ain[j].append(a)
            for i in range(n):
                ent = [(2 * n + a, 1.0) for a in ain[i]] + [(2 * n + a, -1.0) for a in aout[i]]
                row(ent + [(i, -1.0)], -INF, 0)                       # in - out <= x_i
                row(ent + [(i, -1.0), (n + i, float(S))], 0, INF)     # in - out >= x_i - S r_i
            for a, (i, j) in enumerate(arcs):
                row([(2 * n + a, 1.0), (i, -(S - 1.0))], -INF, 0); row([(2 * n + a, 1.0), (j, -(S - 1.0))], -INF, 0)
        self.Astat = sparse.csr_matrix((V, (R, Cc)), shape=(len(lb), self.nv)); self.lstat = np.array(lb); self.ustat = np.array(ub)
        lbv = np.zeros(self.nv); ubv = np.ones(self.nv)
        if conn:
            ubv[2 * n: 2 * n + self.na] = S - 1
        self.lbv, self.ubv = lbv, ubv
        self.integ = np.zeros(self.nv)
        if integral:
            self.integ[:n] = 1
            if conn:
                self.integ[n:2 * n] = 1

    def setup(self, sg, pi, theta, b):
        W = self.W; x = W.x; v = sg * W.y - b * x
        Rx = np.ptp(x); Rv = max(np.ptp(v), 1e-9); kap = math.sqrt(Rv / Rx)
        self.a = (kap * x + v / kap) / 2; self.bb = (kap * x - v / kap) / 2
        self.amu = W.mu * self.a; self.bmu = W.mu * self.bb; self.pi = pi; self.theta = theta
        self.lbv[self.iz] = -float((self.bb ** 2).max()) - 1.0; self.ubv[self.iz] = 0.0

    def true_r(self, members, sg, b):
        W = self.W; t = W.mu[members].sum(); Sx = W.mux[members].sum(); Sy = sg * W.muy[members].sum()
        return cell_r(W, sg, self.pi, self.theta, b, t, Sx, Sy, self.pi[members].sum())

    def solve(self, lo, hi, alphas, time_limit):
        n, W = self.n, self.W; c = np.zeros(self.nv)
        c[:n] = -((lo + hi) * self.amu - lo * hi * W.mu - self.pi); c[self.iz] = -1.0
        dyn = []
        for al in alphas:
            r_ = np.zeros(self.nv); r_[:n] = 2 * al * self.bmu - al * al * W.mu; r_[self.iz] = 1.0; dyn.append(r_)
        r1 = np.zeros(self.nv); r1[:n] = self.amu - hi * W.mu; r2 = np.zeros(self.nv); r2[:n] = lo * W.mu - self.amu
        dyn += [r1, r2]
        A = sparse.vstack([self.Astat, sparse.csr_matrix(np.array(dyn))]).tocsr()
        lb = np.r_[self.lstat, -INF * np.ones(len(dyn))]; ub = np.r_[self.ustat, np.zeros(len(dyn))]
        if self.integral:
            res = milp(c, integrality=self.integ, bounds=Bounds(self.lbv, self.ubv), constraints=[LinearConstraint(A, lb, ub)],
                       options=dict(time_limit=time_limit, mip_rel_gap=1e-7, disp=False))
            if res.status == 2:
                return -INF, None
            bound = -res.mip_dual_bound if res.mip_dual_bound is not None else INF
            if not np.isfinite(bound):
                bound = INF
            xv = None if res.x is None else np.round(res.x[:n])
            return bound - self.theta + SAFE, xv
        # relaxation (i): LP, Neumaier-Shcherbina safe bound from the (approximate) duals, all variables boxed
        Aub = sparse.vstack([A, -A]).tocsr(); bub = np.r_[ub, -lb]; keep = np.isfinite(bub)
        Aub = Aub[np.where(keep)[0]]; bub = bub[keep]
        res = linprog(c, A_ub=Aub, b_ub=bub, bounds=list(zip(self.lbv, self.ubv)), method="highs")
        if res.status == 2:
            return -INF, None
        y = np.maximum(-res.ineqlin.marginals, 0.0)                    # y >= 0 for  max (-c)'x, Aub x <= bub
        red = -c - Aub.T @ y
        safe = float(y @ bub + np.maximum(red * self.lbv, red * self.ubv).sum())
        safe += 1e-12 * (np.abs(y) @ np.abs(bub) + np.abs(red).sum() + 1)
        return safe - self.theta, (np.clip(res.x[:n], 0, 1) if res.x is not None else None)


def certified_pricing(model, sg, pi, theta, b, eps_target, tol_add, time_budget, find=True, J0=6, max_nodes=400, mil_tl=30.0):
    """Spatial B&B over rho_A. Returns dict(eps (valid upper bound on max reduced cost over the class, or None if stopped early
    on finding columns), cols (improving admissible cells), nodes, solves, secs)."""
    t0 = time.time(); model.setup(sg, pi, theta, b); W = model.W
    amin, amax = float(model.a.min()), float(model.a.max()); bmin, bmax = float(model.bb.min()), float(model.bb.max())
    alphas = sorted(set([0.0] + list(np.linspace(bmin, bmax, 9))))
    edges = np.linspace(amin, amax, J0 + 1); heap = [(-INF, float(edges[k]), float(edges[k + 1])) for k in range(J0)]
    closed = []; found = {}; nodes = 0; solves = 0; open_left = []
    while heap:
        negub, lo, hi = heapq.heappop(heap); parent_ub = -negub
        if (time.time() - t0 > time_budget or nodes >= max_nodes) and np.isfinite(parent_ub):
            open_left.append(parent_ub); continue
        nodes += 1; solves += 1
        ub, xv = model.solve(lo, hi, alphas, mil_tl)
        ub = min(ub, parent_ub)
        xs = None
        if xv is not None and xv.sum() > 0:
            integ = bool(np.all(np.abs(xv - np.round(xv)) < 1e-9)); xs = np.where(xv > 0.5)[0]
            if integ and len(xs):
                rt = model.true_r(xs, sg, b)
                if rt > tol_add and (not model.conn or W.connected(xs)):
                    found[tuple(int(i) for i in xs)] = rt
        if find and found and (nodes >= J0 or len(found) >= 4):     # find mode: return a batch of improving columns
            return dict(eps=None, cols=found, nodes=nodes, solves=solves, secs=time.time() - t0)
        if ub <= eps_target or not np.isfinite(ub) and ub < 0:
            closed.append(ub); continue
        if hi - lo < 1e-5 * (amax - amin + 1e-12):
            closed.append(ub); continue
        split = 0.5 * (lo + hi)
        if xv is not None and xv.sum() > 0:
            t = W.mu @ xv; ra = (model.amu @ xv) / t; rb = (model.bmu @ xv) / t
            if lo + 0.15 * (hi - lo) < ra < hi - 0.15 * (hi - lo):
                split = ra
            if min(abs(rb - a_) for a_ in alphas) > 1e-9:
                alphas = sorted(alphas + [rb])
        heapq.heappush(heap, (-ub, lo, split)); heapq.heappush(heap, (-ub, split, hi))
    eps = max(closed + open_left) if (closed or open_left) else -INF
    return dict(eps=float(eps), cols=found, nodes=nodes, solves=solves, secs=time.time() - t0, open=len(open_left))


# ----------------------------------------------------------------------------------------------------------- column generation
def run_cg(W, cls, sg, pool, cols0, opts, model, model_i=None, enum_cells=None):
    """verified CG for the upper slope of sg*Y on X (sg=-1 gives minus the lower slope). Returns a record (scaled units)."""
    t0 = time.time(); cols = list(dict.fromkeys(cols0)); inrmp = set(cols); it = 0; log = []
    eps = None; status = "certified"; mp = None; rng_cg = np.random.default_rng(12345 + sg)
    while True:
        it += 1
        mp = solve_master(pool, cols, sg, cls.K, cls.Dmin)
        if mp is None:
            return dict(status="master_infeasible")
        pi, theta, b = mp["pi"], mp["theta"], mp["b"]
        r = pool.reduced(sg, pi, theta, b); r[list(inrmp)] = -INF
        add = [int(j) for j in np.argsort(-r)[:60] if r[j] > opts["tol_add"]]
        stage = "pool"
        if len(add) < 10:
            stage = "local"; rall = pool.reduced(sg, pi, theta, b)
            top = [int(j) for j in np.argsort(-rall)[:30]]
            for j in top:
                m, rv = improve_cell(W, cls, sg, pi, theta, b, pool.mem[j])
                if rv > opts["tol_add"] and cls.admissible(W, m):
                    jj, new = pool.add(m)
                    if jj not in inrmp:
                        add.append(jj)
        if len(add) < 10:
            stage = "grow"; ns = opts.get("grow_seeds", 0)
            seeds = range(W.n) if (ns <= 0 or ns >= W.n) else rng_cg.choice(W.n, ns, replace=False)
            for rv, m in sorted(grow_cells(W, cls, sg, pi, theta, b, seeds), key=lambda z: -z[0])[:opts.get("grow_add", 40)]:
                if rv > opts["tol_add"]:
                    m2, rv2 = improve_cell(W, cls, sg, pi, theta, b, m)
                    for mm, rr in ((m, rv), (m2, rv2)):
                        if rr > opts["tol_add"] and cls.admissible(W, mm):
                            jj, new = pool.add(mm)
                            if jj not in inrmp:
                                add.append(jj)
        timeout = (time.time() - t0 > opts["cg_time"]) or it > opts["max_it"]
        if not add and not timeout:
            stage = "milp"
            cp = certified_pricing(model, sg, pi, theta, b, opts["eps_target"], opts["tol_add"], opts["price_time"], find=True,
                                   mil_tl=opts["milp_tl"])
            if cp["cols"]:
                for m in cp["cols"]:
                    m = np.array(m)
                    for mm in (m, improve_cell(W, cls, sg, pi, theta, b, m)[0]):
                        if cls.admissible(W, mm):
                            jj, new = pool.add(mm)
                            if jj not in inrmp:
                                add.append(jj)
            else:
                eps = cp; break
        log.append((it, stage, len(add), float(mp["value"]), float(b)))
        if opts.get("verbose"):
            print(f"  [{cls.name} sg={sg}] it={it} stage={stage} add={len(add)} lp={mp['value']:.6f} cols={len(cols)} pool={len(pool.mem)} "
                  f"t={time.time()-t0:.0f}s", file=sys.stderr, flush=True)
        if add and not timeout:
            for jj in dict.fromkeys(add):
                cols.append(jj); inrmp.add(jj)
            continue
        if timeout:
            status = "budget"
            eps = certified_pricing(model, sg, pi, theta, b, opts["eps_target"], opts["tol_add"], opts["final_price_time"], find=False,
                                    mil_tl=opts["milp_tl"])
            break
        # add found nothing although pricing claimed columns (should not happen): certify
        eps = certified_pricing(model, sg, pi, theta, b, opts["eps_target"], opts["tol_add"], opts["final_price_time"], find=False,
                                mil_tl=opts["milp_tl"]); break
    if eps.get("eps") is None:     # find-mode stopped on a column at the very end: recompute in certify mode
        eps = certified_pricing(model, sg, pi, theta, b, opts["eps_target"], opts["tol_add"], opts["final_price_time"], find=False,
                                mil_tl=opts["milp_tl"])
    ub, eps_cols, eps_used = certificate(pool, cols, sg, cls.K, cls.Dmin, pi, theta, b, eps["eps"])
    rec = dict(status=status, iters=it, cols=len(cols), pool=len(pool.mem), lp_value=float(mp["value"]), b=b,
               ub_cert=ub, eps_price=float(eps["eps"]), eps_cols=eps_cols, price_nodes=eps["nodes"], price_open=eps.get("open", 0),
               secs=time.time() - t0, charge=float(ub - (b + max(0.0, (pi.sum() + cls.K * theta) / cls.Dmin))))
    if model_i is not None:          # what relaxation (i) alone would have charged at the final duals
        e_i = certified_pricing(model_i, sg, pi, theta, b, opts["eps_target"], opts["tol_add"], 120.0, find=False, J0=6, max_nodes=200)
        rec["eps_relax_i"] = e_i["eps"]; rec["ub_relax_i"] = certificate(pool, cols, sg, cls.K, cls.Dmin, pi, theta, b, e_i["eps"])[0]
    if enum_cells is not None:       # soundness check against the exact max over enumerated admissible cells
        P2 = Pool(W)
        for m in enum_cells:
            P2.add(m)
        rex = P2.reduced(sg, pi, theta, b); rec["max_r_exact"] = float(rex.max()); rec["pricing_sound"] = bool(eps["eps"] >= rex.max() - 1e-9)
    rec["final_duals"] = dict(theta=float(theta), b=float(b), pi_sum=float(pi.sum()))
    rec["_y"] = mp["y"]; rec["_cols"] = cols
    return rec


# ----------------------------------------------------------------------------------------------------------- witnesses
def random_partition(W, K, rng):
    n = W.n; lab = -np.ones(n, int); seeds = rng.choice(n, K, replace=False); mass = np.zeros(K)
    for k, s in enumerate(seeds):
        lab[s] = k; mass[k] = W.mu[s]
    left = n - K
    while left:
        order = np.argsort(mass + 1e-9 * rng.random(K)); done = False
        for k in order:
            fr = [v for u in np.where(lab == k)[0] for v in W.adj[u] if lab[v] < 0]
            if fr:
                v = fr[rng.integers(len(fr))]; lab[v] = k; mass[k] += W.mu[v]; left -= 1; done = True; break
        if not done:
            return None
    return lab


def part_score(W, cls, sg, cnt, t, Sx, Sy):
    K = cls.K
    if (cnt == 0).any():
        return -1e9, None, None
    nC = (Sx * Sy * sg / t).sum(); dC = (Sx * Sx / t).sum(); beta = nC / dC
    pen = max(0.0, cls.Dmin - dC) * 20 + 20 * (np.maximum(0, cls.tlo - t).sum() + np.maximum(0, t - cls.thi).sum())
    return beta - pen, beta, (pen == 0)


def hill_climb(W, cls, sg, rng, restarts, pool, max_pass=60):
    """iterated local search over contiguous K-partitions maximizing sg*beta subject to balance and D >= Dmin.
    Every cell of every feasible local optimum is added to the pool (admissible columns)."""
    K = cls.K; best = (-INF, None); n = W.n; muy = sg * W.muy
    for rs in range(restarts):
        lab = random_partition(W, K, rng)
        if lab is None:
            continue
        for kick in range(5):
            cnt = np.bincount(lab, minlength=K); t = np.bincount(lab, W.mu, K); Sx = np.bincount(lab, W.mux, K); Sy = np.bincount(lab, W.muy, K)
            cur = part_score(W, cls, 1, cnt, t, Sx, sg * Sy)[0]
            for _ in range(max_pass * n):
                improved = False
                for u in rng.permutation(n):
                    A_ = lab[u]
                    if cnt[A_] == 1:
                        continue
                    targets = set(lab[v] for v in W.adj[u]) - {A_}
                    if not targets:
                        continue
                    restA = None
                    for B_ in targets:
                        t2 = t.copy(); Sx2 = Sx.copy(); Sy2 = Sy.copy(); c2 = cnt.copy()
                        t2[A_] -= W.mu[u]; t2[B_] += W.mu[u]; Sx2[A_] -= W.mux[u]; Sx2[B_] += W.mux[u]; Sy2[A_] -= W.muy[u]; Sy2[B_] += W.muy[u]
                        c2[A_] -= 1; c2[B_] += 1
                        sc = part_score(W, cls, 1, c2, t2, Sx2, sg * Sy2)[0]
                        if sc > cur + 1e-12:
                            if restA is None:
                                rest = np.where(lab == A_)[0]; restA = W.connected(rest[rest != u])
                            if not restA:
                                break
                            lab[u] = B_; t, Sx, Sy, cnt, cur = t2, Sx2, Sy2, c2, sc; improved = True; break
                if not improved:
                    break
            sc, beta, feas = part_score(W, cls, 1, cnt, t, Sx, sg * Sy)
            if feas:
                for k in range(K):
                    pool.add(np.where(lab == k)[0])
                if beta > best[0]:
                    best = (beta, lab.copy())
            # kick: random boundary moves
            for _ in range(max(2, n // 10)):
                u = int(rng.integers(n)); A_ = lab[u]; tg = [lab[v] for v in W.adj[u] if lab[v] != A_]
                if tg and cnt[A_] > 1:
                    rest = np.where(lab == A_)[0]
                    if W.connected(rest[rest != u]):
                        lab[u] = tg[0]; cnt = np.bincount(lab, minlength=K)
    return best


def master_ip_witness(W, cls, sg, pool, cols, lam0, time_limit=60):
    """restricted-master integer programme (Dinkelbach on exact cover by generated columns) -> admissible partition witnesses"""
    cols = np.array(cols); t, sx, sy, nC, dC = pool.arrays(sg); M = pool.M()[cols].T.tocsr(); m = len(cols)
    A = sparse.vstack([M, sparse.csr_matrix(np.ones((1, m))), sparse.csr_matrix(dC[cols][None, :])]).tocsr()
    lb = np.r_[np.ones(W.n), cls.K, cls.Dmin]; ub = np.r_[np.ones(W.n), cls.K, INF]
    lam = lam0; best = None
    for _ in range(8):
        res = milp(-(nC[cols] - lam * dC[cols]), integrality=np.ones(m), bounds=Bounds(0, 1), constraints=[LinearConstraint(A, lb, ub)],
                   options=dict(time_limit=time_limit, disp=False))
        if res.x is None:
            break
        z = res.x > 0.5; val = nC[cols][z].sum() / dC[cols][z].sum()
        if best is None or val > best[0] + 1e-12:
            lab = -np.ones(W.n, int)
            for k, j in enumerate(cols[z]):
                lab[pool.mem[j]] = k
            best = (val, lab)
        if val <= lam + 1e-10:
            break
        lam = val
    return best


def feasible_cover(W, cls, pool):
    """any admissible partition with D >= Dmin among the pool's columns (zero-objective set-partitioning ILP)"""
    t, sx, sy, nC, dC = pool.arrays(1); m = len(pool.mem); M = pool.M().T.tocsr()
    A = sparse.vstack([M, sparse.csr_matrix(np.ones((1, m))), sparse.csr_matrix(dC[None, :])]).tocsr()
    res = milp(np.zeros(m), integrality=np.ones(m), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(A, np.r_[np.ones(W.n), cls.K, cls.Dmin], np.r_[np.ones(W.n), cls.K, INF])], options=dict(time_limit=60))
    return None if res.x is None else np.where(res.x > 0.5)[0]


def verify_partition(W, cls, lab):
    K = cls.K
    if lab is None or (lab < 0).any() or len(set(lab.tolist())) != K:
        return False, None
    for k in range(K):
        if not cls.admissible(W, np.where(lab == k)[0]):
            return False, None
    beta, Dx = W.slope_orig(lab, K)
    return bool(Dx >= 0.25 * W.sx ** 2 * (1 - 1e-12)), float(beta)


# ----------------------------------------------------------------------------------------------------------- drivers
OPTS = dict(tol_add=1e-7, eps_target=2e-5, price_time=300.0, final_price_time=600.0, milp_tl=30.0, cg_time=1500.0, max_it=4000)


def task_big(args):
    nodes, K, beta, seed, opts = args[:5]; dirs = args[5] if len(args) > 5 else (1, -1)
    D = load_data(); W = Window(nodes, D); cls = ZClass(W, K, beta); rng = np.random.default_rng(seed + (0 if dirs == (1, -1) else 7 * dirs[0]))
    rec = dict(n=W.n, req_km=W.req_km, nodes_head=nodes[:3], K=K, beta=beta, sx=W.sx, sy=W.sy, conv=W.sy / W.sx)
    model = PricingModel(W, cls, conn=True, integral=True); model_i = PricingModel(W, cls, conn=False, integral=False)
    out = {}
    for sg, nm in [(s_, "upper" if s_ == 1 else "lower") for s_ in dirs]:
        t0 = time.time(); pool = Pool(W)
        hc = hill_climb(W, cls, sg, rng, restarts=opts.get("restarts", max(40, 4000 // W.n)), pool=pool)
        t_hc = time.time() - t0
        if hc[1] is None:
            out[nm] = dict(status="no_witness", hc_secs=t_hc); continue
        ok, beta_hc = verify_partition(W, cls, hc[1])
        cg = run_cg(W, cls, sg, pool, list(range(len(pool.mem))), opts, model, model_i)
        ip = master_ip_witness(W, cls, sg, pool, cg["_cols"], hc[0], time_limit=opts.get("ip_time", 60.0))
        ok2, beta_ip = verify_partition(W, cls, ip[1]) if ip is not None else (False, None)
        conv = W.sy / W.sx
        cg.pop("_y"); cg.pop("_cols")
        cg.update(hc_secs=t_hc, hc_beta=beta_hc if ok else None, ip_beta=beta_ip if ok2 else None,
                  bound_orig=sg * cg["ub_cert"] * conv, lp_orig=sg * cg["lp_value"] * conv,
                  relax_i_orig=(sg * cg["ub_relax_i"] * conv) if "ub_relax_i" in cg else None)
        out[nm] = cg
    rec.update(out)
    return rec


def bfs_window(D, seed, n):
    seen = [seed]; S = {seed}; q = collections.deque([seed])
    while q and len(seen) < n:
        v = q.popleft()
        for w_ in sorted(D["nbrs"][v]):
            if w_ not in S:
                S.add(w_); seen.append(w_); q.append(w_)
                if len(seen) >= n:
                    break
    return seen if len(seen) >= n else None


def a4_windows(D):
    """replicates the RNG stream of a4_exact_windows.py (seed 5, 20 windows of 12)"""
    rng = np.random.default_rng(5); nbrs = D["nbrs"]; out = []
    for wi in range(20):
        seed = int(rng.integers(D["n"])); sel = [seed]; frontier = set(nbrs[seed])
        while len(sel) < 12 and frontier:
            k = int(rng.choice(sorted(frontier))); sel.append(k); frontier |= nbrs[k]; frontier -= set(sel)
        if len(sel) < 12:
            continue
        _ = int(rng.choice([v for v in sel if v != seed and v not in nbrs[seed]] or [v for v in sel if v != seed]))
        out.append((wi, seed, sel))
    return out


def enum_connected(W):
    out = []
    for mask in range(1, 1 << W.n):
        m = [i for i in range(W.n) if mask >> i & 1]
        if W.connected(m):
            out.append(m)
    return out


def task_a4(args):
    wi, seed, nodes, a4rec, opts = args
    D = load_data(); W = Window(nodes, D); cells = enum_connected(W); conv = W.sy / W.sx; res = dict(window=wi, seed=seed, n_cells=len(cells), classes={})
    for K in (2, 3, 4):
        for beta in (INF, 0.5, 0.25):
            nm = f"K{K}_beta{beta}"; a4c = a4rec["classes"].get(nm, {})
            if a4c.get("count", 0) == 0 or not np.isfinite(a4c["slope"][0]):
                res["classes"][nm] = dict(count=a4c.get("count", 0)); continue
            cls = ZClass(W, K, beta, size_cap=False)                         # A4's column family: every contiguous cell (mass-filtered)
            adm = [m for m in cells if cls.admissible(W, m)]
            row = dict(count=a4c["count"], exact=a4c["slope"], a4_lp=a4c["setpart_K"], n_cols=len(adm))
            for sg, nm2 in ((1, "upper"), (-1, "lower")):
                Pf = Pool(W)
                for m in adm:
                    Pf.add(m)
                mp = solve_master(Pf, list(range(len(adm))), sg, K, cls.Dmin)          # full-column LP (= A4 bisection)
                row[f"full_lp_{nm2}"] = sg * mp["value"] * conv if mp else None
                # verified CG from a hill-climbing start
                rng = np.random.default_rng(1000 + wi); pool = Pool(W)
                hc = hill_climb(W, cls, sg, rng, restarts=6, pool=pool)
                if hc[1] is None:                                               # feasibility seed: any exact cover with D >= Dmin
                    fz = feasible_cover(W, cls, Pf)
                    if fz is None:
                        row[f"cg_{nm2}"] = dict(status="no_witness"); continue
                    for j in fz:
                        pool.add(Pf.mem[j])
                model = PricingModel(W, cls, conn=True, integral=True); model_i = PricingModel(W, cls, conn=False, integral=False)
                cg = run_cg(W, cls, sg, pool, list(range(len(pool.mem))), opts, model, model_i, enum_cells=adm)
                cg.pop("_y"); cg.pop("_cols")
                cg["bound_orig"] = sg * cg["ub_cert"] * conv; cg["relax_i_orig"] = sg * cg["ub_relax_i"] * conv
                row[f"cg_{nm2}"] = cg
            lo_c, hi_c = row["cg_lower"].get("bound_orig"), row["cg_upper"].get("bound_orig")
            ex = row["exact"]
            row["contains_exact"] = bool(lo_c is not None and hi_c is not None and lo_c <= ex[0] + 1e-9 and ex[1] <= hi_c + 1e-9)
            row["gap_to_full_lp"] = [None if lo_c is None else row["full_lp_lower"] - lo_c, None if hi_c is None else hi_c - row["full_lp_upper"]]
            row["sign_exact"] = "neg" if ex[1] < 0 else "pos" if ex[0] > 0 else "both"
            row["sign_cert"] = "neg" if (hi_c is not None and hi_c < 0) else "pos" if (lo_c is not None and lo_c > 0) else "both"
            res["classes"][nm] = row
    return res


def theory_check(D, reps=40, n=10, seed=3):
    """Skeleton check on real 10-OA windows with random prices: (1) max of F over the <=2-fractional skeleton of N (all edges
    of the box-slab polytope, 1-D maximisation on a fine grid + golden refinement; a LOWER bound attained by feasible points);
    (2) the LP branch-and-bound relaxation-(i) bound run until every branch is <= skeleton max + 1e-6 (an UPPER bound);
    agreement proves max_N F is attained on the skeleton to 1e-6; (3) brute-force max over all 2^n subsets (integrality gap of (i));
    (4) KKT half-space structure at the skeleton maximiser."""
    rng = np.random.default_rng(seed); out = []
    for rep in range(reps):
        s0 = int(rng.integers(D["n"])); nodes = bfs_window(D, s0, n)
        if nodes is None:
            continue
        W = Window(nodes, D); K = int(rng.choice([2, 3, 4])); beta = float(rng.choice([INF, 0.5])); cls = ZClass(W, K, beta, size_cap=False)
        b = float(rng.normal(0, 1)); pi = rng.normal(0, 0.05, n) * W.mu * 3; theta = 0.0
        S = np.c_[W.x, W.y]; p = pi / W.mu; M = np.array([[-b, 0.5], [0.5, 0.0]])
        def F(nu):                                                   # nu: (..., n)
            t = nu.sum(-1); Sx = nu @ W.x; Sy = nu @ W.y
            with np.errstate(divide="ignore", invalid="ignore"):
                v = (Sx * Sy - b * Sx * Sx) / t - nu @ p
            return np.where(t > 0, v, -INF)
        best = (-INF, None); grid = np.linspace(0, 1, 401)
        masks = ((np.arange(2 ** (n - 1))[:, None] >> np.arange(n - 1)) & 1).astype(float)
        for u in range(n):                                           # type (a): one free coordinate
            others = [v for v in range(n) if v != u]; base = np.zeros((len(masks), n)); base[:, others] = masks * W.mu[others]
            nu = base[:, None, :] + np.zeros((1, len(grid), n)); nu[:, :, u] = grid[None, :] * W.mu[u]
            t = nu.sum(-1); val = np.where((t >= cls.tlo - 1e-12) & (t <= cls.thi + 1e-12), F(nu), -INF); k = np.unravel_index(np.argmax(val), val.shape)
            if val[k] > best[0]:
                best = (float(val[k]), nu[k].copy())
        masks2 = ((np.arange(2 ** (n - 2))[:, None] >> np.arange(n - 2)) & 1).astype(float)
        for u in range(n):                                           # type (b): two coordinates traded at an active mass bound
            for v in range(u + 1, n):
                others = [w for w in range(n) if w not in (u, v)]; base = np.zeros((len(masks2), n)); base[:, others] = masks2 * W.mu[others]
                for tb in (cls.tlo, cls.thi):
                    rem = tb - base.sum(1)                            # nu_u + nu_v = rem, 0<=nu_u<=mu_u, 0<=nu_v<=mu_v
                    lo_ = np.maximum(0, rem - W.mu[v]); hi_ = np.minimum(W.mu[u], rem); okm = hi_ >= lo_
                    if not okm.any():
                        continue
                    nu = base[okm][:, None, :] + np.zeros((1, len(grid), n)); su = lo_[okm][:, None] + grid[None, :] * (hi_ - lo_)[okm][:, None]
                    nu[:, :, u] = su; nu[:, :, v] = rem[okm][:, None] - su
                    val = F(nu); k = np.unravel_index(np.argmax(val), val.shape)
                    if val[k] > best[0]:
                        best = (float(val[k]), nu[k].copy())
        # golden refinement along the best edge is unnecessary for a lower bound; the grid value is attained.
        cls.smin, cls.smax = 0, n          # the theorem concerns N exactly (box + mass slab): no size rows in this check
        model_i = PricingModel(W, cls, conn=False, integral=False)
        cp = certified_pricing(model_i, 1, pi, theta, b, best[0] + 1e-6, -INF, 600.0, find=False, J0=8, max_nodes=4000)
        # brute-force subsets
        allm = ((np.arange(1, 2 ** n)[:, None] >> np.arange(n)) & 1).astype(float); nu = allm * W.mu; t = nu.sum(1)
        val = np.where((t >= cls.tlo - 1e-12) & (t <= cls.thi + 1e-12), F(nu), -INF); sub_max = float(val.max())
        nu_s = best[1]; ts = nu_s.sum(); m = np.array([nu_s @ W.x, nu_s @ W.y]) / ts; g = 2 * (S @ (M @ m)) - m @ M @ m - p
        frac = (nu_s > 1e-12) & (nu_s < W.mu - 1e-12); sel = nu_s >= W.mu - 1e-12; uns = nu_s <= 1e-12
        thr_ok = bool((not sel.any() or not uns.any()) or g[sel].min() >= g[uns].max() - 1e-6)
        out.append(dict(n=n, K=K, beta=beta, skeleton_max=best[0], bb_upper=cp["eps"], bb_nodes=cp["nodes"], bb_open=cp.get("open", 0),
                        subset_max=sub_max, n_fractional=int(frac.sum()), threshold_structure=thr_ok,
                        agree=bool(cp["eps"] <= best[0] + 2e-6)))
        print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    p = argparse.ArgumentParser(); p.add_argument("--mode", default="a4"); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=30); p.add_argument("--windows", type=int, default=6); p.add_argument("--seed", type=int, default=11)
    p.add_argument("--procs", type=int, default=20); p.add_argument("--only", type=int, default=-1)
    p.add_argument("--classes", default="3:inf,3:0.5,5:inf,5:0.5,8:inf,8:0.5"); p.add_argument("--cg_time", type=float, default=OPTS["cg_time"])
    p.add_argument("--verbose", action="store_true"); p.add_argument("--restarts", type=int, default=0)
    p.add_argument("--split_dirs", action="store_true"); p.add_argument("--final_price_time", type=float, default=OPTS["final_price_time"])
    p.add_argument("--ip_time", type=float, default=60.0); p.add_argument("--max_it", type=int, default=OPTS["max_it"])
    p.add_argument("--grow_seeds", type=int, default=0); p.add_argument("--grow_add", type=int, default=40)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); D = load_data(); opts = dict(OPTS); opts["cg_time"] = a.cg_time
    opts["verbose"] = a.verbose; opts["final_price_time"] = a.final_price_time; opts["ip_time"] = a.ip_time; opts["max_it"] = a.max_it
    opts["grow_seeds"] = a.grow_seeds; opts["grow_add"] = a.grow_add
    if a.restarts:
        opts["restarts"] = a.restarts
    import multiprocessing as mpc
    t_all = time.time()
    if a.mode == "theory":
        rows = theory_check(D); json.dump(rows, open(os.path.join(a.out, "pricing_theory_check.json"), "w"), indent=1, default=float)
    elif a.mode == "a4":
        a4 = json.load(open(paths.results("exp1", "A4", "a4_windows.json")))["windows"]
        wins = a4_windows(D); assert len(wins) == len(a4)
        for (wi, seed, _), r in zip(wins, a4):
            assert r["seed"] == seed and r["window"] == wi, (wi, seed, r["seed"])
        tasks = [(wi, seed, nodes, r, opts) for (wi, seed, nodes), r in zip(wins, a4)]
        if a.only >= 0:
            tasks = tasks[a.only:a.only + 1]
        with mpc.Pool(min(a.procs, len(tasks))) as P:
            rows = []
            for r in P.imap_unordered(task_a4, tasks):
                rows.append(r); print(f"window {r['window']} done ({time.time()-t_all:.0f}s)", flush=True)
                json.dump(sorted(rows, key=lambda z: z["window"]), open(os.path.join(a.out, "pricing_a4.json"), "w"), indent=1, default=float)
    else:
        rng = np.random.default_rng(a.seed); wins = []
        while len(wins) < a.windows:
            s = int(rng.integers(D["n"])); w = bfs_window(D, s, a.n)
            if w is not None:
                wins.append(w)
        cl = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in a.classes.split(",")]
        if a.split_dirs:      # one task per direction (better load balance); records merged per (window, class)
            tasks = [(w, K, beta, 7 + 100 * wi + K, opts, (s_,)) for wi, w in enumerate(wins) for K, beta in cl for s_ in (1, -1)]
        else:
            tasks = [(w, K, beta, 7 + 100 * wi + K, opts) for wi, w in enumerate(wins) for K, beta in cl]
        if a.only >= 0:
            tasks = tasks[a.only:a.only + 1]
        with mpc.Pool(min(a.procs, len(tasks))) as P:
            rows = []; merged = {}
            for r in P.imap_unordered(task_big, tasks):
                r["window"] = [i for i, w in enumerate(wins) if w[:3] == r["nodes_head"]][0]
                key = (r["window"], r["K"], r["beta"])
                if key in merged:
                    merged[key].update({k: v for k, v in r.items() if k in ("upper", "lower")}); r = merged[key]
                else:
                    merged[key] = r; rows.append(r)
                u_, l_ = r.get("upper", {}), r.get("lower", {})
                print(f"win {r['window']} n={r['n']} K={r['K']} beta={r['beta']}: cert=[{l_.get('bound_orig')},{u_.get('bound_orig')}] "
                      f"hc=[{l_.get('hc_beta')},{u_.get('hc_beta')}] ip=[{l_.get('ip_beta')},{u_.get('ip_beta')}] "
                      f"eps=({l_.get('eps_price')},{u_.get('eps_price')}) secs=({l_.get('secs')},{u_.get('secs')}) ({time.time()-t_all:.0f}s)", flush=True)
                json.dump(sorted(rows, key=lambda z: (z["window"], z["K"], z["beta"])), open(os.path.join(a.out, f"pricing_big_n{a.n}.json"), "w"), indent=1, default=float)
    print(f"total {time.time()-t_all:.0f}s")
