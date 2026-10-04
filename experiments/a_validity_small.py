"""a_validity_small.py -- small, exact validity experiments (result keys A1b, A5, A6, A7, A11, A12); results exp1/A_validity.

A1b  deliberate negative cycles (Phi empty), component mass mismatch (ground route), zero objectives, tight value
     faces (U = V* with Gamma <= 0 binding), diagonal arcs crossing an uncertified pixel (must be skipped)
A5   allocation adversaries: 1-D adjacent pixels with non-uniform within-pixel densities; exact increment vs the
     coupling/path bound (Theorem S2) vs the invalid uniform tent shortcut
A6   weighted Lemma S6.1: population concentrated near a face; affine witnesses attain the weighted bound
A7   canonical-path audit of K_a (cf. Theorem S19) against the spectral constant 1/lambda_2 of the pencil on the mean-zero subspace
A11  map-level decomposition (Theorem S10): S_Phi >= J_Phi >= J_in(m) >= Q(psi) on small instances; the three gaps
A12  unequal-mass boundary change (Theorem S18 (i)): interior density of mu_a' - mu_a; tube bound vs transport value
Usage: python a_validity_small.py --out <dir>
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import Poly, solve_max, closure, mcshane, closed_cost, ot_cost, value_face, gradient_only, grid_poly, inner_q1, lemma12
import synth
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigh

p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=31); p.add_argument("--only_a6", action="store_true"); a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); synth.set_rng(rng); R = {}


def a1b():
    out = {}
    # negative cycle: two pixels, increment box [0.5, 1] one way and the reverse constraint forces phi_q - phi_p <= -0.2 via value boxes
    poly = Poly([0, 0], [1, 0.3], [0], [1], [0.5], [1.0])          # phi_q - phi_p >= 0.5 but hi_q - lo_p = 0.3 -> infeasible (augmented negative cycle)
    r = solve_max(poly, np.array([1.0, -1.0])); out["neg_cycle_lp_status"] = r["status"]
    try:
        D = closure(poly); hs, ls = mcshane(poly, D); out["neg_cycle_lattice_feasible"] = bool(np.all(ls <= hs + 1e-12))
    except ValueError as e:
        out["neg_cycle_closure"] = str(e)
    # pure increment negative cycle: 3-cycle with sum of upper bounds negative in a directed loop
    poly = Poly([-9, -9, -9], [9, 9, 9], [0, 1, 2], [1, 2, 0], [-1, -1, -1], [-0.5, -0.5, -0.5])   # phi1-phi0<=-.5, phi2-phi1<=-.5, phi0-phi2<=-.5 : sum<0
    try:
        closure(poly, "fw"); out["inc_neg_cycle_detected"] = False
    except ValueError:
        out["inc_neg_cycle_detected"] = True
    # component mass mismatch: two disconnected pixels, d = (1, -0.5): LP finite via ground, gradient-only infinite
    poly = Poly([0, 0], [1, 2], [], [], [], []); d = np.array([1.0, -0.5])
    r = solve_max(poly, d); T, zm = gradient_only(poly, d); out["mismatch"] = dict(U=r["value"], expected=1.0 * 1 - 0.5 * 0, T_finite=bool(np.isfinite(T)), zero_mass=zm)
    # zero objective
    r = solve_max(poly, np.zeros(2)); out["zero_objective"] = dict(U=r["value"], verified=r["verified"])
    # tight value face: nested pair where every source-sink pair satisfies hi*_p - lo*_q <= D(q,p)  (value-only exact)
    n = 5; lo = np.zeros((n, n)); hi = np.full((n, n), 0.01); G1 = np.zeros((n, n, 2)); G1[..., 0] = -5; G1[..., 1] = 5; G2 = G1.copy()
    poly, _ = grid_poly(lo, hi, G1, G2, 0.1, None); d2 = np.zeros((n, n)); d2[2, 2] = 1; d2 -= 1 / (n * n); d = d2.ravel()
    r = solve_max(poly, d); D = closure(poly, "fw"); hs, ls = mcshane(poly, D); Gam, Vs, V = value_face(poly, D, hs, ls, d)
    out["tight_face"] = dict(U=r["value"], Vstar=Vs, V=V, Gamma=Gam, value_only_exact=bool(abs(r["value"] - Vs) < 1e-12 and Gam <= 1e-12))
    # diagonal arcs crossing an uncertified pixel are skipped: 2x2 block with one corner masked
    lo = np.zeros((2, 2)); hi = np.ones((2, 2)); G1 = np.zeros((2, 2, 2)); G1[..., 1] = 1; G2 = G1.copy(); mk = np.ones((2, 2), bool); mk[1, 0] = False
    poly8, _ = grid_poly(lo, hi, G1, G2, 1.0, mk, neigh=8); poly4, _ = grid_poly(lo, hi, G1, G2, 1.0, mk, neigh=4)
    out["diagonal_skipped_when_block_uncertified"] = bool(poly8.E == poly4.E)
    mk[:] = True; poly8, _ = grid_poly(lo, hi, G1, G2, 1.0, mk, neigh=8); out["diagonal_arcs_full_block"] = int(poly8.E - 4)
    R["A1b"] = out; print("A1b", json.dumps(out, default=float))


def a5():
    """1-D: pixels p=[0,h], q=[h,2h]; densities rho_p, rho_q (normalized); f' in [g-_p, g+_p] on p, [g-_q, g+_q] on q.
    exact increment for a test field; coupling bound with the monotone (quantile) coupling; uniform tent shortcut."""
    h = 1.0; out = []
    x = np.linspace(0, 1, 2001)[:-1] + 0.5 / 2000
    for case in ("uniform", "ramp_opposed", "corner_masses"):
        if case == "uniform":
            rp = np.ones_like(x); rq = np.ones_like(x)
        elif case == "ramp_opposed":
            rp = 2 * (1 - x); rq = 2 * x                     # p mass near its left end, q mass near its right end
        else:
            rp = np.exp(-x / 0.05); rq = np.exp(-(1 - x) / 0.05)
        rp /= rp.mean(); rq /= rq.mean()
        gp_box = (0.5, 1.0); gq_box = (0.5, 1.0)             # gradient boxes (positive: f increasing)
        # test field: f' = g+ everywhere (extreme member)
        fp = lambda t: 1.0 * t; fq = lambda t: 1.0 + 1.0 * (t - 1)       # f(t) = t on [0,2]
        exact = np.mean(fq(1 + x) * rq) - np.mean(fp(x) * rp)
        # coupling bound (quantile coupling): x in p -> T(x) in q ; increment = int_x^{T(x)} f' ; upper bound with g+ per pixel
        Fp = np.cumsum(rp) / len(x); Fq = np.cumsum(rq) / len(x); T = 1 + np.interp(Fp, Fq, x)
        ub = np.mean(((1 - x) * gp_box[1] + (T - 1) * gq_box[1]) * rp)        # path from x (in p) to T(x) (in q)
        lb = np.mean(((1 - x) * gp_box[0] + (T - 1) * gq_box[0]) * rp)
        tent_ub = h / 2 * (gp_box[1] + gq_box[1]); tent_lb = h / 2 * (gp_box[0] + gq_box[0])
        out.append(dict(case=case, exact=float(exact), coupling_lb=float(lb), coupling_ub=float(ub), tent_lb=tent_lb, tent_ub=tent_ub,
                        coupling_sound=bool(lb - 1e-9 <= exact <= ub + 1e-9), tent_sound=bool(tent_lb - 1e-9 <= exact <= tent_ub + 1e-9)))
    R["A5"] = out; print("A5", json.dumps(out))


def a6():
    """2-D pixel [0,h]^2, density concentrated near the left face; f = v- + A1 x1 attains the weighted lower correction."""
    h = 1.0; n = 400; t = (np.arange(n) + 0.5) / n; X1, X2 = np.meshgrid(t, t, indexing="ij")
    out = []
    for tau in (1e9, 0.3, 0.05):
        rho = np.exp(-X1 / tau); rho /= rho.mean()
        g1 = (0.4, 1.2); g2 = (-0.3, 0.5); vminus, vplus = 0.0, 3.0
        A1, B1 = max(g1[0], 0), max(-g1[1], 0); A2, B2 = max(g2[0], 0), max(-g2[1], 0)
        EX1 = np.mean(X1 * rho); E1mX1 = np.mean((1 - X1) * rho); EX2 = np.mean(X2 * rho); E1mX2 = np.mean((1 - X2) * rho)
        lower = vminus + A1 * EX1 + B1 * E1mX1 + A2 * EX2 + B2 * E1mX2
        upper = vplus - (A1 * E1mX1 + B1 * EX1 + A2 * E1mX2 + B2 * EX2)
        uniform_corr = h / 2 * ((max(g1[0], 0) + max(-g1[1], 0)) + (max(g2[0], 0) + max(-g2[1], 0)))
        # affine witness attaining the lower bound: f = v- + A1 x1 (+ A2 x2 if A2>0) ; here g2 straddles 0 so A2 = B2 = 0
        f = vminus + A1 * X1; mean_w = np.mean(f * rho)
        # random members of the class: affine f = c + a1 x1 + a2 x2 with (a1, a2) in the box, kept only if f stays inside [v-, v+] (no clipping)
        viol = 0; kept = 0
        while kept < 200:
            a1 = rng.uniform(g1[0], g1[1]); a2 = rng.uniform(g2[0], g2[1]); c = rng.uniform(vminus - 1, vplus + 1)
            fr = c + a1 * X1 + a2 * X2
            if fr.min() < vminus or fr.max() > vplus:
                continue
            kept += 1; m = np.mean(fr * rho); viol += int(not (lower - 1e-9 <= m <= upper + 1e-9))
        out.append(dict(tau=tau, lower=float(lower), upper=float(upper), uniform_correction=uniform_corr, witness_mean=float(mean_w), attains=bool(abs(mean_w - lower) < 1e-9), random_violations=viol))
    R["A6"] = out; print("A6", json.dumps(out))


def kappa_a(mask, alpha):
    """canonical paths = BFS shortest paths (fixed, lexicographic) on the pixel graph; kappa_a = 1/2 max_e sum_{paths through e} alpha_x alpha_y |path|"""
    ii, jj = np.nonzero(mask); n = len(ii); pix = {(i, j): k for k, (i, j) in enumerate(zip(ii, jj))}
    adj = [[] for _ in range(n)]
    for (i, j), k in pix.items():
        for (di, dj) in ((1, 0), (0, 1)):
            if (i + di, j + dj) in pix:
                adj[k].append(pix[(i + di, j + dj)]); adj[pix[(i + di, j + dj)]].append(k)
    load = {}
    import collections
    for s in range(n):
        prev = [-1] * n; prev[s] = s; dq = collections.deque([s])
        while dq:
            u = dq.popleft()
            for v in sorted(adj[u]):
                if prev[v] < 0:
                    prev[v] = u; dq.append(v)
        for t_ in range(n):
            path = []; v = t_
            while v != s:
                path.append((min(v, prev[v]), max(v, prev[v]))); v = prev[v]
            for e in path:
                load[e] = load.get(e, 0) + alpha[s] * alpha[t_] * len(path)
    kap = 0.5 * max(load.values()) if load else 0.0
    edges = sorted(set(e for e in load)); L = np.zeros((n, n))
    for (u, v) in edges:
        L[u, u] += 1; L[v, v] += 1; L[u, v] -= 1; L[v, u] -= 1
    return kap, L


def a7():
    out = []
    shapes = {"one_pixel": np.ones((1, 1), bool), "two_pixels": np.ones((2, 1), bool), "chain20": np.ones((20, 1), bool), "square6": np.ones((6, 6), bool)}
    bott = np.zeros((9, 4), bool); bott[:4, :] = True; bott[5:, :] = True; bott[4, 0] = True; shapes["bottleneck"] = bott
    for name, mk in shapes.items():
        n = mk.sum(); alpha = np.ones(n) / n
        kap, L = kappa_a(mk, alpha)
        K_a = alpha.max() / np.pi ** 2 + 2 * kap
        if n > 1:
            # pencil (L, diag alpha) on the alpha-mean-zero subspace: lambda_2 = min v'Lv / v'diag(alpha)v over sum alpha v = 0
            W = np.diag(alpha); Z = np.linalg.svd(alpha[None, :], full_matrices=True)[2][1:].T   # basis of {v: alpha.v=0}
            lam2 = eigh(Z.T @ L @ Z, Z.T @ W @ Z, eigvals_only=True)[0]
            spectral_graph = 1.0 / lam2        # Var_alpha(v) <= (1/lam2) sum_e (dv)^2
            out.append(dict(shape=name, n=int(n), kappa=float(kap), K_a=float(K_a), two_kappa=float(2 * kap), one_over_lambda2=float(spectral_graph),
                            ratio_2kappa_over_spectral=float(2 * kap / spectral_graph), sound=bool(2 * kap >= spectral_graph - 1e-12)))
        else:
            out.append(dict(shape=name, n=1, kappa=0.0, K_a=float(K_a), note="pixel Payne-Weinberger only: h^2/pi^2"))
    R["A7"] = out; print("A7", json.dumps(out))


def a11():
    out = []
    for t in range(6):
        n1, n2 = 5, 6; h = 0.25; mk = np.ones((n1, n2), bool); inst = synth.build(n1, n2, h, float(rng.choice([0.0, 0.3])), mk)
        lo, hi, G1, G2, psi = inst["lo"], inst["hi"], inst["G1"], inst["G2"], inst["psi"]
        w = rng.uniform(0.2, 1, (n1, n2))
        cells = np.zeros((n1, n2), int); cells[:, 2:4] = 1; cells[:, 4:] = 2        # three fine cells inside one coarse cell
        ds = []; wts = []
        for c in range(3):
            wa = w * (cells == c); ds.append((wa / wa.sum() - w / w.sum())[mk]); wts.append(wa.sum() / w.sum())
        Dm = np.stack(ds); wts = np.array(wts)
        lo2, hi2, _ = lemma12(lo, hi, G1, G2, h); poly, _ = grid_poly(lo2, hi2, G1, G2, h, mk)
        Q = lambda v: float(np.sum(wts * (Dm @ v) ** 2))
        S = sum(wts[j] * max(solve_max(poly, Dm[j])["value"] ** 2, solve_max(poly, -Dm[j])["value"] ** 2) for j in range(3))
        # J_Phi by exposing vertices of the 3-D projection with many directions (max of a convex function over a polytope is at a vertex)
        best = -1; dirs = rng.normal(size=(400, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        for u in list(dirs) + [np.array(s, float) for s in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]]:
            r = solve_max(poly, Dm.T @ (u * wts)); best = max(best, Q(r["phi"]))
        J_Phi = best
        # J_in(m): same on the Q1 inner family (nodal LPs); m = 2
        m = 2; bestin = -1
        for u in list(dirs[:150]):
            dd = np.zeros((n1, n2)); dd[mk] = Dm.T @ (u * wts)
            v, x = inner_q1(lo, hi, G1, G2, h, dd, m, mk)
            # pixel means of the Q1 witness: average of its 4 nodal values per sub-element, averaged over the m*m sub-elements
            N2 = n2 * m + 1; means = np.zeros((n1, n2))
            for i in range(n1):
                for j in range(n2):
                    acc = 0
                    for aa in range(i * m, i * m + m):
                        for bb in range(j * m, j * m + m):
                            acc += (x[aa * N2 + bb] + x[(aa + 1) * N2 + bb] + x[aa * N2 + bb + 1] + x[(aa + 1) * N2 + bb + 1]) / 4
                    means[i, j] = acc / (m * m)
            bestin = max(bestin, Q(means[mk]))
        Jin = bestin; Qpsi = Q(psi[mk])
        out.append(dict(t=t, S_Phi=S, J_Phi=J_Phi, J_in2=Jin, Q_psi=Qpsi, gap_joint=S - J_Phi, gap_realization_upper=J_Phi - Jin, gap_class=Jin - Qpsi,
                        ordered=bool(S + 1e-9 >= J_Phi >= Jin - 1e-9 and Jin + 1e-9 >= Qpsi), sum_check=float(abs((S - J_Phi) + (J_Phi - Jin) + (Jin - Qpsi) - (S - Qpsi)))))
        print(f"A11 t={t} S={S:.5f} J_Phi={J_Phi:.5f} J_in2={Jin:.5f} Q(psi)={Qpsi:.5f} joint={S-J_Phi:.2e} realiz<= {J_Phi-Jin:.2e} class={Jin-Qpsi:.2e}", flush=True)
    R["A11"] = out


def a12():
    n1, n2 = 6, 6; h = 0.25; mk = np.ones((n1, n2), bool); inst = synth.build(n1, n2, h, 0.0, mk)
    w = rng.uniform(0.2, 1, (n1, n2))
    A = np.zeros((n1, n2), bool); A[1:5, 1:4] = True; A2 = A.copy(); A2[1:5, 4] = True; A2[1, 1] = False        # a' = a + column, minus a corner pixel
    mua, mua2 = w[A].sum(), w[A2].sum(); d = (w * A2 / mua2 - w * A / mua)
    inter = A & A2; interior_density = d[inter]
    tube = (A ^ A2)
    f = inst["psi"]; mean_a = (w * A * f).sum() / mua; mean_a2 = (w * A2 * f).sum() / mua2
    exact = mean_a2 - mean_a
    identity = ((w * (A2 & ~A) * (f - mean_a)).sum() - (w * (A & ~A2) * (f - mean_a)).sum()) / mua2
    tube_bound = w[tube].sum() / mua2 * np.max(np.abs(f[tube] - mean_a))
    lo2, hi2, _ = lemma12(inst["lo"], inst["hi"], inst["G1"], inst["G2"], h); poly, _ = grid_poly(lo2, hi2, inst["G1"], inst["G2"], h, mk)
    U = solve_max(poly, d[mk])["value"]; L = -solve_max(poly, -d[mk])["value"]
    out = dict(mu_a=float(mua), mu_a2=float(mua2), interior_density_max_abs=float(np.abs(interior_density).max()), interior_mass=float(np.abs(interior_density).sum()),
               exact=float(exact), identity=float(identity), identity_ok=bool(abs(exact - identity) < 1e-12), tube_bound=float(tube_bound), transport=[float(L), float(U)],
               transport_contains_exact=bool(L - 1e-9 <= exact <= U + 1e-9), tube_bound_vs_transport_width=float(2 * tube_bound / (U - L)))
    R["A12"] = out; print("A12", json.dumps(out))


if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) > 3 and sys.argv[3] == "--only_a6":
        a6(); json.dump(R, open(os.path.join(a.out, "validity_small_a6.json"), "w"), indent=1, default=float); sys.exit(0)
    a1b(); a5(); a6(); a7(); a11(); a12()
    json.dump(R, open(os.path.join(a.out, "validity_small.json"), "w"), indent=1, default=float); print(f"done {time.time()-t0:.0f}s")
