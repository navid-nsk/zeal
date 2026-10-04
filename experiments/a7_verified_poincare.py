"""a7_verified_poincare.py -- the spectral Poincaré constant of Theorem S19 with a VERIFIED lower bound on lambda_2 (verified
inertia), on the toy shapes of the canonical-path audit and on real population-weighted pixel cells (Georgia counties);
results exp2/A7_verified.

Theorem S19 (ii): for a side-connected pixel cell a with alpha_p = mu(p)/mu(a) > 0 and the unweighted combinatorial Laplacian L_a,
Var_alpha(v) <= (1/lambda_2) sum_{edges} (v_q - v_p)^2 with lambda_2 the second generalized eigenvalue of (L_a, diag alpha).
Certificate (exact algebra): with sum alpha = 1,  lambda_2 >= t  <=>  M_t := L - t diag(alpha) + 2t alpha alpha^T  is positive semidefinite, and lambda_2 > t <=> M_t is positive definite
(for v = c1 + u with alpha.u = 0: v'M_t v = u'Lu - t u'diag(alpha)u + t c^2).  Verified PSD test (Rump 2006 / Higham Thm 10.3): if the floating-point
Cholesky of A = M_t - rho I runs to completion with factor R, then A + dA = R'R with ||dA||_2 <= gamma_{n+1} ||R||_F^2, hence
M_t >= (rho - ||dA||_2 - e_form) I where e_form bounds the rounding error in forming M_t; the test passes iff rho >= ||dA||_2 + e_form.
Then K_spec := 1/t is a verified Poincaré constant (an upper bound on the sharp 1/lambda_2 by at most the declared margin).
Compared with the canonical-path constant 2*kappa_a (canonical-path fallback, a_validity_small.py) and with the unverified 1/lambda_2.
Usage: python a7_verified_poincare.py --out <dir> [--max_pix 3000] [--counties 20]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, collections, io, json, os, sys, time, numpy as np
from scipy.linalg import eigh, cholesky, LinAlgError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--max_pix", type=int, default=3000); p.add_argument("--counties", type=int, default=20); p.add_argument("--margin", type=float, default=1e-6)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); EPS = np.finfo(float).eps


def cell_graph(mask):
    ii, jj = np.nonzero(mask); n = len(ii); pix = {(i, j): k for k, (i, j) in enumerate(zip(ii, jj))}; edges = []
    for (i, j), k in pix.items():
        for (di, dj) in ((1, 0), (0, 1)):
            if (i + di, j + dj) in pix:
                edges.append((k, pix[(i + di, j + dj)]))
    L = np.zeros((n, n))
    for (u, v) in edges:
        L[u, u] += 1; L[v, v] += 1; L[u, v] -= 1; L[v, u] -= 1
    return n, edges, L


def connected(n, edges):
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    seen = {0}; st = [0]
    while st:
        u = st.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); st.append(v)
    return len(seen) == n


def kappa_a(n, edges, alpha):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    load = {}
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
    return 0.5 * max(load.values()) if load else 0.0


def verified_lambda2_lower(L, alpha, t):
    """returns (ok, details): ok = verified  lambda_2(L, diag alpha) >= t  (non-strict) by the RESIDUAL certificate:
    with R the floating-point Cholesky factor of A = fl(M_t - rho I),  M_t = R'R + rho I + E,  ||E||_2 <= e := e_res + e_form, where
    e_res >= ||A - R'R||_F is an outward Frobenius bound of the computed residual (entrywise rounding of the residual evaluation added),
    and e_form >= ||M_t^exact - (A + rho I)||_F bounds the formation error of M_t entrywise (exploiting the diagonal-plus-rank-one structure:
    L integer-exact; entry (i,j) = L_ij - t alpha_i [i=j] + 2 t alpha_i alpha_j, each term with <= 3 roundings).  rho >= e proves M_t >= 0."""
    n = L.shape[0]; D = np.diag(alpha); P = 2 * t * np.outer(alpha, alpha); M = L - t * D + P
    # formation error: |fl(x) - x| <= 3 eps (|L| + t alpha_i + 2 t alpha_i alpha_j) per entry (three operations, each |rel err| <= eps, magnitudes bounded by the term sums)
    e_form = 3 * EPS * np.sqrt(((np.abs(L) + t * D + P) ** 2).sum()) * (1 + 1e-12)
    rho = 0.0
    for _ in range(8):
        A = M - rho * np.eye(n)
        try:
            R = cholesky(A, lower=False, check_finite=False)
        except LinAlgError:
            return False, dict(reason="cholesky failed", rho=rho)
        Res = A - R.T @ R                                    # computed residual; its own rounding: |R'R| products of n terms -> <= (n+2) eps sum_k |R_ki R_kj| per entry, plus one subtraction
        absRR = np.abs(R).T @ np.abs(R)
        e_res = float(np.sqrt((Res ** 2).sum())) * (1 + 1e-12) + (n + 2) * EPS * float(np.sqrt((absRR + np.abs(A)) ** 2).sum())
        e = e_res + e_form
        if rho >= e:
            return True, dict(rho=rho, e_res=e_res, e_form=e_form, e=e, n=n, strict=bool(rho > e))
        rho = 2.0 * e + 1e-300
    return False, dict(reason="margin not reached", rho=rho, e=e)


def analyse(name, n, edges, alpha, with_kappa=True):
    if n == 1:
        return dict(shape=name, n=1, note="one pixel: K = h^2/pi^2 directly (Theorem S19)")
    L = np.zeros((n, n))
    for (u, v) in edges:
        L[u, u] += 1; L[v, v] += 1; L[u, v] -= 1; L[v, u] -= 1
    W = np.diag(alpha); Z = np.linalg.svd(alpha[None, :], full_matrices=True)[2][1:].T
    lam2 = float(eigh(Z.T @ L @ Z, Z.T @ W @ Z, eigvals_only=True)[0])
    ok = False; det = None; t = None
    for mg in (a.margin, 1e-4, 1e-3, 1e-2, 0.05, 0.2):        # the numerical lambda_2 may overshoot the true value; the certificate is for the first margin that passes
        t = (1 - mg) * lam2; ok, det = verified_lambda2_lower(L, alpha, t)
        if ok:
            det["margin"] = mg; break
    # also certify that t cannot be raised much: the unverified lambda_2 is the comparator; a failure at (1+margin)*lam2 is expected (not a certificate)
    row = dict(shape=name, n=int(n), lambda2_numerical=lam2, t_verified=t if ok else None, K_spec_verified=(1.0 / t) if ok else None, verified=bool(ok), conclusion=('lambda_2 > t' if (ok and det.get('strict')) else 'lambda_2 >= t') if ok else None, details=det)
    if with_kappa:
        kap = kappa_a(n, edges, alpha); row.update(kappa_a=float(kap), two_kappa=float(2 * kap), ratio_kappa_t=float(kap * t) if ok else None)   # discrete looseness kappa_a / (1/t) = kappa_a t
    return row


out = []; t0 = time.time()
shapes = {"two_pixels": np.ones((2, 1), bool), "chain20": np.ones((20, 1), bool), "square6": np.ones((6, 6), bool), "square15": np.ones((15, 15), bool)}
bott = np.zeros((9, 4), bool); bott[:4, :] = True; bott[5:, :] = True; bott[4, 0] = True; shapes["bottleneck"] = bott
for name, mk in shapes.items():
    n, edges, L = cell_graph(mk); alpha = np.ones(n) / n; out.append(analyse(name, n, edges, alpha)); print(out[-1]["shape"], out[-1].get("verified"), out[-1].get("conclusion"), out[-1].get("K_spec_verified"), out[-1].get("ratio_kappa_t"), flush=True)
# real cells: Georgia counties with population weights (side-connected component containing the most population; cells up to max_pix pixels)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool); county = np.where(mask, ras["county_idx"], -1); w = np.where(mask, ras["weight"], 0.0)
ids = [c for c in np.unique(county[county >= 0])]; sizes = {c: int((county == c).sum()) for c in ids}
chosen = sorted([c for c in ids if 2 <= sizes[c] <= a.max_pix], key=lambda c: -sizes[c])[:a.counties]
real = []
for c in chosen:
    mk = county == c; ii, jj = np.nonzero(mk); sub = np.zeros((ii.max() - ii.min() + 1, jj.max() - jj.min() + 1), bool); sub[ii - ii.min(), jj - jj.min()] = True
    n, edges, L = cell_graph(sub)
    if not connected(n, edges):
        real.append(dict(county=int(c), n=int(n), note="not side-connected: no whole-cell Poincaré constant from componentwise certificates (between-component variance uncontrolled)")); continue
    ww = w[mk][np.lexsort((jj, ii))]; ww = np.where(ww > 0, ww, ww[ww > 0].min() * 1e-3); alpha = ww / ww.sum()   # positive masses (tiny floor for zero-weight pixels, declared)
    r = analyse(f"county_{c}", n, edges, alpha, with_kappa=(n <= 400)); r["county"] = int(c); real.append(r)
    print(r["shape"], r["n"], r["verified"], r.get("K_spec_verified"), r.get("lambda2_numerical"), f"{time.time()-t0:.0f}s", flush=True)
res = dict(toy=out, real=real, margin=a.margin, n_verified_real=int(sum(1 for r in real if r.get("verified"))), n_real=len(real), seconds=time.time() - t0)
json.dump(res, open(os.path.join(a.out, "a7_verified_poincare.json"), "w"), indent=1, default=float); print("verified real cells", res["n_verified_real"], "/", res["n_real"])
