"""check_spectral.py -- independent checker for the verified spectral Poincaré constants (Supplementary Theorem S19 (ii)): for every cell
of exp2/A7_verified/a7_verified_poincare.json it (1) rebuilds the cell graph and the unweighted Laplacian L (exact integers) and the
normalized masses alpha_i = m_i / S as EXACT rationals (S = sum of the float masses as a Fraction, so sum alpha = 1 exactly; the
producer's declared floor for zero-weight pixels is reproduced), (2) forms M_t = L - t diag(alpha) + 2 t alpha alpha^T in exact rationals
and rounds each entry once to the nearest float (entrywise error <= u |entry|; e_form = u ||M_t||_F upward, plus the rounding of the
diagonal shift), (3) factors A = fl(M_t) - rho I with numpy.linalg.cholesky (an implementation independent of the producer's scipy call)
using the STORED rho, (4) bounds the residual ||A - R^T R||_F outward including its own evaluation rounding, and (5) accepts
lambda_2 >= t iff rho >= e_res + e_form and lambda_2 > t iff rho > e_res + e_form.  Toy shapes: alpha = 1/n exactly.
Usage: python check_spectral.py --out <dir>
"""
try:
    from . import paths                     # imported as part of the package
except ImportError:
    import paths                            # flat import (this directory on sys.path)
import argparse, io, json, os, sys, time, collections, numpy as np
from fractions import Fraction
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
EXP = paths.EXP2
art = json.load(open(os.path.join(EXP, "A7_verified", "a7_verified_poincare.json"), encoding="utf-8")); U = np.finfo(float).eps / 2
def gam(k): return k * U / (1 - k * U)


def cell_graph(mask):
    ii, jj = np.nonzero(mask); n = len(ii); pix = {(i, j): k for k, (i, j) in enumerate(zip(ii, jj))}; edges = []
    for (i, j), k in pix.items():
        for (di, dj) in ((1, 0), (0, 1)):
            if (i + di, j + dj) in pix:
                edges.append((k, pix[(i + di, j + dj)]))
    return n, edges


def check(name, mask, masses_float, t, rho):
    n, edges = cell_graph(mask); t0 = time.time()
    L = np.zeros((n, n))
    for (u, v) in edges:
        L[u, u] += 1; L[v, v] += 1; L[u, v] -= 1; L[v, u] -= 1
    mF = [Fraction(float(m)) for m in masses_float]; S = sum(mF, Fraction(0)); alphaF = [m / S for m in mF]; assert sum(alphaF, Fraction(0)) == 1
    tF = Fraction(float(t)); twot = 2 * tF
    # exact entries -> one rounding each; e_form = u * ||M||_F (upward) + rounding of the diagonal shift
    M = np.empty((n, n)); fro2 = Fraction(0)
    for i in range(n):
        ai = alphaF[i]; row = L[i]
        for j in range(n):
            e = Fraction(int(row[j])) + twot * ai * alphaF[j] - (tF * ai if i == j else 0)
            M[i, j] = float(e); fro2 += e * e
    e_form = float(np.sqrt(float(fro2))) * U * (1 + 1e-12) + U * float(np.sqrt(((np.abs(np.diag(M)) + rho) ** 2).sum())) * (1 + 1e-12)
    A = M - rho * np.eye(n)
    try:
        Rl = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return dict(cell=name, n=n, pass_nonstrict=False, reason="cholesky failed at the stored rho", seconds=time.time() - t0)
    Res = A - Rl @ Rl.T; absRR = np.abs(Rl) @ np.abs(Rl).T
    e_res = float(np.sqrt((Res ** 2).sum())) * (1 + 1e-12) + gam(n + 2) * float(np.sqrt(((absRR + np.abs(A)) ** 2).sum()))
    e = e_res + e_form
    return dict(cell=name, n=n, t=t, rho=rho, e_res=e_res, e_form=e_form, e=e, pass_nonstrict=bool(rho >= e), pass_strict=bool(rho > e), K_spec=1.0 / t, seconds=time.time() - t0)


rows = []
shapes = {"two_pixels": np.ones((2, 1), bool), "chain20": np.ones((20, 1), bool), "square6": np.ones((6, 6), bool), "square15": np.ones((15, 15), bool)}
bott = np.zeros((9, 4), bool); bott[:4, :] = True; bott[5:, :] = True; bott[4, 0] = True; shapes["bottleneck"] = bott
for r in art["toy"]:
    if not r.get("verified"):
        continue
    mk = shapes[r["shape"]]; n = int(mk.sum()); rows.append(check(r["shape"], mk, [1.0] * n, r["t_verified"], r["details"]["rho"])); print(rows[-1], flush=True)
ras = np.load(paths.RASTER["georgia"]); mask = ras["mask"].astype(bool); county = np.where(mask, ras["county_idx"], -1); w = np.where(mask, ras["weight"], 0.0)
for r in art["real"]:
    if not r.get("verified"):
        continue
    c = r["county"]; mk = county == c; ii, jj = np.nonzero(mk); sub = np.zeros((ii.max() - ii.min() + 1, jj.max() - jj.min() + 1), bool); sub[ii - ii.min(), jj - jj.min()] = True
    ww = w[mk][np.lexsort((jj, ii))]; ww = np.where(ww > 0, ww, ww[ww > 0].min() * 1e-3)       # the producer's declared positive floor
    rows.append(check(f"county_{c}", sub, ww, r["t_verified"], r["details"]["rho"])); print({k: v for k, v in rows[-1].items() if k != "seconds"}, f"{rows[-1]['seconds']:.0f}s", flush=True)
summ = dict(n=len(rows), pass_nonstrict=sum(r.get("pass_nonstrict", False) for r in rows), pass_strict=sum(r.get("pass_strict", False) for r in rows))
json.dump(dict(summary=summ, cells=rows), open(os.path.join(a.out, "check_spectral.json"), "w"), indent=1, default=float); print(summ)
