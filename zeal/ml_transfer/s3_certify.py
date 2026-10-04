"""s3_certify.py -- ZEAL certificates for the pooled statistics of the ML-transfer experiment (exp3_ml).

Per (model, eps, verifier method, arc set) and feeder j, the enclosure class is the 1-D difference-constraint polytope
    Phi_j = { phi in R^T : lo_t <= phi_t <= hi_t ,  glo_{t,k} <= phi_{t+k} - phi_t <= ghi_{t,k}  (k in the arc set) }
built from the verifier's boxes (s2_verify.py).  The model's output sequence f_j(y) lies in Phi_j for EVERY admissible series y,
so sup_y d.f_j(y) <= h_Phi(d) for every weight vector d (pooled means, differences, binned slopes).
Bounds computed, for every admissible aggregation choice:
    V      value-only bound  sum d+ hi - sum d- lo                          (the trivial / value-only certificate)
    V*     value face after closure  sum d+ u* - sum d- l*  (u*, l* = top/bottom of Phi; Theorems S3, S4 (i))   [exact for d >= 0]
    Z      ZEAL bound h_Phi(d) = exact LP over Phi (transport step, Theorem S3), reported as the VERIFIED dual upper bound
           (transport_core.solve_max -> verified_upper, outward rounding); evaluated lazily where it can change a conclusion,
           and on a seeded validation sample of choices (for G and the outer/inner ratios).
Closed-form bounds (V, V*) are widened outward by OUT_MARG = 1e-10; u*, l* are computed by Bellman-Ford label correction in float64
(path sums of <= 1008 terms of magnitude < 2: error < 1e-12) and widened by CL_MARG = 1e-11.
Arc sets: 'Z1' = lag-1 increments only; 'ZD' = dyadic lags 1, 2, 4, ..., 128.  Methods with increments: CROWN, alpha-CROWN
(IBP increments are exactly the implied ones, so IBP gives V* = V = Z; reported as value-only).

Declared aggregation classes (fixed before any certificate was computed; t = evaluation hour 0..1007):
  S1a  global window range: all contiguous windows A = [s, s+L), L in [24, 168] h, inside the 6-week span;  sup_{A,B} <f>_A - <f>_B.
  S1b  report-time lookback range: report times e = 1008 - 24 i (i = 0..27), windows A_L = [e-L, e), L in [24, 168];
       R(e) = sup_{L, L'} <f>_{A_L} - <f>_{A_L'}  (nested windows; 145^2 ordered pairs per report time).
  S2   before/after sign: nominal boundaries tau0 in {336, 504, 672}; admissible boundary tau in tau0 + [-24, 24] (hourly) and
       period length L in {24, 48, ..., 168} (1-7 days); Delta = <f>_[tau, tau+L) - <f>_[tau-L, tau)  (343 choices / instance).
  S3   binned-trend sign: a 4-week span [s0, s0+672) with s0 in base + [0, 24] (base in {0, 168, 312}), K in {4, 7, 14, 28} equal-count
       bins (bin length 168/96/48/24 h); statistic = OLS slope of the bin means against bin index, per week  (100 choices / instance).
  S4   ranking: at each report time e (S1b), feeders ranked by the pooled score <f_j>_{A_L}, common L in [24, 168].
  S5   POST-HOC (added after the S1-S4 CROWN results, as a test of the regime theorem's prediction that the increment engine pays
       off only when the transport distance of d is shorter than the model's coherence length; NOT part of the pre-declared classes):
       intra-day contrast pooled over a week w (6 instances per feeder): mean over the 7 days of <f>_[h0, h0+b) - <f>_[h0-b, h0) with
       boundary clock hour h0 in {14..20} and block length b in {1..4} h (28 choices / instance); decision = sign.
Usage: python s3_certify.py [--models mlp,feat] [--procs 16] [--out DIR]   -> DIR/s3_certify_<model>.json, DIR/closure_*.npz
       (DIR defaults to exp3_ml; inputs boxes_/nominal_ are always read from exp3_ml)

LP-failure guard (added 4 Oct 2026, ported from real_s3_certify.lp_scaled; the LP formulation itself is NOT changed here, no
rescaling): a failed LP (HiGHS status != 0 or a non-finite verified bound) now returns +inf = "no certificate" and is counted.
Before, solve_max's NaN reached the consumers and was read as "no improvement" (see lp() below for each consumer).  Failures are
recorded per task (lp_failures, lp_failures_by_class), per S1a record, per S1b report time and per S2/S3/S5 instance (lp_failures),
and per validation-sample bound (lp_failed_U / lp_failed_L, G record lp_failed); the main writes lp_failures_total.
"""
import os, sys, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mlt_common import OUT, T, N_ENT, EPS_GRID, LAGS_DYADIC, jdump
from transport_core import Poly, solve_max

OUT_MARG = 1e-10; CL_MARG = 1e-11
METHODS_INC = ["CROWN", "alpha-CROWN"]; ARCSETS = {"Z1": [1], "ZD": LAGS_DYADIC}
LGRID = np.arange(24, 169)
E_REPORT = np.array([T - 24 * i for i in range(28)])
S2_TAU0 = [336, 504, 672]; S2_L = [24 * k for k in range(1, 8)]; S2_DT = np.arange(-24, 25)
S3_BASE = [0, 168, 312]; S3_K = [4, 7, 14, 28]; S3_SPAN = 672; S3_DS = np.arange(0, 25)
S5_WEEKS = list(range(6)); S5_H0 = list(range(14, 21)); S5_B = [1, 2, 3, 4]      # POST-HOC class S5 (see header)
N_SAMPLE = 4          # validation choices per S2/S3 instance (seeded), both directions
LAZY_CAP = 40         # max LPs per lazy search


# ---------------------------------------------------------------- statistic definitions (d vectors, choice lists)
def s2_choices():
    return [(int(t0 + dt), int(L)) for dt in S2_DT for L in S2_L for t0 in [0]]


def s2_d(tau, L):
    d = np.zeros(T); d[tau:tau + L] += 1.0 / L; d[tau - L:tau] -= 1.0 / L; return d


def s3_d(s0, K):
    b = S3_SPAN // K; c = np.arange(K) - (K - 1) / 2; c = c / np.sum(c * c) * (168.0 / b)      # slope per week of bin means
    d = np.zeros(T)
    for k in range(K):
        d[s0 + k * b:s0 + (k + 1) * b] = c[k] / b
    return d


def s5_d(w, h0, b):
    """intra-day contrast pooled over week w: mean over its 7 days of <f>_[h0, h0+b) - <f>_[h0-b, h0) (clock hours; t = 0 is midnight)"""
    d = np.zeros(T)
    for dd in range(7):
        s_ = 168 * w + 24 * dd; d[s_ + h0:s_ + h0 + b] += 1.0 / (7 * b); d[s_ + h0 - b:s_ + h0] -= 1.0 / (7 * b)
    return d


def s1b_d(e, L, Lp):
    d = np.zeros(T); d[e - L:e] += 1.0 / L; d[e - Lp:e] -= 1.0 / Lp; return d


def win_d(s, L):
    d = np.zeros(T); d[s:s + L] = 1.0 / L; return d


# ---------------------------------------------------------------- closure and bounds
def closure_1d(lo, hi, ep, eq, cm, cp):
    """top u* and bottom l* of Phi by Bellman-Ford label correction; raises if Phi is empty"""
    src = np.concatenate([ep, eq]); dst = np.concatenate([eq, ep]); w = np.concatenate([cp, -cm])     # arc src->dst: phi_dst - phi_src <= w
    n = len(lo); u = hi.copy(); l = lo.copy()
    for it in range(n + 2):
        new = u.copy(); np.minimum.at(new, dst, u[src] + w)
        if np.array_equal(new, u):
            break
        u = new
    else:
        raise ValueError("negative cycle")
    for it in range(n + 2):
        new = l.copy(); np.maximum.at(new, src, l[dst] - w)
        if np.array_equal(new, l):
            break
        l = new
    if np.any(l > u + 1e-12):
        raise ValueError("Phi empty")
    return np.minimum(u + CL_MARG, hi), np.maximum(l - CL_MARG, lo)      # hi, lo are valid bounds themselves


def vbound(d, a, b):
    """upper bound sum d+ a - sum d- b (a = upper values, b = lower values), outward"""
    return float(np.maximum(d, 0) @ a - np.maximum(-d, 0) @ b) + OUT_MARG


def pref(x):
    return np.concatenate([[0.0], np.cumsum(x)])


def wmean(P, s, L):
    return (P[s + L] - P[s]) / L


# ---------------------------------------------------------------- one task = (model, eps, method, arcset, feeder)
G_BOX = {}


def init_worker(models):
    for m in models:
        G_BOX[m] = (dict(np.load(os.path.join(OUT, f"boxes_{m}.npz"))), np.load(os.path.join(OUT, f"nominal_{m}.npz"))["f"].astype(np.float64))


def build_poly(model, eps, meth, arcset, j):
    B, _ = G_BOX[model]; key = f"{meth}|{eps}"
    lo, hi = B[f"lo|{key}"][j].astype(np.float64), B[f"hi|{key}"][j].astype(np.float64)
    ep, eq, cm, cp = [], [], [], []
    for k in ARCSETS[arcset]:
        t = np.arange(T - k); ep.append(t); eq.append(t + k); cm.append(B[f"glo|{key}|{k}"][j]); cp.append(B[f"ghi|{key}|{k}"][j])
    return Poly(lo, hi, np.concatenate(ep), np.concatenate(eq), np.concatenate(cm), np.concatenate(cp))


def lp(poly, d):
    """(verified upper bound, primal value) of h_Phi(d).
    GUARD (4 Oct 2026, as real_s3_certify.lp_scaled): a failed LP returns (+inf, nan) = no certificate, never NaN.  Effect per consumer:
      S1a  Z_at_argmax = +inf (the reported S1a bound V* is unaffected; the record is marked).
      S1b  zv = +inf > best_lp, so Z(e) = +inf: valid (trivially).  Before, NaN > best_lp was False, the failed pair was silently
           dropped from the lazy supremum, and Z(e) could fall BELOW h_Phi at that pair (unsound).
      S2/S3/S5  min(V*, +inf) = V*, max(LV*, -inf) = LV*: the decision bound falls back to the valid closure bound (same as NaN did
           before, but now counted); the validation-sample G / outer-inner records of a failed LP carry Z = +inf (non-finite, so
           excluded from the medians) instead of being recorded as zero saving."""
    r = solve_max(poly, d)
    if r["status"] != 0 or not np.isfinite(r["verified"]):
        return float("inf"), float("nan")
    return float(r["verified"]), float(r["value"])


def s1b_grid(Pa, Pb, e):
    """upper bounds of <phi>_{A_L} - <phi>_{A_L'} for all (L, L') in LGRID^2 at report time e from upper values a and lower values b
    (prefix sums Pa, Pb); exact support function of the box [b, a] for each (cancelled) nested pair"""
    Sa = Pa[e] - Pa[e - LGRID]; Sb = Pb[e] - Pb[e - LGRID]            # sums over the last L hours
    L = LGRID[:, None].astype(float); Lp = LGRID[None, :].astype(float); SaL = Sa[:, None]; SaLp = Sa[None, :]; SbL = Sb[:, None]; SbLp = Sb[None, :]
    lt = (1 / L - 1 / Lp) * SaL - (SbLp - SbL) / Lp                      # L < L'
    gt = (SaL - SaLp) / L - (1 / Lp - 1 / L) * SbLp                      # L > L'
    return np.where(L < Lp, lt, np.where(L > Lp, gt, 0.0))


def run_task(args):
    model, eps, meth, arcset, j = args; t0 = time.time()
    _, Fn = G_BOX[model]; psi = Fn[j]
    poly = build_poly(model, eps, meth, arcset, j); lo, hi = poly.lo, poly.hi
    assert poly.feasible(psi, tol=0.0), "nominal output not in Phi"
    u, l = closure_1d(lo, hi, poly.ep, poly.eq, poly.cm, poly.cp)
    Ph, Plo, Pu, Pl, Pf = pref(hi), pref(lo), pref(u), pref(l), pref(psi)
    res = dict(model=model, eps=eps, meth=meth, arcset=arcset, j=j, u=u, l=l,
               closure_gain_median=float(np.median((hi - lo) - (u - l))), width_ratio_median=float(np.median((u - l) / (hi - lo))))
    nlp = 0; lp_time = 0.0; G_list = []; nfail = 0                      # nfail: GUARD failure counter of this task

    def zlp(d):
        nonlocal nlp, lp_time, nfail
        t1 = time.time(); v = lp(poly, d); nlp += 1; lp_time += time.time() - t1
        if not np.isfinite(v[0]):                                      # GUARD: failed LP (+inf from lp, or from real_s3_certify.lp_scaled)
            nfail += 1
        return v

    def record_G(d, Uv, Us, Uz, tag):
        dpsi = float(d @ psi)
        G_list.append(dict(tag=tag, dpsi=dpsi, V=Uv, Vs=Us, Z=Uz,
                           G=(Us - Uz) / (Us - dpsi) if Us - dpsi > 1e-9 else np.nan,
                           S_total=(Uv - Uz) / (Uv - dpsi) if Uv - dpsi > 1e-9 else np.nan,
                           lp_failed=bool(not np.isfinite(Uz))))              # GUARD: Z = +inf marks a failed LP (G = -inf, excluded)

    # ---- S1a global range (closure/value bounds via max/min window means; LP at the V*-argmax pair)
    best = {}
    for nm, (Pa, Pb) in {"V": (Ph, Plo), "Vs": (Pu, Pl), "nom": (Pf, Pf)}.items():
        mx = (-np.inf, None); mn = (np.inf, None)
        for L in LGRID:
            ma = (Pa[L:] - Pa[:-L]) / L; mb = (Pb[L:] - Pb[:-L]) / L
            ia, ib = int(np.argmax(ma)), int(np.argmin(mb))
            if ma[ia] > mx[0]: mx = (float(ma[ia]), (ia, int(L)))
            if mb[ib] < mn[0]: mn = (float(mb[ib]), (ib, int(L)))
        best[nm] = (mx[0] - mn[0] + (OUT_MARG if nm != "nom" else 0.0), mx[1], mn[1])
    (sA, LA), (sB, LB) = best["Vs"][1], best["Vs"][2]
    f0 = nfail; dpair = win_d(sA, LA) - win_d(sB, LB); zv, zr = zlp(dpair)
    res["S1a"] = dict(V=best["V"][0], Vs=best["Vs"][0], nominal=best["nom"][0], argmax_pair=[sA, LA, sB, LB],
                      nominal_pair=[best["nom"][1][0], best["nom"][1][1], best["nom"][2][0], best["nom"][2][1]],
                      Z_at_argmax=zv, Z_at_argmax_raw=zr, nominal_at_argmax=float(dpair @ psi),
                      Z_range_bracket=[zv, best["Vs"][0]], lp_failures=nfail - f0)          # GUARD: failures in S1a
    # ---- S1b report-time lookback ranges
    s1b = []
    for e in E_REPORT:
        f0 = nfail                                                      # GUARD: failures at this report time
        GV = s1b_grid(Ph, Plo, e); GS = s1b_grid(Pu, Pl, e); GN = s1b_grid(Pf, Pf, e)
        order = np.argsort(-GS, axis=None); best_lp = -np.inf; best_pair = None; k = 0; settled = False
        while k < len(order) and k < LAZY_CAP:
            iL, iLp = np.unravel_index(order[k], GS.shape)
            if GS[iL, iLp] + OUT_MARG <= best_lp:          # every remaining pair has h_Phi <= V* <= best_lp: the sup is settled
                settled = True; break
            d = s1b_d(int(e), int(LGRID[iL]), int(LGRID[iLp])); zv, zr = zlp(d)
            if k == 0:
                record_G(d, float(GV[iL, iLp] + OUT_MARG), float(GS[iL, iLp] + OUT_MARG), zv, "S1b_top")
            if zv > best_lp:
                best_lp, best_pair = zv, (int(LGRID[iL]), int(LGRID[iLp]))
            k += 1
        nxt = float(GS.ravel()[order[k]] + OUT_MARG) if (not settled and k < len(order)) else -np.inf
        Zcert = max(best_lp, nxt)
        iN = np.unravel_index(np.argmax(GN), GN.shape)
        s1b.append(dict(e=int(e), V=float(GV.max() + OUT_MARG), Vs=float(GS.max() + OUT_MARG), Z=float(Zcert), Z_settled=bool(settled), n_lp=k,
                        Z_argmax_pair=best_pair, nominal=float(GN.max()), nominal_pair=[int(LGRID[iN[0]]), int(LGRID[iN[1]])],
                        nominal_at_Zpair=float(s1b_d(int(e), *best_pair) @ psi) if best_pair else np.nan,
                        lp_failures=nfail - f0))                        # GUARD: > 0 means Z(e) = +inf (no certificate)
    res["S1b"] = s1b
    # ---- S2 before/after sign and S3 binned-trend sign: per-choice V, V* bounds (both directions), lazy LP for conclusions
    rng = np.random.default_rng(1000 * j + 7)
    for S, instances in (("S2", [(t0,) for t0 in S2_TAU0]), ("S3", [(b,) for b in S3_BASE]), ("S5", [(w,) for w in S5_WEEKS])):
        out = []
        for inst in instances:
            f0 = nfail                                                  # GUARD: failures in this instance
            if S == "S2":
                ch = [(int(inst[0] + dt), int(L)) for dt in S2_DT for L in S2_L]; dfun = lambda c: s2_d(*c)
            elif S == "S3":
                ch = [(int(inst[0] + ds), int(K)) for ds in S3_DS for K in S3_K]; dfun = lambda c: s3_d(*c)
            else:
                ch = [(int(inst[0]), int(h0), int(b)) for h0 in S5_H0 for b in S5_B]; dfun = lambda c: s5_d(*c)
            Ds = np.array([dfun(c) for c in ch])                                  # [n_choices, T]
            nom = Ds @ psi
            Dp, Dm = np.maximum(Ds, 0), np.maximum(-Ds, 0)
            UV = Dp @ hi - Dm @ lo + OUT_MARG; LV = Dp @ lo - Dm @ hi - OUT_MARG          # value-only upper / lower of the statistic
            US = Dp @ u - Dm @ l + OUT_MARG; LS = Dp @ l - Dm @ u - OUT_MARG              # closure (value face)
            UZ = US.copy(); LZ = LS.copy(); zU = np.zeros(len(ch), bool); zL = np.zeros(len(ch), bool)
            fU = np.zeros(len(ch), bool); fL = np.zeros(len(ch), bool)    # GUARD: the LP of this choice/direction failed

            def lpU(i):                                                 # GUARD: failed -> min(V*, +inf) = V* (valid), flagged in fU
                if not zU[i]: v = zlp(Ds[i])[0]; fU[i] = not np.isfinite(v); UZ[i] = min(UZ[i], v); zU[i] = True
            def lpL(i):
                if not zL[i]: v = zlp(-Ds[i])[0]; fL[i] = not np.isfinite(v); LZ[i] = max(LZ[i], -v); zL[i] = True
            # invariant '+' (all lower > 0): check hardest first, stop at the first failure; same for '-'
            for sgn in (+1, -1):
                key = LZ if sgn > 0 else UZ; f_ = lpL if sgn > 0 else lpU
                bad = np.where(sgn * key <= 0)[0]
                if np.all(sgn * nom > 0):                        # invariance is possible only if every nominal sign agrees
                    for i in bad[np.argsort(sgn * key[bad])]:
                        f_(i)
                        if sgn * key[i] <= 0:
                            break
            # existence of a certified '+' (resp. '-') choice: most promising first
            for sgn in (+1, -1):
                key = LZ if sgn > 0 else UZ; f_ = lpL if sgn > 0 else lpU
                if not np.any(sgn * key > 0):
                    for i in np.argsort(-sgn * key)[:5]:
                        f_(i)
                        if sgn * key[i] > 0:
                            break
            # validation sample (G and outer/inner): seeded random choices, both directions
            samp = rng.choice(len(ch), N_SAMPLE, replace=False)
            for i in samp:
                lpU(i); lpL(i)                                          # GUARD: a failed LP enters the G record as Z = +inf, not as V*
                record_G(Ds[i], float(UV[i]), float(US[i]), np.inf if fU[i] else float(UZ[i]), f"{S}_sample")
                record_G(-Ds[i], float(-LV[i]), float(-LS[i]), np.inf if fL[i] else float(-LZ[i]), f"{S}_sample")

            def cats(Ub, Lb):
                plus, minus = Lb > 0, Ub < 0
                return dict(all_plus=bool(plus.all()), all_minus=bool(minus.all()), any_plus=bool(plus.any()), any_minus=bool(minus.any()),
                            n_cert_plus=int(plus.sum()), n_cert_minus=int(minus.sum()))
            out.append(dict(inst=int(inst[0]), n_choices=len(ch), nominal_min=float(nom.min()), nominal_max=float(nom.max()),
                            nominal_n_plus=int((nom > 0).sum()), nominal_n_minus=int((nom < 0).sum()),
                            V=cats(UV, LV), Vs=cats(US, LS), Z=cats(UZ, LZ), n_lp_U=int(zU.sum()), n_lp_L=int(zL.sum()),
                            sample=[int(i) for i in samp], choices_sample=[ch[i] for i in samp],
                            sample_bounds=[dict(nominal=float(nom[i]), UV=float(UV[i]), US=float(US[i]), UZ=np.inf if fU[i] else float(UZ[i]),
                                                LV=float(LV[i]), LS=float(LS[i]), LZ=-np.inf if fL[i] else float(LZ[i]),
                                                lp_failed_U=bool(fU[i]), lp_failed_L=bool(fL[i])) for i in samp],      # GUARD: marked
                            flip_candidates=[ch[i] for i in np.argsort(np.abs(nom))[:5]],
                            lp_failures=nfail - f0))                    # GUARD: failures in this instance (decisions used V* there)
        res[S] = out
    res["G_records"] = G_list; res["n_lp"] = nlp; res["lp_seconds"] = lp_time; res["seconds"] = time.time() - t0
    # GUARD: failure counts of this task (S1b per report time and S2/S3/S5 per instance are in the records above)
    res["lp_failures"] = nfail
    res["lp_failures_by_class"] = dict(S1a=res["S1a"]["lp_failures"], S1b=int(sum(x["lp_failures"] for x in s1b)),
                                       **{S: int(sum(x["lp_failures"] for x in res[S])) for S in ("S2", "S3", "S5")})
    # discrete exact 1-D theorem check (Theorem S8 in sampled time): gradient-only lag-1 LP == sum_s s_G(B_s), B_s = sum_{t>s} d_t
    if arcset == "Z1":
        d = s2_d(504, 72); Bs = np.cumsum(d[::-1])[::-1][1:]                  # B_s for s = 0..T-2
        gm, gp = poly.cm, poly.cp; closed = float(np.sum(np.where(Bs > 0, gp * Bs, gm * Bs)))
        pg = Poly(np.full(T, -1e3), np.full(T, 1e3), poly.ep, poly.eq, poly.cm, poly.cp); r = solve_max(pg, d)
        ok = r["status"] == 0 and np.isfinite(r["value"])               # GUARD: a check, not a bound; a failure is flagged and counted
        res["exact1d_check"] = dict(closed_form=closed, lp=float(r["value"]), abs_diff=abs(closed - float(r["value"])), lp_failed=not ok)
        res["lp_failures_by_class"]["exact1d"] = int(not ok)
    return res


# ---------------------------------------------------------------- ranking (S4) from the per-feeder u*, l*
def ranking(lo_all, hi_all, f_all):
    """per report time: certified (all L, all perturbations) pairwise order and rank intervals; per-L certified top-3; nominal top-3 sets"""
    out = []
    P = {k: np.array([pref(x) for x in v]) for k, v in (("a", hi_all), ("b", lo_all), ("f", f_all))}
    for e in E_REPORT:
        A = (P["a"][:, e][:, None] - P["a"][:, e - LGRID]) / LGRID[None, :]          # [N_ENT, nL] upper pooled scores
        Bv = (P["b"][:, e][:, None] - P["b"][:, e - LGRID]) / LGRID[None, :]
        Fv = (P["f"][:, e][:, None] - P["f"][:, e - LGRID]) / LGRID[None, :]
        # l_jk = min_L (lower_j - upper_k)  certified lower bound of theta_j - theta_k over all L
        ljk = np.min(Bv[:, None, :] - A[None, :, :], axis=2) - OUT_MARG; np.fill_diagonal(ljk, -np.inf)
        beats = ljk > 0
        rank_lo = 1 + beats.sum(0); rank_hi = N_ENT - beats.sum(1)
        nom_rank = np.argsort(np.argsort(-Fv, axis=0), axis=0) + 1                    # [N_ENT, nL]
        tops = {tuple(sorted(np.argsort(-Fv[:, i])[:3])) for i in range(len(LGRID))}
        cert_top = []
        for i in range(len(LGRID)):
            o = np.argsort(-Fv[:, i]); S_ = o[:3]; R_ = o[3:]
            cert_top.append(bool(np.min(Bv[S_, i]) - np.max(A[R_, i]) > OUT_MARG))
        S_nom = np.argsort(-Fv[:, -1])[:3]; R_nom = np.setdiff1d(np.arange(N_ENT), S_nom)
        top3_invariant = bool(np.all(beats[np.ix_(S_nom, R_nom)]))
        distinct_cert_tops = {tuple(sorted(np.argsort(-Fv[:, i])[:3])) for i in range(len(LGRID)) if cert_top[i]}
        out.append(dict(e=int(e), certified_pairs=int(beats.sum()), rank_interval_width_mean=float(np.mean(rank_hi - rank_lo)),
                        nominal_rank_range_mean=float(np.mean(nom_rank.max(1) - nom_rank.min(1))),
                        nominal_distinct_top3=len(tops), top3_certified_invariant=top3_invariant,
                        n_L_with_certified_top3=int(np.sum(cert_top)), distinct_certified_top3_sets=len(distinct_cert_tops)))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="mlp,feat"); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--eps", default=None); ap.add_argument("--methods", default="CROWN,alpha-CROWN")
    ap.add_argument("--out", default=OUT, help="output directory (default exp3_ml); inputs are always read from exp3_ml"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    METHODS_INC = a.methods.split(",")
    models = a.models.split(","); eps_list = [float(x) for x in a.eps.split(",")] if a.eps else EPS_GRID
    for model in models:
        t0 = time.time(); init_worker([model]); B, Fn = G_BOX[model]
        tasks = [(model, eps, meth, arc, j) for eps in eps_list for meth in METHODS_INC for arc in ARCSETS for j in range(N_ENT)]
        with Pool(a.procs, initializer=init_worker, initargs=([model],)) as pool:
            results = pool.map(run_task, tasks, chunksize=1)
        summary = dict(model=model, declared=dict(LGRID=[24, 168], E_REPORT=E_REPORT, S2_TAU0=S2_TAU0, S2_L=S2_L, S2_DT=[-24, 24],
                                                  S3_BASE=S3_BASE, S3_K=S3_K, S3_SPAN=S3_SPAN, S3_DS=[0, 24], N_SAMPLE=N_SAMPLE, LAZY_CAP=LAZY_CAP,
                                                  S5_post_hoc=dict(WEEKS=S5_WEEKS, H0=S5_H0, B=S5_B),
                                                  OUT_MARG=OUT_MARG, CL_MARG=CL_MARG), configs={})
        # value-only rows (IBP, CROWN, alpha-CROWN boxes): S4 ranking with lo/hi
        for eps in eps_list:
            for meth in ["IBP"] + [m for m in METHODS_INC]:
                key = f"{meth}|{eps}"
                summary["configs"][f"{key}|V"] = dict(S4=ranking(B[f"lo|{key}"].astype(float), B[f"hi|{key}"].astype(float), Fn))
        for eps in eps_list:
            for meth in METHODS_INC:
                for arc in ARCSETS:
                    rs = sorted([r for r in results if r["eps"] == eps and r["meth"] == meth and r["arcset"] == arc], key=lambda r: r["j"])
                    U = np.array([r["u"] for r in rs]); Lw = np.array([r["l"] for r in rs])
                    summary["configs"][f"{meth}|{eps}|{arc}"] = dict(
                        per_feeder=[{k: v for k, v in r.items() if k not in ("u", "l")} for r in rs], S4=ranking(Lw, U, Fn))
                    np.savez_compressed(os.path.join(a.out, f"closure_{model}_{meth}_{eps}_{arc}.npz"), u=U, l=Lw)
        # GUARD: failure totals (zlp LPs; the exact-1-D check LPs are counted separately) and LP volume
        summary["n_lp_total"] = int(sum(r["n_lp"] for r in results))
        summary["lp_failures_total"] = int(sum(r["lp_failures"] for r in results))
        summary["lp_failures_exact1d_total"] = int(sum(r["lp_failures_by_class"].get("exact1d", 0) for r in results))
        summary["lp_failures_by_config"] = {f"{m_}|{e_}|{a_}": {c: int(sum(r["lp_failures_by_class"].get(c, 0) for r in results
                                                                           if (r["meth"], r["eps"], r["arcset"]) == (m_, e_, a_)))
                                                                for c in ("S1a", "S1b", "S2", "S3", "S5", "exact1d")}
                                            for e_ in eps_list for m_ in METHODS_INC for a_ in ARCSETS}
        summary["seconds"] = time.time() - t0
        jdump(summary, os.path.join(a.out, f"s3_certify_{model}.json"))
        print(model, "done", f"{time.time()-t0:.0f}s", "LPs", summary["n_lp_total"], "LP failures", summary["lp_failures_total"],
              "exact1d failures", summary["lp_failures_exact1d_total"], flush=True)
