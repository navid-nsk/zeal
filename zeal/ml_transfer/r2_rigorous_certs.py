"""r2_rigorous_certs.py -- ZEAL certificates on the RIGOROUS boxes (r1_rigorous_boxes.py) versus the auto_LiRPA CROWN boxes, and the
end-to-end chain  rigorous enclosure -> exact-rational LP certificate -> compiled Lean checker.

Statistics (definitions, constants and helpers imported unchanged from s3_certify.py; dyadic arc set ZD; Poly/solve_max from
transport_core.py exactly as s3_certify.py uses them):
  S1b  look-back range at the 28 report times: V, V*, and Z by the same lazy LP search (descending V*, cap LAZY_CAP); the 'top pair'
       record (first LP) gives G = (V* - Z)/(V* - d.psi) as in s3_certify.py.
  S5   (post hoc in exp3_ml) intra-day contrasts, 6 weeks x 28 choices, BOTH directions, with an LP for EVERY choice (s3_certify.py ran LPs
       lazily); G per choice and direction; decision categories (certified +/- choices) from V and from Z.
  Both box sources go through the same code: 'auto' = stored auto_LiRPA CROWN boxes (boxes_<model>.npz, with its 1e-9 margin),
  'rig' = rigorous boxes.  psi = float64 forward pass of the stored network (nominal; used only as the reference point of G).
Lean export (declared subset, rigorous boxes only): feeders EXPORT_FEEDERS; S1b top pair at report-time indices EXPORT_S1B (U direction);
  S5 weeks EXPORT_S5_WEEKS at (h0, b) = EXPORT_S5_CHOICE, U and L directions.  Kind "lp" (cert_export.py format): n = 1008 variables,
  rows phi_{t+k} - phi_t <= ghi and phi_t - phi_{t+k} <= -glo (two rows per arc, all ZD arcs), lo/hi = exact rationals of the rigorous
  floats, objective = the EXACT rational weights of the statistic (1/L, 1/(7b)), y = the HiGHS dual (exact rationals of the floats),
  r = c - A^T y exactly, bound = y.b + sum(r+ hi - r- lo) exactly.  Every file is run through certcheck.exe (PASS/FAIL + exit code).
Real data (--window jul2014|jan2014): sources exp3_ml_real/<w> (auto boxes, models, nominal series, s3_certify cross-check) and
  exp3_ml_real/rigorous/rigorous_boxes_<w>_*.npz; the LP is solved as real_s3_certify.py declares (lp_scaled: centred and scaled
  polytope, verified bound mapped back with outward rounding); for the export the dual y of the SCALED LP is used unchanged in the
  ORIGINAL variables (the scaling is an affine change of variables, so d = A^T y + r is the same equation; only b, lo, hi change), and the
  exact-rational certificate is written for the original, unscaled LP over Phi.
Usage: python r2_rigorous_certs.py [--models mlp,feat,feat24] [--eps 0.01,0.05] [--feeders 0-11] [--procs 12] [--no_lean] [--window w]
       -> exp3_ml/rigorous/r2_summary.json, r2_tasks.json, certs/*.json, lean_results.json
          (real: exp3_ml_real/rigorous/r2_summary_<w>.json, r2_tasks_<w>.json, certs_<w>/*.json, lean_results_<w>.json)
"""
import os as _os, sys as _sys
try:
    from .. import paths                    # imported as part of the package
except ImportError:
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")); import paths
import os, sys, time, json, argparse, subprocess, numpy as np
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from mlt_common import OUT, T, N_ENT, LAGS_DYADIC, jdump
from transport_core import Poly, solve_max, verified_upper
from s3_certify import closure_1d, s1b_grid, s1b_d, s5_d, pref, LGRID, E_REPORT, LAZY_CAP, OUT_MARG, S5_WEEKS, S5_H0, S5_B
from scipy.optimize import linprog
import rigorous_enclosure as RE

REAL_OUT = os.path.abspath(os.path.join(OUT, "..", "exp3_ml_real"))


def ctx(window):
    """(source dir, rigorous-box dir, certificate dir, file tag)"""
    if window is None:
        return OUT, os.path.join(OUT, "rigorous"), os.path.join(OUT, "rigorous", "certs"), ""
    return os.path.join(REAL_OUT, window), os.path.join(REAL_OUT, "rigorous"), os.path.join(REAL_OUT, "rigorous", f"certs_{window}"), f"{window}_"


LEAN = paths.CERTCHECK
EXPORT_FEEDERS = [0, 1]; EXPORT_S1B = [0, 9, 18, 27]; EXPORT_S5_WEEKS = [0, 5]; EXPORT_S5_CHOICE = (17, 2)
_CACHE = {}


def boxes(model, eps, source, window=None):
    key = (model, eps, source, window); SRC, RDIR, _, TAG = ctx(window)
    if key not in _CACHE:
        if source == "auto":
            B = np.load(os.path.join(SRC, f"boxes_{model}.npz")); m = f"CROWN|{eps}"
            d = dict(lo=B[f"lo|{m}"], hi=B[f"hi|{m}"], **{f"glo|{k}": B[f"glo|{m}|{k}"] for k in LAGS_DYADIC}, **{f"ghi|{k}": B[f"ghi|{m}|{k}"] for k in LAGS_DYADIC})
        else:
            B = np.load(os.path.join(RDIR, f"rigorous_boxes_{TAG}{model}_{eps}.npz"))
            d = dict(lo=B["lo"], hi=B["hi"], **{f"glo|{k}": B[f"glo|{k}"] for k in LAGS_DYADIC}, **{f"ghi|{k}": B[f"ghi|{k}"] for k in LAGS_DYADIC})
        _CACHE[key] = {k: np.asarray(v, np.float64) for k, v in d.items()}
    return _CACHE[key]


def nominal(model, window=None):
    key = ("psi", model, window); SRC = ctx(window)[0]
    if key not in _CACHE:
        net = RE.load_net(model, out_dir=SRC); Y = np.load(os.path.join(SRC, f"nominal_{model}.npz"))["y"].astype(np.float64)
        Yw = np.lib.stride_tricks.sliding_window_view(Y, 168, axis=1).reshape(-1, 168)
        _CACHE[key] = (1.0 / (1.0 + np.exp(-RE.forward_float(net, Yw)))).reshape(N_ENT, T)
    return _CACHE[key]


def build_poly(Bx, j):
    ep, eq, cm, cp = [], [], [], []
    for k in LAGS_DYADIC:
        t = np.arange(T - k); ep.append(t); eq.append(t + k); cm.append(Bx[f"glo|{k}"][j]); cp.append(Bx[f"ghi|{k}"][j])
    return Poly(Bx["lo"][j], Bx["hi"][j], np.concatenate(ep), np.concatenate(eq), np.concatenate(cm), np.concatenate(cp))


def solve_with_dual(poly, d, scaled=False):
    """the linprog call of transport_core.solve_max (or of real_s3_certify.lp_scaled's centred/scaled polytope), returning the dual y,
    which is dual-feasible for the ORIGINAL LP in both cases (same A and d); verified bound evaluated on the original LP"""
    d = np.asarray(d, float)
    if scaled:
        c = 0.5 * (poly.lo + poly.hi); w = max(float(np.median(poly.hi - poly.lo)), 1e-15); sc = 1.0 / w; dl = 1e-12
        inc = c[poly.eq] - c[poly.ep]
        P = Poly((poly.lo - c) * sc - dl, (poly.hi - c) * sc + dl, poly.ep, poly.eq, (poly.cm - inc) * sc - dl, (poly.cp - inc) * sc + dl)
        res = linprog(-d, A_ub=P.A, b_ub=P.b, bounds=list(zip(P.lo, P.hi)), method="highs"); val = float(d @ c) + w * (-res.fun)
    else:
        res = linprog(-d, A_ub=poly.A, b_ub=poly.b, bounds=list(zip(poly.lo, poly.hi)), method="highs"); val = -res.fun
    assert res.status == 0, res.message
    y = np.maximum(-res.ineqlin.marginals, 0.0)
    return val, y, verified_upper(poly, d, y)


def rs(q):
    return f"{q.numerator}/{q.denominator}" if q.denominator != 1 else str(q.numerator)


def exact_obj_s1b(e, L, Lp):
    c = [Fraction(0)] * T
    for t in range(e - L, e):
        c[t] += Fraction(1, L)
    for t in range(e - Lp, e):
        c[t] -= Fraction(1, Lp)
    return c


def exact_obj_s5(w, h0, b):
    c = [Fraction(0)] * T
    for dd in range(7):
        s_ = 168 * w + 24 * dd
        for t in range(s_ + h0, s_ + h0 + b):
            c[t] += Fraction(1, 7 * b)
        for t in range(s_ + h0 - b, s_ + h0):
            c[t] -= Fraction(1, 7 * b)
    return c


def export_cert(poly, cF, name, note, cdir, scaled=False):
    """exact-rational LP certificate for max c.phi over Phi (c exact rationals); dual from HiGHS on the float objective"""
    d = np.array([float(v) for v in cF])
    val, y, ver = solve_with_dual(poly, d, scaled)
    E = poly.E; ep, eq = poly.ep, poly.eq
    loF = [Fraction(float(v)) for v in poly.lo]; hiF = [Fraction(float(v)) for v in poly.hi]
    bF = [Fraction(float(v)) for v in poly.cp] + [-Fraction(float(v)) for v in poly.cm]
    yF = [Fraction(float(v)) for v in y]
    r = list(cF)
    for i in np.nonzero(y)[0]:
        i = int(i)
        if i < E:
            r[eq[i]] -= yF[i]; r[ep[i]] += yF[i]
        else:
            r[ep[i - E]] -= yF[i]; r[eq[i - E]] += yF[i]
    rp = [max(v, Fraction(0)) for v in r]; rm = [max(-v, Fraction(0)) for v in r]
    bound = sum((yF[i] * bF[i] for i in np.nonzero(y)[0]), Fraction(0)) + sum((rp[j] * hiF[j] - rm[j] * loF[j] for j in range(T)), Fraction(0))
    rows = [dict(coefs=[[int(eq[k]), "1"], [int(ep[k]), "-1"]], rhs=rs(bF[k])) for k in range(E)] + \
           [dict(coefs=[[int(ep[k]), "1"], [int(eq[k]), "-1"]], rhs=rs(bF[E + k])) for k in range(E)]
    cert = dict(kind="lp", name=name, n=T, rows=rows, obj=[[j, rs(cF[j])] for j in range(T) if cF[j] != 0], lo=[rs(v) for v in loF], hi=[rs(v) for v in hiF],
                cert=dict(y=[rs(v) for v in yF], rp=[rs(v) for v in rp], rm=[rs(v) for v in rm], bound=rs(bound)), note=note)
    path = os.path.join(cdir, name + ".json")
    with open(path, "w") as fh:
        json.dump(cert, fh)
    return dict(name=name, file=path, lp_value=float(val), float_verified=float(ver), exact_bound=float(bound),
                float_verified_ge_exact=bool(ver >= float(bound)), nonzero_y=int(np.count_nonzero(y)), rows=2 * E)


def run_task(args):
    model, eps, source, j, window = args; t0 = time.time()
    Bx = boxes(model, eps, source, window); psi = nominal(model, window)[j]; CDIR = ctx(window)[2]; scaled = window is not None
    if scaled:
        from real_s3_certify import lp_scaled
    poly = build_poly(Bx, j); lo, hi = poly.lo, poly.hi
    feas = poly.feasible(psi, tol=0.0)
    u, l = closure_1d(lo, hi, poly.ep, poly.eq, poly.cm, poly.cp)
    Ph, Plo, Pu, Pl, Pf = pref(hi), pref(lo), pref(u), pref(l), pref(psi)
    nlp = [0]

    def zlp(d):
        nlp[0] += 1
        if scaled:
            return float(lp_scaled(poly, d)[0])
        r = solve_max(poly, d); return float(r["verified"])
    exports = []
    eps_s = str(eps).replace(".", "p"); pre = f"rig_{window}_" if window else "rig_"
    # ---- S1b
    s1b = []
    for ie, e in enumerate(E_REPORT):
        GV = s1b_grid(Ph, Plo, e); GS = s1b_grid(Pu, Pl, e); GN = s1b_grid(Pf, Pf, e)
        order = np.argsort(-GS, axis=None); best_lp = -np.inf; best_pair = None; k = 0; settled = False; top = None
        while k < len(order) and k < LAZY_CAP:
            iL, iLp = np.unravel_index(order[k], GS.shape)
            if GS[iL, iLp] + OUT_MARG <= best_lp:
                settled = True; break
            L, Lp = int(LGRID[iL]), int(LGRID[iLp]); d = s1b_d(int(e), L, Lp); zv = zlp(d)
            if k == 0:
                Vs_ = float(GS[iL, iLp] + OUT_MARG); dpsi = float(d @ psi)
                top = dict(pair=[L, Lp], V=float(GV[iL, iLp] + OUT_MARG), Vs=Vs_, Z=zv, dpsi=dpsi, G=(Vs_ - zv) / (Vs_ - dpsi) if Vs_ - dpsi > 1e-9 else None)
                if source == "rig" and j in EXPORT_FEEDERS and ie in EXPORT_S1B:
                    nm = f"{pre}{model}_e{eps_s}_j{j}_S1b_t{int(e)}_L{L}_Lp{Lp}_U"
                    ex = export_cert(poly, exact_obj_s1b(int(e), L, Lp), nm,
                                     f"{'exp3_ml_real ' + window if window else 'exp3_ml'} rigorous boxes, model {model}, eps {eps}, feeder {j}: S1b look-back pair (L, L') = ({L}, {Lp}) at report time {int(e)}; "
                                     f"upper bound of <phi>_[e-L,e) - <phi>_[e-L',e) over Phi (ZD arcs)", CDIR, scaled)
                    ex.update(stat="S1b", direction="U", Z_reported=zv); exports.append(ex)
            if zv > best_lp:
                best_lp, best_pair = zv, (L, Lp)
            k += 1
        nxt = float(GS.ravel()[order[k]] + OUT_MARG) if (not settled and k < len(order)) else -np.inf
        s1b.append(dict(e=int(e), V=float(GV.max() + OUT_MARG), Vs=float(GS.max() + OUT_MARG), Z=float(max(best_lp, nxt)), settled=bool(settled), n_lp=k,
                        Z_pair=list(best_pair) if best_pair else None, nominal=float(GN.max()), top=top))
    # ---- S5: every choice, both directions
    s5 = []
    for w in S5_WEEKS:
        ch = [(w, h0, b) for h0 in S5_H0 for b in S5_B]
        Ds = np.array([s5_d(*c) for c in ch]); nom = Ds @ psi
        Dp, Dm = np.maximum(Ds, 0), np.maximum(-Ds, 0)
        UV = Dp @ hi - Dm @ lo + OUT_MARG; LV = Dp @ lo - Dm @ hi - OUT_MARG
        US = Dp @ u - Dm @ l + OUT_MARG; LS = Dp @ l - Dm @ u - OUT_MARG
        UZ = np.array([min(US[i], zlp(Ds[i])) for i in range(len(ch))]); LZ = np.array([max(LS[i], -zlp(-Ds[i])) for i in range(len(ch))])
        GU = np.where(US - nom > 1e-9, (US - UZ) / np.maximum(US - nom, 1e-300), np.nan); GL = np.where(nom - LS > 1e-9, (LZ - LS) / np.maximum(nom - LS, 1e-300), np.nan)
        cats = lambda Ub, Lb: dict(all_plus=bool((Lb > 0).all()), all_minus=bool((Ub < 0).all()), n_cert_plus=int((Lb > 0).sum()), n_cert_minus=int((Ub < 0).sum()))
        s5.append(dict(week=w, choices=ch, nominal=nom.tolist(), UV=UV.tolist(), LV=LV.tolist(), US=US.tolist(), LS=LS.tolist(), UZ=UZ.tolist(), LZ=LZ.tolist(),
                       GU=GU.tolist(), GL=GL.tolist(), cat_V=cats(UV, LV), cat_Z=cats(UZ, LZ)))
        if source == "rig" and j in EXPORT_FEEDERS and w in EXPORT_S5_WEEKS:
            h0, b = EXPORT_S5_CHOICE; i = ch.index((w, h0, b)); cF = exact_obj_s5(w, h0, b)
            for dname, sg in (("U", 1), ("L", -1)):
                nm = f"{pre}{model}_e{eps_s}_j{j}_S5_w{w}_h{h0}_b{b}_{dname}"
                ex = export_cert(poly, [sg * v for v in cF], nm,
                                 f"{'exp3_ml_real ' + window if window else 'exp3_ml'} rigorous boxes, model {model}, eps {eps}, feeder {j}: S5 intra-day contrast week {w}, h0 {h0}, b {b}; "
                                 f"{'upper' if sg > 0 else 'lower (negated)'} bound over Phi (ZD arcs)", CDIR, scaled)
                ex.update(stat="S5", direction=dname, Z_reported=float(UZ[i] if sg > 0 else -LZ[i])); exports.append(ex)
    return dict(model=model, eps=eps, source=source, j=j, window=window, psi_in_Phi=bool(feas), S1b=s1b, S5=s5, n_lp=nlp[0], seconds=time.time() - t0, exports=exports)


def med(x):
    x = np.asarray([v for v in x if v is not None], float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def aggregate(tasks, model, eps, feeders, window=None):
    out = {}
    try:
        s3 = json.load(open(os.path.join(ctx(window)[0], f"s3_certify_{model}.json"), encoding="utf-8"))["configs"][f"CROWN|{eps}|ZD"]["per_feeder"]
        s3 = {r["j"]: r for r in s3}
    except Exception as ex_:                                   # stored file absent or being rewritten: no cross-check
        s3 = None; out["stored_s3_certify_error"] = repr(ex_)[:200]
    get = lambda src, j: next(t for t in tasks if t["model"] == model and t["eps"] == eps and t["source"] == src and t["j"] == j)
    for src in ("auto", "rig"):
        S1 = [r for j in feeders for r in get(src, j)["S1b"]]
        S5 = [r for j in feeders for r in get(src, j)["S5"]]
        GU = np.concatenate([r["GU"] for r in S5]); GL = np.concatenate([r["GL"] for r in S5])
        # the s3_certify validation-sample choices (same choices as the stored G records), both directions
        samp = []
        for j in (feeders if s3 is not None else []):
            st = get(src, j)["S5"]
            for inst, rec in zip(s3[j]["S5"], st):
                for c in rec_choices(inst):
                    i = rec["choices"].index(c); samp += [rec["GU"][i], rec["GL"][i]]
        out[src] = dict(S1b_V_median=med([r["V"] for r in S1]), S1b_Z_median=med([r["Z"] for r in S1]), S1b_nominal_median=med([r["nominal"] for r in S1]),
                        S1b_top_G_median=med([r["top"]["G"] for r in S1]), S1b_top_G_p90=float(np.nanpercentile([np.nan if r["top"]["G"] is None else r["top"]["G"] for r in S1], 90)),
                        S1b_settled=int(sum(r["settled"] for r in S1)), S1b_cases=len(S1),
                        S5_G_median_all=med(np.concatenate([GU, GL])), S5_G_median_s3_sample=med(samp),
                        S5_CS_V=int(sum((r["cat_V"]["n_cert_plus"] > 0) and (r["cat_V"]["n_cert_minus"] > 0) for r in S5)),
                        S5_CS_Z=int(sum((r["cat_Z"]["n_cert_plus"] > 0) and (r["cat_Z"]["n_cert_minus"] > 0) for r in S5)),
                        S5_CI_V=int(sum(r["cat_V"]["all_plus"] or r["cat_V"]["all_minus"] for r in S5)),
                        S5_CI_Z=int(sum(r["cat_Z"]["all_plus"] or r["cat_Z"]["all_minus"] for r in S5)), S5_instances=len(S5),
                        psi_in_Phi=int(sum(get(src, j)["psi_in_Phi"] for j in feeders)), n_lp=int(sum(get(src, j)["n_lp"] for j in feeders)),
                        seconds=float(sum(get(src, j)["seconds"] for j in feeders)))
    # paired differences rigorous - auto
    A1 = [r for j in feeders for r in get("auto", j)["S1b"]]; R1 = [r for j in feeders for r in get("rig", j)["S1b"]]
    A5 = [r for j in feeders for r in get("auto", j)["S5"]]; R5 = [r for j in feeders for r in get("rig", j)["S5"]]
    dZ1 = np.array([r["Z"] - a["Z"] for r, a in zip(R1, A1)]); dV1 = np.array([r["V"] - a["V"] for r, a in zip(R1, A1)])
    gnan = lambda g: np.nan if g is None else g
    dG1 = np.array([gnan(r["top"]["G"]) - gnan(a["top"]["G"]) for r, a in zip(R1, A1)])
    dUZ = np.concatenate([np.array(r["UZ"]) - np.array(a["UZ"]) for r, a in zip(R5, A5)]); dLZ = np.concatenate([np.array(r["LZ"]) - np.array(a["LZ"]) for r, a in zip(R5, A5)])
    dUV = np.concatenate([np.array(r["UV"]) - np.array(a["UV"]) for r, a in zip(R5, A5)])
    dG5 = np.concatenate([np.concatenate([r["GU"], r["GL"]]) - np.concatenate([a["GU"], a["GL"]]) for r, a in zip(R5, A5)])
    S5range = np.concatenate([np.array(a["UZ"]) - np.array(a["LZ"]) for a in A5])
    out["rig_minus_auto"] = dict(S1b_Z=dict(median=float(np.median(dZ1)), min=float(dZ1.min()), max=float(dZ1.max()), frac_rig_le_auto=float(np.mean(dZ1 <= 0))),
                                 S1b_V=dict(median=float(np.median(dV1)), min=float(dV1.min()), max=float(dV1.max())),
                                 S1b_top_G=dict(median=float(np.nanmedian(dG1)), min=float(np.nanmin(dG1)), max=float(np.nanmax(dG1))),
                                 S5_UZ=dict(median=float(np.median(dUZ)), min=float(dUZ.min()), max=float(dUZ.max()), frac_rig_le_auto=float(np.mean(dUZ <= 0))),
                                 S5_LZ=dict(median=float(np.median(dLZ)), min=float(dLZ.min()), max=float(dLZ.max()), frac_rig_ge_auto=float(np.mean(dLZ >= 0))),
                                 S5_UV=dict(median=float(np.median(dUV)), min=float(dUV.min()), max=float(dUV.max())),
                                 S5_G=dict(median=float(np.nanmedian(dG5)), min=float(np.nanmin(dG5)), max=float(np.nanmax(dG5))),
                                 S5_relative_Z_change_median=float(np.median(np.abs(np.concatenate([dUZ, dLZ])) / np.concatenate([S5range, S5range]))))
    # cross-check of the auto-box recomputation against the stored s3_certify records (same feeders)
    if s3 is not None:
        st_G1 = [r["G"] for j in feeders for r in s3[j]["G_records"] if r["tag"] == "S1b_top"]
        st_G5 = [r["G"] for j in feeders for r in s3[j]["G_records"] if r["tag"] == "S5_sample"]
        out["stored_s3_certify_same_feeders"] = dict(S1b_top_G_median=med(st_G1), S5_sample_G_median=med(st_G5),
                                                     S1b_Z_median=med([r["Z"] for j in feeders for r in s3[j]["S1b"]]))
    return out


def rec_choices(inst):
    return [tuple(c) for c in inst["choices_sample"]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="mlp,feat,feat24"); ap.add_argument("--eps", default="0.01,0.05")
    ap.add_argument("--feeders", default="0-11"); ap.add_argument("--procs", type=int, default=12); ap.add_argument("--no_lean", action="store_true")
    ap.add_argument("--window", default=None); a = ap.parse_args()
    SRC, RDIR, CDIR, TAG = ctx(a.window); WS = f"_{a.window}" if a.window else ""
    os.makedirs(CDIR, exist_ok=True)
    f0, f1 = (int(x) for x in a.feeders.split("-")); feeders = list(range(f0, f1 + 1))
    models = a.models.split(","); eps_list = [float(x) for x in a.eps.split(",")]
    tasks = [(m, e, s, j, a.window) for m in models for e in eps_list for s in ("rig", "auto") for j in feeders]
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.procs) as pool:
        res = []
        for r in pool.imap_unordered(run_task, tasks, chunksize=1):
            res.append(r); print(f"  task {r['model']} {r['eps']} {r['source']} j{r['j']}: {r['seconds']:.0f}s n_lp {r['n_lp']} psi_in_Phi {r['psi_in_Phi']}", flush=True)
    jdump(res, os.path.join(RDIR, f"r2_tasks{WS}.json"))
    summary = dict(declared=dict(window=a.window, source_dir=SRC, lp="lp_scaled (real_s3_certify.py)" if a.window else "transport_core.solve_max", feeders=feeders, arcset="ZD", lags=LAGS_DYADIC, LAZY_CAP=LAZY_CAP, OUT_MARG=OUT_MARG,
                                 export=dict(feeders=EXPORT_FEEDERS, S1b_report_indices=EXPORT_S1B, S5_weeks=EXPORT_S5_WEEKS, S5_choice=EXPORT_S5_CHOICE)),
                   wall_seconds=time.time() - t0, results={})
    for m in models:
        for e in eps_list:
            summary["results"][f"{m}|{e}"] = ag = aggregate(res, m, e, feeders, a.window)
            print(m, e, json.dumps(ag, indent=None)[:1500], flush=True)
    # ---- Lean checker
    exports = [x for r in res for x in r["exports"]]
    lean = []
    if not a.no_lean:
        for x in sorted(exports, key=lambda z: z["name"]):
            t1 = time.time(); p = subprocess.run([LEAN, x["file"]], capture_output=True, text=True)
            first = (p.stdout.strip().splitlines() or [""])[0]
            lean.append(dict(name=x["name"], exit_code=p.returncode, verdict=first.split(" ")[0], output=p.stdout.strip(), seconds=time.time() - t1,
                             exact_bound=x["exact_bound"], float_verified=x["float_verified"], Z_reported=x["Z_reported"],
                             float_verified_ge_exact=x["float_verified_ge_exact"], rows=x["rows"], nonzero_y=x["nonzero_y"],
                             file_bytes=os.path.getsize(x["file"])))
            print(f"  lean {x['name']}: {first[:90]} exit={p.returncode}", flush=True)
        jdump(lean, os.path.join(RDIR, f"lean_results{WS}.json"))
        summary["lean"] = dict(n=len(lean), n_pass=int(sum(l["verdict"] == "PASS" and l["exit_code"] == 0 for l in lean)),
                               n_fail=int(sum(l["verdict"] != "PASS" or l["exit_code"] != 0 for l in lean)),
                               all_float_verified_ge_exact=bool(all(l["float_verified_ge_exact"] for l in lean)),
                               max_exact_minus_lp_value=float(max(x["exact_bound"] - x["lp_value"] for x in exports)) if exports else None,
                               seconds=float(sum(l["seconds"] for l in lean)), total_bytes=int(sum(l["file_bytes"] for l in lean)))
    summary["exports"] = exports
    jdump(summary, os.path.join(RDIR, f"r2_summary{WS}.json"))
    print("done", f"{time.time() - t0:.0f}s", summary.get("lean"), flush=True)


if __name__ == "__main__":
    main()
