"""real_s3_certify.py -- s3_certify (unchanged functions: run_task, ranking, closure, classes S1a-S5) on the real data of one
evaluation window; the output directory is routed to exp3_ml_real/<window> (also inside the worker processes).
One numerical-conditioning change, declared: the LP h_Phi(d) is solved on the CENTRED AND SCALED polytope xi = (phi - c)/w with
c = (lo + hi)/2 and w = median(hi - lo) (an exact affine reformulation, widened outward by 1e-12 in scaled units), and mapped back
as h = d.c + w * h_xi with outward rounding.  Reason: in the winter window most outputs are ~1e-4 probabilities, the polytope's
slacks (~6e-8) fall below HiGHS's absolute feasibility tolerance (1e-7) and HiGHS declared feasible polytopes infeasible (the
nominal output is feasible).  A failed LP returns +inf (sound) and is counted (lp_failures).
Usage: python real_s3_certify.py --window jul2014 --models mlp,feat,feat24 [--methods CROWN] [--procs 18]"""
import os, sys, time, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import real_common, mlt_common, s3_certify as S3
from transport_core import Poly, solve_max
U_ = np.finfo(float).eps / 2
FAIL = [0]


def lp_scaled(poly, d):
    c = 0.5 * (poly.lo + poly.hi); w = max(float(np.median(poly.hi - poly.lo)), 1e-15); s = 1.0 / w; dl = 1e-12
    inc = c[poly.eq] - c[poly.ep]
    P = Poly((poly.lo - c) * s - dl, (poly.hi - c) * s + dl, poly.ep, poly.eq, (poly.cm - inc) * s - dl, (poly.cp - inc) * s + dl)
    r = solve_max(P, d)
    if r["status"] != 0 or not np.isfinite(r["verified"]):
        FAIL[0] += 1; return float("inf"), float("nan")
    dc = float(d @ c); edc = (len(d) + 2) * U_ * float(np.abs(d * c).sum())
    v = float(r["verified"]) * w; ver = dc + edc + v + 4 * U_ * abs(v) + 4 * U_ * abs(dc + v)
    return ver, dc + float(r["value"]) * w


def init(models, window):
    out = real_common.use_real_out(window); S3.OUT = out; S3.init_worker(models); S3.lp = lp_scaled


def run_task_counted(args):
    FAIL[0] = 0; r = S3.run_task(args); r["lp_failures"] = FAIL[0]; return r


if __name__ == "__main__":
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(); ap.add_argument("--window", required=True); ap.add_argument("--models", default="mlp,feat,feat24")
    ap.add_argument("--methods", default="CROWN"); ap.add_argument("--procs", type=int, default=18); a = ap.parse_args()
    METH = a.methods.split(",")
    for model in a.models.split(","):
        t0 = time.time(); init([model], a.window); B, Fn = S3.G_BOX[model]; OUT = S3.OUT
        tasks = [(model, eps, meth, arc, j) for eps in mlt_common.EPS_GRID for meth in METH for arc in S3.ARCSETS for j in range(mlt_common.N_ENT)]
        with Pool(a.procs, initializer=init, initargs=([model], a.window)) as pool:
            results = pool.map(run_task_counted, tasks, chunksize=1)
        summary = dict(model=model, window=a.window, declared=dict(LGRID=[24, 168], E_REPORT=S3.E_REPORT, S2_TAU0=S3.S2_TAU0, S2_L=S3.S2_L, S2_DT=[-24, 24],
                                                  S3_BASE=S3.S3_BASE, S3_K=S3.S3_K, S3_SPAN=S3.S3_SPAN, S3_DS=[0, 24], N_SAMPLE=S3.N_SAMPLE, LAZY_CAP=S3.LAZY_CAP,
                                                  S5=dict(WEEKS=S3.S5_WEEKS, H0=S3.S5_H0, B=S3.S5_B), OUT_MARG=S3.OUT_MARG, CL_MARG=S3.CL_MARG), configs={})
        for eps in mlt_common.EPS_GRID:                              # identical assembly to s3_certify's main
            for meth in ["IBP"] + METH:
                key = f"{meth}|{eps}"
                summary["configs"][f"{key}|V"] = dict(S4=S3.ranking(B[f"lo|{key}"].astype(float), B[f"hi|{key}"].astype(float), Fn))
        for eps in mlt_common.EPS_GRID:
            for meth in METH:
                for arc in S3.ARCSETS:
                    rs = sorted([r for r in results if r["eps"] == eps and r["meth"] == meth and r["arcset"] == arc], key=lambda r: r["j"])
                    U = np.array([r["u"] for r in rs]); Lw = np.array([r["l"] for r in rs])
                    summary["configs"][f"{meth}|{eps}|{arc}"] = dict(per_feeder=[{k: v for k, v in r.items() if k not in ("u", "l")} for r in rs], S4=S3.ranking(Lw, U, Fn))
                    np.savez_compressed(os.path.join(OUT, f"closure_{model}_{meth}_{eps}_{arc}.npz"), u=U, l=Lw)
        summary["lp_failures_total"] = int(sum(r["lp_failures"] for r in results)); summary["seconds"] = time.time() - t0
        mlt_common.jdump(summary, os.path.join(OUT, f"s3_certify_{model}.json")); print(model, a.window, "done", f"{time.time()-t0:.0f}s", flush=True)
