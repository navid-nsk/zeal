"""cert_run_all.py -- stream EVERY C5 map endpoint (all pairs of all ladders, both directions) through the Lean certificate pipeline:
exact-rational certificate.json (as in cert_export.py) -> compiled verified checker `certcheck.exe` (native evaluation of
checkLPFast, whose soundness is kernel-checked) -> verdict log.  Certificate files are deleted after checking except for a kept
sample (--keep_every); the log records, per endpoint, the certificate's SHA-256, sizes, the exact bound, the producer's reported
endpoint, whether the reported endpoint is certified (>= exact bound in the max convention), the checker verdict and timings.
Usage: python cert_run_all.py --out <dir> [--fields georgia,gm_q4,gm_bad,mx_rwi] [--procs 5] [--keep_every 200]
"""
try:
    from . import paths                     # imported as part of the package
except ImportError:
    import paths                            # flat import (this directory on sys.path)
import argparse, hashlib, io, json, os, subprocess, sys, time, numpy as np, scipy.sparse as sp
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
EXP = paths.results("exp2")
LEAN = paths.LEAN_PROJECT; CHECK = paths.CERTCHECK
src = open(os.path.join(HERE, "check_c5.py"), encoding="utf-8").read()
exec(src[src.index("def lifted_system("):src.index("def dual_bound_outward(")])
F_ = lambda x: Fraction(float(x))
def rs(q): return f"{q.numerator}/{q.denominator}" if q.denominator != 1 else str(q.numerator)
G = {}


def init(field):
    from synth import SPEC
    M = 2 if field == "mx_rwi" else 4
    F = np.load(paths.enclosure(field)); lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
    g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
    ras = np.load(SPEC[field]); mask0 = ras["mask"].astype(bool) & np.isfinite(lo0); N = mask0.shape[0]; h = 2.0 / (N - 1)
    w0 = np.where(mask0, ras["weight"], 0.0).astype(float); fine0 = np.where(mask0, ras["tract_idx"], -1); coarse0 = np.where(mask0, ras["county_idx"], -1)
    globals()['h'] = h            # lifted_system reads the grid spacing from the module globals
    G.update(dict(field=field, M=M, h=h, lo0=lo0.T.copy(), hi0=hi0.T.copy(), g1l0=g1l0.T.copy(), g1h0=g1h0.T.copy(), g2l0=g2l0.T.copy(), g2h0=g2h0.T.copy(),
                  mask0=mask0.T.copy(), w0=w0.T.copy(), fine0=fine0.T.copy(), coarse0=coarse0.T.copy(), ART=os.path.join(EXP, "C5_duals")))


def one_pair(args):
    r, tmpdir, keep = args; field, M = G["field"], G["M"]; t0 = time.time()
    i0, i1, j0, j1 = r["window"]; sl = (slice(i0, i1), slice(j0, j1)); mk = (G["coarse0"] == r["coarse"])[sl]
    A, bvec, lo_n, hi_n, nid, (tail, head, glo, ghi, hs) = lifted_system(G["lo0"][sl], G["hi0"][sl], G["g1l0"][sl], G["g1h0"][sl], G["g2l0"][sl], G["g2h0"][sl], mk, M)
    R = len(tail); hsF = F_(hs); bF = [hsF * F_(ghi[k]) for k in range(R)] + [-(hsF * F_(glo[k])) for k in range(R)]
    wb = G["w0"][sl] * mk; wa = wb * (G["fine0"][sl] == r["fine"]); d2 = wa / wa.sum() - wb / wb.sum(); c = outer_objective(d2, mk, nid, M)
    du = np.load(os.path.join(G["ART"], f"duals_{field}_M{M}", r["dual_file"])); loF = [F_(v) for v in lo_n]; hiF = [F_(v) for v in hi_n]
    rows = [dict(coefs=[[int(head[k]), "1"], [int(tail[k]), "-1"]], rhs=rs(bF[k])) for k in range(R)] + [dict(coefs=[[int(tail[k]), "1"], [int(head[k]), "-1"]], rhs=rs(bF[R + k])) for k in range(R)]
    out = []
    for nm, sgn in (("U", 1), ("L", -1)):
        cF = [F_(sgn * v) for v in c]; y = np.zeros(A.shape[0]); y[du[f"{nm}_idx"]] = du[f"{nm}_val"]; yF = [F_(v) for v in y]
        rr = list(cF)
        for k in np.nonzero(y)[0]:
            k = int(k)
            if k < R:
                rr[head[k]] -= yF[k]; rr[tail[k]] += yF[k]
            else:
                kk = k - R; rr[tail[kk]] -= yF[k]; rr[head[kk]] += yF[k]
        rp = [max(v, Fraction(0)) for v in rr]; rm = [max(-v, Fraction(0)) for v in rr]
        bound = sum((yF[k] * bF[k] for k in np.nonzero(y)[0]), Fraction(0)) + sum((rp[j] * hiF[j] - rm[j] * loF[j] for j in range(len(rr)) if rr[j] != 0), Fraction(0))
        name = f"c5_{field}_M{M}_c{r['coarse']}_f{r['fine']}_{nm}"
        cert = dict(kind="lp", name=name, n=int(A.shape[1]), rows=rows, obj=[[j, rs(cF[j])] for j in range(len(cF)) if cF[j] != 0], lo=[rs(v) for v in loF], hi=[rs(v) for v in hiF],
                    cert=dict(y=[rs(v) for v in yF], rp=[rs(v) for v in rp], rm=[rs(v) for v in rm], bound=rs(bound)))
        data = json.dumps(cert).encode(); sha = hashlib.sha256(data).hexdigest(); path = os.path.join(tmpdir, name + ".json"); open(path, "wb").write(data)
        t1 = time.time(); pr = subprocess.run([CHECK, path], capture_output=True, text=True); t2 = time.time()
        verdict = [l for l in pr.stdout.splitlines() if l.startswith("PASS") or l.startswith("FAIL")]
        art_v = r[f"{nm}_X"]; certified = (art_v >= sgn * float(bound)) if sgn == 1 else (art_v <= sgn * float(bound))
        out.append(dict(name=name, field=field, coarse=r["coarse"], fine=r["fine"], dir=nm, n=int(A.shape[1]), rows=int(2 * R), nnz_y=int(len(np.nonzero(y)[0])), sha256=sha, bytes=len(data),
                        bound_exact=float(sgn * bound), bound_exact_rational=rs(bound), artifact=art_v, artifact_certified=bool(certified), verdict=(verdict[0][:60] if verdict else f"NO VERDICT rc={pr.returncode}"),
                        passed=bool(verdict and verdict[0].startswith("PASS") and pr.returncode == 0), export_s=round(t1 - t0, 2), check_s=round(t2 - t1, 2)))
        if not keep:
            os.remove(path)
    return out


if __name__ == "__main__":
    import multiprocessing as mpc
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--fields", default="georgia,gm_q4,gm_bad,mx_rwi"); p.add_argument("--procs", type=int, default=5)
    p.add_argument("--keep_every", type=int, default=200); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8"); t_all = time.time()
    for field in a.fields.split(","):
        art = json.load(open(os.path.join(EXP, "C5_duals", f"c5_{field}.json"), encoding="utf-8"))["pairs"]
        tmp = os.path.join(a.out, "kept", field); os.makedirs(tmp, exist_ok=True); tasks = [(r, tmp, (k % a.keep_every == 0)) for k, r in enumerate(art)]
        log = []; t0 = time.time()
        with mpc.Pool(a.procs, initializer=init, initargs=(field,)) as P:
            for k, rows in enumerate(P.imap_unordered(one_pair, tasks)):
                log.extend(rows)
                if (k + 1) % 50 == 0:
                    npass = sum(x["passed"] for x in log); ncert = sum(x["artifact_certified"] for x in log)
                    print(f"  {field}: {k+1}/{len(tasks)} pairs, {len(log)} endpoints, PASS {npass}, artifact certified {ncert} ({time.time()-t0:.0f}s)", flush=True)
                    json.dump(log, open(os.path.join(a.out, f"lean_c5_log_{field}.json"), "w"), indent=0)
        json.dump(log, open(os.path.join(a.out, f"lean_c5_log_{field}.json"), "w"), indent=0)
        npass = sum(x["passed"] for x in log); ncert = sum(x["artifact_certified"] for x in log)
        print(f"{field}: {len(log)} endpoints — PASS {npass}, FAIL {len(log)-npass}; producer endpoint certified by the exact bound: {ncert}; {time.time()-t0:.0f}s", flush=True)
    print(f"total {time.time()-t_all:.0f}s")
