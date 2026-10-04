"""s2_verify.py -- local enclosures of the model from the standard verifier (auto_LiRPA 0.7.2): value boxes [lo_t, hi_t] of f(t) and
increment boxes G_{t,k} of f(t+k) - f(t) (two-window model Incr, shared inputs) for k in LAGS_DYADIC, under the declared series box
y_s in [y~_s - eps, y~_s + eps].
Phase 'crown' (float64, CPU): IBP and CROWN (backward); boxes widened outward by MARGIN = 1e-9 (auto_LiRPA has no directed rounding;
  float64 accumulation error here is < 1e-12).  Also (i) a soundness spot-check with 64 random admissible SERIES per eps (32 random
  corners, 32 uniform), and (ii) per-box inner values (PGD on each window / window pair, float32 ascent, float64 re-evaluation) to
  measure the verifier's own tightness.
Phase 'alpha' (float32, GPU): alpha-CROWN (CROWN-Optimized, 20 iterations, lr 0.1); boxes widened by ALPHA_MARGIN = 1e-5 (float32 is
  NOT a rigorous floating-point enclosure; the margin is ~100x the float32 accumulation error observed against the float64 CROWN boxes)
  and then intersected with the float64 CROWN boxes.  float64 alpha-CROWN on this GPU took > 16 min per radius and was abandoned;
  float32 alpha-CROWN takes ~14 min per radius and model, so it was run for the model where increments matter ('feat24') only.
Usage: python s2_verify.py --model mlp --phase crown ; python s2_verify.py --model mlp --phase alpha
       (writes exp3_ml/boxes_{model}.npz, s2_verify_{model}.json)"""
import os, sys, time, argparse, json, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mlp"); ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--phase", default="crown"); ap.add_argument("--eps", default=None); a = ap.parse_args()
torch.set_default_dtype(torch.float64 if a.phase == "crown" else torch.float32)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlt_common import *
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

MARGIN = 1e-9; ALPHA_MARGIN = 1e-5
torch.manual_seed(0)
Fc = load_model(a.model).double()                                   # float64 CPU copy (IBP, CROWN, re-evaluation of witnesses)
F32 = load_model(a.model).float().to(DEV)                           # float32 GPU copy (alpha-CROWN, PGD ascent)
nom = np.load(os.path.join(OUT, f"nominal_{a.model}.npz")); Y = nom["y"].astype(np.float64)          # [N_ENT, T + W - 1]
METHODS = {"IBP": "IBP", "CROWN": "backward", "alpha-CROWN": "CROWN-Optimized"}


def joint(Y, k):
    """joint inputs for lag k: rows (j, t) = Y[j, t : t + W + k], t = 0..T-k-1"""
    return torch.as_tensor(Y).unfold(-1, W + k, 1).reshape(-1, W + k)


def bound(net, X, eps, method):
    if method == "alpha-CROWN":
        dev, X, mg = DEV, X.float(), ALPHA_MARGIN
    else:
        dev, mg = "cpu", MARGIN
    bm = BoundedModule(net, X[:1].to(dev), device=dev, verbose=0,
                       bound_opts={"optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}, "verbosity": 0})
    los, his = [], []
    for i in range(0, len(X), a.chunk):
        xc = X[i:i + a.chunk].to(dev)
        xb = BoundedTensor(xc, PerturbationLpNorm(norm=np.inf, x_L=xc - eps, x_U=xc + eps))
        lb, ub = bm.compute_bounds(x=(xb,), method=METHODS[method])
        los.append(lb.detach().cpu().double().numpy().ravel()); his.append(ub.detach().cpu().double().numpy().ravel())
    return np.concatenate(los) - mg, np.concatenate(his) + mg


def series_f(Ys):
    """float64 model outputs for a batch of perturbed series Ys [B, N_ENT, T + W - 1] -> [B, N_ENT, T]"""
    out_ = []
    with torch.no_grad():
        for Yb in Ys:
            Xw = torch.as_tensor(Yb).unfold(-1, W, 1); out_.append(Fc(Xw.reshape(-1, W)).reshape(Xw.shape[:-1]).numpy())
    return np.array(out_)


def pgd_box(net32, net64, X, eps, sign, steps=60):
    """per-row inner value: max (sign=+1) or min (sign=-1) of the net over the row's own l_inf box (sign-gradient ascent in float32
    on the GPU, 2 starts; the final perturbation is clipped to the box in float64 and re-evaluated in float64 on the CPU)"""
    best = None
    for start in ("centre", "random"):
        Xd = X.float().to(DEV); d = torch.zeros_like(Xd) if start == "centre" else (torch.rand_like(Xd) * 2 - 1) * eps
        for s in range(steps):
            d.requires_grad_(True); v = sign * net32(Xd + d).sum(); g, = torch.autograd.grad(v, d)
            with torch.no_grad():
                d = (d + eps / 6 * g.sign()).clamp(-eps, eps)
        d64 = torch.clamp(d.detach().cpu().double(), -eps, eps)
        with torch.no_grad():
            v = net64(X.double() + d64).ravel().numpy()
        best = v if best is None else (np.maximum(best, v) if sign > 0 else np.minimum(best, v))
    return best


def stats(key, lo, hi, out, t0):
    vw = hi - lo
    return dict(seconds=time.time() - t0, value_width_median=float(np.median(vw)), value_width_mean=float(vw.mean()),
                incr_width_median={k: float(np.median(out[f"ghi|{key}|{k}"] - out[f"glo|{key}|{k}"])) for k in LAGS_DYADIC},
                implied_incr1_width_median=float(np.median(hi[:, 1:] - lo[:, :-1] - (lo[:, 1:] - hi[:, :-1]))),
                kappa1_median=float(np.median((out[f"ghi|{key}|1"] - out[f"glo|{key}|1"]) / np.maximum(vw[:, :-1], 1e-12))),
                kappa_median={k: float(np.median((out[f"ghi|{key}|{k}"] - out[f"glo|{key}|{k}"]) / np.maximum(vw[:, :-k], 1e-12))) for k in LAGS_DYADIC})


def check(eps, out, meths):
    """soundness spot-check (random admissible series) and verifier-over-inner widths, for the listed methods"""
    viol = {}
    for meth in meths:
        key = f"{meth}|{eps}"; v = max(float(np.max(FS[eps] - out[f"hi|{key}"][None])), float(np.max(out[f"lo|{key}"][None] - FS[eps])))
        for k in LAGS_DYADIC:
            inc = FS[eps][:, :, k:] - FS[eps][:, :, :-k]
            v = max(v, float(np.max(inc - out[f"ghi|{key}|{k}"][None])), float(np.max(out[f"glo|{key}|{k}"][None] - inc)))
        viol[meth] = v
    hi_in, lo_in, ghi_in, glo_in = (out[f"{n}|{eps}" + ("|1" if n.startswith("g") else "")] for n in ("hi_in", "lo_in", "ghi_in", "glo_in"))
    tight = {}
    for meth in meths:
        key = f"{meth}|{eps}"
        tight[meth] = dict(value_width_ratio_median=float(np.median((out[f"hi|{key}"] - out[f"lo|{key}"]) / np.maximum(hi_in - lo_in, 1e-12))),
                           incr1_width_ratio_median=float(np.median((out[f"ghi|{key}|1"] - out[f"glo|{key}|1"]) / np.maximum(ghi_in - glo_in, 1e-12))),
                           inner_inside_box=bool(np.all(hi_in <= out[f"hi|{key}"]) and np.all(lo_in >= out[f"lo|{key}"])
                                                 and np.all(ghi_in <= out[f"ghi|{key}|1"]) and np.all(glo_in >= out[f"glo|{key}|1"])))
    return viol, tight


BOXF = os.path.join(OUT, f"boxes_{a.model}.npz"); JF = os.path.join(OUT, f"s2_verify_{a.model}.json")
FS = {}
if a.phase == "crown":
    res = dict(model=a.model, eps_grid=EPS_GRID, lags=LAGS_DYADIC, margin=MARGIN, alpha_margin=ALPHA_MARGIN, runs={}); out = {}
    for eps in EPS_GRID:
        for meth in ("IBP", "CROWN"):
            t0 = time.time(); key = f"{meth}|{eps}"
            lo, hi = bound(Fc, joint(Y, 0), eps, meth); lo = lo.reshape(N_ENT, T); hi = hi.reshape(N_ENT, T)
            out[f"lo|{key}"] = lo; out[f"hi|{key}"] = hi
            for k in LAGS_DYADIC:
                gl, gh = bound(Incr(Fc, k), joint(Y, k), eps, meth)
                out[f"glo|{key}|{k}"] = gl.reshape(N_ENT, T - k); out[f"ghi|{key}|{k}"] = gh.reshape(N_ENT, T - k)
            res["runs"][key] = stats(key, lo, hi, out, t0); print(key, res["runs"][key], flush=True)
        B = 64; rng = np.random.default_rng(1)
        Ys = Y[None] + eps * np.where(rng.random((B,) + Y.shape) < 0.5, -1.0, 1.0); Ys[B // 2:] = Y[None] + eps * (2 * rng.random((B // 2,) + Y.shape) - 1)
        FS[eps] = series_f(Ys)
        Xv = joint(Y, 0); out[f"hi_in|{eps}"] = pgd_box(F32, Fc, Xv, eps, +1).reshape(N_ENT, T); out[f"lo_in|{eps}"] = pgd_box(F32, Fc, Xv, eps, -1).reshape(N_ENT, T)
        X1 = joint(Y, 1)
        out[f"ghi_in|{eps}|1"] = pgd_box(Incr(F32, 1), Incr(Fc, 1), X1, eps, +1).reshape(N_ENT, T - 1)
        out[f"glo_in|{eps}|1"] = pgd_box(Incr(F32, 1), Incr(Fc, 1), X1, eps, -1).reshape(N_ENT, T - 1)
        viol, tight = check(eps, out, ("IBP", "CROWN"))
        res[f"check|{eps}"] = dict(max_violation_random_series=viol, inner_value_width_median=float(np.median(out[f"hi_in|{eps}"] - out[f"lo_in|{eps}"])),
                                   inner_incr1_width_median=float(np.median(out[f"ghi_in|{eps}|1"] - out[f"glo_in|{eps}|1"])), verifier_over_inner=tight)
        print(eps, res[f"check|{eps}"], flush=True)
        np.savez_compressed(os.path.join(OUT, f"randseries_{a.model}_{eps}.npz"), fs=FS[eps].astype(np.float64))
else:
    out = dict(np.load(BOXF)); res = json.load(open(JF, encoding="utf-8"))
    for eps in ([float(x) for x in a.eps.split(",")] if a.eps else EPS_GRID):
        t0 = time.time(); key = f"alpha-CROWN|{eps}"; ck = f"CROWN|{eps}"
        lo, hi = bound(F32, joint(Y, 0), eps, "alpha-CROWN")
        lo = np.maximum(lo.reshape(N_ENT, T), out[f"lo|{ck}"]); hi = np.minimum(hi.reshape(N_ENT, T), out[f"hi|{ck}"])
        out[f"lo|{key}"] = lo; out[f"hi|{key}"] = hi
        for k in LAGS_DYADIC:
            gl, gh = bound(Incr(F32, k), joint(Y, k), eps, "alpha-CROWN")
            out[f"glo|{key}|{k}"] = np.maximum(gl.reshape(N_ENT, T - k), out[f"glo|{ck}|{k}"]); out[f"ghi|{key}|{k}"] = np.minimum(gh.reshape(N_ENT, T - k), out[f"ghi|{ck}|{k}"])
        res["runs"][key] = stats(key, lo, hi, out, t0)
        FS[eps] = np.load(os.path.join(OUT, f"randseries_{a.model}_{eps}.npz"))["fs"]
        viol, tight = check(eps, out, ("alpha-CROWN",))
        res[f"check|{eps}"]["max_violation_random_series"]["alpha-CROWN"] = viol["alpha-CROWN"]; res[f"check|{eps}"]["verifier_over_inner"]["alpha-CROWN"] = tight["alpha-CROWN"]
        print(key, res["runs"][key], viol, tight, flush=True)
        np.savez_compressed(BOXF, **out); jdump(res, JF)                     # save after every radius (long phase)
np.savez_compressed(BOXF, **out)
jdump(res, JF)
