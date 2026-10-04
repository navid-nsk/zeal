"""r1_rigorous_boxes.py -- rigorous floating-point boxes (rigorous_enclosure.py: IBP and CROWN with directed rounding, independent of
auto_LiRPA) for the exp3_ml models at eps in {0.01, 0.05}, ALL 12 feeders x 1008 outputs: value boxes [lo_t, hi_t] and increment boxes
of f(t+k) - f(t) for k in LAGS_DYADIC; comparison with the auto_LiRPA boxes of s2_verify.py (boxes_<model>.npz, float64 + 1e-9 margin).

Comparison per box type (values; each lag) and method (CROWN; IBP), with [rl, rh] = rigorous box, [al, ah] = stored auto_LiRPA box
(margin included) and [al + m, ah - m] its raw float box (m = 1e-9 removed):
  proven_sound   : rigorous box inside the stored auto_LiRPA box  => the stored box contains a rigorous enclosure, so it is SOUND
  raw_proven     : rigorous box inside the raw float box          => the float box would have been provably sound without the margin
  margin_needed  : outputs where only the margin makes the stored box provably sound (raw_proven false, proven_sound true)
  not_proven     : outputs where the stored box does not contain the rigorous box (no proof either way from this enclosure)
  auto_in_rig    : stored box inside the rigorous box
  width ratio    : (rh - rl) / (ah - al)  (median, min, max)
  excess         : max over outputs of max(al - rl, rh - ah) (> 0 = rigorous endpoint beyond the stored box) and the same vs the raw box
Diagnostics (not part of the enclosure): (i) a fresh float64 auto_LiRPA CROWN and IBP run on the LOGIT network (same algorithm
before the sigmoid; isolates rounding); (ii) the stored random admissible series (randseries_*.npz, 64 per eps) and PGD inner values
must lie inside the rigorous boxes; (iii) the float64 nominal logit lies inside the rigorous logit box.
Remaining outputs: where the stored box does not contain the rigorous box, prove_remaining() evaluates the rigorous bound at
auto_LiRPA's OWN sigmoid slopes for that joint window (the rigorous engine is valid for any slopes; it recomputes the intercepts).
Usage: python r1_rigorous_boxes.py [--models mlp,feat,feat24] [--eps 0.01,0.05] [--no_lirpa] [--window jul2014|jan2014] [--prove_only]
       synthetic (default): exp3_ml/            -> exp3_ml/rigorous/rigorous_boxes_<model>_<eps>.npz, r1_summary.json
       real (--window w)  : exp3_ml_real/<w>/   -> exp3_ml_real/rigorous/rigorous_boxes_<w>_<model>_<eps>.npz, r1_summary_<w>.json
       (the real-data models, nominal series and boxes come from exp3_ml_real/<w>, written by real_run.py / s2_verify.py)
"""
import os, sys, time, json, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mlt_common import OUT, W, T, N_ENT, LAGS_DYADIC, jdump
import rigorous_enclosure as RE

MARGIN = 1e-9
REAL_OUT = os.path.abspath(os.path.join(OUT, "..", "exp3_ml_real"))


def paths(window):
    """(source dir, result dir, file tag) for the synthetic run (window None) or a real-data window"""
    if window is None:
        src, rdir, tag = OUT, os.path.join(OUT, "rigorous"), ""
    else:
        src, rdir, tag = os.path.join(REAL_OUT, window), os.path.join(REAL_OUT, "rigorous"), f"{window}_"
    os.makedirs(rdir, exist_ok=True)
    return src, rdir, tag


def compare(rl, rh, al, ah, margin=MARGIN):
    rl, rh, al, ah = (np.asarray(x, float).ravel() for x in (rl, rh, al, ah))
    ins = (rl >= al) & (rh <= ah); raw = (rl >= al + margin) & (rh <= ah - margin); a_in = (al >= rl) & (ah <= rh)
    wr = (rh - rl) / np.maximum(ah - al, 1e-300)
    return dict(n=int(rl.size), proven_sound=int(ins.sum()), raw_proven=int(raw.sum()), margin_needed=int((ins & ~raw).sum()),
                not_proven=int((~ins).sum()), auto_in_rig=int(a_in.sum()),
                width_ratio_median=float(np.median(wr)), width_ratio_min=float(wr.min()), width_ratio_max=float(wr.max()),
                excess_max=float(max(np.max(al - rl), np.max(rh - ah))), excess_raw_max=float(max(np.max(al + margin - rl), np.max(rh - ah + margin))),
                endpoint_absdiff_median=float(np.median(np.concatenate([np.abs(rl - al), np.abs(rh - ah)]))))


def lirpa_logit(model, Yw, eps, src):
    """float64 auto_LiRPA CROWN ('backward') and IBP bounds of the LOGIT network (diagnostic only)"""
    import torch
    torch.set_default_dtype(torch.float64)
    from mlt_common import build

    def load_model(name):
        m = build(name); m.load_state_dict(torch.load(os.path.join(src, f"model_{name}.pt"), map_location="cpu")); m.eval(); return m
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    class Logit(torch.nn.Module):
        def __init__(s, F):
            super().__init__(); s.F = F

        def forward(s, x):
            return s.F.logit(x)
    F = load_model(model).double(); X = torch.as_tensor(Yw)
    bm = BoundedModule(Logit(F), X[:1], verbose=0)
    out = {}
    for meth in ("backward", "IBP"):
        lo, hi = [], []
        for i in range(0, len(X), 4096):
            xc = X[i:i + 4096]; xb = BoundedTensor(xc, PerturbationLpNorm(norm=np.inf, x_L=xc - eps, x_U=xc + eps))
            l, h = bm.compute_bounds(x=(xb,), method=meth); lo.append(l.detach().numpy().ravel()); hi.append(h.detach().numpy().ravel())
        out[meth] = (np.concatenate(lo), np.concatenate(hi))
    return out


def prove_remaining(model, eps, src, net, Y, Rb, B):
    """For every CROWN increment output whose stored auto_LiRPA box does NOT contain the rigorous box (rigorous_enclosure's own slope
    search), read auto_LiRPA's OWN sigmoid relaxation slopes for that joint window (float64 CROWN on Incr(F, k), as s2_verify.py ran it)
    and evaluate the RIGOROUS bound at exactly those slopes (rigorous_enclosure recomputes the intercepts with outward rounding, so the
    result is a rigorous enclosure for any slopes).  If the stored box contains this second rigorous box, the stored box is proven
    sound at that output.  Diagnostic; the saved rigorous boxes are not changed."""
    import torch
    torch.set_default_dtype(torch.float64)
    from mlt_common import build, Incr
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    F = build(model); F.load_state_dict(torch.load(os.path.join(src, f"model_{model}.pt"), map_location="cpu")); F = F.double().eval()
    out = []
    for k in LAGS_DYADIC:
        gl, gh = Rb[f"glo|{k}"], Rb[f"ghi|{k}"]; al, ah = B[f"glo|CROWN|{eps}|{k}"], B[f"ghi|CROWN|{eps}|{k}"]
        for j, t in np.argwhere((gl < al) | (gh > ah)):
            X = torch.as_tensor(Y[j, t:t + W + k][None])
            bm = BoundedModule(Incr(F, k), X, verbose=0)
            bm.compute_bounds(x=(BoundedTensor(X, PerturbationLpNorm(norm=np.inf, x_L=X - eps, x_U=X + eps)),), method="backward")
            Yw = np.stack([Y[j, t + i:t + i + W] for i in range(k + 1)])     # windows t .. t+k; the pair is (0, k)
            cw = RE.crown_windows(net, Yw, eps); cw["eps_up"] = RE.up(float(eps))
            sig = {}
            for nd in bm.nodes():
                if type(nd).__name__ == "BoundSigmoid":                       # identify the branch by its input (logit) interval
                    lo_in = float(nd.inputs[0].lower); w_ = (0, k)[int(np.argmin(np.abs(cw["logit_lo"][[0, k]] - lo_in)))]
                    sig[w_] = (float(nd.lw.reshape(-1)[0]), float(nd.uw.reshape(-1)[0]))
            assert sorted(sig) == [0, k], "branch identification failed"
            res = {}
            for side, (sA, sB) in (("hi", (sig[0][0], sig[k][1])), ("lo", (sig[0][1], sig[k][0]))):
                sl = np.full((k + 1, 1), 0.25); sl[0, 0] = max(sA, 0.0); sl[k, 0] = max(sB, 0.0); cw["slopes"] = sl
                P = RE.SigPieces(cw["logit_lo"][:, None], cw["logit_hi"][:, None]); cw["mu_lo"], cw["mu_hi"] = P.mu_lo(cw["slopes"]), P.mu_hi(cw["slopes"])
                glo2, ghi2, _, _, _ = RE.increment_boxes(cw, (1, k + 1), k, W, refine_rounds=0)
                res[side] = float(ghi2[0, 0]) if side == "hi" else float(glo2[0, 0])
            rec = dict(k=int(k), j=int(j), t=int(t), rig_default=[float(gl[j, t]), float(gh[j, t])], rig_at_auto_slopes=[res["lo"], res["hi"]],
                       stored_auto=[float(al[j, t]), float(ah[j, t])],
                       proven_sound=bool(max(res["lo"], gl[j, t]) >= al[j, t] and min(res["hi"], gh[j, t]) <= ah[j, t]))
            out.append(rec); print("   prove_remaining:", rec, flush=True)
    return dict(n=len(out), n_proven=int(sum(r["proven_sound"] for r in out)), records=out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="mlp,feat,feat24"); ap.add_argument("--eps", default="0.01,0.05")
    ap.add_argument("--no_lirpa", action="store_true"); ap.add_argument("--window", default=None)
    ap.add_argument("--prove_only", action="store_true", help="only run prove_remaining on saved boxes and update the summary"); a = ap.parse_args()
    SRC, RDIR, TAG = paths(a.window)
    SF = os.path.join(RDIR, f"r1_summary{'_' + a.window if a.window else ''}.json")
    summary = json.load(open(SF, encoding="utf-8")) if os.path.exists(SF) else {}
    if a.prove_only:
        for model in a.models.split(","):
            net = RE.load_net(model, out_dir=SRC); Y = np.load(os.path.join(SRC, f"nominal_{model}.npz"))["y"].astype(np.float64)
            B = np.load(os.path.join(SRC, f"boxes_{model}.npz"))
            for eps in [float(x) for x in a.eps.split(",")]:
                key = f"{model}|{eps}"
                if key in summary and summary[key]["compare_with_auto_LiRPA"]["CROWN"]["ALL"]["not_proven"] > 0:
                    Rb = dict(np.load(os.path.join(RDIR, f"rigorous_boxes_{TAG}{model}_{eps}.npz")))
                    summary[key]["remaining_proven_at_auto_slopes"] = prove_remaining(model, eps, SRC, net, Y, Rb, B)
        jdump(summary, SF); return
    summary["declared"] = dict(source_dir=SRC, window=a.window, margin_of_stored_boxes=MARGIN, lags=LAGS_DYADIC, feeders=N_ENT, outputs=T, W=W,
                               method="rigorous_enclosure.py (IBP; CROWN with adaptive ReLU lower slope; exact sigmoid on logit bounds for values; "
                                      "verified-intercept sigmoid relaxation, 6x6 slope grid + 2 rounds golden-section along s_A = s_B, s_A, s_B for increments)")
    for model in a.models.split(","):
        net = RE.load_net(model, out_dir=SRC)
        nomf = np.load(os.path.join(SRC, f"nominal_{model}.npz")); Y = nomf["y"].astype(np.float64)
        B = np.load(os.path.join(SRC, f"boxes_{model}.npz"))
        Yw = np.lib.stride_tricks.sliding_window_view(Y, W, axis=1).reshape(-1, W).copy()
        g_nom = RE.forward_float(net, Yw)
        for eps in [float(x) for x in a.eps.split(",")]:
            t0 = time.time()
            R = RE.enclose_model(net, Y, eps, LAGS_DYADIC)
            wall = time.time() - t0
            rec = dict(seconds_total=wall, seconds=R["seconds"], n_unstable_relu_per_layer=R["n_unstable"].tolist(),
                       refine_gain_max={str(k): v for k, v in R["refine_gain_max"].items()})
            # ---- comparisons with the stored auto_LiRPA boxes
            cmp = {"CROWN": {"value": compare(R["lo"], R["hi"], B[f"lo|CROWN|{eps}"], B[f"hi|CROWN|{eps}"])},
                   "IBP": {"value": compare(R["ibp_lo"], R["ibp_hi"], B[f"lo|IBP|{eps}"], B[f"hi|IBP|{eps}"])}}
            for k in LAGS_DYADIC:
                cmp["CROWN"][f"incr{k}"] = compare(R[f"glo|{k}"], R[f"ghi|{k}"], B[f"glo|CROWN|{eps}|{k}"], B[f"ghi|CROWN|{eps}|{k}"])
                cmp["IBP"][f"incr{k}"] = compare(R[f"ibp_glo|{k}"], R[f"ibp_ghi|{k}"], B[f"glo|IBP|{eps}|{k}"], B[f"ghi|IBP|{eps}|{k}"])
            for meth in cmp:
                tot = {key: sum(cmp[meth][b][key] for b in cmp[meth]) for key in ("n", "proven_sound", "raw_proven", "margin_needed", "not_proven", "auto_in_rig")}
                tot["excess_max"] = max(cmp[meth][b]["excess_max"] for b in cmp[meth]); tot["excess_raw_max"] = max(cmp[meth][b]["excess_raw_max"] for b in cmp[meth])
                cmp[meth]["ALL"] = tot
            rec["compare_with_auto_LiRPA"] = cmp
            if cmp["CROWN"]["ALL"]["not_proven"] > 0 and not a.no_lirpa:
                rec["remaining_proven_at_auto_slopes"] = prove_remaining(model, eps, SRC, net, Y, R, B)
            # ---- kappa (coherence diagnostic) from rigorous vs stored CROWN boxes
            vw = R["hi"] - R["lo"]; vwa = B[f"hi|CROWN|{eps}"] - B[f"lo|CROWN|{eps}"]
            rec["kappa_median"] = {str(k): dict(rigorous=float(np.median((R[f"ghi|{k}"] - R[f"glo|{k}"]) / vw[:, :-k])),
                                                auto=float(np.median((B[f"ghi|CROWN|{eps}|{k}"] - B[f"glo|CROWN|{eps}|{k}"]) / vwa[:, :-k])))
                                   for k in LAGS_DYADIC}
            rec["value_width_median"] = dict(rigorous=float(np.median(vw)), auto=float(np.median(vwa)))
            # ---- diagnostics: nominal logit, random admissible series, PGD inner values
            rec["nominal_logit_inside"] = bool(np.all((g_nom >= R["logit_lo"].ravel()) & (g_nom <= R["logit_hi"].ravel())))
            fs = np.load(os.path.join(SRC, f"randseries_{model}_{eps}.npz"))["fs"]
            v = max(float(np.max(fs - R["hi"][None])), float(np.max(R["lo"][None] - fs)))
            for k in LAGS_DYADIC:
                inc = fs[:, :, k:] - fs[:, :, :-k]
                v = max(v, float(np.max(inc - R[f"ghi|{k}"][None])), float(np.max(R[f"glo|{k}"][None] - inc)))
            vin = max(float(np.max(B[f"hi_in|{eps}"] - R["hi"])), float(np.max(R["lo"] - B[f"lo_in|{eps}"])),
                      float(np.max(B[f"ghi_in|{eps}|1"] - R["ghi|1"])), float(np.max(R["glo|1"] - B[f"glo_in|{eps}|1"])))
            rec["spot_check"] = dict(max_violation_random_series=v, n_series=int(fs.shape[0]), max_violation_pgd_inner=vin,
                                     rigorous_over_pgd_inner_value_width_median=float(np.median(vw / np.maximum(B[f"hi_in|{eps}"] - B[f"lo_in|{eps}"], 1e-12))),
                                     rigorous_over_pgd_inner_incr1_width_median=float(np.median((R["ghi|1"] - R["glo|1"]) / np.maximum(B[f"ghi_in|{eps}|1"] - B[f"glo_in|{eps}|1"], 1e-12))))
            # ---- logit-level comparison with a fresh float64 auto_LiRPA run (same algorithm up to the sigmoid)
            if not a.no_lirpa:
                t1 = time.time(); L = lirpa_logit(model, Yw, eps, SRC)
                ll = {}
                for meth, (rl, rh) in (("backward", (R["logit_lo"].ravel(), R["logit_hi"].ravel())), ("IBP", (R["ibp_logit_lo"].ravel(), R["ibp_logit_hi"].ravel()))):
                    al_, ah_ = L[meth]
                    ll[meth] = dict(max_abs_diff=float(max(np.max(np.abs(rl - al_)), np.max(np.abs(rh - ah_)))),
                                    median_abs_diff=float(np.median(np.concatenate([np.abs(rl - al_), np.abs(rh - ah_)]))),
                                    rigorous_contains_float=int(np.sum((rl <= al_) & (rh >= ah_))), n=int(rl.size),
                                    max_float_outside_rigorous=float(max(np.max(rl - al_), np.max(ah_ - rh))))
                ll["seconds_auto_LiRPA"] = time.time() - t1
                rec["logit_level_vs_fresh_auto_LiRPA_float64"] = ll
            # ---- save the boxes
            sv = dict(lo=R["lo"], hi=R["hi"], logit_lo=R["logit_lo"], logit_hi=R["logit_hi"], ibp_lo=R["ibp_lo"], ibp_hi=R["ibp_hi"],
                      ibp_logit_lo=R["ibp_logit_lo"], ibp_logit_hi=R["ibp_logit_hi"])
            for k in LAGS_DYADIC:
                sv[f"glo|{k}"] = R[f"glo|{k}"]; sv[f"ghi|{k}"] = R[f"ghi|{k}"]; sv[f"ibp_glo|{k}"] = R[f"ibp_glo|{k}"]; sv[f"ibp_ghi|{k}"] = R[f"ibp_ghi|{k}"]
                sv[f"combo_hi|{k}"] = R[f"combo_hi|{k}"].astype(np.int8); sv[f"combo_lo|{k}"] = R[f"combo_lo|{k}"].astype(np.int8)
            np.savez_compressed(os.path.join(RDIR, f"rigorous_boxes_{TAG}{model}_{eps}.npz"), **sv)
            summary[f"{model}|{eps}"] = rec
            c = cmp["CROWN"]
            print(f"{a.window or 'synthetic'} {model} eps={eps}: {wall:.0f}s  CROWN value: proven {c['value']['proven_sound']}/{c['value']['n']} raw {c['value']['raw_proven']} "
                  f"ratio med {c['value']['width_ratio_median']:.4f} [{c['value']['width_ratio_min']:.4f},{c['value']['width_ratio_max']:.4f}] | "
                  f"ALL: proven {c['ALL']['proven_sound']}/{c['ALL']['n']} margin_needed {c['ALL']['margin_needed']} not_proven {c['ALL']['not_proven']} "
                  f"excess {c['ALL']['excess_max']:.2e} | IBP ALL proven {cmp['IBP']['ALL']['proven_sound']}/{cmp['IBP']['ALL']['n']} | spot {rec['spot_check']}", flush=True)
            if "logit_level_vs_fresh_auto_LiRPA_float64" in rec:
                print("   logit level:", rec["logit_level_vs_fresh_auto_LiRPA_float64"], flush=True)
            for k in LAGS_DYADIC:
                ck = c[f"incr{k}"]
                print(f"   lag {k:3d}: proven {ck['proven_sound']}/{ck['n']} raw {ck['raw_proven']} not {ck['not_proven']} ratio med {ck['width_ratio_median']:.4f} "
                      f"[{ck['width_ratio_min']:.4f},{ck['width_ratio_max']:.4f}] excess {ck['excess_max']:.2e} kappa rig {rec['kappa_median'][str(k)]['rigorous']:.3f} auto {rec['kappa_median'][str(k)]['auto']:.3f}", flush=True)
            jdump(summary, SF)


if __name__ == "__main__":
    main()
