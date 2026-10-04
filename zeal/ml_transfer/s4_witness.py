"""s4_witness.py -- realizable witnesses (inner values) by PGD on the declared SERIES box, and the final assembly of exp3_ml.

A witness is an explicit admissible series y = y~ + delta (|delta_s| <= eps, checked in float64) and an explicit aggregation choice; its
value d.f(y) is evaluated with the actual network in float64.  It is a LOWER bound of the true supremum (realizable), never a claim
of optimality.  Witness targets: S1a/S1b ranges at the certified argmax pair (alpha-CROWN, ZD; the same witness is a valid lower bound
for every configuration), the seeded validation choices of S2/S3 (both directions), and sign flips of S2/S3 instances whose nominal
sign is constant (the 5 choices of smallest nominal |statistic|).
Outputs: exp3_ml/s4_witness_<model>.json (raw witnesses) and exp3_ml/summary_<model>.json (all tables used in findings.md).
Usage: python s4_witness.py [--models mlp,feat] [--out DIR]   (DIR holds s3_certify_<model>.json and receives the outputs; default exp3_ml)"""
import os, sys, time, argparse, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlt_common import *
import s3_certify as S3

ap = argparse.ArgumentParser(); ap.add_argument("--models", default="mlp,feat"); ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--out", default=OUT, help="directory of s3_certify_<model>.json and of the outputs (default exp3_ml); "
                                           "nominal_ and s2_verify_ are always read from exp3_ml"); a = ap.parse_args()
BEST = "alpha-CROWN|{eps}|ZD"


def pgd(F, Ybase, Dm, eps, steps):
    """maximize sum_t D[p,t] f_p(t; Ybase[p] + delta_p) over |delta| <= eps; returns float64 values and deltas"""
    Yb = torch.as_tensor(Ybase, dtype=torch.float32, device=DEV); Dt = torch.as_tensor(Dm, dtype=torch.float32, device=DEV)
    best_v = np.full(len(Yb), -np.inf); best_d = np.zeros(Ybase.shape)
    for start in ("zero", "random"):
        g = torch.Generator(device=DEV).manual_seed(5)
        dl = torch.zeros_like(Yb) if start == "zero" else eps * (2 * (torch.rand(Yb.shape, generator=g, device=DEV) > 0.5).float() - 1)
        for s in range(steps):
            dl.requires_grad_(True)
            f = F(windows(Yb + dl).reshape(-1, W)).reshape(Yb.shape[0], T)
            obj = (Dt * f).sum(); gr, = torch.autograd.grad(obj, dl)
            step = eps / 8 if s < steps * 2 // 3 else eps / 40
            with torch.no_grad():
                dl = (dl + step * gr.sign()).clamp(-eps, eps)
        d64 = np.clip(dl.detach().cpu().double().numpy(), -eps, eps)                 # exact box membership in float64
        with torch.no_grad():
            f64 = F64(windows(torch.as_tensor(Ybase + d64)).reshape(-1, W)).reshape(len(Yb), T).numpy()
        v = np.sum(Dm * f64, axis=1)
        upd = v > best_v; best_v[upd] = v[upd]; best_d[upd] = d64[upd]
    return best_v, best_d


def med(x):
    x = np.asarray([v for v in x if np.isfinite(v)]); return float(np.median(x)) if len(x) else np.nan


def q(x, p):
    x = np.asarray([v for v in x if np.isfinite(v)]); return float(np.quantile(x, p)) if len(x) else np.nan


for model in a.models.split(","):
    t0 = time.time()
    F = load_model(model).to(DEV); F64 = load_model(model).double()
    nomz = np.load(os.path.join(OUT, f"nominal_{model}.npz")); Y = nomz["y"].astype(np.float64); f0 = nomz["f"].astype(np.float64)
    C = json.load(open(os.path.join(a.out, f"s3_certify_{model}.json"), encoding="utf-8"))
    V2 = json.load(open(os.path.join(OUT, f"s2_verify_{model}.json"), encoding="utf-8"))
    eps_list = sorted({float(k.split("|")[1]) for k in C["configs"]})
    WIT = {}; summary = dict(model=model, verifier=dict(), configs={})
    for eps in eps_list:
        bk = BEST.format(eps=eps) if BEST.format(eps=eps) in C["configs"] else f"CROWN|{eps}|ZD"
        best = C["configs"][bk]["per_feeder"]
        reqs = []                                                   # (tag, j, d)
        for r in best:
            j = r["j"]; sA, LA, sB, LB = r["S1a"]["argmax_pair"]; reqs.append((("S1a", j), j, S3.win_d(sA, LA) - S3.win_d(sB, LB)))
            sA, LA, sB, LB = r["S1a"]["nominal_pair"]; reqs.append((("S1a_nom", j), j, S3.win_d(sA, LA) - S3.win_d(sB, LB)))
            for row in r["S1b"]:
                if row["Z_argmax_pair"]:
                    reqs.append((("S1b", j, row["e"]), j, S3.s1b_d(row["e"], *row["Z_argmax_pair"])))
                reqs.append((("S1b_nom", j, row["e"]), j, S3.s1b_d(row["e"], *row["nominal_pair"])))
            for S in ("S2", "S3", "S5"):
                for inst in r[S]:
                    dfun = {"S2": S3.s2_d, "S3": S3.s3_d, "S5": S3.s5_d}[S]
                    for c in inst["choices_sample"]:
                        d = dfun(*c); reqs.append(((S, j, inst["inst"], tuple(c), +1), j, d)); reqs.append(((S, j, inst["inst"], tuple(c), -1), j, -d))
                    if inst["nominal_n_plus"] == 0 or inst["nominal_n_minus"] == 0:
                        sg = 1 if inst["nominal_n_plus"] > 0 else -1
                        for c in inst["flip_candidates"]:
                            reqs.append(((S + "flip", j, inst["inst"], tuple(c)), j, -sg * dfun(*c)))
        vals = np.zeros(len(reqs))
        for i in range(0, len(reqs), 192):
            chunk = reqs[i:i + 192]
            v, _ = pgd(F, np.array([Y[c[1]] for c in chunk]), np.array([c[2] for c in chunk]), eps, a.steps); vals[i:i + 192] = v
        W_ = {c[0]: float(v) for c, v in zip(reqs, vals)}; WIT[eps] = W_
        print(model, eps, len(reqs), "witnesses", f"{time.time()-t0:.0f}s", flush=True)
        # ---------------- assembly per configuration
        for ck, cv in C["configs"].items():
            meth, e_, arc = ck.split("|")
            if float(e_) != eps:
                continue
            row = dict(S4=cv["S4"])
            if arc != "V":
                pf = cv["per_feeder"]
                # S1a
                s1a = []
                for r in pf:
                    w = max(W_[("S1a", r["j"])], W_[("S1a_nom", r["j"])], r["S1a"]["nominal"]); nomr = r["S1a"]["nominal"]
                    s1a.append(dict(j=r["j"], nominal=nomr, witness=w, V=r["S1a"]["V"], Vs=r["S1a"]["Vs"], Z_lo=r["S1a"]["Z_range_bracket"][0], Z_hi=r["S1a"]["Z_range_bracket"][1]))
                row["S1a"] = dict(per_feeder=s1a,
                                  median=dict({k: med([x[k] for x in s1a]) for k in ("nominal", "witness", "V", "Vs", "Z_lo", "Z_hi")}),
                                  excess_ratio_V=med([(x["V"] - x["nominal"]) / (x["witness"] - x["nominal"]) for x in s1a if x["witness"] - x["nominal"] > 1e-6]),
                                  excess_ratio_Vs=med([(x["Vs"] - x["nominal"]) / (x["witness"] - x["nominal"]) for x in s1a if x["witness"] - x["nominal"] > 1e-6]),
                                  ratio_V=med([x["V"] / x["witness"] for x in s1a]), ratio_Zhi=med([x["Z_hi"] / x["witness"] for x in s1a]),
                                  Z_bracket_rel_width=med([(x["Z_hi"] - x["Z_lo"]) / x["Z_hi"] for x in s1a]))
                # S1b
                s1b = []
                for r in pf:
                    for x in r["S1b"]:
                        w = max(W_.get(("S1b", r["j"], x["e"]), -np.inf), W_[("S1b_nom", r["j"], x["e"])], x["nominal"])
                        s1b.append(dict(V=x["V"], Vs=x["Vs"], Z=x["Z"], nominal=x["nominal"], witness=w, settled=x["Z_settled"], n_lp=x["n_lp"]))
                ex = lambda k: [(x[k] - x["nominal"]) / (x["witness"] - x["nominal"]) for x in s1b if x["witness"] - x["nominal"] > 1e-6]
                row["S1b"] = dict(n=len(s1b), settled=int(sum(x["settled"] for x in s1b)), lp_median=med([x["n_lp"] for x in s1b]),
                                  median=dict({k: med([x[k] for x in s1b]) for k in ("nominal", "witness", "V", "Vs", "Z")}),
                                  excess_ratio=dict(V=med(ex("V")), Vs=med(ex("Vs")), Z=med(ex("Z"))),
                                  excess_ratio_q90=dict(V=q(ex("V"), .9), Vs=q(ex("Vs"), .9), Z=q(ex("Z"), .9)),
                                  ratio=dict(V=med([x["V"] / x["witness"] for x in s1b]), Vs=med([x["Vs"] / x["witness"] for x in s1b]), Z=med([x["Z"] / x["witness"] for x in s1b])),
                                  Z_over_V=med([(x["Z"] - x["nominal"]) / (x["V"] - x["nominal"]) for x in s1b if x["V"] - x["nominal"] > 1e-9]))
                # G records (regime theorem) and outer/inner on the validation sample
                Gs = {}
                for r in pf:
                    for g in r["G_records"]:
                        Gs.setdefault(g["tag"], []).append(g)
                row["G"] = {tag: dict(n=len(v), G_median=med([g["G"] for g in v]), G_q90=q([g["G"] for g in v], .9), G_max=q([g["G"] for g in v], 1.0),
                                      S_total_median=med([g["S_total"] for g in v]),
                                      closure_share_median=med([(g["V"] - g["Vs"]) / (g["V"] - g["dpsi"]) for g in v if g["V"] - g["dpsi"] > 1e-9]))
                            for tag, v in Gs.items()}
                oi = {}
                for S in ("S2", "S3", "S5"):
                    rr = []
                    for r in pf:
                        for inst in r[S]:
                            for c, b in zip(inst["choices_sample"], inst["sample_bounds"]):
                                for sg, (Ux, Us, Uz) in ((+1, (b["UV"], b["US"], b["UZ"])), (-1, (-b["LV"], -b["LS"], -b["LZ"]))):
                                    w = W_[(S, r["j"], inst["inst"], tuple(c), sg)]; dn = sg * b["nominal"]
                                    if w - dn > 1e-7:
                                        rr.append(dict(V=(Ux - dn) / (w - dn), Vs=(Us - dn) / (w - dn), Z=(Uz - dn) / (w - dn)))
                    oi[S] = dict(n=len(rr), **{k: dict(median=med([x[k] for x in rr]), q10=q([x[k] for x in rr], .1), q90=q([x[k] for x in rr], .9)) for k in ("V", "Vs", "Z")})
                row["outer_over_inner_excess"] = oi
                # decisions S2/S3 with witnessed flips
                dec = {}
                for S in ("S2", "S3", "S5"):
                    D_ = {}
                    for bt in ("V", "Vs", "Z"):
                        cnt = dict(certified_invariant=0, certified_sensitive=0, witnessed_sensitive_nominal=0, witnessed_flip_perturbation=0, open=0, n=0)
                        for r in pf:
                            for inst in r[S]:
                                cb = inst[bt]; cnt["n"] += 1
                                mixed = inst["nominal_n_plus"] > 0 and inst["nominal_n_minus"] > 0
                                if cb["all_plus"] or cb["all_minus"]:
                                    cnt["certified_invariant"] += 1
                                elif cb["any_plus"] and cb["any_minus"]:
                                    cnt["certified_sensitive"] += 1
                                elif mixed:
                                    cnt["witnessed_sensitive_nominal"] += 1
                                else:
                                    fl = [W_.get((S + "flip", r["j"], inst["inst"], tuple(c)), -np.inf) for c in inst["flip_candidates"]]
                                    if max(fl) > 0:
                                        cnt["witnessed_flip_perturbation"] += 1
                                    else:
                                        cnt["open"] += 1
                        D_[bt] = cnt
                    D_["nominal_mixed_sign_instances"] = int(sum((i["nominal_n_plus"] > 0 and i["nominal_n_minus"] > 0) for r in pf for i in r[S]))
                    D_["lp_calls"] = int(sum(i["n_lp_U"] + i["n_lp_L"] for r in pf for i in r[S]))
                    dec[S] = D_
                row["decisions"] = dec
                row["closure_gain_median"] = med([r["closure_gain_median"] for r in pf]); row["width_ratio_median"] = med([r["width_ratio_median"] for r in pf])
                row["n_lp"] = int(sum(r["n_lp"] for r in pf)); row["lp_seconds"] = float(sum(r["lp_seconds"] for r in pf))
                if arc == "Z1":
                    row["exact1d_check_max_abs_diff"] = max(r["exact1d_check"]["abs_diff"] for r in pf)
            summary["configs"][ck] = row
        summary["verifier"][str(eps)] = {m: dict(V2["runs"][f"{m}|{eps}"], **({} if f"check|{eps}" not in V2 else {"check": V2[f"check|{eps}"]["verifier_over_inner"].get(m)}))
                                         for m in ("IBP", "CROWN", "alpha-CROWN") if f"{m}|{eps}" in V2["runs"]}
        summary["verifier"][str(eps)]["soundness_max_violation_random_series"] = V2[f"check|{eps}"]["max_violation_random_series"]
        summary["verifier"][str(eps)]["inner_value_width_median"] = V2[f"check|{eps}"]["inner_value_width_median"]
        summary["verifier"][str(eps)]["inner_incr1_width_median"] = V2[f"check|{eps}"]["inner_incr1_width_median"]
    jdump({str(k): {"|".join(map(str, kk)): v for kk, v in d.items()} for k, d in WIT.items()}, os.path.join(a.out, f"s4_witness_{model}.json"))
    for k_ in ("n_lp_total", "lp_failures_total", "lp_failures_exact1d_total", "lp_failures_by_config"):   # LP-failure guard counts (if present)
        if k_ in C:
            summary[k_] = C[k_]
    summary["seconds"] = time.time() - t0
    jdump(summary, os.path.join(a.out, f"summary_{model}.json")); print(model, "summary written", flush=True)
