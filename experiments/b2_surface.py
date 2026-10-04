"""b2_surface.py -- boundary-displacement certificates (Definition S2 class C_delta; Supplementary Theorems S3-S5, S7,
S18; results exp2/B2_surface) on the pixel rasters -- the regime experiment for increment coupling.

For a real coarse cell a (county / MSOA / municipio window) we form a' by removing a strip of width delta pixels along a random
contiguous segment of its boundary (about a quarter of the boundary), so d = alpha_{a'} - alpha_a is supported on the tube plus the
interior normalization component (Theorem S18 (i)). We certify the change of the cell mean <f>_{a'} - <f>_a over the fitted field's
box class by: value-only V; the transport programme on Phi' (tent + Lemma S6.1); the lifted outer X^(1) (Theorem S7, pure network); all
with verified dual bounds; the realizable witness from X^(1)'s optimizer (Q_2 field); and the realized change of the midpoint
reference. Reported by delta: U/V (does the increment information matter?), outer/witness, absolute widths; the tube bound of
Theorem S18 (i) (value-only on the tube) beside the transport value.
Usage: python b2_surface.py --field georgia [--cells 30] [--deltas 1,2,4,8] [--max_pix 1500]
"""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, time, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transport_core import solve_max, lemma12, grid_poly, project_to_phi, verified_solve_system
from outer_x import Mesh
def verified_solve(ms, obj):
    A, b = ms.system(); return verified_solve_system(A, b, ms.lo_n, ms.hi_n, obj)
from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--cells", type=int, default=30); p.add_argument("--deltas", default="1,2,4,8")
p.add_argument("--max_pix", type=int, default=1500); p.add_argument("--seed", type=int, default=17); p.add_argument("--out", default=None); p.add_argument("--mode", default="remove", choices=["remove", "swap"]); a = p.parse_args()
OUT = a.out or paths.results("exp2", "B2_surface"); os.makedirs(OUT, exist_ok=True)
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float), F["hi"].astype(float)
g1l0, g1h0, g2l0, g2h0 = (F[k].astype(float) for k in ("g1l", "g1h", "g2l", "g2h"))
ras = np.load(SPEC[a.field]); mask = ras["mask"].astype(bool) & np.isfinite(lo0); N = mask.shape[0]; h = 2.0 / (N - 1)
w = np.where(mask, ras["weight"], 0.0).astype(float); coarse = np.where(mask, ras["county_idx"], -1)
lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, coarse = (A.T.copy() for A in (lo0, hi0, g1l0, g1h0, g2l0, g2h0, mask, w, coarse))
G1 = np.stack([g1l0, g1h0], -1); G2 = np.stack([g2l0, g2h0], -1); lam = (lo0 + hi0) / 2
rng = np.random.default_rng(a.seed)
sizes = {int(b): int((coarse == b).sum()) for b in np.unique(coarse[coarse >= 0])}; cand = [b for b, s_ in sizes.items() if 60 <= s_ <= a.max_pix]


def boundary_pixels(cell):
    inner = cell.copy(); inner[1:, :] &= cell[:-1, :]; inner[:-1, :] &= cell[1:, :]; inner[:, 1:] &= cell[:, :-1]; inner[:, :-1] &= cell[:, 1:]
    return cell & ~inner


def segment(cell, frac=0.25):
    bd = boundary_pixels(cell); ii, jj = np.nonzero(bd)
    if len(ii) == 0:
        return None
    start = int(rng.integers(len(ii))); target = max(3, int(frac * len(ii))); seg = {(ii[start], jj[start])}; frontier = [(ii[start], jj[start])]; bset = set(zip(ii.tolist(), jj.tolist()))
    while frontier and len(seg) < target:
        i, j = frontier.pop(0)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = (i + di, j + dj)
            if q in bset and q not in seg:
                seg.add(q); frontier.append(q)
    return seg


def tube_of(seg, delta, shape):
    rem = np.zeros(shape, bool)
    for (i, j) in seg:
        rem[max(0, i - delta + 1):i + delta, max(0, j - delta + 1):j + delta] = True
    return rem


def largest_piece(new):
    from scipy.ndimage import label as cclabel
    lab, k = cclabel(new)
    if k == 0:
        return None
    if k > 1:
        big = np.argmax(np.bincount(lab[lab > 0])); new = lab == big
    return new


def displace(cell, domain, delta, mode):
    """mode 'remove': a' = a minus a delta-strip along a boundary segment (unequal mass -> interior component);
    mode 'swap': a' = (a minus the strip) plus the delta-strip of the OUTSIDE domain along the same segment (approximately mass-preserving: d supported near the boundary)."""
    seg = segment(cell)
    if seg is None:
        return None
    tube = tube_of(seg, delta, cell.shape)
    if mode == "remove":
        return largest_piece(cell & ~tube)
    out_strip = tube & domain & ~cell
    new = (cell & ~tube) | out_strip
    return largest_piece(new)


rows = []; t0 = time.time()
for ci in range(a.cells):
    if not cand:
        break
    b = int(rng.choice(cand)); sel = coarse == b; pad = max(int(x) for x in a.deltas.split(",")) + 1
    i0, i1 = np.nonzero(sel.any(1))[0][[0, -1]]; j0, j1 = np.nonzero(sel.any(0))[0][[0, -1]]; sl = (slice(max(0, i0 - pad), min(N, i1 + 1 + pad)), slice(max(0, j0 - pad), min(N, j1 + 1 + pad)))
    cell = sel[sl]; dom = mask[sl]; wc = w[sl] * cell
    for delta in [int(x) for x in a.deltas.split(",")]:
        new = displace(cell, dom, delta, a.mode)
        if new is None or new.sum() < 0.5 * cell.sum():
            continue
        region = cell | new                      # LP domain: the union (all certified pixels)
        lo2, hi2, _ = lemma12(lo0[sl], hi0[sl], G1[sl], G2[sl], h); polyC, _ = grid_poly(lo2, hi2, G1[sl], G2[sl], h, region)
        wn = w[sl] * new; d2 = wn / wn.sum() - wc / wc.sum(); d = d2[region]
        tube = cell ^ new; out = dict(coarse=b, delta=delta, mode=a.mode, n_pix=int(region.sum()), removed_share=float(wc[cell & ~new].sum() / wc.sum()), added_share=float(wn[new & ~cell].sum() / wn.sum()), mass_ratio=float(wn.sum() / wc.sum()),
                   interior_mass_share=float(np.abs(d2[cell & new]).sum() / np.abs(d2).sum()))
        psi = project_to_phi(polyC, lam[sl][region]); cellmask = region
        for sgn, nm in ((1, "U"), (-1, "L")):
            dd = sgn * d; dp, dm = np.maximum(dd, 0), np.maximum(-dd, 0)
            out[f"{nm}_V"] = sgn * float(dp @ polyC.hi - dm @ polyC.lo)
            r = solve_max(polyC, dd); out[f"{nm}_PhiC"] = sgn * r["verified"]
            ms = Mesh(lo0[sl], hi0[sl], G1[sl], G2[sl], h, region, 2); vX, xX, ubX = verified_solve(ms, sgn * ms.obj_centre(d2)); out[f"{nm}_X1"] = sgn * ubX
            out[f"{nm}_witness"] = sgn * float((sgn * ms.obj_q1(d2)) @ xX) if xX is not None else float("nan")
        out["realized_mid"] = float(d @ psi)
        out["width_V"] = out["U_V"] - out["L_V"]; out["width_PhiC"] = out["U_PhiC"] - out["L_PhiC"]; out["width_X1"] = out["U_X1"] - out["L_X1"]; out["width_witness"] = out["U_witness"] - out["L_witness"]
        out["PhiC_over_V"] = out["width_PhiC"] / out["width_V"]; out["X1_over_V"] = out["width_X1"] / out["width_V"]; out["X1_over_witness"] = out["width_X1"] / out["width_witness"] if out["width_witness"] > 1e-9 else float("nan")
        # tube bound of Theorem S18 (i) (value-only form): |removed mass| / mu(a') * max |f - <f>_a| over the tube, using the boxes
        mean_box = (float((wc[cell] / wc.sum()) @ lo2[cell]), float((wc[cell] / wc.sum()) @ hi2[cell]))
        out["tube_bound_width"] = 2 * (w[sl] * tube).sum() / wn.sum() * float(max(np.max(hi2[tube]) - mean_box[0], mean_box[1] - np.min(lo2[tube])))
        rows.append(out)
    print(f"  cell {b} (n={cell.sum()}, {a.mode}): " + " | ".join(f"d={r['delta']}: Phi'/V {r['PhiC_over_V']:.3f} X1/V {r['X1_over_V']:.3f} X1/wit {r['X1_over_witness']:.3f} wV {r['width_V']:.4f}" for r in rows if r["coarse"] == b) + f"  {time.time()-t0:.0f}s", flush=True)
S = {}
for delta in sorted(set(r["delta"] for r in rows)):
    rr = [r for r in rows if r["delta"] == delta]
    S[delta] = dict(n=len(rr), removed_share_median=float(np.median([r["removed_share"] for r in rr])), interior_mass_share_median=float(np.median([r["interior_mass_share"] for r in rr])),
                    PhiC_over_V_median=float(np.median([r["PhiC_over_V"] for r in rr])), X1_over_V_median=float(np.median([r["X1_over_V"] for r in rr])), X1_over_witness_median=float(np.nanmedian([r["X1_over_witness"] for r in rr])),
                    share_PhiC_below_0p9V=float(np.mean([r["PhiC_over_V"] < 0.9 for r in rr])), width_V_median=float(np.median([r["width_V"] for r in rr])), width_X1_median=float(np.median([r["width_X1"] for r in rr])),
                    tube_over_X1_median=float(np.median([r["tube_bound_width"] / r["width_X1"] for r in rr])), realized_abs_median=float(np.median([abs(r["realized_mid"]) for r in rr])))
    print(f"{a.field} delta={delta}: n={S[delta]['n']} removed {S[delta]['removed_share_median']:.3f} Phi'/V {S[delta]['PhiC_over_V_median']:.3f} X1/V {S[delta]['X1_over_V_median']:.3f} X1/wit {S[delta]['X1_over_witness_median']:.3f} share<0.9: {S[delta]['share_PhiC_below_0p9V']:.2f} widths V {S[delta]['width_V_median']:.4f} X1 {S[delta]['width_X1_median']:.4f} tube/X1 {S[delta]['tube_over_X1_median']:.2f}", flush=True)
json.dump(dict(field=a.field, mode=a.mode, summary=S, rows=rows), open(os.path.join(OUT, f"b2s_{a.field}_{a.mode}.json"), "w"), indent=1, default=float)
