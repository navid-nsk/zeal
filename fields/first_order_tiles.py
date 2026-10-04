"""Certified first-order tiles at pixel resolution: for every in-domain pixel p of the raster, a sound interval for the
field value [lo_p, hi_p] and sound intervals for both components of its gradient [g_p^-, g_p^+] (field lambda = softplus(h)).

Value intervals: CROWN on h with the 2-D location as the perturbed input, tile = pixel (then softplus, monotone),
intersected with the mean-value form h(c) +- (h/2) * max|g| per axis evaluated from the gradient intervals.
Gradient intervals: the intersection of (a) the projected enclosure (exact sin/cos ranges, Jacobian bounds of g over
the feature box, J^T applied in interval arithmetic, per component) and (b) end-to-end CROWN on the explicit gradient
network with x as input; then multiplied by the softplus slope interval [sigmoid(h_lo), sigmoid(h_hi)].
Output: data/first_order_<name>.npz with lo, hi, g1l, g1h, g2l, g2h (N x N, NaN outside the domain) and statistics.
Usage: python first_order_tiles.py --field georgia|gm_q4|mx_rwi [--batch 256]
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, math, os, sys, time, numpy as np, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from zeal_m1 import Field
from zeal_manifold_cert import encoder_box, _JacW, iv_prod
from osc_lib import _Pre
from gradnet import GradNet

p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); p.add_argument("--batch", type=int, default=256); p.add_argument("--no_e2e", action="store_true"); p.add_argument("--sample", type=int, default=0, help="evaluate a random subset of pixels only (tightness test)")
a = p.parse_args(); dev = "cuda"
FIELDS = {"georgia": (paths.field_file("local_cert_net.pt"), paths.RASTER["georgia"]),
          "gm_q4": (paths.field_file("uk_field_q4_net.pt"), paths.RASTER["gm_q4"]),
          "gm_bad": (paths.field_file("uk_field_bad_net.pt"), paths.RASTER["gm_bad"]),
          "mx_rwi": (paths.field_file("ladder_field_mx_rwi_net.pt"), paths.RASTER["mx_rwi"])}
ck_path, ras_path = FIELDS[a.field]
sd = torch.load(ck_path, map_location=dev, weights_only=False); sd = sd if isinstance(sd, dict) and "B" in sd else sd["state"]
m = sd["B"].shape[0]; w = sd["net.0.weight"].shape[0]; depth = sum(1 for k in sd if k.startswith("net.") and k.endswith(".weight")) - 1
net = Field(m=m, sigma=1.0, w=w, depth=depth, act="tanh").to(dev); net.load_state_dict(sd); net.eval()
ras = np.load(ras_path); mask = ras["mask"].astype(bool); N = mask.shape[0]; h = 2.0 / (N - 1)
ii, jj = np.nonzero(mask)
if a.sample:
    sel = np.random.default_rng(0).choice(len(ii), size=a.sample, replace=False); ii, jj = ii[sel], jj[sel]
P = len(ii)
xc = torch.tensor(np.stack([-1 + h * jj, -1 + h * ii], 1), dtype=torch.float32, device=dev)      # pixel centres (x = column, y = row)
XL, XH = xc - h / 2, xc + h / 2
print(f"[{a.field}] m={m} w={w} depth={depth} | {P:,} pixel tiles of side {h:.5f} ({N}^2 raster)")

B, g = net.B.to(dev), net.net.to(dev).eval()
bm_j = BoundedModule(_JacW(g), torch.zeros(a.batch, 2 * m, device=dev), device=dev, bound_opts={"sparse_intermediate_bounds": False})
bm_h = BoundedModule(_Pre(net).to(dev).eval(), torch.zeros(a.batch, 2, device=dev), device=dev)
gn = GradNet(net).to(dev).eval(); bm_g = BoundedModule(gn, torch.zeros(a.batch, 2, device=dev), device=dev, bound_opts={"sparse_intermediate_bounds": False})
coef = 2 * math.pi / math.sqrt(m)


def projected_components(xl, xh):
    sL, sH, cL, cH, zL, zH = encoder_box(B, m, xl, xh)
    zb = BoundedTensor((zL + zH) / 2, PerturbationLpNorm(norm=np.inf, x_L=zL, x_U=zH))
    lb, ub = bm_j.compute_jacobian_bounds((zb,), optimize=False); dgL, dgH = lb.detach().reshape(xl.shape[0], -1), ub.detach().reshape(xl.shape[0], -1)
    al, ah = iv_prod(cL, cH, dgL[:, :m], dgH[:, :m]); bl, bh = iv_prod(sL, sH, dgL[:, m:], dgH[:, m:]); uL, uH = al - bh, ah - bl
    out = []
    for k in range(2):
        Bp, Bn = B[:, k].clamp(min=0), B[:, k].clamp(max=0)
        out.append((coef * ((uL * Bp).sum(1) + (uH * Bn).sum(1)), coef * ((uH * Bp).sum(1) + (uL * Bn).sum(1))))
    return out


def e2e_components(xl, xh):
    xb = BoundedTensor((xl + xh) / 2, PerturbationLpNorm(norm=np.inf, x_L=xl, x_U=xh))
    lb, ub = bm_g.compute_bounds(x=(xb,), method="CROWN")
    return [(lb[:, k], ub[:, k]) for k in range(2)]


LO = np.full((N, N), np.nan, np.float32); HI = LO.copy(); G = [LO.copy() for _ in range(4)]
stat = dict(proj_tighter=0, e2e_tighter=0, n=0); t0 = time.time()
with torch.no_grad():
    for s in range(0, P, a.batch):
        xl, xh = XL[s:s + a.batch], XH[s:s + a.batch]; n = xl.shape[0]
        if n < a.batch:
            xl = torch.cat([xl, xl[-1:].expand(a.batch - n, 2)]); xh = torch.cat([xh, xh[-1:].expand(a.batch - n, 2)])
        xb = BoundedTensor((xl + xh) / 2, PerturbationLpNorm(norm=np.inf, x_L=xl, x_U=xh)); hlo, hhi = bm_h.compute_bounds(x=(xb,), method="CROWN")   # bounds of h over the pixel
        pc = projected_components(xl, xh)
        comps = []
        if a.no_e2e:
            comps = pc
        else:
            ec = e2e_components(xl, xh)
            for (pl, ph), (el, eh) in zip(pc, ec):
                comps.append((torch.maximum(pl, el), torch.minimum(ph, eh)))
            stat["proj_tighter"] += int(((pc[0][1] - pc[0][0]) < (ec[0][1] - ec[0][0])).sum()); stat["e2e_tighter"] += int(((ec[0][1] - ec[0][0]) < (pc[0][1] - pc[0][0])).sum()); stat["n"] += n
        # mean-value form on h: h(c) +- (h/2) * sum_k max|g_k|
        hc = net.net(net.feat((xl + xh) / 2)).squeeze(-1)
        rad = (h / 2) * sum(torch.maximum(cl.abs(), ch.abs()) for cl, ch in comps)
        hlo, hhi = torch.maximum(hlo.reshape(-1), hc - rad), torch.minimum(hhi.reshape(-1), hc + rad)
        slo, shi = torch.sigmoid(hlo), torch.sigmoid(hhi)                        # softplus slope interval
        lo, hi = torch.nn.functional.softplus(hlo), torch.nn.functional.softplus(hhi)
        r_, c_ = ii[s:s + n], jj[s:s + n]
        LO[r_, c_] = lo[:n].cpu().numpy(); HI[r_, c_] = hi[:n].cpu().numpy()
        for k, (cl, ch) in enumerate(comps):
            gl, gh = iv_prod(slo, shi, cl, ch)                                   # grad lambda = s * grad h
            G[2 * k][r_, c_] = gl[:n].cpu().numpy(); G[2 * k + 1][r_, c_] = gh[:n].cpu().numpy()
        if (s // a.batch) % 200 == 0:
            print(f"  {s + n:,}/{P:,} pixels, {time.time() - t0:.0f}s", flush=True)
# empirical check: autograd gradient and value at pixel centres must lie inside the intervals
with torch.enable_grad():
    xr = xc.clone().requires_grad_(True); lam = net(xr); gr = torch.autograd.grad(lam.sum(), xr)[0].detach().cpu().numpy(); lam = lam.detach().cpu().numpy()
viol_v = int(((lam < LO[ii, jj] - 1e-5) | (lam > HI[ii, jj] + 1e-5)).sum())
viol_g = int(((gr[:, 0] < G[0][ii, jj] - 1e-4) | (gr[:, 0] > G[1][ii, jj] + 1e-4) | (gr[:, 1] < G[2][ii, jj] - 1e-4) | (gr[:, 1] > G[3][ii, jj] + 1e-4)).sum())
gw = np.median(np.maximum(G[1][ii, jj] - G[0][ii, jj], G[3][ii, jj] - G[2][ii, jj])); gmax = float(np.sqrt(np.maximum(G[0][ii, jj] ** 2, G[1][ii, jj] ** 2) + np.maximum(G[2][ii, jj] ** 2, G[3][ii, jj] ** 2)).max())
gemp = float(np.linalg.norm(gr, axis=1).max())
info = dict(field=a.field, pixels=P, side=h, seconds=time.time() - t0, value_width_median=float(np.median(HI[ii, jj] - LO[ii, jj])), field_range=float(np.nanmax(HI) - np.nanmin(LO)),
            grad_width_median=float(gw), grad_norm_bound_max=gmax, grad_norm_empirical_max=gemp, ratio_max=gmax / gemp,
            grad_ratio_median=float(np.median(np.sqrt(np.maximum(G[0][ii, jj] ** 2, G[1][ii, jj] ** 2) + np.maximum(G[2][ii, jj] ** 2, G[3][ii, jj] ** 2)) / np.maximum(np.linalg.norm(gr, axis=1), 1e-6))),
            violations_value=viol_v, violations_grad=viol_g, projected_tighter_share=stat["proj_tighter"] / max(stat["n"], 1), e2e_tighter_share=stat["e2e_tighter"] / max(stat["n"], 1))
print(json.dumps(info, indent=1))
os.makedirs(os.path.dirname(paths.enclosure(a.field)), exist_ok=True)
np.savez_compressed(paths.enclosure(a.field), lo=LO, hi=HI, g1l=G[0], g1h=G[1], g2l=G[2], g2h=G[3], info=json.dumps(info))
print("saved", paths.rel(paths.enclosure(a.field)))
