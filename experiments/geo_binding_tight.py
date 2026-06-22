"""A tight and binding cross-zoning example.

The certified envelope L_cert*d is the worst-case re-aggregation movement over ALL
fields with Lipschitz <= L_cert. On Georgia the opportunity surface is smooth and
its gradient is nearly orthogonal to the tract<->county re-aggregation direction,
so it realises a tiny fraction of that worst case -> the envelope looks ~1600x
loose. Here we compute the WORST-CASE field for the SAME re-aggregation and show
the SAME certificate binds to a small factor. The Georgia looseness is therefore a
property of the stable opportunity surface, not a defect of the certificate.

The worst unit-Lipschitz LINEAR field (a planar opportunity gradient aligned to the
county geometry) is exact: its movement is sqrt(lambda_max(M)), M the count-weighted
second moment of the tract->county centroid displacement. Linear fields are the
case the certifier recovers to 1.02x (validation V2), so its L_cert is essentially
exact and the envelope binds.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import numpy as np, torch, json, time
import geo_m4_multistate as M
from zeal_m1 import dev

s = M.S("13"); N = s.N
mi = torch.tensor(s.mask.reshape(-1), device=dev)
w = s.w; W = w / w.sum()
h = 2.0 / (N - 1)


def cen_of(lab, K):
    cnt = torch.bincount(lab, minlength=K).clamp_min(1).float()
    return torch.zeros(K, 2, device=dev).index_add_(0, lab, s.coords) / cnt[:, None]


def agg(lab, K, g):
    num = torch.zeros(K, device=dev).index_add_(0, lab, g * w)
    den = torch.zeros(K, device=dev).index_add_(0, lab, w)
    return (num / den.clamp_min(1e-9))[lab]


def movement(g):
    a = agg(s.inv_tr, s.Ktr, g); b = agg(s.inv_co, s.Kco, g)
    return float((w * (a - b) ** 2).sum().div(w.sum()).sqrt())


m2 = mi.reshape(N, N)
valid = (m2[2:, 1:-1] & m2[:-2, 1:-1] & m2[1:-1, 2:] & m2[1:-1, :-2] & m2[1:-1, 1:-1])


def grid_lip(g):
    grid = torch.zeros(N * N, device=dev).masked_scatter(mi, g).reshape(N, N)
    dx = (grid[2:, 1:-1] - grid[:-2, 1:-1]) / (2 * h)
    dy = (grid[1:-1, 2:] - grid[1:-1, :-2]) / (2 * h)
    return float((dx * dx + dy * dy + 1e-12).sqrt()[valid].max())


def d_geo():
    def cs(lab, K):
        cnt = torch.bincount(lab, minlength=K).clamp_min(1).float()
        cen = torch.zeros(K, 2, device=dev).index_add_(0, lab, s.coords) / cnt[:, None]
        d2 = ((s.coords - cen[lab]) ** 2).sum(1)
        sp = (torch.zeros(K, device=dev).index_add_(0, lab, d2) / cnt).sqrt()
        return cen, sp
    cA, spA = cs(s.inv_tr, s.Ktr); cB, spB = cs(s.inv_co, s.Kco)
    return float((((cA[s.inv_tr] - cB[s.inv_co]).norm(dim=1) + spA[s.inv_tr] + spB[s.inv_co]) ** 2).mean().sqrt())


d = d_geo()

# ---- exact worst-case LINEAR field: top eigenvector of the centroid-displacement second moment ----
cen_tr = cen_of(s.inv_tr, s.Ktr); cen_co = cen_of(s.inv_co, s.Kco)
dcen = cen_tr[s.inv_tr] - cen_co[s.inv_co]                    # per-pixel tract->county centroid shift
Mmat = (W[:, None, None] * dcen[:, :, None] * dcen[:, None, :]).sum(0)    # 2x2, count-weighted
evals, evecs = torch.linalg.eigh(Mmat)
u = evecs[:, -1]                                             # worst direction (unit vector => Lip = 1)
field_lin = (s.coords @ u)                                  # the planar field u . x  (Lip = |u| = 1)
mov_lin = movement(field_lin); lip_lin = grid_lip(field_lin)
d_low = float(Mmat.trace().sqrt())                          # centroid-displacement metric (reference)

# ---- the actual opportunity field (paper, beta=0.5) ----
torch.manual_seed(0); net = s.train(0.5)
with torch.no_grad():
    pred = net(s.coords)
mov_opp = movement(pred); lip_opp = grid_lip(pred)

# ---- ratios (empirical envelope = Lip_emp * d; the certified envelope = L_cert * d adds the certifier slack) ----
env_emp_lin = lip_lin * d / mov_lin                         # worst-case-aligned field: binds tightly
env_emp_opp = lip_opp * d / mov_opp                         # opportunity surface: loose
frac_lin = mov_lin / (lip_lin * d)                          # fraction of the bound realised (worst-case)
frac_opp = mov_opp / (lip_opp * d)                          # fraction realised (opportunity)

out = {"d": d, "d_low": d_low,
       "worst_linear": {"movement": mov_lin, "Lip": lip_lin, "u": u.tolist(),
                        "fraction_of_bound": frac_lin, "envelope_over_movement_emp": env_emp_lin},
       "opportunity": {"movement": mov_opp, "Lip_emp": lip_opp,
                       "fraction_of_bound": frac_opp, "envelope_over_movement_emp": env_emp_opp},
       "binding_advantage": frac_lin / frac_opp}
json.dump(out, open(paths.data("binding_tight.json"), "w"), indent=2)

print("=== TIGHT + BINDING demonstration : Georgia tract<->county ===")
print(f"transport distance d = {d:.4f}   (centroid proxy d_low = {d_low:.4f})")
print(f"WORST-CASE-aligned field (planar, u={u.tolist()}):")
print(f"   movement = {mov_lin:.4f}   Lip = {lip_lin:.3f}   realises {100*frac_lin:.1f}% of its bound   "
      f"-> empirical envelope/movement = {env_emp_lin:.2f}x")
print(f"OPPORTUNITY surface (beta=0.5):")
print(f"   movement = {mov_opp:.4f}   Lip_emp = {lip_opp:.2f}   realises {100*frac_opp:.3f}% of its bound  "
      f"-> empirical envelope/movement = {env_emp_opp:.0f}x")
print(f"-> the worst-case-aligned field is {out['binding_advantage']:.0f}x closer to binding the bound.")
print("Linear fields are certified to 1.02x (validation V2), so the CERTIFIED envelope binds to")
print(f"   ~{env_emp_lin*1.02:.1f}x for the worst-case field, vs the opportunity surface's loose bound.")
print("saved zeal/data/binding_tight.json")
