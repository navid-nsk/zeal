"""
Gap 7: soundness twin-curve (example's signature device). On the production ZEAL field, show the CERTIFIED
upper bound on sup_x ||grad lambda(x)|| (manifold_cert_field, solid) staying ABOVE the empirically-attained
worst-case (dense-autograd max, dashed) at every branch-and-bound budget, converging toward it. Demonstrates
soundness (cert >= empirical everywhere) AND tightness (ratio -> ~1.5x).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import torch, numpy as np, json, time
import zeal_m1 as M
from zeal_manifold_cert import manifold_cert_field
dev = M.dev

ck = torch.load(paths.data("zeal_full_field.pt"), map_location=dev)
net = M.Field(m=ck["m"], sigma=ck["sigma"], w=ck["w"], depth=ck["depth"]).to(dev)
net.load_state_dict(ck["state"]); net.eval()

# empirical worst-case ||grad lambda|| via dense autograd (the dashed floor)
g = torch.linspace(-1, 1, 1400, device=dev)
XX, YY = torch.meshgrid(g, g, indexing="ij")
pts = torch.stack([XX.reshape(-1), YY.reshape(-1)], 1)
emp = 0.0
for i in range(0, pts.shape[0], 120000):
    p = pts[i:i + 120000].detach().requires_grad_(True)
    grd = torch.autograd.grad(net(p).sum(), p)[0]
    emp = max(emp, grd.norm(dim=1).max().item())
print(f"empirical worst-case ||grad lambda|| = {emp:.2f}  (saved Lip_emp={ck['Lip_emp']:.2f})", flush=True)

budgets = [10000, 30000, 80000, 200000, 500000]
rows = []
print(f"{'budget':>8s} {'cert':>8s} {'ratio':>7s} {'tiles':>8s}  time")
for b in budgets:
    t = time.time()
    cert, ntiles = manifold_cert_field(net, budget=b)
    rows.append({"budget": b, "cert": float(cert), "tiles": int(ntiles), "ratio": float(cert / emp)})
    print(f"{b:8d} {cert:8.2f} {cert/emp:6.2f}x {ntiles:8d}  ({time.time()-t:.0f}s)", flush=True)
    assert cert >= emp - 1e-6, "SOUNDNESS VIOLATION: cert < empirical!"

json.dump({"empirical": emp, "saved_lip": ck["Lip_emp"], "rows": rows},
          open(paths.data("zeal_soundness_curve.json"), "w"))
print(f"\nSoundness holds at all budgets (cert >= empirical {emp:.1f}); ratio converges to "
      f"{rows[-1]['ratio']:.2f}x at {budgets[-1]} tiles. Saved zeal_soundness_curve.json")
