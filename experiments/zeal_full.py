"""
Train and SAVE the FULL production ZEAL field (m=128, w=128, depth=3) at a chosen beta,
so all four certificates (empirical, spectral-product, LipSDP, alpha-CROWN) run on ONE model.

Monitoring:
  - print+flush R2 every checkpoint,
  - SANITY ASSERT after warmup (abort in ~seconds if the field is not learning),
  - report Lip_emp and save weights+B.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths
import torch, numpy as np, time, sys
import zeal_m1 as M, g0_numerics as g0
M.load_sandbox()

BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
OUT = paths.data("zeal_full_field.pt")

net = M.Field(m=128, sigma=6.0, w=128, depth=3).to(M.dev)
opt = torch.optim.Adam(net.parameters(), 2e-3)
t = time.time()
print(f"training full field m=128 w=128 depth=3 beta={BETA} on {M.dev}", flush=True)
for it in range(2600):
    opt.zero_grad()
    pred = net(M.coords)
    rec = sum(((M.cell_mean(pred, inv, K) - tm) ** 2).mean()
              for inv, K, tm in M.train_idx) / len(M.train_idx)
    (rec + BETA * M.smoothness(pred.view(M.N, M.N))).backward()
    opt.step()
    if it in (100, 300, 800, 1500, 2599):
        with torch.no_grad():
            r2 = M.r2(net(M.coords), M.test_idx)
        print(f"  it={it:4d}  loss={(rec).item():.4f}  R2_te={r2:.3f}  ({time.time()-t:.1f}s)", flush=True)
        if it == 300 and r2 < 0.3:
            raise SystemExit("SANITY FAIL: R2_te<0.3 after warmup -> aborting (do not waste time).")
with torch.no_grad():
    grid = net(M.coords).view(M.N, M.N).cpu().numpy()
    r2 = M.r2(net(M.coords), M.test_idx)
Lip_emp = g0.lipschitz(grid, M.H_COORD)
print(f"DONE  R2_te={r2:.3f}  Lip_emp={Lip_emp:.2f}  ({time.time()-t:.1f}s)", flush=True)
torch.save({"state": net.state_dict(), "B": net.B.detach().cpu(),
            "m": 128, "w": 128, "depth": 3, "sigma": 6.0, "beta": BETA,
            "Lip_emp": Lip_emp, "R2_te": r2}, OUT)
np.save(paths.data("zeal_full_grid.npy"), grid)
print(f"saved {OUT} + grid", flush=True)
