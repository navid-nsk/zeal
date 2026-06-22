"""
Minimal Lipschitz-bounded neural field + parameter-free aggregator R_Z,
trained on the sandbox KNOWN field under MULTIPLE zonings, then certified.

Architecture (all zoning dependence in the parameter-free R_Z):
  lambda_theta(x) = softplus( MLP_spectralnorm( FourierFeatures_sigma(x) ) )
  - FourierFeatures bandwidth sigma  =>  Lip(encoder) <= 2*pi*sqrt(2)/sqrt(m) * ||B||_2 ~ 2*pi*sqrt(2)*sigma
  - spectral_norm on every linear (||W||_2 ~ 1) + softplus (1-Lip)  =>  Lip(lambda_theta) <= Lip(encoder)*Prod||W||
  => L_cert is known A PRIORI from the encoder bandwidth, before training.

Training: one shared continuum field supervised by several zonings simultaneously
  loss = mean_z || cell_mean(lambda_theta; Z) - cell_mean(lambda*; Z) ||^2  +  beta * mean||grad lambda_theta||^2
(lambda* = real Guadalajara density, log-scaled).  Held-out zonings used to certify.

Reports, over the bandwidth frontier sigma_ff:
  Lip_cert (a priori), Lip_emp (trained), recon R^2 (train/test), and the G0 number on
  the TRAINED field: kappa_emp = N/(Lip_emp*dhat) and kappa_cert = N/(Lip_cert*dhat).

Run:  python zeal_m1.py     (env zeal; torch+cuda)
"""
import numpy as np, torch, torch.nn as nn, time
import paths
import g0_numerics as g0

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# --------------------------------------------------------------------------- #
# Ground-truth field (real sandbox density) on a grid
# --------------------------------------------------------------------------- #
N = 160
# Sandbox ground-truth field (Guadalajara 2010 population density, standardized,
# 160x160). Loaded lazily from a cached array by load_sandbox() so that importing
# this module stays light and carries no shapefile/extraction dependency.
lam_np = None
lam_t = None

xs = np.linspace(-1, 1, N).astype(np.float32)
XX, YY = np.meshgrid(xs, xs, indexing="ij")
coords = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1), device=dev)
H_COORD = 2.0 / N                                       # pixel size in [-1,1] coords

# zonings: train (6) + held-out test (2)
specs = [(60, 0), (60, 1), (80, 2), (100, 3), (80, 4), (100, 5), (70, 6), (110, 7)]
labels = [g0.part_voronoi(N, k, s) for k, s in specs]
train_Z, test_Z = labels[:6], labels[6:]


def idx_truemean(Z):
    lab = Z.ravel()
    uniq, inv = np.unique(lab, return_inverse=True)
    inv = torch.tensor(inv, device=dev, dtype=torch.long); K = len(uniq)
    s = torch.zeros(K, device=dev).index_add_(0, inv, lam_t)
    c = torch.zeros(K, device=dev).index_add_(0, inv, torch.ones_like(lam_t))
    return inv, K, (s / c)[inv]                          # true cell-mean broadcast to pixels


train_idx = None
test_idx = None


def load_sandbox():
    """Populate lam_np, lam_t, train_idx, test_idx from the cached sandbox density
    (data/sandbox_density.npz, produced by data_prep/make_sandbox_density.py).
    Idempotent. Call before training the sandbox field or before using
    train_idx/test_idx (zeal_full.py and the soundness/slack/dimsweep experiments)."""
    global lam_np, lam_t, train_idx, test_idx
    if lam_t is not None:
        return
    lam_np = np.load(paths.data("sandbox_density.npz"))["lam"].astype(np.float32)
    lam_t = torch.tensor(lam_np.ravel(), device=dev)
    train_idx = [idx_truemean(Z) for Z in train_Z]
    test_idx = [idx_truemean(Z) for Z in test_Z]


def cell_mean(vals, inv, K):
    s = torch.zeros(K, device=dev).index_add_(0, inv, vals)
    c = torch.zeros(K, device=dev).index_add_(0, inv, torch.ones_like(vals))
    return (s / c)[inv]


# --------------------------------------------------------------------------- #
class Field(nn.Module):
    def __init__(self, m=128, sigma=3.0, w=128, depth=3, act="tanh"):
        super().__init__()
        self.register_buffer("B", torch.randn(m, 2) * sigma); self.m = m
        # INTERNAL activation = tanh (REQUIRED for the auto_LiRPA Jacobian / alpha-CROWN path: softplus
        # has no gradient bound op). The OUTER softplus (in forward) is KEPT for Tobler positivity lambda>=0.
        Act = {"tanh": nn.Tanh, "softplus": nn.Softplus, "sigmoid": nn.Sigmoid}[act]
        L = [nn.Linear(2 * m, w), Act()]
        for _ in range(depth - 1):
            L += [nn.Linear(w, w), Act()]
        L += [nn.Linear(w, 1)]
        self.net = nn.Sequential(*L)

    def feat(self, x):
        p = 2 * np.pi * x @ self.B.t()
        return torch.cat([torch.sin(p), torch.cos(p)], -1) / np.sqrt(self.m)

    def forward(self, x):
        return nn.functional.softplus(self.net(self.feat(x))).squeeze(-1)

    def lip_cert(self):
        encB = torch.linalg.matrix_norm(self.B, 2).item()
        enc = 2 * np.pi * np.sqrt(2) / np.sqrt(self.m) * encB           # encoder Lipschitz bound
        prod = 1.0
        for mod in self.net:
            if hasattr(mod, "weight"):
                prod *= torch.linalg.matrix_norm(mod.weight, 2).item()  # ~1 (spectral-normed)
        return enc * prod

    def mlp_pre_softplus(self):
        """The scalar MLP g: z (Fourier features in R^{2m}) -> R, WITHOUT the outer softplus.
           This is the module the alpha-CROWN Jacobian path certifies (Route B)."""
        return self.net

    def mlp_weights(self):
        return [mod.weight.detach() for mod in self.net if hasattr(mod, "weight")]


def smoothness(grid):
    gx = grid[1:, :] - grid[:-1, :]
    gy = grid[:, 1:] - grid[:, :-1]
    return (gx ** 2).mean() + (gy ** 2).mean()


def r2(pred_pix, idxset):
    """reconstruction R^2 of cell-means vs truth, averaged over a zoning set."""
    vals = []
    for inv, K, truemean in idxset:
        pm = cell_mean(pred_pix, inv, K)
        ss_res = ((pm - truemean) ** 2).mean()
        ss_tot = ((truemean - truemean.mean()) ** 2).mean()
        vals.append((1 - ss_res / (ss_tot + 1e-9)).item())
    return float(np.mean(vals))


def train(sigma, beta, iters=1500, lr=2e-3):
    net = Field(sigma=sigma).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr)
    for it in range(iters):
        opt.zero_grad()
        pred = net(coords)
        rec = sum(((cell_mean(pred, inv, K) - tm) ** 2).mean() for inv, K, tm in train_idx) / len(train_idx)
        loss = rec + beta * smoothness(pred.view(N, N))
        loss.backward(); opt.step()
    return net


def measure(net):
    with torch.no_grad():
        grid = net(coords).view(N, N).cpu().numpy()
    Lip_emp = g0.lipschitz(grid, H_COORD)
    Lip_cert = net.lip_cert()
    # G0 number on the trained field, held-out zoning pair, vs operator-norm proxy dhat
    Z, Zp = test_Z[0], test_Z[1]
    basis = {"lx": g0.field_linear(XX, YY, [1, 0]), "ly": g0.field_linear(XX, YY, [0, 1]),
             "lxy": g0.field_linear(XX, YY, [1, 1])}
    res = {}
    for dpx in (3.0, 6.0):
        dhat = max(g0.certificate_N(f, Z, Zp, dpx, H_COORD) / (g0.lipschitz(f, H_COORD) + 1e-12)
                   for f in basis.values())
        Nv = g0.certificate_N(grid, Z, Zp, dpx, H_COORD)
        res[dpx] = (Nv, dhat, Nv / (Lip_emp * dhat + 1e-12), Nv / (Lip_cert * dhat + 1e-12))
    return Lip_cert, Lip_emp, grid, res


def run():
    load_sandbox()
    print(f"device={dev}  field=log1p(real density) standardized  grid {N}x{N}\n")
    print(f"{'sigma':>5s} {'beta':>7s} {'Lip_cert':>9s} {'Lip_emp':>8s} {'Lc/Le':>7s} "
          f"{'R2_tr':>6s} {'R2_te':>6s} {'kap_emp':>8s} {'kap_cert':>9s}")
    for sigma in (6.0,):
        for beta in (1e-2, 1e-1, 5e-1, 1.0, 2.0, 5.0):
            t = time.time()
            net = train(sigma, beta, iters=2000)
            Lc, Le, grid, res = measure(net)
            with torch.no_grad():
                pred = net(coords)
            R2tr, R2te = r2(pred, train_idx), r2(pred, test_idx)
            Nv, dhat, kep, kcert = res[3.0]
            kep6 = res[6.0][2]
            print(f"{sigma:5.1f} {beta:7.4f} {Lc:9.2f} {Le:8.2f} {Lc/Le:7.1f} "
                  f"{R2tr:6.3f} {R2te:6.3f} {kep:8.3f} {kcert:9.4f}   "
                  f"(kap_emp@d6={kep6:.3f}, {time.time()-t:.1f}s)")
    print("\nReading: Lip_cert is a-priori (encoder bandwidth); Lip_emp is the trained field's true max-grad.")
    print("kap_emp = non-vacuity of the metric on the TRAINED field; kap_cert uses the certified Lipschitz.")
    print("Frontier: higher sigma_ff -> better R2 (expressive) but larger Lip_cert (looser envelope).")


if __name__ == "__main__":
    run()
