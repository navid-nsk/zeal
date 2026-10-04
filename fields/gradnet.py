"""Explicit gradient network of a Field: x (N,2) -> grad h(x) (N,2), written with sin/cos/tanh/mul/linear only
(so that CROWN can bound it with the 2-D location as the perturbed input)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zeal")); import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import math, torch, torch.nn as nn


class GradNet(nn.Module):
    """x (N,2) -> grad h(x) (N,2) for h = g(gamma(x)), g a Tanh perceptron, written with sin/cos/tanh/mul/linear only."""
    def __init__(self, f):
        super().__init__()
        self.register_buffer("B", f.B.detach().clone()); self.m = f.m
        lins = [mod for mod in f.net if isinstance(mod, nn.Linear)]
        self.Ws = nn.ParameterList([nn.Parameter(l.weight.detach().clone(), requires_grad=False) for l in lins])
        self.bs = nn.ParameterList([nn.Parameter(l.bias.detach().clone(), requires_grad=False) for l in lins])
        self.scale = 2 * math.pi / math.sqrt(self.m)

    def forward(self, x):
        pphi = (2 * math.pi) * x @ self.B.t()
        s, c = torch.sin(pphi), torch.cos(pphi)
        z = torch.cat([s, c], -1) / math.sqrt(self.m)
        ts = []; h = z
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = torch.tanh(h @ W.t() + b); ts.append(h)
        u = self.Ws[-1].expand(x.shape[0], -1) * (1.0 - ts[-1] * ts[-1])   # (N, w): through the last tanh'
        for i in range(len(ts) - 2, -1, -1):
            u = (u @ self.Ws[i + 1]) * (1.0 - ts[i] * ts[i])          # back through W_{i+1} and tanh'(a_i)
        v = u @ self.Ws[0]                                            # (N, 2m): grad of g w.r.t. the features
        vs, vc = v[:, :self.m], v[:, self.m:]
        return self.scale * ((vs * c) @ self.B - (vc * s) @ self.B)  # J^T v, chain rule in closed form


