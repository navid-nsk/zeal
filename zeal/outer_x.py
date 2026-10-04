"""outer_x.py -- the lifted outer set Phi_X (pixel means + edge means + vertex values) and its lens-cut version
Phi_XL; Supplementary Theorem S7 (lifted outer-inner theorem; Definition S3).

Phi_X^(m): one variable per node of the h/(2m) mesh. Centre nodes of the m x m sub-pixels carry SUB-PIXEL MEANS,
mid-edge nodes carry SUB-EDGE MEANS, corner nodes carry POINT VALUES. Constraints = those of the inner Q_{2m} LP
(every mesh edge slope in the box of the pixel containing it; on a pixel boundary the intersection of both boxes;
node values in the intersection of adjacent value boxes). Sound because
   mean_R - mean_pixel = (1/h'^2) int x1 d1f  in (h'/2) G1      (half-tent identity, sub-pixel side h')
   vertex - mean_edge  = (1/h') int s * (tangential derivative)  in (h'/2) G~   (trace derivative in both boxes)
Objectives on the SAME feasible set:
   outer  U_X(m)   = max sum_p d_p * (mean of the centre nodes of p's sub-pixels)
   inner  U_Q(2m)  = max sum_p d_p * (Q1 mean of p)      (a genuine field of the class: Theorem S6)
Theorem S7: U_Q(2m) <= U_F <= U_X(m) <= U_Q(2m) + (h/(8m)) sum_p |d_p| (w1p + w2p); with exact lenses h/(16m).
Lens cuts (Phi_XL): Agarwal-Dragomir/Iyengar inequality, line-averaged (Jensen), K tangents (K odd), per sub-pixel
axis on (mid_L, centre, mid_R) and per sub-edge on (vertex, mid, vertex) with the edge's (intersected) box.
"""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog


class Mesh:
    def __init__(self, lo, hi, G1, G2, h, mask, M):
        n1, n2 = lo.shape; self.n1, self.n2, self.M, self.h = n1, n2, M, h
        N1, N2 = n1 * M + 1, n2 * M + 1; self.N2 = N2
        nid = lambda a, b: a * N2 + b; self.nid = nid
        V = N1 * N2
        lo_n = np.full(V, -np.inf); hi_n = np.full(V, np.inf)
        edges = {}
        def inter(key, g):
            if key in edges:
                e = edges[key]; edges[key] = (max(e[0], g[0]), min(e[1], g[1]))
            else:
                edges[key] = (g[0], g[1])
        self.pix = [(i, j) for i in range(n1) for j in range(n2) if mask[i, j]]
        for (i, j) in self.pix:
            for a in range(i * M, i * M + M + 1):
                for b in range(j * M, j * M + M + 1):
                    k = nid(a, b); lo_n[k] = max(lo_n[k], lo[i, j]); hi_n[k] = min(hi_n[k], hi[i, j])
            for a in range(i * M, i * M + M):
                for b in range(j * M, j * M + M + 1):
                    inter((nid(a, b), nid(a + 1, b)), G1[i, j])
            for a in range(i * M, i * M + M + 1):
                for b in range(j * M, j * M + M):
                    inter((nid(a, b), nid(a, b + 1)), G2[i, j])
        used = np.where(np.isfinite(lo_n))[0]
        self.col = -np.ones(V, np.int64); self.col[used] = np.arange(len(used)); self.nv = len(used)
        self.lo_n, self.hi_n = lo_n[used], hi_n[used]
        self.edges = edges; hs = h / M; self.hs = hs
        R = len(edges); keys = list(edges.keys())
        k1 = self.col[[k[0] for k in keys]]; k2 = self.col[[k[1] for k in keys]]
        smin = np.array([edges[k][0] for k in keys]); smax = np.array([edges[k][1] for k in keys])
        rows = np.concatenate([np.arange(R), np.arange(R), R + np.arange(R), R + np.arange(R)])
        cols = np.concatenate([k2, k1, k1, k2]); vals = np.concatenate([np.ones(R), -np.ones(R), np.ones(R), -np.ones(R)])
        self.A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * R, self.nv)); self.b = np.concatenate([hs * smax, -hs * smin])
        self.empty_edge = bool(np.any(smin > smax + 1e-15))
        self.extra_A = []; self.extra_b = []

    def c(self, a, b):
        return self.col[self.nid(a, b)]

    def obj_q1(self, d):
        M = self.M; o = np.zeros(self.nv)
        for (i, j) in self.pix:
            for a in range(i * M, i * M + M):
                for b in range(j * M, j * M + M):
                    for (x, y) in ((a, b), (a + 1, b), (a, b + 1), (a + 1, b + 1)):
                        o[self.c(x, y)] += d[i, j] / (4 * M * M)
        return o

    def obj_centre(self, d):
        M = self.M; m = M // 2; assert M == 2 * m; o = np.zeros(self.nv)
        for (i, j) in self.pix:
            for k in range(m):
                for l in range(m):
                    o[self.c(i * M + 2 * k + 1, j * M + 2 * l + 1)] += d[i, j] / (m * m)
        return o

    def add_lens(self, G1, G2, K=9):
        """Agarwal-Dragomir/Iyengar lens cuts (K tangents, K odd so that the apex tangent is included)."""
        M = self.M; m = M // 2; hs = self.hs; rows = []; rhs = []
        seen = set()
        def lens(A, C, B, ga, gb, length):
            w = gb - ga
            if w <= 1e-14:
                return
            for t in np.linspace(0, 1, K):
                m0 = ga + t * w; rho0 = (length / (2 * w)) * (gb - m0) * (m0 - ga); sl = (length / (2 * w)) * (ga + gb - 2 * m0)
                for sgn in (1, -1):
                    rows.append({C: sgn, A: -sgn / 2 + sl / length, B: -sgn / 2 - sl / length}); rhs.append(rho0 - sl * m0)
        for (i, j) in self.pix:
            for k in range(m):
                for l in range(m):
                    ca, cb = i * M + 2 * k + 1, j * M + 2 * l + 1
                    lens(self.c(ca - 1, cb), self.c(ca, cb), self.c(ca + 1, cb), G1[i, j, 0], G1[i, j, 1], 2 * hs)
                    lens(self.c(ca, cb - 1), self.c(ca, cb), self.c(ca, cb + 1), G2[i, j, 0], G2[i, j, 1], 2 * hs)
                    # the four sub-edges of this sub-pixel (deduplicated); box = intersected mesh-edge box
                    for (A_, C_, B_) in (((ca - 1, cb - 1), (ca - 1, cb), (ca - 1, cb + 1)), ((ca + 1, cb - 1), (ca + 1, cb), (ca + 1, cb + 1)),
                                         ((ca - 1, cb - 1), (ca, cb - 1), (ca + 1, cb - 1)), ((ca - 1, cb + 1), (ca, cb + 1), (ca + 1, cb + 1))):
                        key = (A_, B_)
                        if key in seen:
                            continue
                        seen.add(key)
                        g = self.edges[(self.nid(*A_), self.nid(*C_))]; g2 = self.edges[(self.nid(*C_), self.nid(*B_))]
                        ga, gb = max(g[0], g2[0]), min(g[1], g2[1])
                        lens(self.c(*A_), self.c(*C_), self.c(*B_), ga, gb, 2 * hs)
        R = len(rows); r_, c_, v_ = [], [], []
        for r, row in enumerate(rows):
            for k, v in row.items():
                r_.append(r); c_.append(k); v_.append(v)
        self.extra_A.append(sp.csr_matrix((v_, (r_, c_)), shape=(R, self.nv))); self.extra_b.append(np.array(rhs))
        return R

    def system(self):
        A = sp.vstack([self.A] + self.extra_A).tocsr(); b = np.concatenate([self.b] + self.extra_b)
        return A, b

    def solve(self, obj):
        A, b = self.system()
        res = linprog(-obj, A_ub=A, b_ub=b, bounds=list(zip(self.lo_n, self.hi_n)), method="highs")
        if res.status != 0:
            return np.nan, None
        return -res.fun, res.x

    def deviation_bound(self, d, G1, G2, lens=False):
        """sum_p |d_p| * (per-pixel bound on |Q1 mean - centre mean|) of Theorem S7, using intersected edge widths."""
        M = self.M; m = M // 2; hs = self.hs; tot = 0.0
        for (i, j) in self.pix:
            w1 = G1[i, j, 1] - G1[i, j, 0]; w2 = G2[i, j, 1] - G2[i, j, 0]; hp = 2 * hs   # sub-pixel side
            for k in range(m):
                for l in range(m):
                    ca, cb = i * M + 2 * k + 1, j * M + 2 * l + 1; we = 0.0
                    for (A_, C_) in (((ca - 1, cb - 1), (ca - 1, cb)), ((ca + 1, cb - 1), (ca + 1, cb)), ((ca - 1, cb - 1), (ca, cb - 1)), ((ca - 1, cb + 1), (ca, cb + 1))):
                        g = self.edges[(self.nid(*A_), self.nid(*C_))]; we += g[1] - g[0]
                    f = 0.5 if lens else 1.0
                    tot += abs(d[i, j]) / (m * m) * f * (3 * hp / 32 * (w1 + w2) + hp / 64 * we)
        return tot


def phi_X(lo, hi, G1, G2, h, mask, d, m=1, lens=False, K=9, both_signs=True):
    """returns dict(U_X, L_X, U_Q, L_Q, bound) on the Q_{2m}-structured lifted set (lens optional)."""
    ms = Mesh(lo, hi, G1, G2, h, mask, 2 * m)
    if lens:
        ms.add_lens(G1, G2, K)
    oc, oq = ms.obj_centre(d), ms.obj_q1(d)
    out = {}
    for sgn, nm in ((1, "U"), (-1, "L")):
        if sgn < 0 and not both_signs:
            continue
        vX, xX = ms.solve(sgn * oc); vQ, xQ = ms.solve(sgn * oq)
        out[f"{nm}_X"] = sgn * vX; out[f"{nm}_Q"] = sgn * vQ
        if xX is not None:   # the Q_{2m} witness built from the outer maximizer (Theorem S7)
            out[f"{nm}_witness"] = sgn * float(sgn * oq @ xX)
    out["bound"] = ms.deviation_bound(d, G1, G2, lens=lens)
    out["bound_simple"] = float(h / (8 * m) * (0.5 if lens else 1.0) * np.sum(np.abs(d[mask]) * ((G1[..., 1] - G1[..., 0])[mask] + (G2[..., 1] - G2[..., 0])[mask])))
    out["nv"] = ms.nv; out["nrows"] = ms.system()[0].shape[0]
    return out, ms
