"""c5_glue_audit.py -- gluing audit (results exp2/C5): are the per-county single-field map witnesses one continuous field on the
raster? For every pair of adjacent coarse cells the shared mesh nodes (corners and edge midpoints on the common boundary) carry two
independently optimized values; we report the mismatch |x1 - x2| (its max and population-weighted RMS) against the local value-box width
and the gradient-bound tolerance (continuity requires equality: tolerance 0). Also the share of shared nodes with mismatch below
1e-9 (glued by chance). Outcome: the product-class bracket is exact; the global-class lower bound needs one glued field (c5_global_witness_bf.py).
Usage: python c5_glue_audit.py --field georgia."""
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("zeal", "fields")]; import paths  # data/results roots: ZEAL_DATA_ROOT, ZEAL_RESULTS_ROOT
import argparse, io, json, os, sys, glob, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from synth import SPEC
p = argparse.ArgumentParser(); p.add_argument("--field", default="georgia"); a = p.parse_args()
OUT = paths.results("exp2", "C5"); FD = os.path.join(OUT, "mapwitness_fields")
F = np.load(paths.enclosure(a.field)); lo0, hi0 = F["lo"].astype(float).T, F["hi"].astype(float).T
ras = np.load(SPEC[a.field]); mask = ras["mask"].astype(bool).T & np.isfinite(lo0); coarse = np.where(mask, ras["county_idx"].T, -1)
files = sorted(glob.glob(os.path.join(FD, f"{a.field}_cell*.npz"))); fields = {}
for f in files:
    z = np.load(f); b = int(os.path.basename(f).split("cell")[1].split(".")[0]); n1, n2, Mz, mk = int(z["n1"]), int(z["n2"]), int(z["M"]), z["mask"]
    # reconstruct the Mesh's compaction: used nodes = union of the (M+1)^2 nodes of every in-mask pixel, ordered by nid = a*N2 + b
    N2 = n2 * Mz + 1; used = np.zeros((n1 * Mz + 1) * N2, bool)
    for i, j in zip(*np.nonzero(mk)):
        for a_ in range(i * Mz, i * Mz + Mz + 1):
            used[a_ * N2 + j * Mz: a_ * N2 + j * Mz + Mz + 1] = True
    col = -np.ones(used.size, np.int64); col[used] = np.arange(used.sum()); assert used.sum() == z["x"].shape[0], (f, used.sum(), z["x"].shape)
    fields[b] = dict(x=z["x"], i0=int(z["i0"]), j0=int(z["j0"]), n1=n1, n2=n2, mask=mk, M=Mz, col=col)
M = next(iter(fields.values()))["M"] if fields else 4
def node_val(fd, gi, gj):
    """value of global mesh node (gi, gj) in patch fd, or None if outside / not adjacent to an in-mask pixel of that patch"""
    a_, b_ = gi - fd["i0"] * M, gj - fd["j0"] * M; N2 = fd["n2"] * M + 1
    if a_ < 0 or b_ < 0 or a_ > fd["n1"] * M or b_ > fd["n2"] * M:
        return None
    adj = False
    for i in (a_ // M - (1 if a_ % M == 0 else 0), a_ // M):
        for j in (b_ // M - (1 if b_ % M == 0 else 0), b_ // M):
            if 0 <= i < fd["n1"] and 0 <= j < fd["n2"] and fd["mask"][i, j]:
                adj = True
    k = fd["col"][a_ * N2 + b_]
    return float(fd["x"][k]) if (adj and k >= 0) else None
# adjacent cell pairs and their shared boundary nodes: for every pixel edge between pixels of different cells, the M+1 mesh nodes on that edge
mism = []; shared = 0; glued = 0; pairs = set()
n1g, n2g = mask.shape
for i in range(n1g):
    for j in range(n2g):
        if not mask[i, j] or coarse[i, j] not in fields:
            continue
        c1 = coarse[i, j]
        for (di, dj) in ((1, 0), (0, 1)):
            i2, j2 = i + di, j + dj
            if i2 >= n1g or j2 >= n2g or not mask[i2, j2] or coarse[i2, j2] == c1 or coarse[i2, j2] not in fields:
                continue
            c2 = coarse[i2, j2]; pairs.add((min(c1, c2), max(c1, c2)))
            # shared edge nodes: if di == 1 the edge is at mesh a = (i+1)*M, b from j*M..(j+1)*M
            for t in range(M + 1):
                gi, gj = ((i + 1) * M, j * M + t) if di == 1 else (i * M + t, (j + 1) * M)
                v1, v2 = node_val(fields[c1], gi, gj), node_val(fields[c2], gi, gj)
                if v1 is None or v2 is None:
                    continue
                shared += 1; d = abs(v1 - v2); mism.append((d, hi0[i, j] - lo0[i, j])); glued += int(d < 1e-9)
mism = np.array(mism) if mism else np.zeros((0, 2))
S = dict(field=a.field, cells_with_fields=len(fields), adjacent_cell_pairs=len(pairs), shared_boundary_nodes=int(shared), glued_share=float(glued / max(shared, 1)),
         mismatch_median=float(np.median(mism[:, 0])) if len(mism) else None, mismatch_max=float(np.max(mism[:, 0])) if len(mism) else None,
         mismatch_over_local_box_width_median=float(np.median(mism[:, 0] / np.maximum(mism[:, 1], 1e-12))) if len(mism) else None,
         verdict="the per-cell witnesses are one field of the PRODUCT class (independent patches with free boundaries); they are NOT a continuous global field unless mismatch == 0 everywhere; the global-class lower bound needs one glued field (c5_global_witness.py, c5_global_witness_bf.py)")
json.dump(S, open(os.path.join(OUT, f"c5_glue_audit_{a.field}.json"), "w"), indent=1); print(json.dumps(S, indent=1))
