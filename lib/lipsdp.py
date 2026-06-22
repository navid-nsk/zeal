"""
LipSDP-Neuron (Fazlyab et al. 2019) with GENERAL per-layer slope sectors.

Certifies an ell2 Lipschitz upper bound for a feedforward net
   f(u) = W_L phi_L( ... phi_1( W_0 u ) ... )
where each activation layer k applies a coordinatewise slope-restricted nonlinearity
phi_k with incremental slope sector [a_k, b_k]:
   softplus / relu / sigmoid : [0, 1]
   sine / cosine             : [-1, 1]     (the user's IQC extension)

Derivation (S-procedure). For each hidden neuron i with pre-activation diff p_i and
output diff q_i, the sector gives the incremental QC
   (q_i - a_i p_i)(b_i p_i - q_i) >= 0.
With stacked z=[dx_0; dx_1; ...; dx_L], A z = all pre-activations, B z = all outputs,
the squared-Lipschitz rho is certified if there is diagonal Lambda=diag(lam)>=0 with
   M = rho * E0^T E0  -  E_L^T W_L^T W_L E_L
       - [ 0.5(A^T D1 B + B^T D1 A) - A^T D2 A - B^T Lambda B ]  >= 0,
   D1 = diag(lam*(a+b)),  D2 = diag(lam*a*b).
Minimize rho.  Returns sqrt(rho).
"""
import numpy as np
import cvxpy as cp


def lipschitz_sdp(Ws, sectors, solver="CLARABEL", verbose=False):
    """
    Ws      : [W_0, ..., W_L]  numpy arrays. L hidden activation layers (phi after W_0..W_{L-1}),
              W_L is the linear output map.  W_k has shape (n_{k+1}, n_k).
    sectors : list of (a_k, b_k), length L, one per hidden activation layer.
    returns : Lipschitz upper bound (float).
    """
    Ws = [np.asarray(W, dtype=float) for W in Ws]
    L = len(Ws) - 1
    assert len(sectors) == L, "one sector per hidden activation layer"
    n0 = Ws[0].shape[1]
    sizes = [Ws[k].shape[0] for k in range(L)]      # hidden sizes n_1..n_L
    n = sum(sizes)
    dim = n0 + n
    off = [0, n0]
    for s in sizes:
        off.append(off[-1] + s)                     # block starts of x_0,x_1,...,x_L

    A = np.zeros((n, dim)); B = np.zeros((n, dim))
    a_vec = np.zeros(n); b_vec = np.zeros(n)
    r = 0
    for k in range(L):
        A[r:r + sizes[k], off[k]:off[k] + Ws[k].shape[1]] = Ws[k]      # preact of layer k+1 = W_k x_k
        B[r:r + sizes[k], off[k + 1]:off[k + 1] + sizes[k]] = np.eye(sizes[k])  # output x_{k+1}
        a_vec[r:r + sizes[k]] = sectors[k][0]
        b_vec[r:r + sizes[k]] = sectors[k][1]
        r += sizes[k]

    E0 = np.zeros((n0, dim)); E0[:, 0:n0] = np.eye(n0)
    EL = np.zeros((sizes[-1], dim)); EL[:, off[L]:off[L] + sizes[-1]] = np.eye(sizes[-1])
    WL = Ws[-1]
    C_out = EL.T @ WL.T @ WL @ EL
    E0E0 = E0.T @ E0
    ab_sum = a_vec + b_vec
    ab_prod = a_vec * b_vec

    rho = cp.Variable(nonneg=True)
    lam = cp.Variable(n, nonneg=True)

    def rowscale(dvec, Mat):                # = diag(dvec) @ Mat, WITHOUT forming an n x n diagonal
        return cp.multiply(cp.reshape(dvec, (n, 1), order="C"), Mat)

    d1 = cp.multiply(lam, ab_sum)           # lam_i (a_i + b_i)
    d2 = cp.multiply(lam, ab_prod)          # lam_i a_i b_i
    X = A.T @ rowscale(d1, B)               # A^T diag(d1) B
    QC = 0.5 * (X + X.T) - A.T @ rowscale(d2, A) - B.T @ rowscale(lam, B)
    M = rho * E0E0 - C_out - QC
    prob = cp.Problem(cp.Minimize(rho), [M >> 0])
    prob.solve(solver=solver, verbose=verbose)
    if rho.value is None:
        raise RuntimeError(f"SDP failed: {prob.status}")
    return float(np.sqrt(max(rho.value, 0.0)))


if __name__ == "__main__":
    # sanity: 1 hidden layer, softplus sector [0,1]; LipSDP must lie in [Lip_true, ||W1||*||W0||]
    rng = np.random.default_rng(0)
    W0 = rng.standard_normal((8, 3)); W1 = rng.standard_normal((1, 8))
    prod = np.linalg.norm(W1, 2) * np.linalg.norm(W0, 2)
    L = lipschitz_sdp([W0, W1], [(0.0, 1.0)])
    print(f"1-hidden-layer softplus:  LipSDP={L:.3f}   product||W1||||W0||={prod:.3f}   (LipSDP<=product expected)")
