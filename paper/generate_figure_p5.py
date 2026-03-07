#!/usr/bin/env python3
"""
Generate Figure 1 for Paper 5: TC vs S (perfect line) alongside T_ZZ vs S (noisy cloud).
Two-panel figure showing the exact identity vs the single-channel proxy.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build_heisenberg_hamiltonian(N, seed):
    rng = np.random.RandomState(seed)
    dim = 2**N
    sx = sparse.csr_matrix(np.array([[0,1],[1,0]], dtype=complex))
    sy = sparse.csr_matrix(np.array([[0,-1j],[1j,0]], dtype=complex))
    sz = sparse.csr_matrix(np.array([[1,0],[0,-1]], dtype=complex))
    eye = sparse.eye(2, dtype=complex)

    def kron_op(op, i, N):
        ops = [eye]*N
        ops[i] = op
        result = ops[0]
        for o in ops[1:]:
            result = sparse.kron(result, o, format='csr')
        return result

    H = sparse.csr_matrix((dim, dim), dtype=complex)
    for i in range(N):
        for j in range(i+1, N):
            Jx, Jy, Jz = rng.uniform(-1, 1, 3)
            Si_x, Sj_x = kron_op(sx, i, N), kron_op(sx, j, N)
            Si_y, Sj_y = kron_op(sy, i, N), kron_op(sy, j, N)
            Si_z, Sj_z = kron_op(sz, i, N), kron_op(sz, j, N)
            H += Jx * Si_x @ Sj_x + Jy * Si_y @ Sj_y + Jz * Si_z @ Sj_z
    return H

def partial_trace(psi, N, keep):
    keep = sorted(keep)
    trace_out = sorted(set(range(N)) - set(keep))
    k, t = len(keep), len(trace_out)
    psi_tensor = psi.reshape([2]*N)
    psi_tensor = np.transpose(psi_tensor, keep + trace_out)
    psi_mat = psi_tensor.reshape(2**k, 2**t)
    return psi_mat @ psi_mat.conj().T

def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-14]
    return -np.sum(evals * np.log2(evals))

def connected_zz(psi, N, i, j):
    dim = 2**N
    def z_op(qubit):
        op = np.ones(dim, dtype=float)
        for bit in range(dim):
            if (bit >> (N-1-qubit)) & 1:
                op[bit] = -1.0
        return op
    zi, zj = z_op(i), z_op(j)
    probs = np.abs(psi)**2
    return abs(np.sum(probs * zi * zj) - np.sum(probs * zi) * np.sum(probs * zj))

# Collect data from multiple seeds
N = 12
k = 6
n_seeds = 5
n_subsets = 100  # More points for visual density

all_TC = []
all_S = []
all_T = []
all_seeds_label = []

print("Generating figure data...")
for seed in range(n_seeds):
    print(f"  Seed {seed}...")
    H = build_heisenberg_hamiltonian(N, seed)
    vals, vecs = eigsh(H, k=1, which='SA')
    psi = vecs[:, 0]

    single_entropies = np.array([von_neumann_entropy(partial_trace(psi, N, [i])) for i in range(N)])

    rng = np.random.RandomState(seed + 50000)
    for _ in range(n_subsets):
        subset = sorted(rng.choice(list(range(N)), k, replace=False).tolist())
        rho_A = partial_trace(psi, N, subset)
        S_obs = von_neumann_entropy(rho_A)
        sum_Si = sum(single_entropies[i] for i in subset)
        TC = sum_Si - S_obs

        T_obs = 0
        n_pairs = 0
        for qi in range(len(subset)):
            for qj in range(qi+1, len(subset)):
                T_obs += connected_zz(psi, N, subset[qi], subset[qj])
                n_pairs += 1
        T_obs /= n_pairs

        all_TC.append(TC)
        all_S.append(S_obs)
        all_T.append(T_obs)
        all_seeds_label.append(seed)

all_TC = np.array(all_TC)
all_S = np.array(all_S)
all_T = np.array(all_T)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

# Colors for different seeds
colors = plt.cm.tab10(np.array(all_seeds_label) / n_seeds)

# Panel (a): TC vs S — perfect anti-correlation
ax1.scatter(all_S, all_TC, s=4, alpha=0.6, c=colors, edgecolors='none')
# Add the identity line: TC = k - S (since sum S_i = k = 6)
s_range = np.array([min(all_S), max(all_S)])
ax1.plot(s_range, k - s_range, 'k--', lw=1.2, label=r'$\mathrm{TC} = k - S$')
ax1.set_xlabel(r'$S(\rho_A)$ [bits]')
ax1.set_ylabel(r'$\mathrm{TC}(A)$ [bits]')
ax1.set_title(r'(a) $r(\mathrm{TC}, S) = -1.0000$')
ax1.legend(fontsize=8, loc='upper right')

# Panel (b): T_ZZ vs S — noisy cloud
ax2.scatter(all_S, all_T, s=4, alpha=0.6, c=colors, edgecolors='none')
r_val = pearsonr(all_T, all_S)[0]
ax2.set_xlabel(r'$S(\rho_A)$ [bits]')
ax2.set_ylabel(r'$T_\mathrm{obs}$ (mean $|C_{ij}^{ZZ}|$)')
ax2.set_title(rf'(b) $r(T_{{ZZ}}, S) = {r_val:.2f}$')

plt.tight_layout()
plt.savefig('figures/fig_p5_budget.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_p5_budget.png', bbox_inches='tight', dpi=300)
print(f"\nFigure saved to figures/fig_p5_budget.pdf and .png")
print(f"Overall r(T_ZZ, S) = {r_val:.4f}")
