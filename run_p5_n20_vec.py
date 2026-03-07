#!/usr/bin/env python3
"""
Paper 5 N=20 Verification — VECTORIZED.
Uses numpy vectorized operations for Hamiltonian construction.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr
import json
import time
import sys

def build_heisenberg_vectorized(N, seed):
    """Build Heisenberg Hamiltonian using vectorized operations."""
    rng = np.random.RandomState(seed)
    dim = 2**N
    states = np.arange(dim, dtype=np.int64)

    # Accumulate diagonal (ZZ) terms
    diag = np.zeros(dim, dtype=np.float64)

    # Accumulate off-diagonal (XX+YY) terms
    off_rows = []
    off_cols = []
    off_vals = []

    for i in range(N):
        for j in range(i+1, N):
            Jx = rng.uniform(-1, 1)
            Jy = rng.uniform(-1, 1)
            Jz = rng.uniform(-1, 1)

            bit_i = N - 1 - i
            bit_j = N - 1 - j

            # ZZ: si * sj where si = 1 - 2*bit
            si = 1 - 2 * ((states >> bit_i) & 1).astype(np.float64)
            sj = 1 - 2 * ((states >> bit_j) & 1).astype(np.float64)
            diag += Jz * si * sj

            # XX+YY: flip both bits i and j
            flipped = states ^ (1 << bit_i) ^ (1 << bit_j)

            # si_bit and sj_bit for YY sign
            si_bit = (states >> bit_i) & 1
            sj_bit = (states >> bit_j) & 1
            # aligned (same bits): YY gives -1; anti-aligned: YY gives +1
            yy_sign = np.where(si_bit == sj_bit, -1.0, 1.0)

            coeff = Jx + Jy * yy_sign

            # Only add nonzero entries
            mask = np.abs(coeff) > 1e-15
            if np.any(mask):
                off_rows.append(flipped[mask])
                off_cols.append(states[mask])
                off_vals.append(coeff[mask])

    # Build sparse matrix
    all_rows = np.concatenate([states] + off_rows)
    all_cols = np.concatenate([states] + off_cols)
    all_vals = np.concatenate([diag] + off_vals)

    H = sparse.csr_matrix((all_vals, (all_rows, all_cols)), shape=(dim, dim))
    return H

def partial_trace(psi, N, keep):
    keep = sorted(keep)
    trace_out = sorted(set(range(N)) - set(keep))
    k = len(keep)
    t = len(trace_out)
    psi_tensor = psi.reshape([2]*N)
    perm = keep + trace_out
    psi_tensor = np.transpose(psi_tensor, perm)
    psi_mat = psi_tensor.reshape(2**k, 2**t)
    return psi_mat @ psi_mat.conj().T

def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-14]
    return -np.sum(evals * np.log2(evals))

def connected_zz_vectorized(psi, N, pairs):
    """Compute connected ZZ for multiple pairs at once."""
    dim = 2**N
    probs = np.abs(psi)**2
    states = np.arange(dim, dtype=np.int64)

    results = []
    for i, j in pairs:
        bit_i = N - 1 - i
        bit_j = N - 1 - j
        zi = 1 - 2 * ((states >> bit_i) & 1).astype(np.float64)
        zj = 1 - 2 * ((states >> bit_j) & 1).astype(np.float64)
        ez_i = np.dot(probs, zi)
        ez_j = np.dot(probs, zj)
        ez_ij = np.dot(probs, zi * zj)
        results.append(abs(ez_ij - ez_i * ez_j))
    return results

# =============================================================
N = 20
n_seeds = 5
n_subsets = 50

results = {'N': N, 'n_seeds': n_seeds, 'n_subsets': n_subsets, 'seeds': []}

print("=" * 60, flush=True)
print(f"PAPER 5 N={N} VERIFICATION (Hilbert dim = {2**N:,})", flush=True)
print("=" * 60, flush=True)

for k_obs in [10, 6]:
    print(f"\n--- k = {k_obs} ---", flush=True)

    for seed in range(n_seeds):
        t0 = time.time()
        print(f"\nSeed {seed}, k={k_obs}...", flush=True)

        t_ham = time.time()
        H = build_heisenberg_vectorized(N, seed)
        print(f"  Hamiltonian: {time.time()-t_ham:.1f}s, nnz={H.nnz:,}", flush=True)

        t_gs = time.time()
        vals, vecs = eigsh(H, k=1, which='SA')
        psi = vecs[:, 0].real  # Ground state is real for real H
        print(f"  Lanczos: {time.time()-t_gs:.1f}s, E0={vals[0]:.4f}", flush=True)

        # Free Hamiltonian memory
        del H

        # Single-qubit entropies
        single_entropies = np.array([von_neumann_entropy(partial_trace(psi, N, [i])) for i in range(N)])
        se_mean = np.mean(single_entropies)
        se_std = np.std(single_entropies)
        print(f"  Single-qubit S: mean={se_mean:.6f}, std={se_std:.6f}", flush=True)

        rng = np.random.RandomState(seed + 30000 + k_obs)
        TC_values = []
        S_values = []
        T_values = []

        for sub_idx in range(n_subsets):
            subset = sorted(rng.choice(list(range(N)), k_obs, replace=False).tolist())

            rho_A = partial_trace(psi, N, subset)
            S_obs = von_neumann_entropy(rho_A)
            sum_Si = sum(single_entropies[i] for i in subset)
            TC = sum_Si - S_obs

            pairs = [(subset[qi], subset[qj]) for qi in range(len(subset)) for qj in range(qi+1, len(subset))]
            czz = connected_zz_vectorized(psi, N, pairs)
            T_obs = np.mean(czz)

            TC_values.append(TC)
            S_values.append(S_obs)
            T_values.append(T_obs)

            if (sub_idx + 1) % 10 == 0:
                print(f"    subset {sub_idx+1}/{n_subsets}", flush=True)

        TC_values = np.array(TC_values)
        S_values = np.array(S_values)
        T_values = np.array(T_values)

        r_TC_S, p_TC_S = pearsonr(TC_values, S_values)
        r_T_S = np.nan
        if np.std(T_values) > 1e-10 and np.std(S_values) > 1e-10:
            r_T_S, _ = pearsonr(T_values, S_values)

        print(f"  r(TC, S) = {r_TC_S:.6f} (p={p_TC_S:.2e})", flush=True)
        if not np.isnan(r_T_S):
            print(f"  r(T_ZZ, S) = {r_T_S:.4f}", flush=True)
        else:
            print(f"  r(T_ZZ, S) = undefined", flush=True)
        print(f"  Total: {time.time()-t0:.1f}s", flush=True)

        results['seeds'].append({
            'seed': seed,
            'k': k_obs,
            'r_TC_S': float(r_TC_S),
            'p_TC_S': float(p_TC_S),
            'r_T_S': float(r_T_S) if not np.isnan(r_T_S) else None,
            'single_entropy_mean': float(se_mean),
            'single_entropy_std': float(se_std),
        })

# Summary
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)

for k_obs in [10, 6]:
    seed_data = [s for s in results['seeds'] if s['k'] == k_obs]
    r_vals = [s['r_TC_S'] for s in seed_data]
    se_stds = [s['single_entropy_std'] for s in seed_data]
    print(f"\nk={k_obs}:", flush=True)
    print(f"  mean r(TC, S) = {np.mean(r_vals):.6f} +/- {np.std(r_vals):.6f}", flush=True)
    print(f"  mean single-qubit entropy std = {np.mean(se_stds):.8f}", flush=True)

with open('results/p5_n20_verification.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/p5_n20_verification.json", flush=True)
