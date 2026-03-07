#!/usr/bin/env python3
"""
Paper 5 Verification: Four predictions from the analytic derivation.

Prediction 1: TC(A) vs S(A) should give |r| → 1
Prediction 2: Single-qubit entropy variance across subsets should be small
Prediction 3: ORC on complete (unthresholded) graph should show NO κ-T coupling
Prediction 4: Product states should show NO complementarity
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr
from scipy.optimize import linprog
from itertools import combinations
import json
import time

def build_heisenberg_hamiltonian(N, seed):
    """Build random all-to-all Heisenberg H with uniform couplings."""
    rng = np.random.RandomState(seed)
    dim = 2**N

    # Pauli matrices
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
            Jx = rng.uniform(-1, 1)
            Jy = rng.uniform(-1, 1)
            Jz = rng.uniform(-1, 1)

            Si_x = kron_op(sx, i, N)
            Sj_x = kron_op(sx, j, N)
            Si_y = kron_op(sy, i, N)
            Sj_y = kron_op(sy, j, N)
            Si_z = kron_op(sz, i, N)
            Sj_z = kron_op(sz, j, N)

            H += Jx * Si_x @ Sj_x + Jy * Si_y @ Sj_y + Jz * Si_z @ Sj_z

    return H

def get_ground_state(H, N):
    """Get ground state via sparse Lanczos."""
    vals, vecs = eigsh(H, k=1, which='SA')
    return vecs[:, 0]

def partial_trace(psi, N, keep):
    """Partial trace to get reduced density matrix on 'keep' qubits."""
    keep = sorted(keep)
    trace_out = sorted(set(range(N)) - set(keep))
    k = len(keep)
    t = len(trace_out)

    # Reshape into tensor
    psi_tensor = psi.reshape([2]*N)

    # Move kept indices to front, traced to back
    perm = keep + trace_out
    psi_tensor = np.transpose(psi_tensor, perm)

    # Reshape to (2^k, 2^t)
    psi_mat = psi_tensor.reshape(2**k, 2**t)

    # ρ = Tr_B(|ψ⟩⟨ψ|) = M @ M†
    rho = psi_mat @ psi_mat.conj().T
    return rho

def von_neumann_entropy(rho):
    """Von Neumann entropy in bits."""
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-14]
    return -np.sum(evals * np.log2(evals))

def mutual_information(psi, N, i, j, subset):
    """MI between qubits i,j within observed subset."""
    rho_i = partial_trace(psi, N, [i])
    rho_j = partial_trace(psi, N, [j])
    rho_ij = partial_trace(psi, N, [i, j])

    return von_neumann_entropy(rho_i) + von_neumann_entropy(rho_j) - von_neumann_entropy(rho_ij)

def connected_zz(psi, N, i, j):
    """Connected ZZ correlation |⟨ZiZj⟩ - ⟨Zi⟩⟨Zj⟩|."""
    dim = 2**N

    # Build Z_i and Z_j operators
    def z_op(qubit, N):
        sz = np.array([1, -1], dtype=float)
        op = np.ones(dim, dtype=float)
        for bit in range(dim):
            if (bit >> (N-1-qubit)) & 1:
                op[bit] = -1.0
            else:
                op[bit] = 1.0
        return op

    zi = z_op(i, N)
    zj = z_op(j, N)

    probs = np.abs(psi)**2
    ez_i = np.sum(probs * zi)
    ez_j = np.sum(probs * zj)
    ez_ij = np.sum(probs * zi * zj)

    return abs(ez_ij - ez_i * ez_j)

def compute_orc(mi_matrix, subset, threshold_percentile=50, alpha=0.5):
    """Compute ORC on MI graph. If threshold_percentile=None, use complete graph."""
    k = len(subset)

    # Build distance matrix
    mi_vals = []
    for i in range(k):
        for j in range(i+1, k):
            if mi_matrix[i,j] > 1e-14:
                mi_vals.append(mi_matrix[i,j])

    if not mi_vals:
        return {}, None

    # Threshold
    if threshold_percentile is not None:
        threshold = np.percentile(mi_vals, threshold_percentile)
    else:
        threshold = 0  # Keep all edges

    # Build adjacency and distance
    adj = np.zeros((k, k), dtype=bool)
    dist = np.full((k, k), np.inf)
    np.fill_diagonal(dist, 0)

    for i in range(k):
        for j in range(i+1, k):
            if mi_matrix[i,j] > max(threshold, 1e-14):
                adj[i,j] = adj[j,i] = True
                d = 1.0 / mi_matrix[i,j]
                dist[i,j] = dist[j,i] = d

    # Floyd-Warshall for shortest paths
    sp = dist.copy()
    for m in range(k):
        for i in range(k):
            for j in range(k):
                if sp[i,m] + sp[m,j] < sp[i,j]:
                    sp[i,j] = sp[i,m] + sp[m,j]

    # Compute ORC for each edge
    orc_values = {}
    for i in range(k):
        for j in range(i+1, k):
            if not adj[i,j]:
                continue

            # Lazy random walk measures
            neighbors_i = [n for n in range(k) if adj[i,n]]
            neighbors_j = [n for n in range(k) if adj[j,n]]
            di = len(neighbors_i)
            dj = len(neighbors_j)

            if di == 0 or dj == 0:
                continue

            # Build distributions
            mu_i = np.zeros(k)
            mu_i[i] = alpha
            for n in neighbors_i:
                mu_i[n] += (1-alpha) / di

            mu_j = np.zeros(k)
            mu_j[j] = alpha
            for n in neighbors_j:
                mu_j[n] += (1-alpha) / dj

            # Wasserstein-1 via LP
            supply = mu_i - mu_j
            sources = np.where(supply > 1e-14)[0]
            sinks = np.where(supply < -1e-14)[0]

            if len(sources) == 0 or len(sinks) == 0:
                orc_values[(i,j)] = 1.0
                continue

            n_vars = len(sources) * len(sinks)
            c = np.zeros(n_vars)
            for si, s in enumerate(sources):
                for ti, t in enumerate(sinks):
                    c[si * len(sinks) + ti] = sp[s, t]

            # Supply constraints
            A_eq_rows = []
            b_eq = []

            for si, s in enumerate(sources):
                row = np.zeros(n_vars)
                for ti in range(len(sinks)):
                    row[si * len(sinks) + ti] = 1.0
                A_eq_rows.append(row)
                b_eq.append(supply[s])

            for ti, t in enumerate(sinks):
                row = np.zeros(n_vars)
                for si in range(len(sources)):
                    row[si * len(sinks) + ti] = 1.0
                A_eq_rows.append(row)
                b_eq.append(-supply[t])

            A_eq = np.array(A_eq_rows)
            b_eq = np.array(b_eq)

            bounds = [(0, None)] * n_vars

            try:
                result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                if result.success:
                    w1 = result.fun
                    kappa = 1.0 - w1 / sp[i,j]
                    orc_values[(i,j)] = kappa
            except:
                pass

    return orc_values, adj

def make_product_state(N, seed):
    """Make a random product state (no entanglement)."""
    rng = np.random.RandomState(seed)
    psi = np.ones(1, dtype=complex)
    for i in range(N):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2*np.pi)
        qubit = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
        psi = np.kron(psi, qubit)
    return psi / np.linalg.norm(psi)

# =============================================================
# MAIN EXPERIMENTS
# =============================================================

N = 12
k = 6  # half-system
n_seeds = 10
n_subsets = 50

results = {
    'prediction1_TC_vs_S': [],
    'prediction2_qubit_entropy_variance': [],
    'prediction3_complete_graph_orc': [],
    'prediction4_product_states': [],
}

print("=" * 60)
print("PAPER 5 VERIFICATION EXPERIMENTS")
print(f"N={N}, k={k}, {n_seeds} seeds, {n_subsets} subsets each")
print("=" * 60)

# ---- PREDICTIONS 1, 2, 3: Entangled ground states ----

for seed in range(n_seeds):
    t0 = time.time()
    print(f"\nSeed {seed}...")

    H = build_heisenberg_hamiltonian(N, seed)
    psi = get_ground_state(H, N)

    # Compute all single-qubit entropies
    single_entropies = []
    for i in range(N):
        rho_i = partial_trace(psi, N, [i])
        single_entropies.append(von_neumann_entropy(rho_i))
    single_entropies = np.array(single_entropies)

    print(f"  Single-qubit entropies: mean={np.mean(single_entropies):.4f}, "
          f"std={np.std(single_entropies):.4f}")

    # Random subsets
    all_qubits = list(range(N))
    rng = np.random.RandomState(seed + 10000)

    TC_values = []
    S_values = []
    T_values = []

    for sub_idx in range(n_subsets):
        subset = sorted(rng.choice(all_qubits, k, replace=False).tolist())

        # S_obs = entanglement entropy of subset
        rho_A = partial_trace(psi, N, subset)
        S_obs = von_neumann_entropy(rho_A)

        # TC(A) = Σ S(ρ_i) - S(ρ_A)
        sum_Si = sum(single_entropies[i] for i in subset)
        TC = sum_Si - S_obs

        # T_obs = mean |connected ZZ|
        T_obs = 0
        n_pairs = 0
        for qi in range(len(subset)):
            for qj in range(qi+1, len(subset)):
                T_obs += connected_zz(psi, N, subset[qi], subset[qj])
                n_pairs += 1
        T_obs /= n_pairs

        TC_values.append(TC)
        S_values.append(S_obs)
        T_values.append(T_obs)

    TC_values = np.array(TC_values)
    S_values = np.array(S_values)
    T_values = np.array(T_values)

    # Prediction 1: TC vs S
    r_TC_S, p_TC_S = pearsonr(TC_values, S_values)
    r_T_S, p_T_S = pearsonr(T_values, S_values)

    print(f"  r(TC, S) = {r_TC_S:.4f} (p={p_TC_S:.2e}) | r(T_ZZ, S) = {r_T_S:.4f}")

    results['prediction1_TC_vs_S'].append({
        'seed': seed,
        'r_TC_S': float(r_TC_S),
        'p_TC_S': float(p_TC_S),
        'r_T_S': float(r_T_S),
        'p_T_S': float(p_T_S),
    })

    # Prediction 2: qubit entropy variance
    # Variance of Σ_{i∈A} S(ρ_i) across subsets
    sum_Si_values = np.array([sum(single_entropies[i] for i in
                    sorted(rng.choice(all_qubits, k, replace=False).tolist()))
                    for _ in range(200)])

    results['prediction2_qubit_entropy_variance'].append({
        'seed': seed,
        'single_entropy_mean': float(np.mean(single_entropies)),
        'single_entropy_std': float(np.std(single_entropies)),
        'sum_Si_mean': float(np.mean(sum_Si_values)),
        'sum_Si_std': float(np.std(sum_Si_values)),
        'sum_Si_cv': float(np.std(sum_Si_values) / np.mean(sum_Si_values)),
    })

    print(f"  Σ S(ρ_i) across subsets: mean={np.mean(sum_Si_values):.4f}, "
          f"std={np.std(sum_Si_values):.4f}, CV={np.std(sum_Si_values)/np.mean(sum_Si_values):.4f}")

    # Prediction 3: ORC on complete vs thresholded graph (one subset)
    subset = sorted(rng.choice(all_qubits, k, replace=False).tolist())
    mi_matrix = np.zeros((k, k))
    t_matrix = np.zeros((k, k))
    for qi in range(k):
        for qj in range(qi+1, k):
            mi = mutual_information(psi, N, subset[qi], subset[qj], subset)
            mi_matrix[qi, qj] = mi_matrix[qj, qi] = mi
            t_val = connected_zz(psi, N, subset[qi], subset[qj])
            t_matrix[qi, qj] = t_matrix[qj, qi] = t_val

    # Thresholded ORC
    orc_thresh, adj_thresh = compute_orc(mi_matrix, subset, threshold_percentile=50)
    # Complete ORC
    orc_complete, adj_complete = compute_orc(mi_matrix, subset, threshold_percentile=None)

    # Correlate ORC with T for both
    if len(orc_thresh) >= 3:
        kappas_t = []
        ts_t = []
        for (i,j), kap in orc_thresh.items():
            kappas_t.append(kap)
            ts_t.append(t_matrix[i,j])
        if np.std(kappas_t) > 1e-10 and np.std(ts_t) > 1e-10:
            r_thresh, _ = pearsonr(kappas_t, ts_t)
        else:
            r_thresh = 0.0
    else:
        r_thresh = float('nan')

    if len(orc_complete) >= 3:
        kappas_c = []
        ts_c = []
        for (i,j), kap in orc_complete.items():
            kappas_c.append(kap)
            ts_c.append(t_matrix[i,j])
        if np.std(kappas_c) > 1e-10 and np.std(ts_c) > 1e-10:
            r_complete, _ = pearsonr(kappas_c, ts_c)
        else:
            r_complete = 0.0
    else:
        r_complete = float('nan')

    print(f"  ORC-T coupling: thresholded r={r_thresh:.4f}, complete r={r_complete:.4f}")

    results['prediction3_complete_graph_orc'].append({
        'seed': seed,
        'r_kappa_T_thresholded': float(r_thresh),
        'r_kappa_T_complete': float(r_complete),
        'n_edges_thresh': len(orc_thresh),
        'n_edges_complete': len(orc_complete),
    })

    print(f"  Time: {time.time()-t0:.1f}s")

# ---- PREDICTION 4: Product states ----
print("\n" + "=" * 60)
print("PREDICTION 4: Product states")
print("=" * 60)

for seed in range(n_seeds):
    psi = make_product_state(N, seed + 5000)

    rng = np.random.RandomState(seed + 20000)
    TC_values = []
    S_values = []
    T_values = []

    single_entropies = []
    for i in range(N):
        rho_i = partial_trace(psi, N, [i])
        single_entropies.append(von_neumann_entropy(rho_i))
    single_entropies = np.array(single_entropies)

    for sub_idx in range(n_subsets):
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

        TC_values.append(TC)
        S_values.append(S_obs)
        T_values.append(T_obs)

    TC_values = np.array(TC_values)
    S_values = np.array(S_values)
    T_values = np.array(T_values)

    # For product states, S_obs = Σ S(ρ_i) exactly, so TC = 0 for all subsets
    # And T_obs = 0 for all subsets (no correlations)
    r_T_S = float('nan')
    if np.std(T_values) > 1e-10 and np.std(S_values) > 1e-10:
        r_T_S, _ = pearsonr(T_values, S_values)

    print(f"  Seed {seed}: mean TC={np.mean(TC_values):.6f}, "
          f"mean T={np.mean(T_values):.6f}, mean S={np.mean(S_values):.4f}, "
          f"r(T,S)={r_T_S:.4f}" if not np.isnan(r_T_S) else
          f"  Seed {seed}: mean TC={np.mean(TC_values):.6f}, "
          f"mean T={np.mean(T_values):.6f}, mean S={np.mean(S_values):.4f}, "
          f"r(T,S)=undefined (no variance)")

    results['prediction4_product_states'].append({
        'seed': seed,
        'mean_TC': float(np.mean(TC_values)),
        'mean_T': float(np.mean(T_values)),
        'mean_S': float(np.mean(S_values)),
        'std_TC': float(np.std(TC_values)),
        'std_T': float(np.std(T_values)),
        'r_T_S': float(r_T_S) if not np.isnan(r_T_S) else None,
    })

# ---- SUMMARY ----
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Prediction 1
r_TC_S_vals = [r['r_TC_S'] for r in results['prediction1_TC_vs_S']]
r_T_S_vals = [r['r_T_S'] for r in results['prediction1_TC_vs_S']]
print(f"\nPrediction 1 (TC vs S should give |r|→1):")
print(f"  mean r(TC, S) = {np.mean(r_TC_S_vals):.4f} ± {np.std(r_TC_S_vals):.4f}")
print(f"  mean r(T_ZZ, S) = {np.mean(r_T_S_vals):.4f} ± {np.std(r_T_S_vals):.4f}")
print(f"  RATIO: |r(TC,S)| / |r(T,S)| = {abs(np.mean(r_TC_S_vals))/abs(np.mean(r_T_S_vals)):.2f}")

# Prediction 2
cvs = [r['sum_Si_cv'] for r in results['prediction2_qubit_entropy_variance']]
print(f"\nPrediction 2 (Σ S_i should be ~constant across subsets):")
print(f"  mean CV of Σ S(ρ_i) = {np.mean(cvs):.4f} ± {np.std(cvs):.4f}")

# Prediction 3
r_thresh_vals = [r['r_kappa_T_thresholded'] for r in results['prediction3_complete_graph_orc'] if not np.isnan(r['r_kappa_T_thresholded'])]
r_complete_vals = [r['r_kappa_T_complete'] for r in results['prediction3_complete_graph_orc'] if not np.isnan(r['r_kappa_T_complete'])]
print(f"\nPrediction 3 (ORC-T coupling absent on complete graph):")
print(f"  Thresholded: mean r(κ,T) = {np.mean(r_thresh_vals):.4f} ± {np.std(r_thresh_vals):.4f}")
print(f"  Complete:    mean r(κ,T) = {np.mean(r_complete_vals):.4f} ± {np.std(r_complete_vals):.4f}")

# Prediction 4
product_T = [r['mean_T'] for r in results['prediction4_product_states']]
product_TC = [r['mean_TC'] for r in results['prediction4_product_states']]
print(f"\nPrediction 4 (Product states: no complementarity):")
print(f"  mean T_obs = {np.mean(product_T):.6f}")
print(f"  mean TC = {np.mean(product_TC):.6f}")

# Save results
with open('results/p5_verification.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/p5_verification.json")
