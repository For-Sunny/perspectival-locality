#!/usr/bin/env python3
"""
Paper 4: MATTER-GEOMETRY COMPLEMENTARITY
=========================================

Investigation of T_obs vs S_obs anti-correlation across observers.

Different half-observers (k of N) of the same quantum state see a strong
anti-correlation between stress-energy T_obs (mean connected ZZ correlations)
and entanglement entropy S_obs. Observers who see more "matter" see less
"geometry" and vice versa.

This script investigates:
  1. UNIVERSALITY: Does the T-S anti-correlation hold at N=12, 14, 16?
  2. PARTIALITY DEPENDENCE: How does r(T,S) change with k/N at N=12?
  3. EDGE-LEVEL COUPLING: Does kappa~T strengthen where complementarity peaks?
  4. EIGENSTATES: Does T-S anti-correlation change across energy eigenstates?

Built by Opus Warrior, March 6 2026.
"""

import sys
import os
import json
import time
import numpy as np
from scipy import stats
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantum import (
    random_all_to_all, random_all_to_all_sparse,
    ground_state, ground_state_sparse,
    heisenberg_all_to_all_sparse,
    partial_trace, von_neumann_entropy,
    mutual_information_matrix,
)
from src.curvature import ollivier_ricci


# ─────────────────────────────────────────────────────────────
# Core measurement functions
# ─────────────────────────────────────────────────────────────

def precompute_zz(psi, n_qubits):
    """Precompute ZZ building blocks: probs, signs, <Z_q> for all qubits."""
    dim = len(psi)
    probs = np.abs(psi) ** 2
    indices = np.arange(dim, dtype=np.int64)
    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]
    signs = np.zeros((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        signs[q] = 1.0 - 2.0 * ((indices & bit_masks[q]) != 0).astype(np.float64)
    ez = np.array([np.dot(signs[q], probs) for q in range(n_qubits)])
    return probs, signs, ez


def compute_T_obs(probs, signs, ez, subset):
    """T_obs = mean |connected ZZ correlation| over all pairs in subset."""
    vals = []
    for i, j in combinations(subset, 2):
        ezizj = np.dot(signs[i] * signs[j], probs)
        connected = ezizj - ez[i] * ez[j]
        vals.append(abs(connected))
    return float(np.mean(vals)) if vals else 0.0


def compute_S_obs(psi, n_qubits, subset):
    """S_obs = von Neumann entropy of reduced state on subset."""
    rho = partial_trace(psi, list(subset), n_qubits)
    return von_neumann_entropy(rho)


def compute_T_S_for_subsets(psi, n_qubits, k, n_subsets, rng_seed):
    """Compute T_obs and S_obs for random k-subsets. Returns (T_list, S_list)."""
    probs, signs, ez = precompute_zz(psi, n_qubits)
    rng = np.random.default_rng(rng_seed)
    all_sites = list(range(n_qubits))

    T_list = []
    S_list = []
    seen = set()
    attempts = 0
    count = 0
    while count < n_subsets and attempts < n_subsets * 20:
        sub = tuple(sorted(rng.choice(all_sites, size=k, replace=False)))
        if sub not in seen:
            seen.add(sub)
            T_obs = compute_T_obs(probs, signs, ez, sub)
            S_obs = compute_S_obs(psi, n_qubits, sub)
            T_list.append(T_obs)
            S_list.append(S_obs)
            count += 1
        attempts += 1

    return np.array(T_list), np.array(S_list)


def compute_edge_kappa_T(psi, n_qubits, subset, probs, signs, ez,
                         orc_threshold=0.5, orc_alpha=0.5):
    """
    Compute edge-level ORC kappa(i,j) and T_ZZ(i,j) for a single subset.
    Returns (kappa_vals, T_vals) as parallel arrays.
    """
    # MI matrix for subset
    MI = mutual_information_matrix(psi, n_qubits, sites=list(subset))
    orc_result = ollivier_ricci(MI, threshold=orc_threshold, alpha=orc_alpha)

    if not orc_result['edge_curvatures']:
        return np.array([]), np.array([])

    subset_list = sorted(subset)
    kappa_vals = []
    T_vals = []

    for (li, lj), kappa in orc_result['edge_curvatures'].items():
        # li, lj are LOCAL indices in the subset
        qi = subset_list[li]
        qj = subset_list[lj]
        ezizj = np.dot(signs[qi] * signs[qj], probs)
        connected = abs(ezizj - ez[qi] * ez[qj])
        kappa_vals.append(kappa)
        T_vals.append(connected)

    return np.array(kappa_vals), np.array(T_vals)


# ─────────────────────────────────────────────────────────────
# PART 1: UNIVERSALITY
# ─────────────────────────────────────────────────────────────

def run_universality():
    print("\n" + "=" * 70)
    print("PART 1: UNIVERSALITY — Does T-S anti-correlation hold at N=12,14,16?")
    print("=" * 70)

    configs = [
        {'N': 12, 'k': 6, 'n_subsets': 100, 'seeds': list(range(6000, 6010)), 'sparse': False},
        {'N': 14, 'k': 7, 'n_subsets': 100, 'seeds': list(range(9000, 9010)), 'sparse': True},
        {'N': 16, 'k': 8, 'n_subsets': 50,  'seeds': list(range(6000, 6005)), 'sparse': True},
    ]

    universality_results = []

    for cfg in configs:
        N = cfg['N']
        k = cfg['k']
        n_subsets = cfg['n_subsets']
        seeds = cfg['seeds']
        use_sparse = cfg['sparse']

        print(f"\n--- N={N}, k={k}, {len(seeds)} seeds, {n_subsets} subsets/seed ---")
        seed_r_values = []

        for si, seed in enumerate(seeds):
            t0 = time.time()
            if use_sparse:
                H_sp, couplings = random_all_to_all_sparse(N, seed=seed)
                E0, psi = ground_state_sparse(H_sp)
            else:
                H, couplings = random_all_to_all(N, seed=seed)
                E0, psi = ground_state(H)

            T_arr, S_arr = compute_T_S_for_subsets(psi, N, k, n_subsets, rng_seed=seed + 100000)

            if len(T_arr) >= 5:
                r_TS, p_TS = stats.pearsonr(T_arr, S_arr)
            else:
                r_TS, p_TS = 0.0, 1.0

            seed_r_values.append(r_TS)
            elapsed = time.time() - t0
            print(f"  Seed {seed}: r(T,S)={r_TS:+.4f} (p={p_TS:.2e}) [{elapsed:.1f}s]")

        mean_r = float(np.mean(seed_r_values))
        std_r = float(np.std(seed_r_values))
        n_neg = sum(1 for r in seed_r_values if r < 0)
        n_sig = sum(1 for r in seed_r_values if r < -0.2)  # meaningfully negative

        result = {
            'N': N, 'k': k, 'n_seeds': len(seeds), 'n_subsets': n_subsets,
            'mean_r_TS': mean_r, 'std_r_TS': std_r,
            'individual_r': seed_r_values,
            'n_negative': n_neg, 'n_strongly_negative': n_sig,
        }
        universality_results.append(result)

        print(f"  SUMMARY N={N}: r(T,S) = {mean_r:+.4f} +/- {std_r:.4f}")
        print(f"  Negative: {n_neg}/{len(seeds)}, Strongly negative (<-0.2): {n_sig}/{len(seeds)}")

    return universality_results


# ─────────────────────────────────────────────────────────────
# PART 2: PARTIALITY DEPENDENCE
# ─────────────────────────────────────────────────────────────

def run_partiality():
    print("\n" + "=" * 70)
    print("PART 2: PARTIALITY DEPENDENCE — r(T,S) vs k/N at N=12")
    print("=" * 70)

    N = 12
    K_VALUES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    N_SUBSETS = 100
    SEEDS = list(range(6000, 6010))

    partiality_results = []

    # Build all Hamiltonians and ground states once
    states = {}
    for seed in SEEDS:
        H, couplings = random_all_to_all(N, seed=seed)
        E0, psi = ground_state(H)
        states[seed] = psi

    for k in K_VALUES:
        print(f"\n--- k={k} (k/N={k/N:.3f}) ---")
        seed_r_values = []

        for seed in SEEDS:
            psi = states[seed]

            if k == N:
                # Full system: T_obs is fixed, S_obs is 0 (pure state). No correlation possible.
                # Compute T once for reference
                probs, signs, ez = precompute_zz(psi, N)
                T_full = compute_T_obs(probs, signs, ez, list(range(N)))
                seed_r_values.append(0.0)  # undefined, no variance
                continue

            T_arr, S_arr = compute_T_S_for_subsets(psi, N, k, N_SUBSETS, rng_seed=seed + 200000 + k)

            if len(T_arr) >= 5 and np.std(T_arr) > 1e-12 and np.std(S_arr) > 1e-12:
                r_TS, p_TS = stats.pearsonr(T_arr, S_arr)
            else:
                r_TS = 0.0

            seed_r_values.append(r_TS)

        mean_r = float(np.mean(seed_r_values))
        std_r = float(np.std(seed_r_values))

        partiality_results.append({
            'k': k, 'k_over_N': k / N,
            'mean_r_TS': mean_r, 'std_r_TS': std_r,
            'individual_r': seed_r_values,
        })

        print(f"  k={k} (k/N={k/N:.2f}): r(T,S) = {mean_r:+.4f} +/- {std_r:.4f}")

    return partiality_results


# ─────────────────────────────────────────────────────────────
# PART 3: EDGE-LEVEL COUPLING IN COMPLEMENTARITY REGIME
# ─────────────────────────────────────────────────────────────

def run_edge_coupling(partiality_results):
    print("\n" + "=" * 70)
    print("PART 3: EDGE-LEVEL kappa vs T_ZZ — does Einstein strengthen at peak complementarity?")
    print("=" * 70)

    N = 12
    SEEDS = list(range(6000, 6010))
    N_SUBSETS = 50
    ORC_THRESHOLD = 0.5
    ORC_ALPHA = 0.5

    # Find the k with strongest (most negative) r(T,S) from Part 2
    if partiality_results:
        best = min(partiality_results, key=lambda x: x['mean_r_TS'])
        k_peak = best['k']
        # Don't use k=N or too small k
        if k_peak >= N or k_peak < 3:
            k_peak = N // 2
    else:
        k_peak = N // 2

    k_comparison = [k_peak, N - 1]  # peak complementarity vs near-full
    if k_peak == N - 1:
        k_comparison = [N // 2, N - 1]

    print(f"  Peak complementarity at k={k_peak}")
    print(f"  Comparing k={k_comparison[0]} vs k={k_comparison[1]}")

    edge_results = {}

    for k in k_comparison:
        all_kappa = []
        all_T_edge = []

        for seed in SEEDS:
            H, couplings = random_all_to_all(N, seed=seed)
            E0, psi = ground_state(H)
            probs, signs, ez = precompute_zz(psi, N)

            rng = np.random.default_rng(seed + 300000 + k)
            all_sites = list(range(N))

            for _ in range(N_SUBSETS):
                sub = tuple(sorted(rng.choice(all_sites, size=k, replace=False)))
                kv, tv = compute_edge_kappa_T(psi, N, sub, probs, signs, ez,
                                               ORC_THRESHOLD, ORC_ALPHA)
                if len(kv) > 0:
                    all_kappa.extend(kv.tolist())
                    all_T_edge.extend(tv.tolist())

        all_kappa = np.array(all_kappa)
        all_T_edge = np.array(all_T_edge)

        if len(all_kappa) >= 10:
            r_kT, p_kT = stats.pearsonr(all_kappa, all_T_edge)
        else:
            r_kT, p_kT = 0.0, 1.0

        edge_results[k] = {
            'k': k, 'n_edges': len(all_kappa),
            'r_kappa_T': float(r_kT), 'p_kappa_T': float(p_kT),
            'kappa_mean': float(np.mean(all_kappa)) if len(all_kappa) > 0 else 0.0,
            'T_edge_mean': float(np.mean(all_T_edge)) if len(all_T_edge) > 0 else 0.0,
        }

        print(f"  k={k}: r(kappa, T_ZZ) = {r_kT:+.4f} (p={p_kT:.2e}), n_edges={len(all_kappa)}")

    return edge_results


# ─────────────────────────────────────────────────────────────
# PART 4: EIGENSTATES
# ─────────────────────────────────────────────────────────────

def run_eigenstates():
    print("\n" + "=" * 70)
    print("PART 4: EIGENSTATES — Does T-S anti-correlation change with energy?")
    print("=" * 70)

    N = 12
    k = 6
    N_SUBSETS = 100
    SEEDS = list(range(6000, 6010))
    EIGENSTATE_INDICES = [0, 5, 10, 15]  # ground state, excited states

    eigenstate_results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()
        H, couplings = random_all_to_all(N, seed=seed)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        for ei in EIGENSTATE_INDICES:
            if ei >= len(eigenvalues):
                continue
            psi = eigenvectors[:, ei]
            E = eigenvalues[ei]

            T_arr, S_arr = compute_T_S_for_subsets(psi, N, k, N_SUBSETS, rng_seed=seed + 400000 + ei)

            if len(T_arr) >= 5 and np.std(T_arr) > 1e-12 and np.std(S_arr) > 1e-12:
                r_TS, p_TS = stats.pearsonr(T_arr, S_arr)
            else:
                r_TS, p_TS = 0.0, 1.0

            eigenstate_results.append({
                'seed': seed, 'eigenstate_idx': ei, 'energy': float(E),
                'r_TS': float(r_TS), 'p_TS': float(p_TS),
                'T_mean': float(np.mean(T_arr)), 'S_mean': float(np.mean(S_arr)),
            })

        elapsed = time.time() - t0
        print(f"  Done ({elapsed:.1f}s)")

    return eigenstate_results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    total_t0 = time.time()

    print("=" * 70)
    print("PAPER 4: MATTER-GEOMETRY COMPLEMENTARITY INVESTIGATION")
    print("=" * 70)

    # Run all parts
    universality_results = run_universality()
    partiality_results = run_partiality()
    edge_results = run_edge_coupling(partiality_results)
    eigenstate_results = run_eigenstates()

    total_elapsed = time.time() - total_t0

    # ─────────────────────────────────────────────────────────
    # COMPREHENSIVE SUMMARY
    # ─────────────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("COMPREHENSIVE SUMMARY: MATTER-GEOMETRY COMPLEMENTARITY")
    print("=" * 70)

    # Table 1: Universality
    print("\n--- TABLE 1: r(T_obs, S_obs) vs System Size ---")
    print(f"{'N':>4} {'k':>4} {'k/N':>6} {'seeds':>6} {'subs':>6} {'mean r(T,S)':>14} {'std':>8} {'neg/total':>10}")
    print("-" * 65)
    for u in universality_results:
        print(f"{u['N']:4d} {u['k']:4d} {u['k']/u['N']:6.2f} {u['n_seeds']:6d} {u['n_subsets']:6d} "
              f"{u['mean_r_TS']:+14.4f} {u['std_r_TS']:8.4f} {u['n_negative']:3d}/{u['n_seeds']:3d}")

    # Table 2: Partiality
    print("\n--- TABLE 2: r(T_obs, S_obs) vs k/N at N=12 ---")
    print(f"{'k':>4} {'k/N':>6} {'mean r(T,S)':>14} {'std':>8}")
    print("-" * 35)
    for p in partiality_results:
        print(f"{p['k']:4d} {p['k_over_N']:6.3f} {p['mean_r_TS']:+14.4f} {p['std_r_TS']:8.4f}")

    # Find peak
    if partiality_results:
        peak = min(partiality_results, key=lambda x: x['mean_r_TS'])
        print(f"\n  Peak anti-correlation at k={peak['k']} (k/N={peak['k_over_N']:.3f}): "
              f"r = {peak['mean_r_TS']:+.4f}")

        # Is it monotonic or peaked?
        r_vals = [p['mean_r_TS'] for p in partiality_results if p['k'] < 12]
        if len(r_vals) >= 3:
            diffs = [r_vals[i+1] - r_vals[i] for i in range(len(r_vals)-1)]
            all_decreasing = all(d <= 0.05 for d in diffs)
            all_increasing = all(d >= -0.05 for d in diffs)
            if all_decreasing:
                print("  Trend: MONOTONICALLY DECREASING (more negative with more partiality)")
            elif all_increasing:
                print("  Trend: MONOTONICALLY INCREASING")
            else:
                print("  Trend: NON-MONOTONIC (peaked)")

    # Table 3: Edge-level coupling
    print("\n--- TABLE 3: Edge-level kappa vs T_ZZ ---")
    print(f"{'k':>4} {'r(kappa,T)':>14} {'p-value':>12} {'n_edges':>10}")
    print("-" * 45)
    for k, er in sorted(edge_results.items()):
        print(f"{k:4d} {er['r_kappa_T']:+14.4f} {er['p_kappa_T']:12.2e} {er['n_edges']:10d}")

    if len(edge_results) >= 2:
        keys = sorted(edge_results.keys())
        r_peak = edge_results[keys[0]]['r_kappa_T']
        r_full = edge_results[keys[-1]]['r_kappa_T']
        if abs(r_peak) > abs(r_full):
            print(f"\n  Edge-level Einstein STRONGER at peak complementarity "
                  f"(|{r_peak:.4f}| > |{r_full:.4f}|)")
        else:
            print(f"\n  Edge-level Einstein NOT stronger at peak "
                  f"(|{r_peak:.4f}| vs |{r_full:.4f}|)")

    # Table 4: Eigenstates
    print("\n--- TABLE 4: r(T,S) vs Eigenstate Energy ---")
    # Aggregate by eigenstate index
    eigen_agg = {}
    for er in eigenstate_results:
        ei = er['eigenstate_idx']
        if ei not in eigen_agg:
            eigen_agg[ei] = {'r_vals': [], 'E_vals': []}
        eigen_agg[ei]['r_vals'].append(er['r_TS'])
        eigen_agg[ei]['E_vals'].append(er['energy'])

    print(f"{'state':>6} {'mean E':>10} {'mean r(T,S)':>14} {'std':>8} {'n_neg':>6}")
    print("-" * 48)
    for ei in sorted(eigen_agg.keys()):
        r_vals = eigen_agg[ei]['r_vals']
        E_vals = eigen_agg[ei]['E_vals']
        n_neg = sum(1 for r in r_vals if r < 0)
        print(f"{ei:6d} {np.mean(E_vals):10.3f} {np.mean(r_vals):+14.4f} "
              f"{np.std(r_vals):8.4f} {n_neg:3d}/{len(r_vals):3d}")

    # ─────────────────────────────────────────────────────────
    # HONEST ASSESSMENT
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HONEST ASSESSMENT")
    print("=" * 70)

    # Collect evidence
    all_universal = all(u['mean_r_TS'] < -0.15 for u in universality_results)
    strong_universal = all(u['mean_r_TS'] < -0.3 for u in universality_results)
    n16_result = [u for u in universality_results if u['N'] == 16]
    n16_strong = n16_result[0]['mean_r_TS'] < -0.3 if n16_result else False

    has_peak = False
    if partiality_results:
        filtered = [p for p in partiality_results if p['k'] < 12]
        if filtered:
            r_at_half = [p for p in filtered if p['k'] == 6]
            r_at_extreme = [p for p in filtered if p['k'] in [3, 11]]
            if r_at_half and r_at_extreme:
                half_r = r_at_half[0]['mean_r_TS']
                extreme_r = np.mean([p['mean_r_TS'] for p in r_at_extreme])
                has_peak = half_r < extreme_r - 0.1

    eigen_stable = False
    if eigen_agg:
        eigen_r_means = [np.mean(eigen_agg[ei]['r_vals']) for ei in sorted(eigen_agg.keys())]
        if len(eigen_r_means) >= 2:
            eigen_stable = np.std(eigen_r_means) < 0.15

    if strong_universal and n16_strong:
        print("  STRONG EVIDENCE: T-S anti-correlation is UNIVERSAL across system sizes.")
        print("  Persists at N=16 — unlikely to be purely a finite-size artifact.")
    elif all_universal:
        print("  MODERATE EVIDENCE: T-S anti-correlation present at all sizes tested,")
        print("  but magnitude may be weakening. Further N scaling needed.")
    else:
        print("  WEAK EVIDENCE: Anti-correlation not consistent across system sizes.")
        print("  May be a finite-size artifact or Hamiltonian-dependent.")

    if has_peak:
        print("  PARTIALITY: Complementarity PEAKS at intermediate k/N — genuinely")
        print("  requires partial observation. Not an edge effect.")
    else:
        print("  PARTIALITY: No clear peak — relationship may be monotonic.")

    if eigen_stable:
        print("  EIGENSTATES: Complementarity is STABLE across energy eigenstates —")
        print("  a property of the Hilbert space structure, not the energy.")
    else:
        print("  EIGENSTATES: Complementarity varies with energy — may depend on")
        print("  the specific nature of the state, not just the observer partition.")

    print(f"\n  Total computation time: {total_elapsed:.1f}s")

    # ─────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────
    results = {
        'experiment': 'p4_complementarity',
        'description': 'Matter-geometry complementarity: T_obs vs S_obs anti-correlation',
        'universality': universality_results,
        'partiality': partiality_results,
        'edge_coupling': {str(k): v for k, v in edge_results.items()},
        'eigenstates': eigenstate_results,
        'total_elapsed_s': total_elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'p4_complementarity.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
