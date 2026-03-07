#!/usr/bin/env python3
"""
Paper 4, Experiment 3: Discrete Bianchi Identity

Does the curvature on the MI graph satisfy a discrete conservation law?

In GR, the contracted Bianchi identity says div(G_ab) = 0 (stress-energy
conservation). We test whether a discrete analog holds on MI graphs by
comparing the L2 norm of the curvature divergence vector to a null model
(shuffled edge curvatures).

Two divergence definitions:
  div_R(i)  = sum_{j~i} kappa(i,j) * MI(i,j)           [weight-weighted]
  div2_R(i) = sum_{j~i} [kappa(i,j) - kappa_mean_i] * (1/MI(i,j))  [distance-weighted deviation]

Null model: 1000 random permutations of edge curvatures (graph + weights fixed).

Ratio < 1.0 means curvature is more conserved than random.
Ratio << 1.0 means genuine conservation law.

Built by Opus Warrior, March 6 2026.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantum import (
    random_all_to_all, ground_state,
    random_all_to_all_sparse, ground_state_sparse,
    mutual_information_matrix,
)
from src.curvature import ollivier_ricci


def compute_divergences(mi_matrix, edge_curvatures, n):
    """
    Compute curvature divergence vectors for all nodes.

    Returns:
        div_R: array of length n, div_R(i) = sum_{j~i} kappa(i,j) * MI(i,j)
        div_R_norm: array of length n, div_R(i) / sum_{j~i} MI(i,j)
        div2_R: array of length n, sum_{j~i} [kappa(i,j) - kappa_mean_i] * (1/MI(i,j))
    """
    div_R = np.zeros(n)
    div_R_norm = np.zeros(n)
    div2_R = np.zeros(n)

    # Build adjacency from edge_curvatures
    node_edges = {i: [] for i in range(n)}
    for (u, v), kappa in edge_curvatures.items():
        mi_val = mi_matrix[u, v]
        node_edges[u].append((v, kappa, mi_val))
        node_edges[v].append((u, kappa, mi_val))

    for i in range(n):
        edges = node_edges[i]
        if not edges:
            continue

        kappas = np.array([e[1] for e in edges])
        mi_vals = np.array([e[2] for e in edges])

        # Definition 1: weight-weighted divergence
        div_R[i] = np.sum(kappas * mi_vals)
        w_sum = np.sum(mi_vals)
        if w_sum > 1e-14:
            div_R_norm[i] = div_R[i] / w_sum

        # Definition 2: distance-weighted deviation
        kappa_mean = np.mean(kappas)
        distances = 1.0 / np.maximum(mi_vals, 1e-14)
        div2_R[i] = np.sum((kappas - kappa_mean) * distances)

    return div_R, div_R_norm, div2_R


def shuffle_curvatures(edge_curvatures, rng):
    """Randomly permute curvature values across edges (keep structure)."""
    edges = list(edge_curvatures.keys())
    kappas = np.array(list(edge_curvatures.values()))
    rng.shuffle(kappas)
    return dict(zip(edges, kappas.tolist()))


def run_single(N, seed, use_sparse=False, n_shuffles=1000):
    """Run Bianchi identity test for one Hamiltonian instance."""
    print(f"  Seed {seed}, N={N}...")
    t0 = time.time()

    # Build Hamiltonian and find ground state
    if use_sparse:
        H, couplings = random_all_to_all_sparse(N, seed=seed)
        E0, psi = ground_state_sparse(H)
    else:
        H, couplings = random_all_to_all(N, seed=seed)
        E0, psi = ground_state(H)

    # MI matrix (full, k=N)
    MI = mutual_information_matrix(psi, N)

    # ORC edge curvatures
    orc = ollivier_ricci(MI, threshold=0.5, alpha=0.5)
    edge_curvatures = orc['edge_curvatures']
    n_edges = orc['n_edges']

    if n_edges < 3:
        print(f"    Too few edges ({n_edges}), skipping.")
        return None

    # Actual divergence
    div_R, div_R_norm, div2_R = compute_divergences(MI, edge_curvatures, N)
    norm_actual = np.linalg.norm(div_R)
    norm_actual_normalized = np.linalg.norm(div_R_norm)
    norm2_actual = np.linalg.norm(div2_R)

    # Null model: shuffle curvatures
    rng = np.random.default_rng(seed + 100000)
    norms_shuffled = np.zeros(n_shuffles)
    norms_shuffled_norm = np.zeros(n_shuffles)
    norms2_shuffled = np.zeros(n_shuffles)

    for s in range(n_shuffles):
        ec_shuf = shuffle_curvatures(edge_curvatures, rng)
        d1, d1n, d2 = compute_divergences(MI, ec_shuf, N)
        norms_shuffled[s] = np.linalg.norm(d1)
        norms_shuffled_norm[s] = np.linalg.norm(d1n)
        norms2_shuffled[s] = np.linalg.norm(d2)

    mean_shuffled = np.mean(norms_shuffled)
    mean_shuffled_norm = np.mean(norms_shuffled_norm)
    mean2_shuffled = np.mean(norms2_shuffled)

    ratio = norm_actual / mean_shuffled if mean_shuffled > 1e-14 else float('nan')
    ratio_norm = norm_actual_normalized / mean_shuffled_norm if mean_shuffled_norm > 1e-14 else float('nan')
    ratio2 = norm2_actual / mean2_shuffled if mean2_shuffled > 1e-14 else float('nan')

    # p-value: fraction of shuffled with norm <= actual
    p_value = np.mean(norms_shuffled <= norm_actual)
    p_value_norm = np.mean(norms_shuffled_norm <= norm_actual_normalized)
    p_value2 = np.mean(norms2_shuffled <= norm2_actual)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s. ratio={ratio:.4f}, ratio_norm={ratio_norm:.4f}, "
          f"ratio2={ratio2:.4f}, edges={n_edges}")

    return {
        'seed': seed,
        'N': N,
        'n_edges': n_edges,
        'E0': E0,
        'norm_actual': norm_actual,
        'norm_actual_normalized': norm_actual_normalized,
        'norm2_actual': norm2_actual,
        'mean_norm_shuffled': mean_shuffled,
        'mean_norm_shuffled_normalized': mean_shuffled_norm,
        'mean_norm2_shuffled': mean2_shuffled,
        'std_norm_shuffled': float(np.std(norms_shuffled)),
        'std_norm_shuffled_normalized': float(np.std(norms_shuffled_norm)),
        'std_norm2_shuffled': float(np.std(norms2_shuffled)),
        'ratio': ratio,
        'ratio_normalized': ratio_norm,
        'ratio2': ratio2,
        'p_value': float(p_value),
        'p_value_normalized': float(p_value_norm),
        'p_value2': float(p_value2),
        'div_R_vector': div_R.tolist(),
        'div_R_norm_vector': div_R_norm.tolist(),
        'div2_R_vector': div2_R.tolist(),
        'elapsed_s': elapsed,
    }


def main():
    print("=" * 70)
    print("Paper 4, Experiment 3: Discrete Bianchi Identity")
    print("=" * 70)

    seeds = list(range(8000, 8030))
    n_shuffles = 1000

    all_results = {}

    for N, use_sparse in [(12, False), (14, True)]:
        print(f"\n{'='*50}")
        print(f"N = {N} ({'sparse' if use_sparse else 'dense'}), {len(seeds)} seeds")
        print(f"{'='*50}")

        results = []
        for seed in seeds:
            r = run_single(N, seed, use_sparse=use_sparse, n_shuffles=n_shuffles)
            if r is not None:
                results.append(r)

        if not results:
            print(f"  No valid results for N={N}!")
            all_results[f'N{N}'] = {'error': 'no valid results'}
            continue

        # Aggregate
        ratios = [r['ratio'] for r in results]
        ratios_norm = [r['ratio_normalized'] for r in results]
        ratios2 = [r['ratio2'] for r in results]
        p_values = [r['p_value'] for r in results]
        p_values_norm = [r['p_value_normalized'] for r in results]
        p_values2 = [r['p_value2'] for r in results]

        summary = {
            'N': N,
            'n_seeds': len(results),
            'n_shuffles': n_shuffles,
            'div1_weighted': {
                'mean_ratio': float(np.mean(ratios)),
                'std_ratio': float(np.std(ratios)),
                'median_ratio': float(np.median(ratios)),
                'min_ratio': float(np.min(ratios)),
                'max_ratio': float(np.max(ratios)),
                'mean_p_value': float(np.mean(p_values)),
                'frac_ratio_below_1': float(np.mean(np.array(ratios) < 1.0)),
                'frac_ratio_below_0_5': float(np.mean(np.array(ratios) < 0.5)),
                'frac_p_below_0_05': float(np.mean(np.array(p_values) < 0.05)),
            },
            'div1_normalized': {
                'mean_ratio': float(np.mean(ratios_norm)),
                'std_ratio': float(np.std(ratios_norm)),
                'median_ratio': float(np.median(ratios_norm)),
                'min_ratio': float(np.min(ratios_norm)),
                'max_ratio': float(np.max(ratios_norm)),
                'mean_p_value': float(np.mean(p_values_norm)),
                'frac_ratio_below_1': float(np.mean(np.array(ratios_norm) < 1.0)),
                'frac_ratio_below_0_5': float(np.mean(np.array(ratios_norm) < 0.5)),
                'frac_p_below_0_05': float(np.mean(np.array(p_values_norm) < 0.05)),
            },
            'div2_distance_deviation': {
                'mean_ratio': float(np.mean(ratios2)),
                'std_ratio': float(np.std(ratios2)),
                'median_ratio': float(np.median(ratios2)),
                'min_ratio': float(np.min(ratios2)),
                'max_ratio': float(np.max(ratios2)),
                'mean_p_value': float(np.mean(p_values2)),
                'frac_ratio_below_1': float(np.mean(np.array(ratios2) < 1.0)),
                'frac_ratio_below_0_5': float(np.mean(np.array(ratios2) < 0.5)),
                'frac_p_below_0_05': float(np.mean(np.array(p_values2) < 0.05)),
            },
            'per_seed': results,
        }
        all_results[f'N{N}'] = summary

        # Print summary
        print(f"\n--- N={N} Summary ({len(results)} seeds) ---")
        print(f"\nDiv1 (weight-weighted):")
        print(f"  Mean ratio ||div_R|| / mean(||div_R_shuffled||): {np.mean(ratios):.4f} +/- {np.std(ratios):.4f}")
        print(f"  Median ratio: {np.median(ratios):.4f}")
        print(f"  Range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
        print(f"  Mean p-value: {np.mean(p_values):.4f}")
        print(f"  Fraction with ratio < 1.0: {np.mean(np.array(ratios) < 1.0):.2%}")
        print(f"  Fraction with ratio < 0.5: {np.mean(np.array(ratios) < 0.5):.2%}")
        print(f"  Fraction with p < 0.05:    {np.mean(np.array(p_values) < 0.05):.2%}")

        print(f"\nDiv1 (normalized):")
        print(f"  Mean ratio: {np.mean(ratios_norm):.4f} +/- {np.std(ratios_norm):.4f}")
        print(f"  Median ratio: {np.median(ratios_norm):.4f}")
        print(f"  Mean p-value: {np.mean(p_values_norm):.4f}")
        print(f"  Fraction with ratio < 1.0: {np.mean(np.array(ratios_norm) < 1.0):.2%}")
        print(f"  Fraction with p < 0.05:    {np.mean(np.array(p_values_norm) < 0.05):.2%}")

        print(f"\nDiv2 (distance-weighted deviation):")
        print(f"  Mean ratio: {np.mean(ratios2):.4f} +/- {np.std(ratios2):.4f}")
        print(f"  Median ratio: {np.median(ratios2):.4f}")
        print(f"  Mean p-value: {np.mean(p_values2):.4f}")
        print(f"  Fraction with ratio < 1.0: {np.mean(np.array(ratios2) < 1.0):.2%}")
        print(f"  Fraction with p < 0.05:    {np.mean(np.array(p_values2) < 0.05):.2%}")

    # Comparison across N
    print(f"\n{'='*70}")
    print("CROSS-N COMPARISON")
    print(f"{'='*70}")
    for key in ['N12', 'N14']:
        if key in all_results and 'error' not in all_results[key]:
            s = all_results[key]
            print(f"\n{key}:")
            print(f"  Div1 weighted:  ratio = {s['div1_weighted']['mean_ratio']:.4f}, "
                  f"p = {s['div1_weighted']['mean_p_value']:.4f}")
            print(f"  Div1 normalized: ratio = {s['div1_normalized']['mean_ratio']:.4f}, "
                  f"p = {s['div1_normalized']['mean_p_value']:.4f}")
            print(f"  Div2 deviation:  ratio = {s['div2_distance_deviation']['mean_ratio']:.4f}, "
                  f"p = {s['div2_distance_deviation']['mean_p_value']:.4f}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    for key in ['N12', 'N14']:
        if key in all_results and 'error' not in all_results[key]:
            r1 = all_results[key]['div1_weighted']['mean_ratio']
            if r1 < 0.5:
                verdict = "STRONG conservation law (ratio << 1.0)"
            elif r1 < 0.8:
                verdict = "MODERATE conservation tendency (ratio < 1.0)"
            elif r1 < 1.0:
                verdict = "WEAK conservation tendency (ratio slightly < 1.0)"
            elif r1 < 1.2:
                verdict = "NO conservation (ratio ~ 1.0, consistent with random)"
            else:
                verdict = "ANTI-conservation (ratio > 1.0, less conserved than random)"
            print(f"  {key}: {verdict}")

    # Save
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'p4_exp3_bianchi.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
