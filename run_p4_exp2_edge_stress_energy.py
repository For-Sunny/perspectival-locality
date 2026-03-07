#!/usr/bin/env python3
"""
Paper 4, Experiment 2: Edge-level Stress-Energy Tensor

Does ORC edge curvature kappa(i,j) correlate with a "stress-energy"
quantity T(i,j) defined from the quantum state?

For each edge (i,j) in the MI graph we define:
  T_ZZ(i,j) = |<Z_i Z_j> - <Z_i><Z_j>|   (connected ZZ correlation)
  T_MI(i,j) = MI(i,j)                       (mutual information)

We correlate kappa(i,j) vs T_ZZ and T_MI across edges, seeds, and
eigenstates (ground + excited). We also compute partial correlations
controlling for node degree to check if the coupling is a confound.

N=12, 20 seeds (7000-7019), eigenstates [0, 1, 5, 10].
Dense diagonalization.

Built by Opus Warrior, March 6 2026.
"""

import sys
import os
import json
import time
import numpy as np
from scipy import stats
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from quantum import (
    random_all_to_all,
    mutual_information_matrix,
)
from curvature import ollivier_ricci


# ─────────────────────────────────────────────────────────
# Fast ZZ connected correlation via computational basis
# ─────────────────────────────────────────────────────────

def fast_zz_correlations(psi, n_qubits):
    """
    Compute full connected ZZ correlation matrix using bit manipulation.
    O(N^2 * 2^N) instead of building dense operator matrices.

    For pure state |psi>, in computational basis:
      <Z_i> = sum_x (-1)^{x_i} |<x|psi>|^2
      <Z_i Z_j> = sum_x (-1)^{x_i + x_j} |<x|psi>|^2
      C(i,j) = <Z_i Z_j> - <Z_i><Z_j>

    Returns NxN matrix of |C(i,j)| (absolute connected correlations).
    """
    dim = 2 ** n_qubits
    probs = np.abs(psi) ** 2  # |<x|psi>|^2, shape (dim,)

    # Precompute sign arrays: signs[q, x] = (-1)^{x_q}
    indices = np.arange(dim, dtype=np.int64)
    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]

    signs = np.empty((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        signs[q] = 1.0 - 2.0 * ((indices & bit_masks[q]) != 0).astype(np.float64)

    # <Z_q> for each qubit
    expect_z = signs @ probs  # shape (n_qubits,)

    # <Z_i Z_j> and connected correlation
    C = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            expect_zz = (signs[i] * signs[j]) @ probs
            connected = expect_zz - expect_z[i] * expect_z[j]
            C[i, j] = abs(connected)
            C[j, i] = abs(connected)

    return C


def partial_correlation(x, y, z):
    """
    Partial correlation of x and y controlling for z.
    Uses residuals from OLS regression.
    Returns (r_partial, p_value).
    """
    if len(x) < 4:
        return 0.0, 1.0

    # Regress x on z
    z_arr = np.array(z).reshape(-1, 1)
    x_arr = np.array(x)
    y_arr = np.array(y)

    # Add constant
    Z = np.column_stack([z_arr, np.ones(len(z_arr))])

    # Residuals of x ~ z
    beta_x = np.linalg.lstsq(Z, x_arr, rcond=None)[0]
    res_x = x_arr - Z @ beta_x

    # Residuals of y ~ z
    beta_y = np.linalg.lstsq(Z, y_arr, rcond=None)[0]
    res_y = y_arr - Z @ beta_y

    if np.std(res_x) < 1e-14 or np.std(res_y) < 1e-14:
        return 0.0, 1.0

    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def run_experiment():
    N = 12
    seeds = list(range(7000, 7020))
    eigenstate_indices = [0, 1, 5, 10]
    n_states = len(eigenstate_indices)
    max_eigen_idx = max(eigenstate_indices) + 1  # need at least 11 eigenstates

    print(f"Paper 4, Experiment 2: Edge-level Stress-Energy Tensor")
    print(f"N={N}, {len(seeds)} seeds, eigenstates={eigenstate_indices}")
    print(f"ORC: threshold=0.5, alpha=0.5")
    print("=" * 70)

    # Collectors per eigenstate index
    results_by_state = {idx: {
        'r_kappa_tzz': [],
        'r_kappa_tmi': [],
        'p_kappa_tzz': [],
        'p_kappa_tmi': [],
        'r_partial_tzz_degree': [],
        'p_partial_tzz_degree': [],
        'r_partial_tmi_degree': [],
        'p_partial_tmi_degree': [],
        'n_edges': [],
    } for idx in eigenstate_indices}

    all_seed_results = []
    t_total = time.time()

    for si, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n--- Seed {seed} ({si+1}/{len(seeds)}) ---")

        # Build Hamiltonian and diagonalize fully
        H, couplings = random_all_to_all(N, seed=seed)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        seed_data = {'seed': seed, 'states': {}}

        for state_idx in eigenstate_indices:
            psi = eigenvectors[:, state_idx]
            energy = float(eigenvalues[state_idx])

            # MI matrix (full, k=N=12)
            MI = mutual_information_matrix(psi, N)

            # ORC with edge curvatures
            orc = ollivier_ricci(MI, threshold=0.5, alpha=0.5)
            edge_curvatures = orc['edge_curvatures']

            if len(edge_curvatures) < 3:
                print(f"  state={state_idx}: only {len(edge_curvatures)} edges, skipping")
                continue

            # Fast ZZ correlations
            C_zz = fast_zz_correlations(psi, N)

            # Collect edge-level data
            kappa_vals = []
            tzz_vals = []
            tmi_vals = []
            degree_vals = []

            # Build graph for degree computation
            from curvature import mi_to_graph
            G = mi_to_graph(MI, threshold=0.5)

            for (i, j), kappa in edge_curvatures.items():
                kappa_vals.append(kappa)
                tzz_vals.append(C_zz[i, j])
                tmi_vals.append(MI[i, j])
                # Average degree of endpoints
                deg_i = G.degree(i) if G.has_node(i) else 0
                deg_j = G.degree(j) if G.has_node(j) else 0
                degree_vals.append(0.5 * (deg_i + deg_j))

            kappa_arr = np.array(kappa_vals)
            tzz_arr = np.array(tzz_vals)
            tmi_arr = np.array(tmi_vals)
            degree_arr = np.array(degree_vals)

            # Pearson correlations
            r_tzz, p_tzz = stats.pearsonr(kappa_arr, tzz_arr)
            r_tmi, p_tmi = stats.pearsonr(kappa_arr, tmi_arr)

            # Partial correlations controlling for degree
            rp_tzz, pp_tzz = partial_correlation(kappa_arr, tzz_arr, degree_arr)
            rp_tmi, pp_tmi = partial_correlation(kappa_arr, tmi_arr, degree_arr)

            n_edges = len(kappa_vals)

            print(f"  state={state_idx} (E={energy:.3f}): {n_edges} edges | "
                  f"r(k,Tzz)={r_tzz:.3f} r(k,Tmi)={r_tmi:.3f} | "
                  f"partial r(k,Tzz|deg)={rp_tzz:.3f} r(k,Tmi|deg)={rp_tmi:.3f}")

            # Store
            rb = results_by_state[state_idx]
            rb['r_kappa_tzz'].append(float(r_tzz))
            rb['r_kappa_tmi'].append(float(r_tmi))
            rb['p_kappa_tzz'].append(float(p_tzz))
            rb['p_kappa_tmi'].append(float(p_tmi))
            rb['r_partial_tzz_degree'].append(float(rp_tzz))
            rb['p_partial_tzz_degree'].append(float(pp_tzz))
            rb['r_partial_tmi_degree'].append(float(rp_tmi))
            rb['p_partial_tmi_degree'].append(float(pp_tmi))
            rb['n_edges'].append(n_edges)

            seed_data['states'][str(state_idx)] = {
                'energy': energy,
                'n_edges': n_edges,
                'scalar_curvature': orc['scalar_curvature'],
                'r_kappa_tzz': float(r_tzz),
                'p_kappa_tzz': float(p_tzz),
                'r_kappa_tmi': float(r_tmi),
                'p_kappa_tmi': float(p_tmi),
                'r_partial_tzz_degree': float(rp_tzz),
                'p_partial_tzz_degree': float(pp_tzz),
                'r_partial_tmi_degree': float(rp_tmi),
                'p_partial_tmi_degree': float(pp_tmi),
                'mean_tzz': float(np.mean(tzz_arr)),
                'mean_tmi': float(np.mean(tmi_arr)),
                'mean_kappa': float(np.mean(kappa_arr)),
                'mean_degree': float(np.mean(degree_arr)),
            }

        all_seed_results.append(seed_data)
        elapsed = time.time() - t0
        print(f"  Seed time: {elapsed:.1f}s")

    total_time = time.time() - t_total

    # ─────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Edge-level Stress-Energy Tensor")
    print("=" * 70)

    summary = {}
    for state_idx in eigenstate_indices:
        rb = results_by_state[state_idx]
        if not rb['r_kappa_tzz']:
            print(f"\n  State {state_idx}: NO DATA")
            continue

        mean_r_tzz = np.mean(rb['r_kappa_tzz'])
        std_r_tzz = np.std(rb['r_kappa_tzz'])
        mean_r_tmi = np.mean(rb['r_kappa_tmi'])
        std_r_tmi = np.std(rb['r_kappa_tmi'])
        mean_rp_tzz = np.mean(rb['r_partial_tzz_degree'])
        std_rp_tzz = np.std(rb['r_partial_tzz_degree'])
        mean_rp_tmi = np.mean(rb['r_partial_tmi_degree'])
        std_rp_tmi = np.std(rb['r_partial_tmi_degree'])

        # Fraction significant (p < 0.05)
        frac_sig_tzz = np.mean(np.array(rb['p_kappa_tzz']) < 0.05)
        frac_sig_tmi = np.mean(np.array(rb['p_kappa_tmi']) < 0.05)
        frac_sig_partial_tzz = np.mean(np.array(rb['p_partial_tzz_degree']) < 0.05)
        frac_sig_partial_tmi = np.mean(np.array(rb['p_partial_tmi_degree']) < 0.05)

        mean_edges = np.mean(rb['n_edges'])

        print(f"\n  Eigenstate index {state_idx} ({len(rb['r_kappa_tzz'])} seeds, mean {mean_edges:.0f} edges):")
        print(f"    r(kappa, T_ZZ)       = {mean_r_tzz:+.4f} +/- {std_r_tzz:.4f}  ({frac_sig_tzz*100:.0f}% sig)")
        print(f"    r(kappa, T_MI)       = {mean_r_tmi:+.4f} +/- {std_r_tmi:.4f}  ({frac_sig_tmi*100:.0f}% sig)")
        print(f"    partial r(k,Tzz|deg) = {mean_rp_tzz:+.4f} +/- {std_rp_tzz:.4f}  ({frac_sig_partial_tzz*100:.0f}% sig)")
        print(f"    partial r(k,Tmi|deg) = {mean_rp_tmi:+.4f} +/- {std_rp_tmi:.4f}  ({frac_sig_partial_tmi*100:.0f}% sig)")

        summary[str(state_idx)] = {
            'mean_r_kappa_tzz': float(mean_r_tzz),
            'std_r_kappa_tzz': float(std_r_tzz),
            'mean_r_kappa_tmi': float(mean_r_tmi),
            'std_r_kappa_tmi': float(std_r_tmi),
            'mean_partial_r_tzz_degree': float(mean_rp_tzz),
            'std_partial_r_tzz_degree': float(std_rp_tzz),
            'mean_partial_r_tmi_degree': float(mean_rp_tmi),
            'std_partial_r_tmi_degree': float(std_rp_tmi),
            'frac_sig_tzz': float(frac_sig_tzz),
            'frac_sig_tmi': float(frac_sig_tmi),
            'frac_sig_partial_tzz': float(frac_sig_partial_tzz),
            'frac_sig_partial_tmi': float(frac_sig_partial_tmi),
            'mean_n_edges': float(mean_edges),
            'n_seeds': len(rb['r_kappa_tzz']),
        }

    # Overall assessment
    print("\n" + "-" * 70)
    all_r_tmi = []
    all_rp_tmi = []
    for state_idx in eigenstate_indices:
        rb = results_by_state[state_idx]
        all_r_tmi.extend(rb['r_kappa_tmi'])
        all_rp_tmi.extend(rb['r_partial_tmi_degree'])

    if all_r_tmi:
        grand_r_tmi = np.mean(all_r_tmi)
        grand_rp_tmi = np.mean(all_rp_tmi)
        print(f"  Grand mean r(kappa, T_MI) across all states/seeds: {grand_r_tmi:+.4f}")
        print(f"  Grand mean partial r(kappa, T_MI | degree):        {grand_rp_tmi:+.4f}")
        print()

        if abs(grand_rp_tmi) < 0.1:
            verdict = "CONFOUND: kappa~T_MI coupling is mostly explained by node degree"
        elif abs(grand_rp_tmi) > 0.3:
            verdict = "REAL: kappa~T_MI coupling persists after controlling for degree"
        else:
            verdict = "MIXED: partial correlation is moderate -- degree explains some but not all"
        print(f"  VERDICT: {verdict}")

    print(f"\n  Total time: {total_time:.1f}s")

    # ─────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    output = {
        'experiment': 'p4_exp2_edge_stress_energy',
        'description': 'Edge-level stress-energy tensor: correlate ORC kappa(i,j) with T_ZZ and T_MI',
        'parameters': {
            'N': N,
            'seeds': seeds,
            'eigenstate_indices': eigenstate_indices,
            'orc_threshold': 0.5,
            'orc_alpha': 0.5,
        },
        'summary': summary,
        'per_seed': all_seed_results,
        'total_time_s': total_time,
    }

    out_path = os.path.join(results_dir, 'p4_exp2_edge_stress_energy.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    run_experiment()
