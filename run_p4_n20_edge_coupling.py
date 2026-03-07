#!/usr/bin/env python3
"""
Paper 4: N=20 Edge-Level Curvature-Matter Coupling

Tests whether kappa(i,j) ~ -T_ZZ(i,j) survives at N=20.

At N=12, edge-level ORC curvature anti-correlates with connected ZZ
correlations at r=-0.44, surviving degree control. This experiment
checks whether the same coupling holds at N=20 where:
  - MI graphs have more edges (~95 after 50th percentile pruning of C(20,2)=190)
  - ORC Wasserstein LP runs on ~20-node graphs (feasible but slow)
  - Ground state computed via sparse Lanczos (2^20 = 1M dim)

N=20, 5 seeds (10000-10004), ground state only.
Sparse Hamiltonian + Lanczos eigensolver.

Built by Opus Warrior, March 6 2026.
"""

import sys
import os
import json
import time
import gc
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from quantum import (
    random_all_to_all_sparse,
    ground_state_sparse,
    mutual_information_matrix,
)
from curvature import ollivier_ricci, mi_to_graph


# ─────────────────────────────────────────────────────
# Fast ZZ connected correlation via computational basis
# ─────────────────────────────────────────────────────

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

    z_arr = np.array(z).reshape(-1, 1)
    x_arr = np.array(x)
    y_arr = np.array(y)

    Z = np.column_stack([z_arr, np.ones(len(z_arr))])

    beta_x = np.linalg.lstsq(Z, x_arr, rcond=None)[0]
    res_x = x_arr - Z @ beta_x

    beta_y = np.linalg.lstsq(Z, y_arr, rcond=None)[0]
    res_y = y_arr - Z @ beta_y

    if np.std(res_x) < 1e-14 or np.std(res_y) < 1e-14:
        return 0.0, 1.0

    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def run_experiment():
    N = 20
    seeds = list(range(10000, 10005))

    print(f"Paper 4: N=20 Edge-Level Curvature-Matter Coupling")
    print(f"N={N}, {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}), ground state only")
    print(f"ORC: threshold=0.5, alpha=0.5")
    print(f"Sparse Lanczos eigensolver, dim=2^{N}={2**N:,}")
    print("=" * 70)

    all_seed_results = []
    r_tzz_all = []
    rp_tzz_all = []
    t_total = time.time()

    for si, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"Seed {seed} ({si+1}/{len(seeds)})")
        print(f"{'='*70}")

        # Step 1: Build sparse Hamiltonian
        print("  [1/5] Building sparse Hamiltonian...")
        t1 = time.time()
        H_sparse, couplings = random_all_to_all_sparse(N, seed=seed)
        print(f"         Done in {time.time()-t1:.1f}s")

        # Step 2: Ground state via Lanczos
        print("  [2/5] Lanczos ground state...")
        t1 = time.time()
        E0, psi = ground_state_sparse(H_sparse)
        print(f"         E0={E0:.6f}, done in {time.time()-t1:.1f}s")

        # Free Hamiltonian
        del H_sparse
        gc.collect()

        # Step 3: Full MI matrix (190 pairs)
        print("  [3/5] Full MI matrix (190 pairs)...")
        t1 = time.time()
        MI = mutual_information_matrix(psi, N, list(range(N)))
        n_nonzero = np.sum(MI > 1e-14) // 2  # upper triangle
        print(f"         {n_nonzero} nonzero MI pairs, done in {time.time()-t1:.1f}s")

        # Step 4: ORC with edge curvatures
        print("  [4/5] Ollivier-Ricci curvature (this is the slow step)...")
        t1 = time.time()
        orc = ollivier_ricci(MI, threshold=0.5, alpha=0.5)
        edge_curvatures = orc['edge_curvatures']
        n_edges = orc['n_edges']
        print(f"         {n_edges} edges, scalar R={orc['scalar_curvature']:.4f}, done in {time.time()-t1:.1f}s")

        if n_edges < 5:
            print(f"  WARNING: Only {n_edges} edges, skipping seed")
            all_seed_results.append({
                'seed': seed, 'n_edges': n_edges, 'skipped': True
            })
            gc.collect()
            continue

        # Step 5: ZZ correlations and edge-level coupling
        print("  [5/5] ZZ correlations and edge-level coupling...")
        t1 = time.time()
        C_zz = fast_zz_correlations(psi, N)

        # Build graph for degree info
        G = mi_to_graph(MI, threshold=0.5)

        kappa_vals = []
        tzz_vals = []
        tmi_vals = []
        degree_vals = []

        for (i, j), kappa in edge_curvatures.items():
            kappa_vals.append(kappa)
            tzz_vals.append(C_zz[i, j])
            tmi_vals.append(MI[i, j])
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

        elapsed_step = time.time() - t1
        elapsed_total = time.time() - t0

        print(f"         Done in {elapsed_step:.1f}s")
        print()
        print(f"  RESULTS for seed {seed}:")
        print(f"    n_edges           = {n_edges}")
        print(f"    r(kappa, T_ZZ)    = {r_tzz:+.4f}  (p={p_tzz:.4f})")
        print(f"    r(kappa, T_MI)    = {r_tmi:+.4f}  (p={p_tmi:.4f})")
        print(f"    partial r(k,Tzz|deg) = {rp_tzz:+.4f}  (p={pp_tzz:.4f})")
        print(f"    partial r(k,Tmi|deg) = {rp_tmi:+.4f}  (p={pp_tmi:.4f})")
        print(f"    Seed total time   = {elapsed_total:.1f}s")

        r_tzz_all.append(float(r_tzz))
        rp_tzz_all.append(float(rp_tzz))

        seed_result = {
            'seed': seed,
            'skipped': False,
            'E0': float(E0),
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
            'mean_kappa': float(np.mean(kappa_arr)),
            'std_kappa': float(np.std(kappa_arr)),
            'mean_tzz': float(np.mean(tzz_arr)),
            'mean_tmi': float(np.mean(tmi_arr)),
            'mean_degree': float(np.mean(degree_arr)),
            'seed_time_s': elapsed_total,
        }
        all_seed_results.append(seed_result)

        # Clean up
        del psi, MI, C_zz, G, orc, edge_curvatures
        del kappa_arr, tzz_arr, tmi_arr, degree_arr
        gc.collect()

    total_time = time.time() - t_total

    # ─────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: N=20 Edge-Level Curvature-Matter Coupling")
    print("=" * 70)

    valid_results = [r for r in all_seed_results if not r.get('skipped', False)]
    n_valid = len(valid_results)

    if n_valid == 0:
        print("  NO VALID RESULTS. All seeds skipped.")
    else:
        print(f"\n  Per-seed results ({n_valid} valid seeds):")
        print(f"  {'Seed':<8} {'n_edges':<10} {'r(k,Tzz)':<12} {'partial_r':<12} {'p_partial':<12}")
        print(f"  {'-'*54}")
        for r in valid_results:
            print(f"  {r['seed']:<8} {r['n_edges']:<10} {r['r_kappa_tzz']:+.4f}      "
                  f"{r['r_partial_tzz_degree']:+.4f}      {r['p_partial_tzz_degree']:.4f}")

        mean_r = np.mean(r_tzz_all)
        std_r = np.std(r_tzz_all)
        mean_rp = np.mean(rp_tzz_all)
        std_rp = np.std(rp_tzz_all)

        print(f"\n  Mean r(kappa, T_ZZ) across {n_valid} seeds:    {mean_r:+.4f} +/- {std_r:.4f}")
        print(f"  Mean partial r(k, Tzz|deg):              {mean_rp:+.4f} +/- {std_rp:.4f}")
        print(f"\n  Comparison to N=12: r(kappa, T_ZZ) = -0.44")
        print(f"  N=20 result:        r(kappa, T_ZZ) = {mean_r:+.4f}")

        # Verdict
        print()
        if mean_r < -0.3 and mean_rp < -0.2:
            verdict = ("CONFIRMED: Edge-level curvature-matter coupling kappa(i,j) ~ -T_ZZ(i,j) "
                       f"survives at N=20 (r={mean_r:+.3f}, partial_r={mean_rp:+.3f}). "
                       "Relationship strengthens or holds with system size.")
        elif mean_r < -0.2:
            verdict = (f"WEAKENED but present: r={mean_r:+.3f} at N=20 vs r=-0.44 at N=12. "
                       "Coupling exists but may dilute with system size.")
        elif mean_r < -0.1:
            verdict = (f"MARGINAL: r={mean_r:+.3f} at N=20 suggests the coupling is weakening "
                       "and may not survive the thermodynamic limit.")
        else:
            verdict = (f"FAILED: r={mean_r:+.3f} at N=20. Edge-level curvature-matter coupling "
                       "does not survive at this system size.")

        print(f"  VERDICT: {verdict}")

    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ─────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    output = {
        'experiment': 'p4_n20_edge_coupling',
        'description': ('N=20 edge-level curvature-matter coupling: '
                        'does kappa(i,j) ~ -T_ZZ(i,j) survive at N=20?'),
        'parameters': {
            'N': N,
            'seeds': seeds,
            'orc_threshold': 0.5,
            'orc_alpha': 0.5,
            'eigensolver': 'sparse_lanczos',
        },
        'summary': {
            'n_valid_seeds': n_valid,
            'mean_r_kappa_tzz': float(mean_r) if n_valid > 0 else None,
            'std_r_kappa_tzz': float(std_r) if n_valid > 0 else None,
            'mean_partial_r_tzz_degree': float(mean_rp) if n_valid > 0 else None,
            'std_partial_r_tzz_degree': float(std_rp) if n_valid > 0 else None,
            'n12_reference_r': -0.44,
            'verdict': verdict if n_valid > 0 else 'NO DATA',
        },
        'per_seed': all_seed_results,
        'total_time_s': total_time,
    }

    out_path = os.path.join(results_dir, 'p4_n20_edge_coupling.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    run_experiment()
