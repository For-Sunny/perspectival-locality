#!/usr/bin/env python3
"""
Paper 4: N=20 CONFIRMATION — MATTER-GEOMETRY COMPLEMENTARITY
=============================================================

Critical test: does the T_obs vs S_obs anti-correlation survive at N=20
(Hilbert dim = 2^20 = 1,048,576)?

Prior results (half-system observers, k=N/2):
  N=12: r(T,S) = -0.57
  N=14: r(T,S) = -0.66
  N=16: r(T,S) = -0.67

If N=20 continues this trend, the complementarity is confirmed as a robust
property that strengthens (or saturates) with system size.

10 random all-to-all Heisenberg Hamiltonians, seeds 10000-10009.
100 random k=10 subsets (half-system) per seed.
Also 100 random k=5 subsets (quarter-system) per seed for partiality check.

Built by Opus Warrior, March 6 2026.
"""

import sys
import os
import json
import time
import gc
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantum import (
    random_all_to_all_sparse,
    ground_state_sparse,
    partial_trace,
    von_neumann_entropy,
)


# ─────────────────────────────────────────────────────────────
# Core measurement functions (optimized for N=20)
# ─────────────────────────────────────────────────────────────

def precompute_zz(psi, n_qubits):
    """
    Precompute ZZ building blocks from |psi|^2.
    Returns probs, signs, ez where:
      probs[x] = |psi(x)|^2
      signs[q][x] = (-1)^{x_q}
      ez[q] = <Z_q> = sum_x signs[q][x] * probs[x]
    """
    dim = len(psi)
    probs = np.abs(psi) ** 2
    indices = np.arange(dim, dtype=np.int64)
    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]

    # signs[q] = +1 if bit q is 0, -1 if bit q is 1
    signs = np.zeros((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        signs[q] = 1.0 - 2.0 * ((indices & bit_masks[q]) != 0).astype(np.float64)

    # <Z_q> for all qubits
    ez = np.array([np.dot(signs[q], probs) for q in range(n_qubits)])

    return probs, signs, ez


def compute_T_obs(probs, signs, ez, subset):
    """
    T_obs = mean |<Z_i Z_j> - <Z_i><Z_j>| over all pairs (i,j) in subset.
    Fast: <Z_i Z_j> = sum_x signs[i]*signs[j]*probs, all vectorized.
    """
    from itertools import combinations
    vals = []
    for i, j in combinations(subset, 2):
        ezizj = np.dot(signs[i] * signs[j], probs)
        connected = ezizj - ez[i] * ez[j]
        vals.append(abs(connected))
    return float(np.mean(vals)) if vals else 0.0


def compute_S_obs(psi, n_qubits, subset):
    """
    S_obs = von Neumann entropy of reduced density matrix on subset.
    For k=10, N=20: produces 1024x1024 density matrix.
    """
    rho = partial_trace(psi, list(subset), n_qubits)
    return von_neumann_entropy(rho)


def compute_T_S_for_subsets(psi, n_qubits, k, n_subsets, rng_seed,
                             probs=None, signs=None, ez=None):
    """
    Compute (T_obs, S_obs) for n_subsets random k-subsets.
    Reuses precomputed ZZ data if provided.
    """
    if probs is None or signs is None or ez is None:
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
            if count % 25 == 0:
                print(f"      {count}/{n_subsets} subsets done (k={k})")
        attempts += 1

    return np.array(T_list), np.array(S_list)


# ─────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────

def main():
    total_t0 = time.time()

    N = 20
    SEEDS = list(range(10000, 10010))
    N_SUBSETS = 100
    K_HALF = 10   # k/N = 0.5 (half-system)
    K_QUARTER = 5  # k/N = 0.25 (quarter-system)

    print("=" * 70)
    print(f"PAPER 4: N={N} CONFIRMATION — MATTER-GEOMETRY COMPLEMENTARITY")
    print(f"Hilbert dim = 2^{N} = {2**N:,}")
    print(f"Seeds: {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} total)")
    print(f"Subsets per seed: {N_SUBSETS}")
    print(f"k values: {K_HALF} (k/N=0.5), {K_QUARTER} (k/N=0.25)")
    print("=" * 70)

    results_k10 = []  # per-seed r(T,S) for k=10
    results_k5 = []   # per-seed r(T,S) for k=5

    all_data = []  # full per-seed data for JSON output

    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({si+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        # ── Build sparse Hamiltonian and find ground state ──
        t_ham = time.time()
        H_sp, couplings = random_all_to_all_sparse(N, seed=seed)
        ham_time = time.time() - t_ham

        t_eig = time.time()
        E0, psi = ground_state_sparse(H_sp)
        eig_time = time.time() - t_eig

        print(f"  Hamiltonian: {ham_time:.1f}s, Eigensolver: {eig_time:.1f}s")
        print(f"  E0 = {E0:.6f}, |psi| = {np.linalg.norm(psi):.10f}")

        # Free the sparse matrix immediately
        del H_sp
        gc.collect()

        # ── Precompute ZZ data (shared across all subsets) ──
        t_zz = time.time()
        probs, signs, ez = precompute_zz(psi, N)
        zz_time = time.time() - t_zz
        print(f"  ZZ precompute: {zz_time:.1f}s")

        seed_data = {
            'seed': seed,
            'E0': float(E0),
            'ham_time_s': ham_time,
            'eig_time_s': eig_time,
        }

        # ── k=10 (half-system) ──
        print(f"\n  --- k={K_HALF} (half-system, k/N=0.5) ---")
        t_k10 = time.time()
        T10, S10 = compute_T_S_for_subsets(
            psi, N, K_HALF, N_SUBSETS, rng_seed=seed + 100000,
            probs=probs, signs=signs, ez=ez
        )
        k10_time = time.time() - t_k10

        if len(T10) >= 5:
            r10, p10 = stats.pearsonr(T10, S10)
        else:
            r10, p10 = 0.0, 1.0

        results_k10.append(r10)
        seed_data['k10'] = {
            'r_TS': float(r10), 'p_TS': float(p10),
            'T_mean': float(np.mean(T10)), 'T_std': float(np.std(T10)),
            'S_mean': float(np.mean(S10)), 'S_std': float(np.std(S10)),
            'n_subsets': len(T10),
            'time_s': k10_time,
        }
        print(f"  k=10: r(T,S) = {r10:+.4f} (p={p10:.2e}), "
              f"T_mean={np.mean(T10):.4f}, S_mean={np.mean(S10):.4f} [{k10_time:.1f}s]")

        # ── k=5 (quarter-system) ──
        print(f"\n  --- k={K_QUARTER} (quarter-system, k/N=0.25) ---")
        t_k5 = time.time()
        T5, S5 = compute_T_S_for_subsets(
            psi, N, K_QUARTER, N_SUBSETS, rng_seed=seed + 200000,
            probs=probs, signs=signs, ez=ez
        )
        k5_time = time.time() - t_k5

        if len(T5) >= 5:
            r5, p5 = stats.pearsonr(T5, S5)
        else:
            r5, p5 = 0.0, 1.0

        results_k5.append(r5)
        seed_data['k5'] = {
            'r_TS': float(r5), 'p_TS': float(p5),
            'T_mean': float(np.mean(T5)), 'T_std': float(np.std(T5)),
            'S_mean': float(np.mean(S5)), 'S_std': float(np.std(S5)),
            'n_subsets': len(T5),
            'time_s': k5_time,
        }
        print(f"  k=5:  r(T,S) = {r5:+.4f} (p={p5:.2e}), "
              f"T_mean={np.mean(T5):.4f}, S_mean={np.mean(S5):.4f} [{k5_time:.1f}s]")

        all_data.append(seed_data)

        # Free state-specific data
        del psi, probs, signs, ez, T10, S10, T5, S5
        gc.collect()

        seed_elapsed = time.time() - t_ham
        print(f"\n  Seed {seed} total: {seed_elapsed:.1f}s")

    total_elapsed = time.time() - total_t0

    # ─────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY: N=20 MATTER-GEOMETRY COMPLEMENTARITY")
    print("=" * 70)

    # Per-seed table
    print(f"\n{'seed':>8} {'r(T,S) k=10':>14} {'r(T,S) k=5':>14}")
    print("-" * 40)
    for i, seed in enumerate(SEEDS):
        print(f"{seed:8d} {results_k10[i]:+14.4f} {results_k5[i]:+14.4f}")

    # Aggregate statistics
    mean_r10 = float(np.mean(results_k10))
    std_r10 = float(np.std(results_k10))
    mean_r5 = float(np.mean(results_k5))
    std_r5 = float(np.std(results_k5))

    n_neg_k10 = sum(1 for r in results_k10 if r < 0)
    n_neg_k5 = sum(1 for r in results_k5 if r < 0)
    n_strong_k10 = sum(1 for r in results_k10 if r < -0.3)
    n_strong_k5 = sum(1 for r in results_k5 if r < -0.3)

    print(f"\n--- Aggregate ---")
    print(f"  k=10 (k/N=0.50): mean r(T,S) = {mean_r10:+.4f} +/- {std_r10:.4f}")
    print(f"    Negative: {n_neg_k10}/{len(SEEDS)}, Strongly negative (<-0.3): {n_strong_k10}/{len(SEEDS)}")
    print(f"  k=5  (k/N=0.25): mean r(T,S) = {mean_r5:+.4f} +/- {std_r5:.4f}")
    print(f"    Negative: {n_neg_k5}/{len(SEEDS)}, Strongly negative (<-0.3): {n_strong_k5}/{len(SEEDS)}")

    # Comparison to N=12-16 trend
    print(f"\n--- Comparison to N=12-16 trend ---")
    print(f"  N=12: r(T,S) = -0.5680 (k=6,  k/N=0.50)")
    print(f"  N=14: r(T,S) = -0.6564 (k=7,  k/N=0.50)")
    print(f"  N=16: r(T,S) = -0.6662 (k=8,  k/N=0.50)")
    print(f"  N=20: r(T,S) = {mean_r10:+.4f} (k=10, k/N=0.50)  <-- THIS EXPERIMENT")

    # Trend analysis
    prior_r = [-0.5680, -0.6564, -0.6662]
    prior_N = [12, 14, 16]

    if mean_r10 < -0.55:
        # Check if it continues the trend (stays strong or strengthens)
        if mean_r10 <= prior_r[-1] - 0.01:
            trend = "STRENGTHENING"
        elif abs(mean_r10 - prior_r[-1]) < 0.05:
            trend = "SATURATING (plateau)"
        else:
            trend = "WEAKENING slightly but still strong"
    elif mean_r10 < -0.3:
        trend = "WEAKENING but still present"
    elif mean_r10 < -0.1:
        trend = "WEAK — possible finite-size artifact"
    else:
        trend = "ABSENT — complementarity does NOT survive at N=20"

    print(f"\n  TREND: {trend}")

    # Partiality comparison at N=20
    print(f"\n--- Partiality at N=20 ---")
    print(f"  k=10 (k/N=0.50): r = {mean_r10:+.4f}")
    print(f"  k=5  (k/N=0.25): r = {mean_r5:+.4f}")
    if abs(mean_r5) > abs(mean_r10) + 0.05:
        print(f"  Quarter-system observers see STRONGER complementarity (more partial = stronger)")
    elif abs(mean_r5) < abs(mean_r10) - 0.05:
        print(f"  Half-system observers see stronger complementarity")
    else:
        print(f"  Similar strength at both partiality levels")

    # VERDICT
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if mean_r10 < -0.55 and n_neg_k10 >= 8:
        print("  CONFIRMED: Matter-geometry complementarity SURVIVES at N=20.")
        print(f"  r(T,S) = {mean_r10:+.4f} with {n_neg_k10}/10 seeds showing negative correlation.")
        print("  The effect is NOT a finite-size artifact. It is a robust property of")
        print("  partial observation in random quantum systems.")
        if mean_r10 <= prior_r[-1]:
            print("  The trend is strengthening/saturating — consistent with a fundamental bound.")
    elif mean_r10 < -0.3 and n_neg_k10 >= 6:
        print("  LIKELY CONFIRMED: Complementarity present but weakened at N=20.")
        print(f"  r(T,S) = {mean_r10:+.4f} — still meaningfully negative.")
        print("  May be approaching a thermodynamic limit, or convergence is slower.")
    else:
        print("  NOT CONFIRMED: Complementarity weakens significantly at N=20.")
        print(f"  r(T,S) = {mean_r10:+.4f} — may be a finite-size artifact at N=12-16.")

    print(f"\n  Total computation time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # ─────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────
    output = {
        'experiment': 'p4_n20_complementarity',
        'description': 'N=20 confirmation of matter-geometry complementarity',
        'N': N,
        'n_seeds': len(SEEDS),
        'seeds': SEEDS,
        'n_subsets_per_seed': N_SUBSETS,
        'k_half': K_HALF,
        'k_quarter': K_QUARTER,
        'summary': {
            'k10_mean_r_TS': mean_r10,
            'k10_std_r_TS': std_r10,
            'k10_individual_r': [float(r) for r in results_k10],
            'k10_n_negative': n_neg_k10,
            'k10_n_strongly_negative': n_strong_k10,
            'k5_mean_r_TS': mean_r5,
            'k5_std_r_TS': std_r5,
            'k5_individual_r': [float(r) for r in results_k5],
            'k5_n_negative': n_neg_k5,
            'k5_n_strongly_negative': n_strong_k5,
        },
        'prior_results': {
            'N12_mean_r': -0.5680,
            'N14_mean_r': -0.6564,
            'N16_mean_r': -0.6662,
        },
        'trend': trend,
        'per_seed': all_data,
        'total_elapsed_s': total_elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'p4_n20_complementarity.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
