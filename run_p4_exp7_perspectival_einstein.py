#!/usr/bin/env python3
"""
Paper 4, Experiment 7: Perspectival Einstein Equation
=====================================================

Question: Does R_obs depend on T_obs?

Different observers of the same quantum state see different geometry (Paper 3).
Do they see different geometry BECAUSE they access different stress-energy?

For each ground state |psi>, many random k=7 observers each see:
  - R_obs: ORC scalar curvature of their MI subgraph
  - T_obs: mean connected ZZ correlation across their observed pairs
  - S_obs: entanglement entropy of their k=7 subsystem

If R_obs = f(T_obs) holds across observers of the SAME state,
that's a perspectival Einstein equation: each observer's curvature
is sourced by that observer's matter.

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
    random_all_to_all_sparse, ground_state_sparse,
    mutual_information_matrix, partial_trace, von_neumann_entropy
)
from src.curvature import ollivier_ricci


# ─────────────────────────────────────────────────────────────
# Fast ZZ correlations via bit manipulation on full state
# ─────────────────────────────────────────────────────────────

def compute_zz_correlations_fast(psi, n_qubits, subset):
    """
    Compute connected ZZ correlations for all pairs in subset.
    Uses bit manipulation on the full N-qubit state vector.

    <Z_i> = sum_x (-1)^{x_i} |psi(x)|^2
    <Z_i Z_j> = sum_x (-1)^{x_i + x_j} |psi(x)|^2
    Connected: <Z_i Z_j> - <Z_i><Z_j>

    Returns dict mapping (i,j) -> |connected correlation|
    """
    dim = len(psi)
    probs = np.abs(psi) ** 2  # |psi(x)|^2 for each basis state x
    indices = np.arange(dim, dtype=np.int64)

    bit_masks = {q: 1 << (n_qubits - 1 - q) for q in subset}

    # Precompute signs: sign_q[x] = (-1)^{x_q} = 1 - 2*bit_q(x)
    signs = {}
    for q in subset:
        bit_q = ((indices & bit_masks[q]) != 0).astype(np.float64)
        signs[q] = 1.0 - 2.0 * bit_q

    # <Z_q> for each qubit in subset
    ez = {}
    for q in subset:
        ez[q] = np.dot(signs[q], probs)

    # Connected correlations for all pairs
    corr = {}
    for i, j in combinations(subset, 2):
        ezizj = np.dot(signs[i] * signs[j], probs)
        connected = ezizj - ez[i] * ez[j]
        corr[(i, j)] = abs(connected)

    return corr


def compute_obs_stress_energy(psi, n_qubits, subset):
    """T_obs = mean |connected ZZ correlation| over all pairs in subset."""
    corr = compute_zz_correlations_fast(psi, n_qubits, subset)
    if not corr:
        return 0.0
    return float(np.mean(list(corr.values())))


def compute_obs_entropy(psi, n_qubits, subset):
    """S_obs = von Neumann entropy of the reduced state on subset."""
    rho = partial_trace(psi, list(subset), n_qubits)
    return von_neumann_entropy(rho)


def partial_correlation(x, y, z):
    """
    Partial correlation of x and y controlling for z.
    r_{xy.z} = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    """
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]

    denom = np.sqrt(max(1e-30, (1 - r_xz**2) * (1 - r_yz**2)))
    return (r_xy - r_xz * r_yz) / denom


# ─────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────

def run_experiment():
    N = 14
    K = 7  # half-observers
    N_SEEDS = 10
    N_SUBSETS = 50
    SEEDS = list(range(9000, 9000 + N_SEEDS))
    ORC_THRESHOLD = 0.5  # percentile
    ORC_ALPHA = 0.5

    print("=" * 70)
    print("EXPERIMENT 7: Perspectival Einstein Equation")
    print(f"N={N}, k={K}, {N_SEEDS} seeds, {N_SUBSETS} subsets/seed")
    print(f"ORC: threshold={ORC_THRESHOLD}, alpha={ORC_ALPHA}")
    print("=" * 70)

    all_results = []
    seed_summaries = []

    total_t0 = time.time()

    for si, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({si+1}/{N_SEEDS}) ---")
        t0 = time.time()

        # Build Hamiltonian and find ground state
        H_sp, couplings = random_all_to_all_sparse(N, seed=seed)
        E0, psi = ground_state_sparse(H_sp)
        print(f"  Ground state: E0={E0:.6f}, |psi| dim={len(psi)}")

        # Precompute ZZ building blocks once (shared across subsets)
        dim = len(psi)
        probs = np.abs(psi) ** 2
        indices = np.arange(dim, dtype=np.int64)
        bit_masks_all = [1 << (N - 1 - q) for q in range(N)]
        signs_all = np.zeros((N, dim), dtype=np.float64)
        for q in range(N):
            signs_all[q] = 1.0 - 2.0 * ((indices & bit_masks_all[q]) != 0).astype(np.float64)
        ez_all = np.array([np.dot(signs_all[q], probs) for q in range(N)])

        # Generate random subsets
        rng = np.random.default_rng(seed + 100000)
        all_sites = list(range(N))
        subsets = []
        seen = set()
        attempts = 0
        while len(subsets) < N_SUBSETS and attempts < N_SUBSETS * 10:
            sub = tuple(sorted(rng.choice(all_sites, size=K, replace=False)))
            if sub not in seen:
                seen.add(sub)
                subsets.append(list(sub))
            attempts += 1

        R_obs_list = []
        T_obs_list = []
        S_obs_list = []
        subset_records = []

        for idx, subset in enumerate(subsets):
            # --- T_obs: mean |connected ZZ| over pairs in subset ---
            t_vals = []
            for qi, qj in combinations(subset, 2):
                ezizj = np.dot(signs_all[qi] * signs_all[qj], probs)
                connected = ezizj - ez_all[qi] * ez_all[qj]
                t_vals.append(abs(connected))
            T_obs = float(np.mean(t_vals))

            # --- S_obs: entanglement entropy of subset ---
            S_obs = compute_obs_entropy(psi, N, subset)

            # --- R_obs: ORC scalar curvature of MI subgraph ---
            MI = mutual_information_matrix(psi, N, sites=subset)
            orc = ollivier_ricci(MI, threshold=ORC_THRESHOLD, alpha=ORC_ALPHA)
            R_obs = orc['scalar_curvature']

            R_obs_list.append(R_obs)
            T_obs_list.append(T_obs)
            S_obs_list.append(S_obs)

            subset_records.append({
                'subset': subset,
                'R_obs': R_obs,
                'T_obs': T_obs,
                'S_obs': S_obs,
                'n_edges': orc['n_edges'],
            })

            if (idx + 1) % 10 == 0:
                print(f"  Subset {idx+1}/{N_SUBSETS}: R={R_obs:.4f}, T={T_obs:.4f}, S={S_obs:.4f}")

        # Within-state correlations
        R = np.array(R_obs_list)
        T = np.array(T_obs_list)
        S = np.array(S_obs_list)

        r_RT, p_RT = stats.pearsonr(R, T)
        r_RS, p_RS = stats.pearsonr(R, S)
        r_TS, p_TS = stats.pearsonr(T, S)

        # Partial correlations
        r_RT_S = partial_correlation(R, T, S)  # R vs T controlling for S
        r_RS_T = partial_correlation(R, S, T)  # R vs S controlling for T

        elapsed = time.time() - t0
        print(f"  r(R,T) = {r_RT:.4f} (p={p_RT:.2e})")
        print(f"  r(R,S) = {r_RS:.4f} (p={p_RS:.2e})")
        print(f"  r(T,S) = {r_TS:.4f} (p={p_TS:.2e})")
        print(f"  r(R,T|S) = {r_RT_S:.4f}  [partial, controlling S]")
        print(f"  r(R,S|T) = {r_RS_T:.4f}  [partial, controlling T]")
        print(f"  Time: {elapsed:.1f}s")

        seed_summary = {
            'seed': seed,
            'E0': E0,
            'n_subsets': len(subsets),
            'r_RT': r_RT, 'p_RT': p_RT,
            'r_RS': r_RS, 'p_RS': p_RS,
            'r_TS': r_TS, 'p_TS': p_TS,
            'r_RT_given_S': r_RT_S,
            'r_RS_given_T': r_RS_T,
            'R_mean': float(np.mean(R)), 'R_std': float(np.std(R)),
            'T_mean': float(np.mean(T)), 'T_std': float(np.std(T)),
            'S_mean': float(np.mean(S)), 'S_std': float(np.std(S)),
            'subset_records': subset_records,
            'elapsed_s': elapsed,
        }
        seed_summaries.append(seed_summary)

    total_elapsed = time.time() - total_t0

    # ─────────────────────────────────────────────────────────
    # Aggregate across seeds
    # ─────────────────────────────────────────────────────────
    r_RTs = [s['r_RT'] for s in seed_summaries]
    r_RSs = [s['r_RS'] for s in seed_summaries]
    r_RT_Ss = [s['r_RT_given_S'] for s in seed_summaries]
    r_RS_Ts = [s['r_RS_given_T'] for s in seed_summaries]
    p_RTs = [s['p_RT'] for s in seed_summaries]

    mean_r_RT = float(np.mean(r_RTs))
    std_r_RT = float(np.std(r_RTs))
    mean_r_RS = float(np.mean(r_RSs))
    std_r_RS = float(np.std(r_RSs))
    mean_r_RT_S = float(np.mean(r_RT_Ss))
    std_r_RT_S = float(np.std(r_RT_Ss))
    mean_r_RS_T = float(np.mean(r_RS_Ts))
    std_r_RS_T = float(np.std(r_RS_Ts))

    # How many seeds have significant R-T correlation?
    n_sig_RT = sum(1 for p in p_RTs if p < 0.05)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (across 10 seeds)")
    print("=" * 70)
    print(f"  r(R_obs, T_obs)     = {mean_r_RT:.4f} +/- {std_r_RT:.4f}")
    print(f"  r(R_obs, S_obs)     = {mean_r_RS:.4f} +/- {std_r_RS:.4f}")
    print(f"  r(R_obs, T_obs | S) = {mean_r_RT_S:.4f} +/- {std_r_RT_S:.4f}  [partial]")
    print(f"  r(R_obs, S_obs | T) = {mean_r_RS_T:.4f} +/- {std_r_RS_T:.4f}  [partial]")
    print(f"  Seeds with p(R,T) < 0.05: {n_sig_RT}/{N_SEEDS}")
    print(f"  Total time: {total_elapsed:.1f}s")

    # Interpretation
    print("\n--- INTERPRETATION ---")
    if abs(mean_r_RT) > 0.3 and n_sig_RT >= 7:
        print("  STRONG: R_obs correlates with T_obs across observers.")
        if abs(mean_r_RT_S) > 0.2:
            print("  Partial r(R,T|S) survives => T_obs has INDEPENDENT predictive power.")
            print("  => PERSPECTIVAL EINSTEIN EQUATION IS REAL.")
            print("  Each observer's curvature is sourced by that observer's matter.")
        else:
            print("  But partial r(R,T|S) collapses => S_obs mediates the relationship.")
            print("  => R and T both depend on entanglement, not directly on each other.")
    elif abs(mean_r_RT) > 0.15:
        print("  MODERATE: Weak R-T correlation. Suggestive but not definitive.")
    else:
        print("  WEAK/ABSENT: R_obs does NOT track T_obs. No perspectival Einstein equation.")

    # Save results
    results = {
        'experiment': 'p4_exp7_perspectival_einstein',
        'description': 'Does R_obs depend on T_obs? Perspectival Einstein equation test.',
        'parameters': {
            'N': N, 'K': K, 'n_seeds': N_SEEDS, 'n_subsets': N_SUBSETS,
            'seeds': SEEDS,
            'orc_threshold': ORC_THRESHOLD, 'orc_alpha': ORC_ALPHA,
        },
        'aggregate': {
            'mean_r_RT': mean_r_RT, 'std_r_RT': std_r_RT,
            'mean_r_RS': mean_r_RS, 'std_r_RS': std_r_RS,
            'mean_r_RT_given_S': mean_r_RT_S, 'std_r_RT_given_S': std_r_RT_S,
            'mean_r_RS_given_T': mean_r_RS_T, 'std_r_RS_given_T': std_r_RS_T,
            'n_significant_RT': n_sig_RT,
            'individual_r_RT': r_RTs,
            'individual_r_RS': r_RSs,
            'individual_r_RT_given_S': r_RT_Ss,
            'individual_r_RS_given_T': r_RS_Ts,
        },
        'seed_summaries': [{k: v for k, v in s.items() if k != 'subset_records'}
                           for s in seed_summaries],
        'total_elapsed_s': total_elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'p4_exp7_perspectival_einstein.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_experiment()
