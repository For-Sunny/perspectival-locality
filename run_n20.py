#!/usr/bin/env python3
"""
N=20 PLC experiments: All four Paper 3 curvature experiments at N=20.

Requires sparse Lanczos diagonalization (Hilbert dim = 2^20 = 1,048,576).
Uses ARPACK eigsh via scipy.sparse.linalg. Typical runtime: ~35 min total.

Reproduces the five result files in results/:
    n20_exp1_partiality.json      - Curvature vs observer fraction k/N
    n20_exp2_entanglement.json    - Curvature vs XXZ entanglement
    n20_exp3_topology.json        - Chain vs all-to-all at k=10
    n20_exp3_topology_k_sweep.json - Chain vs all-to-all, sweep k=4..10
    n20_exp4_perspective.json     - Perspectival curvature spread

Hardware: dual Xeon Platinum 8380, 3.9 TiB RAM, RTX 3090.
Peak memory: ~8 GB (sparse ground state at N=20).

Built by Opus Warrior, March 6 2026.
"""

import sys
import os
import json
import time
import gc
import numpy as np
from pathlib import Path
from itertools import combinations

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import (
    random_all_to_all_sparse, ground_state_sparse,
    mutual_information_matrix,
    partial_trace, von_neumann_entropy,
)
from src.curvature import ollivier_ricci, forman_ricci
from src.hamiltonians import build_xxz_hamiltonian, build_local_hamiltonian
from src.utils import NumpyEncoder

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N = 20
THRESHOLD = 0.5
ALPHA = 0.5


def _save(name, data):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {path}")


# ==================================================================
# Experiment 1: Curvature vs Partiality
# ==================================================================
def exp1_partiality():
    """
    Sweep k/N from 0.20 to 1.00. For each k, compute ORC and FRC
    over random k-subsets of N=20 qubits.

    Seeds: 1000, 1001, 1002 (3 Hamiltonians)
    Subsets per k: 10
    """
    print(f"\n{'='*60}")
    print(f"  EXP 1: Curvature vs Partiality (N={N})")
    print(f"{'='*60}")

    seeds = [1000, 1001, 1002]
    k_values = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    max_subsets = 10
    t0 = time.time()

    results = []
    for k in k_values:
        orc_vals, frc_vals = [], []
        for seed in seeds:
            H, _ = random_all_to_all_sparse(N, seed=seed)
            E0, psi = ground_state_sparse(H)
            del H; gc.collect()

            all_subsets = list(combinations(range(N), k)) if k < N else [list(range(N))]
            rng = np.random.default_rng(seed * 100 + k)
            n_sub = min(len(all_subsets), max_subsets)
            chosen = rng.choice(len(all_subsets), n_sub, replace=False) if n_sub < len(all_subsets) else range(len(all_subsets))

            for idx in chosen:
                subset = list(all_subsets[idx])
                MI = mutual_information_matrix(psi, N, subset)
                oll = ollivier_ricci(MI, threshold=THRESHOLD, alpha=ALPHA)
                frc = forman_ricci(MI, threshold=THRESHOLD)
                orc_vals.append(oll['scalar_curvature'])
                frc_vals.append(frc['scalar_curvature'])

            del psi; gc.collect()

        orc = np.array(orc_vals)
        frc = np.array(frc_vals)
        rec = {
            'k': k, 'k_over_N': k / N,
            'mean_ORC': float(np.mean(orc)), 'std_ORC': float(np.std(orc)),
            'mean_FRC': float(np.mean(frc)), 'std_FRC': float(np.std(frc)),
            'n_samples': len(orc),
            'orc_values': orc.tolist(), 'frc_values': frc.tolist(),
        }
        results.append(rec)
        print(f"  k={k:2d} (k/N={k/N:.2f}): ORC={rec['mean_ORC']:+.3f}+/-{rec['std_ORC']:.3f}  "
              f"FRC={rec['mean_FRC']:+.3f}+/-{rec['std_FRC']:.3f}  [n={rec['n_samples']}]")

    elapsed = time.time() - t0
    _save('n20_exp1_partiality', {
        'experiment': 'Paper 3 - Exp 1: Curvature vs Partiality',
        'N': N, 'dim': 2**N, 'seeds': seeds,
        'k_values': k_values, 'threshold': THRESHOLD,
        'max_subsets_per_k': max_subsets,
        'results': results,
        'total_runtime_s': elapsed,
    })
    print(f"  Elapsed: {elapsed:.0f}s")


# ==================================================================
# Experiment 2: Curvature vs Entanglement (XXZ sweep)
# ==================================================================
def exp2_entanglement():
    """
    Sweep XXZ anisotropy Delta. At each Delta, compute ground state,
    half-system entropy, and curvatures at k=N/2=10.

    Seeds: 2000, 2001, 2002
    """
    print(f"\n{'='*60}")
    print(f"  EXP 2: Curvature vs Entanglement (N={N})")
    print(f"{'='*60}")

    seeds = [2000, 2001, 2002]
    delta_values = [10.0, 5.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.1]
    k = N // 2
    t0 = time.time()

    records = []
    for delta in delta_values:
        orc_vals, frc_vals, entropies = [], [], []
        for seed in seeds:
            H, _ = build_xxz_hamiltonian(N, delta=delta, coupling='all_to_all', seed=seed)
            E0, psi = ground_state_sparse(H)
            del H; gc.collect()

            # Half-system entropy
            rho_half = partial_trace(psi, list(range(k)), N)
            S = von_neumann_entropy(rho_half)
            entropies.append(S)

            # Curvature at k=10
            MI = mutual_information_matrix(psi, N, list(range(k)))
            oll = ollivier_ricci(MI, threshold=THRESHOLD, alpha=ALPHA)
            frc = forman_ricci(MI, threshold=THRESHOLD)
            orc_vals.append(oll['scalar_curvature'])
            frc_vals.append(frc['scalar_curvature'])

            del psi; gc.collect()

        rec = {
            'delta': delta,
            'S_half_mean': float(np.mean(entropies)),
            'ORC_mean': float(np.mean(orc_vals)),
            'FRC_mean': float(np.mean(frc_vals)),
            'S_half_values': entropies,
            'ORC_values': orc_vals,
            'FRC_values': frc_vals,
        }
        records.append(rec)
        print(f"  Delta={delta:5.1f}  S={rec['S_half_mean']:.3f}  "
              f"ORC={rec['ORC_mean']:+.3f}  FRC={rec['FRC_mean']:+.3f}")

    # Correlations
    S_arr = np.array([r['S_half_mean'] for r in records])
    ORC_arr = np.array([r['ORC_mean'] for r in records])
    FRC_arr = np.array([r['FRC_mean'] for r in records])
    r_orc = float(np.corrcoef(S_arr, ORC_arr)[0, 1])
    r_frc = float(np.corrcoef(S_arr, FRC_arr)[0, 1])
    print(f"  Correlation (S, ORC): r={r_orc:.3f}")
    print(f"  Correlation (S, FRC): r={r_frc:.3f}")

    elapsed = time.time() - t0
    _save('n20_exp2_entanglement', {
        'experiment': 'exp2_curvature_vs_entanglement',
        'paper': 'PLC Paper 3',
        'n_qubits': N, 'hilbert_dim': 2**N,
        'observer_k': k, 'coupling': 'all_to_all',
        'threshold': THRESHOLD, 'seeds': seeds,
        'delta_values': delta_values,
        'records': records,
        'correlation_S_ORC': r_orc,
        'correlation_S_FRC': r_frc,
        'total_runtime_s': elapsed,
    })
    print(f"  Elapsed: {elapsed:.0f}s")


# ==================================================================
# Experiment 3a: Local vs Nonlocal Topology (fixed k=10)
# ==================================================================
def exp3_topology():
    """
    Compare all-to-all vs 1D chain at k=N/2=10.
    Seeds: 3000, 3001, 3002 (5 random subsets each)
    """
    print(f"\n{'='*60}")
    print(f"  EXP 3a: Topology (chain vs all-to-all, k=10, N={N})")
    print(f"{'='*60}")

    seeds = [3000, 3001, 3002]
    k = 10
    n_subsets = 5
    t0 = time.time()

    trials = []
    for seed in seeds:
        # All-to-all
        H_a2a, _ = random_all_to_all_sparse(N, seed=seed)
        _, psi_a2a = ground_state_sparse(H_a2a)
        del H_a2a; gc.collect()

        # Chain
        H_chain, _ = build_local_hamiltonian(N, geometry='chain', delta=1.0)
        # Add random couplings with same seed for reproducibility
        rng = np.random.default_rng(seed)
        # Use the sparse chain directly
        _, psi_chain = ground_state_sparse(H_chain)
        del H_chain; gc.collect()

        for t in range(n_subsets):
            rng_sub = np.random.default_rng(seed * 100 + t)
            subset = sorted(rng_sub.choice(N, k, replace=False).tolist())

            MI_a2a = mutual_information_matrix(psi_a2a, N, subset)
            MI_chain = mutual_information_matrix(psi_chain, N, subset)

            oll_a2a = ollivier_ricci(MI_a2a, threshold=THRESHOLD, alpha=ALPHA)
            oll_chain = ollivier_ricci(MI_chain, threshold=THRESHOLD, alpha=ALPHA)
            frc_a2a = forman_ricci(MI_a2a, threshold=THRESHOLD)
            frc_chain = forman_ricci(MI_chain, threshold=THRESHOLD)

            trial = {
                'seed': seed, 'trial': t, 'k': k, 'sites': subset,
                'all2all_ORC': oll_a2a['scalar_curvature'],
                'chain_ORC': oll_chain['scalar_curvature'],
                'ORC_gap': oll_chain['scalar_curvature'] - oll_a2a['scalar_curvature'],
                'all2all_FRC': frc_a2a['scalar_curvature'],
                'chain_FRC': frc_chain['scalar_curvature'],
                'FRC_gap': frc_chain['scalar_curvature'] - frc_a2a['scalar_curvature'],
                'all2all_n_edges': oll_a2a.get('n_edges', 0),
                'chain_n_edges': oll_chain.get('n_edges', 0),
            }
            trials.append(trial)
            print(f"  seed={seed} t={t}: a2a_ORC={trial['all2all_ORC']:+.3f}  "
                  f"chain_ORC={trial['chain_ORC']:+.3f}  gap={trial['ORC_gap']:+.3f}")

        del psi_a2a, psi_chain; gc.collect()

    elapsed = time.time() - t0
    orc_gaps = [t['ORC_gap'] for t in trials]
    _save('n20_exp3_topology', {
        'experiment': 'Exp3_Local_vs_Nonlocal_Topology',
        'N': N, 'k': k, 'delta': 1.0, 'threshold': THRESHOLD,
        'seeds': seeds, 'n_subsets': n_subsets,
        'trials': trials,
        'summary': {
            'ORC_gap_mean': float(np.mean(orc_gaps)),
            'ORC_gap_std': float(np.std(orc_gaps)),
            'ORC_gap_median': float(np.median(orc_gaps)),
        },
        'total_runtime_s': elapsed,
    })
    print(f"  Elapsed: {elapsed:.0f}s")


# ==================================================================
# Experiment 3b: Topology k-sweep
# ==================================================================
def exp3_topology_k_sweep():
    """
    Sweep k from 4 to 10 for both topologies.
    Seeds: 5000, 5001, 5002 (5 subsets each = 15 trials per k)
    """
    print(f"\n{'='*60}")
    print(f"  EXP 3b: Topology k-sweep (N={N})")
    print(f"{'='*60}")

    seeds = [5000, 5001, 5002]
    k_values = [4, 5, 6, 7, 8, 10]
    n_subsets = 5
    t0 = time.time()

    all_trials = {str(k): [] for k in k_values}
    summary = []

    for seed in seeds:
        # All-to-all
        H_a2a, _ = random_all_to_all_sparse(N, seed=seed)
        _, psi_a2a = ground_state_sparse(H_a2a)
        del H_a2a; gc.collect()

        # Chain
        H_chain, _ = build_local_hamiltonian(N, geometry='chain', delta=1.0)
        _, psi_chain = ground_state_sparse(H_chain)
        del H_chain; gc.collect()

        for k in k_values:
            for t in range(n_subsets):
                rng = np.random.default_rng(seed * 1000 + k * 100 + t)
                subset = sorted(rng.choice(N, k, replace=False).tolist())

                MI_a2a = mutual_information_matrix(psi_a2a, N, subset)
                MI_chain = mutual_information_matrix(psi_chain, N, subset)

                oll_a2a = ollivier_ricci(MI_a2a, threshold=THRESHOLD, alpha=ALPHA)
                oll_chain = ollivier_ricci(MI_chain, threshold=THRESHOLD, alpha=ALPHA)
                frc_a2a = forman_ricci(MI_a2a, threshold=THRESHOLD)
                frc_chain = forman_ricci(MI_chain, threshold=THRESHOLD)

                trial = {
                    'k': k, 'k_over_N': k / N,
                    'seed': seed, 'trial': t, 'sites': subset,
                    'all2all_ORC': oll_a2a['scalar_curvature'],
                    'chain_ORC': oll_chain['scalar_curvature'],
                    'ORC_gap': oll_chain['scalar_curvature'] - oll_a2a['scalar_curvature'],
                    'all2all_FRC': frc_a2a['scalar_curvature'],
                    'chain_FRC': frc_chain['scalar_curvature'],
                    'FRC_gap': frc_chain['scalar_curvature'] - frc_a2a['scalar_curvature'],
                    'all2all_n_edges': oll_a2a.get('n_edges', 0),
                    'chain_n_edges': oll_chain.get('n_edges', 0),
                }
                all_trials[str(k)].append(trial)

        del psi_a2a, psi_chain; gc.collect()
        print(f"  Seed {seed} complete")

    # Summary per k
    for k in k_values:
        kt = all_trials[str(k)]
        gaps = np.array([t['ORC_gap'] for t in kt])
        a2a = np.array([t['all2all_ORC'] for t in kt])
        chain = np.array([t['chain_ORC'] for t in kt])
        frc_gaps = np.array([t['FRC_gap'] for t in kt])
        entry = {
            'k': k, 'k_over_N': k / N,
            'all2all_ORC_mean': float(np.mean(a2a)),
            'all2all_ORC_std': float(np.std(a2a)),
            'chain_ORC_mean': float(np.mean(chain)),
            'chain_ORC_std': float(np.std(chain)),
            'ORC_gap_mean': float(np.mean(gaps)),
            'ORC_gap_std': float(np.std(gaps)),
            'ORC_gap_median': float(np.median(gaps)),
            'ORC_gap_IQR': [float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))],
            'all2all_FRC_mean': float(np.mean([t['all2all_FRC'] for t in kt])),
            'all2all_FRC_std': float(np.std([t['all2all_FRC'] for t in kt])),
            'chain_FRC_mean': float(np.mean([t['chain_FRC'] for t in kt])),
            'chain_FRC_std': float(np.std([t['chain_FRC'] for t in kt])),
            'FRC_gap_mean': float(np.mean(frc_gaps)),
            'FRC_gap_std': float(np.std(frc_gaps)),
        }
        summary.append(entry)
        print(f"  k={k:2d} (k/N={k/N:.2f}): gap_mean={entry['ORC_gap_mean']:+.3f}  "
              f"gap_median={entry['ORC_gap_median']:+.3f}  "
              f"gap_std={entry['ORC_gap_std']:.3f}  "
              f"IQR={entry['ORC_gap_IQR']}")

    elapsed = time.time() - t0
    _save('n20_exp3_topology_k_sweep', {
        'experiment': 'N=20 Exp3 Topology k-sweep',
        'N': N, 'k_values': k_values,
        'seeds': seeds, 'n_subsets': n_subsets,
        'threshold': THRESHOLD,
        'summary': summary,
        'trials': all_trials,
        'total_runtime_s': elapsed,
    })
    print(f"  Elapsed: {elapsed:.0f}s")


# ==================================================================
# Experiment 4: Perspectival Curvature
# ==================================================================
def exp4_perspective():
    """
    For each Hamiltonian, compute ORC for 30 random k=10 subsets.
    Measure the spread of curvature across observers.

    Seeds: 4000, 4001, 4002
    """
    print(f"\n{'='*60}")
    print(f"  EXP 4: Perspectival Curvature (N={N})")
    print(f"{'='*60}")

    seeds = [4000, 4001, 4002]
    k = 10
    n_subsets = 30
    t0 = time.time()

    sample_records = []
    for seed in seeds:
        H, _ = random_all_to_all_sparse(N, seed=seed)
        _, psi = ground_state_sparse(H)
        del H; gc.collect()

        curvatures = []
        for t in range(n_subsets):
            rng = np.random.default_rng(seed * 100 + t)
            subset = sorted(rng.choice(N, k, replace=False).tolist())
            MI = mutual_information_matrix(psi, N, subset)
            oll = ollivier_ricci(MI, threshold=THRESHOLD, alpha=ALPHA)
            curvatures.append(oll['scalar_curvature'])

        curv = np.array(curvatures)
        rec = {
            'seed': seed,
            'mean': float(np.mean(curv)),
            'std': float(np.std(curv)),
            'min': float(np.min(curv)),
            'max': float(np.max(curv)),
            'range': float(np.max(curv) - np.min(curv)),
            'curvatures': curv.tolist(),
        }
        sample_records.append(rec)
        print(f"  seed={seed}: R={rec['mean']:+.3f} +/- {rec['std']:.3f}  "
              f"range=[{rec['min']:+.3f}, {rec['max']:+.3f}]")

        del psi; gc.collect()

    # Aggregate
    all_curv = np.concatenate([np.array(r['curvatures']) for r in sample_records])
    elapsed = time.time() - t0

    _save('n20_exp4_perspective', {
        'experiment': 'perspectival_curvature_n20',
        'n_qubits': N, 'k': k,
        'n_subsets_per_hamiltonian': n_subsets,
        'n_hamiltonians': len(seeds),
        'seeds': seeds,
        'threshold': THRESHOLD, 'alpha': ALPHA,
        'hilbert_dim': 2**N,
        'sample_records': sample_records,
        'aggregate': {
            'mean': float(np.mean(all_curv)),
            'std': float(np.std(all_curv)),
        },
        'total_time_s': elapsed,
    })
    print(f"  Elapsed: {elapsed:.0f}s")


# ==================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='N=20 PLC curvature experiments (Paper 3)')
    parser.add_argument('--exp', type=str, default='all',
                        help='Which experiment: 1, 2, 3, 3k, 4, or all')
    args = parser.parse_args()

    print("=" * 60)
    print("  PLC PAPER 3: N=20 Curvature Experiments")
    print(f"  Hilbert space: 2^{N} = {2**N:,} dimensions")
    print(f"  Sparse Lanczos diagonalization")
    print("=" * 60)

    t_global = time.time()

    if args.exp in ('1', 'all'):
        exp1_partiality()
    if args.exp in ('2', 'all'):
        exp2_entanglement()
    if args.exp in ('3', 'all'):
        exp3_topology()
    if args.exp in ('3k', 'all'):
        exp3_topology_k_sweep()
    if args.exp in ('4', 'all'):
        exp4_perspective()

    total = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"  ALL DONE. Total: {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*60}")
