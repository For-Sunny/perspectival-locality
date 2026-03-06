#!/usr/bin/env python3
"""
Extended topology k-sweep: 10 Hamiltonians × 20 subsets = 200 trials per k.

Replaces the thin 3×5=15 trial dataset that produced the outlier-driven
headline in the original Paper 3 draft.

Expected runtime: ~30 min on dual Xeon Platinum 8380.
"""

import sys
import json
import time
import gc
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import (
    random_all_to_all_sparse, ground_state_sparse,
    mutual_information_matrix,
)
from src.curvature import ollivier_ricci, forman_ricci
from src.hamiltonians import build_local_hamiltonian
from src.utils import NumpyEncoder

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N = 20
THRESHOLD = 0.5
ALPHA = 0.5
N_SEEDS = 10
N_SUBSETS = 20
K_VALUES = [4, 5, 6, 7, 8, 10]
SEEDS = list(range(5000, 5000 + N_SEEDS))

print("=" * 60)
print(f"  EXTENDED TOPOLOGY K-SWEEP (N={N})")
print(f"  {N_SEEDS} Hamiltonians × {N_SUBSETS} subsets = {N_SEEDS * N_SUBSETS} trials/k")
print(f"  k values: {K_VALUES}")
print(f"  Seeds: {SEEDS[0]}..{SEEDS[-1]}")
print("=" * 60)

t0 = time.time()
all_trials = {str(k): [] for k in K_VALUES}

for si, seed in enumerate(SEEDS):
    t_seed = time.time()

    # All-to-all ground state
    H_a2a, _ = random_all_to_all_sparse(N, seed=seed)
    _, psi_a2a = ground_state_sparse(H_a2a)
    del H_a2a; gc.collect()

    # Chain ground state (fixed topology, random coupling via seed)
    H_chain, _ = build_local_hamiltonian(N, geometry='chain', delta=1.0)
    _, psi_chain = ground_state_sparse(H_chain)
    del H_chain; gc.collect()

    for k in K_VALUES:
        for t in range(N_SUBSETS):
            rng = np.random.default_rng(seed * 10000 + k * 100 + t)
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
    elapsed_seed = time.time() - t_seed
    print(f"  Seed {seed} ({si+1}/{N_SEEDS}): {elapsed_seed:.0f}s")

# Summary
print(f"\n{'='*60}")
print(f"  RESULTS ({N_SEEDS * N_SUBSETS} trials per k)")
print(f"{'='*60}")

summary = []
for k in K_VALUES:
    kt = all_trials[str(k)]
    gaps = np.array([t['ORC_gap'] for t in kt])
    frc_gaps = np.array([t['FRC_gap'] for t in kt])

    # Robust stats
    med = np.median(gaps)
    iqr = [float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))]
    # Wilcoxon signed-rank test (nonparametric, no normality assumption)
    from scipy.stats import wilcoxon
    try:
        stat, p_val = wilcoxon(gaps)
    except Exception:
        p_val = 1.0

    entry = {
        'k': k, 'k_over_N': k / N,
        'n_trials': len(gaps),
        'ORC_gap_mean': float(np.mean(gaps)),
        'ORC_gap_std': float(np.std(gaps)),
        'ORC_gap_median': float(med),
        'ORC_gap_IQR': iqr,
        'ORC_gap_p_wilcoxon': float(p_val),
        'FRC_gap_mean': float(np.mean(frc_gaps)),
        'FRC_gap_median': float(np.median(frc_gaps)),
        'FRC_gap_std': float(np.std(frc_gaps)),
        'n_outliers_abs_gt_10': int(np.sum(np.abs(gaps) > 10)),
    }
    summary.append(entry)

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  k={k:2d} (k/N={k/N:.2f}): median={med:+.3f}  IQR={iqr}  "
          f"mean={entry['ORC_gap_mean']:+.3f}  std={entry['ORC_gap_std']:.3f}  "
          f"p={p_val:.4f} {sig}  "
          f"outliers={entry['n_outliers_abs_gt_10']}")

elapsed = time.time() - t0
print(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Save
outpath = RESULTS_DIR / 'n20_exp3_topology_k_sweep_extended.json'
with open(outpath, 'w') as f:
    json.dump({
        'experiment': 'N=20 Exp3 Topology k-sweep (extended)',
        'N': N, 'k_values': K_VALUES,
        'seeds': SEEDS, 'n_subsets': N_SUBSETS,
        'n_trials_per_k': N_SEEDS * N_SUBSETS,
        'threshold': THRESHOLD,
        'summary': summary,
        'trials': all_trials,
        'total_runtime_s': elapsed,
    }, f, indent=2, cls=NumpyEncoder)

print(f"  Saved: {outpath}")
