#!/usr/bin/env python3
"""
Symmetry-Breaking Control: Does PLC effect persist without SU(2)?

Referee objection: the emergent locality might be an artifact of SU(2) symmetry
in the Heisenberg model, not a genuine consequence of partial observation.

Test: three Hamiltonian families with decreasing symmetry:
  1. Heisenberg (SU(2) symmetric): H = sum J_ij (XX + YY + ZZ)
  2. XXZ delta=0.5 (U(1) only):   H = sum J_ij (XX + YY + 0.5*ZZ)
  3. Random Pauli (NO symmetry):   H = sum (Jx*XX + Jy*YY + Jz*ZZ), independent J's

For each: compute dim_ratio and Pearson r (correlation decay with MI distance).
Also test XX vs ZZ correlators separately to see anisotropy effects.

If PLC effect persists across all three -> partiality is the cause, not symmetry.

Built by Opus Warrior, March 5 2026.
"""

import numpy as np
import json
import time
import sys
from pathlib import Path
from itertools import combinations
from scipy import stats as sp_stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import (
    random_all_to_all, xxz_all_to_all, random_pauli_all_to_all,
    ground_state_gpu, ground_state,
    mutual_information_matrix, correlation_matrix,
    connected_correlation,
)

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

RESULTS_DIR = Path(__file__).parent / "results"


def _mi_to_distance(MI: np.ndarray) -> np.ndarray:
    """Convert mutual information matrix to distance matrix."""
    n = MI.shape[0]
    D = np.zeros_like(MI)
    for i in range(n):
        for j in range(i + 1, n):
            if MI[i, j] > 1e-14:
                D[i, j] = 1.0 / MI[i, j]
            else:
                D[i, j] = 1e6
            D[j, i] = D[i, j]
    return D


def _effective_dimension(D: np.ndarray, threshold: float = 0.9) -> float:
    """Effective embedding dimension via classical MDS."""
    n = D.shape[0]
    if n < 3:
        return 1.0
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = eigenvalues[::-1]
    pos_eig = eigenvalues[eigenvalues > 1e-10]
    if len(pos_eig) == 0:
        return 1.0
    total = np.sum(pos_eig)
    if total < 1e-14:
        return 1.0
    cumulative = np.cumsum(pos_eig) / total
    dim = np.searchsorted(cumulative, threshold) + 1
    return float(dim)


def compute_decay_pearson(psi, n_qubits, subset, pauli='Z'):
    """
    Compute Pearson r between log|correlation| and MI-distance
    for a given observer subset. Negative r = decay = locality.
    """
    MI_obs = mutual_information_matrix(psi, n_qubits, subset)
    k = len(subset)

    distances = []
    correlations = []
    for i in range(k):
        for j in range(i + 1, k):
            mi = MI_obs[i, j]
            if mi > 1e-14:
                d = 1.0 / mi
                c = abs(connected_correlation(psi, subset[i], subset[j], n_qubits, pauli))
                if c > 1e-14:
                    distances.append(d)
                    correlations.append(c)

    if len(distances) < 3:
        return float('nan'), float('nan')

    distances = np.array(distances)
    log_corr = np.log(np.array(correlations))

    if np.std(distances) < 1e-14 or np.std(log_corr) < 1e-14:
        return 0.0, float('nan')

    r, p = sp_stats.pearsonr(distances, log_corr)
    return float(r), float(p)


def compute_dim_ratio(psi, n_qubits, subset):
    """Compute effective dimension ratio (observer / full)."""
    MI_full = mutual_information_matrix(psi, n_qubits)
    D_full = _mi_to_distance(MI_full)
    dim_full = _effective_dimension(D_full)

    MI_obs = mutual_information_matrix(psi, n_qubits, subset)
    D_obs = _mi_to_distance(MI_obs)
    dim_obs = _effective_dimension(D_obs)

    ratio = dim_obs / dim_full if dim_full > 0 else float('nan')
    return float(dim_full), float(dim_obs), float(ratio)


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    values = np.array([v for v in values if np.isfinite(v)])
    if len(values) < 3:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(np.mean(values)), float(lo), float(hi)


def run_symmetry_breaking(n_qubits=8, n_trials=20):
    """Main experiment: test PLC effect across symmetry classes."""

    print("\n" + "=" * 70)
    print("  SYMMETRY-BREAKING CONTROL EXPERIMENT")
    print("  Does PLC effect persist when SU(2) is broken?")
    print(f"  N = {n_qubits}, {n_trials} trials per model")
    print(f"  GPU: {HAS_GPU}")
    print("=" * 70)

    diag_fn = ground_state_gpu if HAS_GPU else ground_state
    t0 = time.time()

    # k values: k/N ~ 0.375 and 0.5
    k_values = [max(3, round(n_qubits * 0.375)), n_qubits // 2]
    # Remove duplicates
    k_values = sorted(set(k_values))
    print(f"  Observer sizes k: {k_values} (k/N: {[k/n_qubits for k in k_values]})")

    models = {
        'heisenberg': {
            'name': 'Heisenberg (SU(2))',
            'symmetry': 'SU(2)',
            'build': lambda seed: random_all_to_all(n_qubits, seed=seed),
        },
        'xxz_0.5': {
            'name': 'XXZ delta=0.5 (U(1))',
            'symmetry': 'U(1)',
            'build': lambda seed: xxz_all_to_all(n_qubits, delta=0.5, seed=seed),
        },
        'random_pauli': {
            'name': 'Random Pauli (no symmetry)',
            'symmetry': 'none',
            'build': lambda seed: random_pauli_all_to_all(n_qubits, seed=seed),
        },
    }

    all_results = {}

    for model_key, model_info in models.items():
        print(f"\n{'─' * 70}")
        print(f"  MODEL: {model_info['name']}")
        print(f"  Symmetry: {model_info['symmetry']}")
        print(f"{'─' * 70}")

        model_data = {
            'name': model_info['name'],
            'symmetry': model_info['symmetry'],
            'trials': [],
        }

        for trial in range(n_trials):
            seed = 5000 + trial
            H_result = model_info['build'](seed)
            # random_all_to_all returns (H, couplings), others return (H, couplings/dict)
            H = H_result[0]

            E0, psi = diag_fn(H)

            trial_data = {
                'trial': trial,
                'seed': seed,
                'energy': float(E0),
                'k_results': {},
            }

            for k in k_values:
                k_over_n = k / n_qubits

                # Sample several observer subsets
                all_subsets = list(combinations(range(n_qubits), k))
                rng = np.random.default_rng(42 + trial * 100 + k)
                n_sample = min(len(all_subsets), 10)
                indices = rng.choice(len(all_subsets), n_sample, replace=False)

                dim_ratios = []
                pearson_zz = []
                pearson_xx = []

                for idx in indices:
                    subset = list(all_subsets[idx])

                    # Effective dimension ratio
                    _, _, ratio = compute_dim_ratio(psi, n_qubits, subset)
                    dim_ratios.append(ratio)

                    # Pearson r for ZZ correlations
                    r_zz, _ = compute_decay_pearson(psi, n_qubits, subset, pauli='Z')
                    pearson_zz.append(r_zz)

                    # Pearson r for XX correlations
                    r_xx, _ = compute_decay_pearson(psi, n_qubits, subset, pauli='X')
                    pearson_xx.append(r_xx)

                trial_data['k_results'][str(k)] = {
                    'k': k,
                    'k_over_N': float(k_over_n),
                    'dim_ratios': [float(x) for x in dim_ratios],
                    'pearson_zz': [float(x) for x in pearson_zz if np.isfinite(x)],
                    'pearson_xx': [float(x) for x in pearson_xx if np.isfinite(x)],
                    'mean_dim_ratio': float(np.nanmean(dim_ratios)),
                    'mean_pearson_zz': float(np.nanmean([x for x in pearson_zz if np.isfinite(x)])) if any(np.isfinite(x) for x in pearson_zz) else float('nan'),
                    'mean_pearson_xx': float(np.nanmean([x for x in pearson_xx if np.isfinite(x)])) if any(np.isfinite(x) for x in pearson_xx) else float('nan'),
                }

            model_data['trials'].append(trial_data)

            if (trial + 1) % 5 == 0:
                # Print progress
                last_k = str(k_values[-1])
                dr = trial_data['k_results'][last_k]['mean_dim_ratio']
                rzz = trial_data['k_results'][last_k]['mean_pearson_zz']
                rxx = trial_data['k_results'][last_k]['mean_pearson_xx']
                print(f"    Trial {trial+1}/{n_trials}: dim_ratio={dr:.3f}, r_ZZ={rzz:.3f}, r_XX={rxx:.3f}")

        all_results[model_key] = model_data

    # ─────────────────────────────────────────────────────────
    # Aggregate and compare across models
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    summary = {}

    for model_key, model_data in all_results.items():
        print(f"\n  {model_data['name']} ({model_data['symmetry']})")
        model_summary = {}

        for k in k_values:
            k_str = str(k)
            k_over_n = k / n_qubits

            # Collect across trials
            all_dim_ratios = []
            all_pearson_zz = []
            all_pearson_xx = []

            for trial_data in model_data['trials']:
                kr = trial_data['k_results'][k_str]
                all_dim_ratios.extend(kr['dim_ratios'])
                all_pearson_zz.extend(kr['pearson_zz'])
                all_pearson_xx.extend(kr['pearson_xx'])

            # Bootstrap CIs
            dr_mean, dr_lo, dr_hi = bootstrap_ci(all_dim_ratios)
            rzz_mean, rzz_lo, rzz_hi = bootstrap_ci(all_pearson_zz)
            rxx_mean, rxx_lo, rxx_hi = bootstrap_ci(all_pearson_xx)

            model_summary[k_str] = {
                'k': k,
                'k_over_N': float(k_over_n),
                'dim_ratio': {'mean': dr_mean, 'ci_lo': dr_lo, 'ci_hi': dr_hi, 'n': len(all_dim_ratios)},
                'pearson_zz': {'mean': rzz_mean, 'ci_lo': rzz_lo, 'ci_hi': rzz_hi, 'n': len(all_pearson_zz)},
                'pearson_xx': {'mean': rxx_mean, 'ci_lo': rxx_lo, 'ci_hi': rxx_hi, 'n': len(all_pearson_xx)},
            }

            print(f"    k/N = {k_over_n:.3f} (k={k}):")
            print(f"      dim_ratio:  {dr_mean:.3f}  [{dr_lo:.3f}, {dr_hi:.3f}]")
            print(f"      r_ZZ:       {rzz_mean:.3f}  [{rzz_lo:.3f}, {rzz_hi:.3f}]")
            print(f"      r_XX:       {rxx_mean:.3f}  [{rxx_lo:.3f}, {rxx_hi:.3f}]")

        summary[model_key] = {
            'name': model_data['name'],
            'symmetry': model_data['symmetry'],
            'k_results': model_summary,
        }

    # ─────────────────────────────────────────────────────────
    # Cross-model comparison table
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-MODEL COMPARISON")
    print("=" * 70)

    for k in k_values:
        k_str = str(k)
        k_over_n = k / n_qubits
        print(f"\n  k/N = {k_over_n:.3f}:")
        print(f"  {'Model':<30} {'dim_ratio':>10} {'r_ZZ':>10} {'r_XX':>10}")
        print(f"  {'─' * 62}")

        for model_key in ['heisenberg', 'xxz_0.5', 'random_pauli']:
            s = summary[model_key]['k_results'][k_str]
            dr = s['dim_ratio']['mean']
            rzz = s['pearson_zz']['mean']
            rxx = s['pearson_xx']['mean']
            name = summary[model_key]['name']
            print(f"  {name:<30} {dr:>10.3f} {rzz:>10.3f} {rxx:>10.3f}")

    # ─────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERDICT: Is the PLC effect a symmetry artifact?")
    print("=" * 70)

    # Check: does effect persist in random Pauli (no symmetry)?
    rp = summary['random_pauli']
    k_str = str(k_values[-1])  # use larger k for comparison
    rp_dr = rp['k_results'][k_str]['dim_ratio']['mean']
    rp_rzz = rp['k_results'][k_str]['pearson_zz']['mean']

    heis = summary['heisenberg']
    h_dr = heis['k_results'][k_str]['dim_ratio']['mean']
    h_rzz = heis['k_results'][k_str]['pearson_zz']['mean']

    print(f"\n  Heisenberg (SU(2)):    dim_ratio = {h_dr:.3f}, r_ZZ = {h_rzz:.3f}")
    print(f"  Random Pauli (none):   dim_ratio = {rp_dr:.3f}, r_ZZ = {rp_rzz:.3f}")

    # Effect persists if:
    # 1. dim_ratio < 1 in random Pauli (dimensionality reduction from partiality)
    # 2. r_ZZ < 0 in random Pauli (correlation decay with MI distance)
    dr_persists = rp_dr < 1.0
    decay_persists = rp_rzz < -0.1

    if dr_persists and decay_persists:
        print(f"\n  >>> EFFECT PERSISTS WITHOUT SU(2). <<<")
        print(f"  >>> PLC is about PARTIALITY, not symmetry. Paper STRENGTHENED. <<<")
        verdict = "PERSISTS"
    elif dr_persists or decay_persists:
        print(f"\n  >>> PARTIAL persistence. Nuanced discussion needed. <<<")
        verdict = "PARTIAL"
    else:
        print(f"\n  >>> EFFECT DOES NOT PERSIST. Symmetry artifact. Paper needs revision. <<<")
        verdict = "ARTIFACT"

    # XX vs ZZ anisotropy check
    print(f"\n  XX vs ZZ Anisotropy Check:")
    for model_key in ['heisenberg', 'xxz_0.5', 'random_pauli']:
        s = summary[model_key]['k_results'][k_str]
        rzz = s['pearson_zz']['mean']
        rxx = s['pearson_xx']['mean']
        name = summary[model_key]['name']
        diff = abs(rzz - rxx)
        print(f"    {name:<30}: r_ZZ={rzz:.3f}, r_XX={rxx:.3f}, |diff|={diff:.3f}")
        if model_key == 'heisenberg':
            if diff < 0.05:
                print(f"      -> XX ~ ZZ (expected: SU(2) symmetric)")
            else:
                print(f"      -> XX != ZZ (unexpected for SU(2))")
        elif model_key == 'xxz_0.5':
            print(f"      -> {'XX != ZZ' if diff > 0.05 else 'XX ~ ZZ'} (U(1): XX=YY but may differ from ZZ)")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    # Save results
    output = {
        'experiment': 'symmetry_breaking_control',
        'n_qubits': n_qubits,
        'n_trials': n_trials,
        'k_values': k_values,
        'summary': summary,
        'verdict': verdict,
        'elapsed_seconds': elapsed,
        'raw_data': {k: {
            'name': v['name'],
            'symmetry': v['symmetry'],
            'trial_count': len(v['trials']),
        } for k, v in all_results.items()},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS_DIR / "symmetry_breaking.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    print(f"\n  Results saved to {outpath}")
    return output


if __name__ == '__main__':
    run_symmetry_breaking(n_qubits=8, n_trials=20)
