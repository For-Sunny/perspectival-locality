#!/usr/bin/env python3
"""
Distance Robustness Test for PLC Simulation.

Referee question: does the emergent locality result depend on the specific
MI-to-distance transformation? We test 5 different distance definitions
and show whether dim_ratio and Pearson r are robust across all of them.

Built by Opus Warrior, March 5 2026.
"""

import numpy as np
import json
import time
import sys
from itertools import combinations
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import (
    random_all_to_all, ground_state_gpu, ground_state,
    mutual_information_matrix, correlation_matrix,
)
from src.experiments import _effective_dimension
from src.statistics import bootstrap_ci
from src.utils import NumpyEncoder


# ─────────────────────────────────────────────────────────────
# Distance definitions
# ─────────────────────────────────────────────────────────────

def d_subtract(MI):
    """I_max - I(i:j). Linear in MI, zero for max-correlated pairs."""
    D = MI.max() - MI
    np.fill_diagonal(D, 0.0)
    return D

def d_inverse(MI):
    """1/I(i:j). The original distance used in experiments.py."""
    D = 1.0 / (MI + 1e-10)
    np.fill_diagonal(D, 0.0)
    return D

def d_neglog(MI):
    """-log(I/I_max). Logarithmic scaling, infinite for zero MI."""
    mi_max = MI.max() if MI.max() > 0 else 1.0
    D = -np.log(MI / mi_max + 1e-10)
    np.fill_diagonal(D, 0.0)
    return D

def d_normalized(MI):
    """1 - I/I_max. Bounded [0,1], linear."""
    mi_max = MI.max() if MI.max() > 0 else 1.0
    D = 1.0 - MI / mi_max
    np.fill_diagonal(D, 0.0)
    return D

def d_sqrt(MI):
    """sqrt(I_max - I). Sub-linear, compresses large distances."""
    D = np.sqrt(np.maximum(MI.max() - MI, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


DISTANCE_FNS = {
    "subtract":   d_subtract,
    "inverse":    d_inverse,
    "neglog":     d_neglog,
    "normalized": d_normalized,
    "sqrt":       d_sqrt,
}


# Alias for backward compatibility
effective_dimension = _effective_dimension


# ─────────────────────────────────────────────────────────────
# Triangle inequality check
# ─────────────────────────────────────────────────────────────

def triangle_inequality_rate(D):
    """Fraction of (i,j,k) triples satisfying triangle inequality for all orderings."""
    n = D.shape[0]
    n_violations = 0
    n_checks = 0
    for i, j, k in combinations(range(n), 3):
        for a, b, c in [(i, j, k), (j, k, i), (k, i, j)]:
            n_checks += 1
            if D[a, c] > D[a, b] + D[b, c] + 1e-10:
                n_violations += 1
    rate = 1.0 - n_violations / n_checks if n_checks > 0 else 1.0
    return rate


# ─────────────────────────────────────────────────────────────
# Pearson r between |C_ij| and distance
# ─────────────────────────────────────────────────────────────

def pearson_r_corr_vs_dist(C, D):
    """Pearson r between |C_ij| and D_ij for all off-diagonal pairs.

    Note: zero distance (max-MI pair) is meaningful, not degenerate.
    We only filter infinite/NaN distances and truly zero correlations.
    """
    n = C.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    corr_vals = np.abs(C[triu_i, triu_j])
    dist_vals = D[triu_i, triu_j]

    # Only filter infinite/NaN distances. Zero distance is valid (max-MI pair).
    valid = np.isfinite(dist_vals) & (dist_vals < 1e6)
    if np.sum(valid) < 3:
        return 0.0

    cv = corr_vals[valid]
    dv = dist_vals[valid]

    if np.std(dv) < 1e-14 or np.std(cv) < 1e-14:
        return 0.0

    return float(np.corrcoef(dv, cv)[0, 1])


# ─────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────

def run_distance_robustness(N=8, n_trials=20, k_over_N_values=(0.375, 0.5), use_gpu=True):
    """Run the full distance robustness test."""

    print("\n" + "=" * 70)
    print("  DISTANCE ROBUSTNESS TEST")
    print(f"  N={N}, {n_trials} trials, k/N = {k_over_N_values}")
    print(f"  5 distance metrics: {list(DISTANCE_FNS.keys())}")
    print("=" * 70)

    t0 = time.time()
    diag_fn = ground_state_gpu if use_gpu else ground_state

    k_values = [max(3, int(round(r * N))) for r in k_over_N_values]
    # Deduplicate while preserving order
    seen = set()
    k_values_unique = []
    for k in k_values:
        if k not in seen:
            k_values_unique.append(k)
            seen.add(k)
    k_values = k_values_unique

    print(f"  k values: {k_values} (from k/N = {k_over_N_values})")

    # Storage: metric -> k -> lists of measurements
    data = {}
    for metric_name in DISTANCE_FNS:
        data[metric_name] = {}
        for k in k_values:
            data[metric_name][k] = {
                "dim_ratio": [],
                "pearson_r": [],
                "triangle_ineq_rate": [],
            }

    for trial in range(n_trials):
        seed = 9000 + trial
        H, couplings = random_all_to_all(N, seed=seed)
        E0, psi = diag_fn(H)

        # Compute full MI and correlation matrices once
        MI_full = mutual_information_matrix(psi, N)
        C_full = correlation_matrix(psi, N)

        for k in k_values:
            # Sample observer subsets
            all_subsets = list(combinations(range(N), k))
            rng = np.random.default_rng(seed + k * 100)
            n_sub = min(len(all_subsets), 10)
            sub_indices = rng.choice(len(all_subsets), n_sub, replace=False)

            for si in sub_indices:
                subset = list(all_subsets[si])
                MI_obs = mutual_information_matrix(psi, N, subset)
                C_obs = correlation_matrix(psi, N, subset)

                for metric_name, dist_fn in DISTANCE_FNS.items():
                    # Observer distance
                    D_obs = dist_fn(MI_obs)

                    # Full system distance (for dim ratio)
                    D_full = dist_fn(MI_full)

                    # Effective dimensions
                    dim_full = effective_dimension(D_full)
                    dim_obs = effective_dimension(D_obs)
                    dim_ratio = dim_obs / dim_full if dim_full > 0 else 0.0

                    # Pearson r between |C| and distance
                    r = pearson_r_corr_vs_dist(C_obs, D_obs)

                    # Triangle inequality
                    tri_rate = triangle_inequality_rate(D_obs)

                    data[metric_name][k]["dim_ratio"].append(dim_ratio)
                    data[metric_name][k]["pearson_r"].append(r)
                    data[metric_name][k]["triangle_ineq_rate"].append(tri_rate)

        if (trial + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Trial {trial+1}/{n_trials} ({elapsed:.1f}s)")

    # Compute bootstrap CIs and build results
    results = {}
    for metric_name in DISTANCE_FNS:
        results[metric_name] = {}
        for k in k_values:
            k_over_N = round(k / N, 4)
            d = data[metric_name][k]

            dim_arr = np.array(d["dim_ratio"])
            r_arr = np.array(d["pearson_r"])
            tri_arr = np.array(d["triangle_ineq_rate"])

            dim_ci = bootstrap_ci(dim_arr, n_bootstrap=5000, seed=42)
            r_ci = bootstrap_ci(r_arr, n_bootstrap=5000, seed=43)

            results[metric_name][str(k_over_N)] = {
                "dim_ratio": {
                    "estimate": dim_ci["estimate"],
                    "ci_low": dim_ci["ci_low"],
                    "ci_high": dim_ci["ci_high"],
                    "se": dim_ci["se"],
                    "n": dim_ci["n"],
                },
                "pearson_r": {
                    "estimate": r_ci["estimate"],
                    "ci_low": r_ci["ci_low"],
                    "ci_high": r_ci["ci_high"],
                    "se": r_ci["se"],
                    "n": r_ci["n"],
                },
                "triangle_ineq_rate": float(np.mean(tri_arr)),
                "triangle_ineq_std": float(np.std(tri_arr)),
                "n_trials": len(dim_arr),
            }

    total_elapsed = time.time() - t0

    # ─── Print summary table ───
    print("\n" + "=" * 70)
    print("  DISTANCE ROBUSTNESS RESULTS")
    print("=" * 70)

    for k in k_values:
        k_over_N = round(k / N, 4)
        k_str = str(k_over_N)

        print(f"\n  k/N = {k_over_N} (k={k}, N={N})")
        print(f"  {'Metric':<12} {'dim_ratio':>12} {'95% CI':>20} {'Pearson r':>12} {'95% CI':>20} {'Triangle%':>10}")
        print(f"  {'-'*12} {'-'*12} {'-'*20} {'-'*12} {'-'*20} {'-'*10}")

        for metric_name in DISTANCE_FNS:
            r = results[metric_name][k_str]
            dr = r["dim_ratio"]
            pr = r["pearson_r"]
            tri = r["triangle_ineq_rate"]

            print(f"  {metric_name:<12} "
                  f"{dr['estimate']:>12.4f} "
                  f"[{dr['ci_low']:.4f}, {dr['ci_high']:.4f}] "
                  f"{pr['estimate']:>12.4f} "
                  f"[{pr['ci_low']:.4f}, {pr['ci_high']:.4f}] "
                  f"{tri*100:>9.1f}%")

    # Cross-metric consistency check
    print(f"\n  {'='*70}")
    print(f"  CROSS-METRIC CONSISTENCY")
    print(f"  {'='*70}")

    for k in k_values:
        k_over_N = round(k / N, 4)
        k_str = str(k_over_N)

        dim_estimates = [results[m][k_str]["dim_ratio"]["estimate"] for m in DISTANCE_FNS]
        r_estimates = [results[m][k_str]["pearson_r"]["estimate"] for m in DISTANCE_FNS]

        dim_range = max(dim_estimates) - min(dim_estimates)
        r_range = max(r_estimates) - min(r_estimates)
        dim_cv = np.std(dim_estimates) / np.mean(dim_estimates) if np.mean(dim_estimates) > 0 else 0
        r_cv = np.std(r_estimates) / abs(np.mean(r_estimates)) if abs(np.mean(r_estimates)) > 0 else 0

        # Check sign agreement for Pearson r
        n_negative = sum(1 for r in r_estimates if r < 0)
        sign_agreement = n_negative == len(r_estimates) or n_negative == 0

        print(f"\n  k/N = {k_over_N}:")
        print(f"    dim_ratio: mean={np.mean(dim_estimates):.4f}, "
              f"range={dim_range:.4f}, CV={dim_cv:.2%}")
        print(f"    pearson_r: mean={np.mean(r_estimates):.4f}, "
              f"range={r_range:.4f}, CV={r_cv:.2%}")
        print(f"    Sign agreement (Pearson r): {n_negative}/{len(r_estimates)} negative "
              f"{'[ALL AGREE]' if sign_agreement else '[MIXED - CAUTION]'}")

        # Verdict: direction consistency matters more than magnitude matching
        all_dim_below_1 = all(results[m][k_str]["dim_ratio"]["estimate"] < 1.0
                              for m in DISTANCE_FNS)
        n_dim_below_1 = sum(1 for m in DISTANCE_FNS
                            if results[m][k_str]["dim_ratio"]["estimate"] < 1.0)

        if sign_agreement and all_dim_below_1:
            print(f"    VERDICT: ROBUST - all 5 metrics show negative r AND dim_ratio < 1")
        elif n_negative >= 4 and n_dim_below_1 >= 4:
            print(f"    VERDICT: MOSTLY ROBUST - {n_negative}/5 negative r, "
                  f"{n_dim_below_1}/5 dim_ratio < 1")
        elif sign_agreement:
            print(f"    VERDICT: CORRELATION ROBUST (all negative r), "
                  f"dim_ratio mixed ({n_dim_below_1}/5 < 1)")
        else:
            print(f"    VERDICT: SENSITIVE - result depends on distance definition")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "distance_robustness.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_distance_robustness(N=8, n_trials=20, k_over_N_values=(0.375, 0.5), use_gpu=True)
