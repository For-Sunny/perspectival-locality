#!/usr/bin/env python3
"""
Stronger null models for PLC correlation decay.

Tests whether the PLC effect (correlations decay with MI-distance) survives
null models that preserve MORE structure than simple shuffling.

4 null models, ordered by what they preserve:
  1. Shuffle (original) - destroys ALL structure
  2. Eigenvalue-preserving - keeps MI spectrum, randomizes assignment
  3. Degree-preserving - keeps per-qubit total MI, randomizes pairwise
  4. Random Hamiltonian - uses wrong state's MI on right state's correlations

If PLC survives all 4 -> the effect is real, specific, and publishable.

Built by Opus Warrior, March 5 2026.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from itertools import combinations

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum import (
    random_all_to_all, ground_state, ground_state_gpu,
    mutual_information_matrix, correlation_matrix,
)
from src.statistics import (
    shuffled_null_pearson_r,
    eigenvalue_preserving_null,
    degree_preserving_null,
    random_hamiltonian_null,
    bootstrap_ci,
    _pearson_r_log_corr_vs_dist,
)
from src.utils import NumpyEncoder


def run_null_model_battery(N: int = 8, n_trials: int = 10,
                            k_over_N_list: list[float] = [0.375, 0.5],
                            n_shuffles_fast: int = 200,
                            n_shuffles_hamiltonian: int = 20,
                            use_gpu: bool = True,
                            seed_base: int = 42000) -> dict:
    """
    Run all 4 null models for given parameters.

    For each trial and k/N ratio:
      - Pick a random subset of k qubits
      - Compute real Pearson r
      - Run all 4 null models
      - Collect distributions

    Returns structured results for analysis.
    """
    diag_fn = ground_state_gpu if use_gpu else ground_state
    results = {}

    for k_ratio in k_over_N_list:
        k = max(3, round(k_ratio * N))
        key = f"N={N}_k={k}_ratio={k_ratio}"
        print(f"\n{'='*60}")
        print(f"  N={N}, k={k}, k/N={k_ratio}")
        print(f"{'='*60}")

        data = {
            "N": N, "k": k, "k_over_N": k_ratio,
            "real_r": [],
            "shuffle": {"rs": [], "means": [], "stds": [], "p_values": [], "effect_sizes": []},
            "eigenvalue": {"rs": [], "means": [], "stds": [], "p_values": [], "effect_sizes": []},
            "degree": {"rs": [], "means": [], "stds": [], "p_values": [], "effect_sizes": []},
            "hamiltonian": {"rs": [], "means": [], "stds": [], "p_values": [], "effect_sizes": []},
        }

        all_subsets = list(combinations(range(N), k))

        for trial in range(n_trials):
            t0 = time.time()
            seed = seed_base + N * 1000 + trial
            rng = np.random.default_rng(seed)

            # Generate random Hamiltonian and ground state
            H, couplings = random_all_to_all(N, seed=seed)
            E0, psi = diag_fn(H)

            # Pick a random subset
            sub_idx = rng.integers(0, len(all_subsets))
            subset = list(all_subsets[sub_idx])

            # Compute MI and correlation matrices for observer
            MI_obs = mutual_information_matrix(psi, N, subset)
            C_obs = correlation_matrix(psi, N, subset)

            # Real Pearson r
            n_sub = len(subset)
            triu_i, triu_j = np.triu_indices(n_sub, k=1)
            mi_vals = MI_obs[triu_i, triu_j]
            corr_vals = np.abs(C_obs[triu_i, triu_j])
            real_r = _pearson_r_log_corr_vs_dist(mi_vals, corr_vals)
            data["real_r"].append(float(real_r))

            # --- Null 1: Shuffle (original, weak) ---
            s1 = shuffled_null_pearson_r(MI_obs, C_obs,
                                          n_shuffles=n_shuffles_fast, seed=seed)
            data["shuffle"]["rs"].append(s1["real_r"])
            data["shuffle"]["means"].append(s1["shuffled_r_mean"])
            data["shuffle"]["stds"].append(s1["shuffled_r_std"])
            data["shuffle"]["p_values"].append(s1["p_value"])
            data["shuffle"]["effect_sizes"].append(s1["effect_size"])

            # --- Null 2: Eigenvalue-preserving ---
            s2 = eigenvalue_preserving_null(MI_obs, C_obs,
                                             n_shuffles=n_shuffles_fast, seed=seed + 1)
            data["eigenvalue"]["means"].append(s2["null_r_mean"])
            data["eigenvalue"]["stds"].append(s2["null_r_std"])
            data["eigenvalue"]["p_values"].append(s2["p_value"])
            data["eigenvalue"]["effect_sizes"].append(s2["effect_size"])

            # --- Null 3: Degree-preserving ---
            s3 = degree_preserving_null(MI_obs, C_obs,
                                         n_shuffles=n_shuffles_fast, seed=seed + 2)
            data["degree"]["means"].append(s3["null_r_mean"])
            data["degree"]["stds"].append(s3["null_r_std"])
            data["degree"]["p_values"].append(s3["p_value"])
            data["degree"]["effect_sizes"].append(s3["effect_size"])

            # --- Null 4: Random Hamiltonian (expensive) ---
            s4 = random_hamiltonian_null(psi, C_obs, N, k, subset,
                                          n_shuffles=n_shuffles_hamiltonian,
                                          seed=seed + 3, use_gpu=use_gpu)
            data["hamiltonian"]["means"].append(s4["null_r_mean"])
            data["hamiltonian"]["stds"].append(s4["null_r_std"])
            data["hamiltonian"]["p_values"].append(s4["p_value"])
            data["hamiltonian"]["effect_sizes"].append(s4["effect_size"])

            elapsed = time.time() - t0
            print(f"  Trial {trial+1}/{n_trials}: real_r={real_r:.4f} | "
                  f"shuffle_p={s1['p_value']:.3f} eigen_p={s2['p_value']:.3f} "
                  f"degree_p={s3['p_value']:.3f} hamil_p={s4['p_value']:.3f} | "
                  f"{elapsed:.1f}s")

        # --- Summary statistics ---
        real_arr = np.array(data["real_r"])

        summary = {
            "N": N, "k": k, "k_over_N": k_ratio, "n_trials": n_trials,
            "real_r_mean": float(np.mean(real_arr)),
            "real_r_std": float(np.std(real_arr, ddof=1)),
            "real_r_ci": bootstrap_ci(real_arr, seed=seed_base),
        }

        for null_name in ["shuffle", "eigenvalue", "degree", "hamiltonian"]:
            null_data = data[null_name]
            means = np.array(null_data["means"])
            p_vals = np.array(null_data["p_values"])
            es_vals = np.array(null_data["effect_sizes"])

            # Effect size: (real_r - null_mean) / null_std, averaged across trials
            mean_es = float(np.mean(es_vals))
            # Fraction of trials where null was rejected
            frac_sig = float(np.mean(p_vals < 0.05))

            # Bootstrap CI on effect sizes
            es_ci = bootstrap_ci(es_vals, seed=seed_base + hash(null_name) % 10000)

            summary[null_name] = {
                "null_r_mean": float(np.mean(means)),
                "null_r_std": float(np.std(means, ddof=1)),
                "mean_p_value": float(np.mean(p_vals)),
                "median_p_value": float(np.median(p_vals)),
                "frac_significant": frac_sig,
                "mean_effect_size": mean_es,
                "effect_size_ci": es_ci,
                "interpretation": _interpret_null(null_name, mean_es, frac_sig),
            }

        # Print summary
        print(f"\n  --- Summary for N={N}, k={k} (k/N={k_ratio}) ---")
        print(f"  Real r: {summary['real_r_mean']:.4f} +/- {summary['real_r_std']:.4f}")
        print(f"  {'Null Model':<25} {'Null r':<12} {'Effect Size':<15} {'Frac p<0.05':<12} {'Verdict'}")
        print(f"  {'-'*80}")
        for null_name in ["shuffle", "eigenvalue", "degree", "hamiltonian"]:
            ns = summary[null_name]
            print(f"  {null_name:<25} {ns['null_r_mean']:<12.4f} "
                  f"{ns['mean_effect_size']:<15.2f} {ns['frac_significant']:<12.1%} "
                  f"{ns['interpretation']}")

        data["summary"] = summary
        results[key] = data

    return results


def _interpret_null(null_name: str, effect_size: float, frac_sig: float) -> str:
    """Interpret null model results for the paper."""
    if frac_sig > 0.8 and abs(effect_size) > 2.0:
        verdict = "STRONG REJECTION"
    elif frac_sig > 0.5 and abs(effect_size) > 1.0:
        verdict = "MODERATE REJECTION"
    elif frac_sig > 0.2 or abs(effect_size) > 0.5:
        verdict = "WEAK REJECTION"
    else:
        verdict = "FAILS TO REJECT"

    what_it_means = {
        "shuffle": "structure exists",
        "eigenvalue": "qubit-pair assignments matter, not just spectrum",
        "degree": "pairwise structure matters, not just connectivity",
        "hamiltonian": "state-specific MI matters, not generic ground state",
    }

    return f"{verdict} -> {what_it_means.get(null_name, '?')}"


def main():
    print("=" * 70)
    print("  PLC NULL MODEL BATTERY")
    print("  Testing whether correlation decay survives stronger nulls")
    print("=" * 70)

    t_start = time.time()

    # Run for N=8 with 10 trials at k/N = 0.375 and 0.5
    results = run_null_model_battery(
        N=8,
        n_trials=10,
        k_over_N_list=[0.375, 0.5],
        n_shuffles_fast=200,
        n_shuffles_hamiltonian=100,
        use_gpu=True,
        seed_base=42000,
    )

    elapsed = time.time() - t_start

    # Prepare output (strip large arrays for JSON, keep summaries)
    output = {
        "metadata": {
            "N": 8,
            "n_trials": 10,
            "k_over_N_list": [0.375, 0.5],
            "n_shuffles_fast": 200,
            "n_shuffles_hamiltonian": 100,
            "total_time_seconds": elapsed,
        },
        "results": {},
    }

    for key, data in results.items():
        output["results"][key] = {
            "summary": data["summary"],
            "real_r_values": data["real_r"],
            "per_null": {},
        }
        for null_name in ["shuffle", "eigenvalue", "degree", "hamiltonian"]:
            output["results"][key]["per_null"][null_name] = {
                "means": data[null_name]["means"],
                "stds": data[null_name]["stds"],
                "p_values": data[null_name]["p_values"],
                "effect_sizes": data[null_name]["effect_sizes"],
            }

    # Save
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "null_models.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*70}")
    print(f"  COMPLETE. {elapsed:.1f}s total. Saved to {out_path}")
    print(f"{'='*70}")

    # Final verdict
    print(f"\n  FINAL ANALYSIS:")
    print(f"  {'='*60}")
    for key, data in results.items():
        s = data["summary"]
        print(f"\n  {key}:")
        print(f"    Real r = {s['real_r_mean']:.4f}")
        all_pass = True
        for null_name in ["shuffle", "eigenvalue", "degree", "hamiltonian"]:
            ns = s[null_name]
            status = "PASS" if ns["frac_significant"] > 0.5 or abs(ns["mean_effect_size"]) > 1.0 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"    {null_name:<25}: effect_size={ns['mean_effect_size']:.2f}, "
                  f"frac_sig={ns['frac_significant']:.1%} -> {status}")
        verdict = "PLC EFFECT IS REAL" if all_pass else "PLC EFFECT NEEDS MORE INVESTIGATION"
        print(f"    >>> {verdict}")


if __name__ == "__main__":
    main()
