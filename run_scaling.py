#!/usr/bin/env python3
"""
PLC Scaling Study: N=10 and N=12
Experiments 2 (emergent metric) and 5 (correlation decay) at larger system sizes.

For Physical Review Letters — clean N-scaling with full statistical rigor.
Built by Opus Warrior, March 5 2026.
"""

import sys
import os
import json
import time
import numpy as np
from itertools import combinations
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import (
    random_all_to_all, ground_state_gpu,
    mutual_information_matrix, correlation_matrix,
)
from src.experiments import _mi_to_distance, _effective_dimension
from src.statistics import (
    bootstrap_ci, permutation_test_dim_ratio,
    pvalue_r_negative, shuffled_null_pearson_r, _pearson_r_log_corr_vs_dist,
)
from src.utils import NumpyEncoder

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 10
SEEDS_BASE = 9000
N_BOOTSTRAP = 2000
K_RATIOS = [0.3, 0.5, 0.7]
MAX_SUBSETS_N12 = 5  # limit observer subset sampling for N=12


def closest_k(N, ratio):
    """Get closest valid k for a given k/N ratio."""
    k = max(3, round(N * ratio))
    k = min(k, N - 1)
    return k


def run_experiment_2_scaling(N, n_trials, seeds_base):
    """Experiment 2: Emergent metric (dim_ratio) for given N."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 2 — EMERGENT METRIC — N={N}")
    print(f"  Hilbert space: 2^{N} = {2**N} dimensions")
    print(f"  {n_trials} trials, seeds {seeds_base}+")
    print(f"{'='*60}")

    k_values = [closest_k(N, r) for r in K_RATIOS]
    k_values = sorted(set(k_values))
    print(f"  k values: {k_values} (ratios: {[k/N for k in k_values]})")

    # Storage: key = k -> lists
    data = {k: {"dim_full": [], "dim_obs": [], "ratio": []} for k in k_values}

    for trial in range(n_trials):
        seed = seeds_base + trial
        t0 = time.time()

        print(f"\n  Trial {trial+1}/{n_trials} (seed={seed})...", end="", flush=True)

        # Build Hamiltonian and find ground state
        H, couplings = random_all_to_all(N, seed=seed, use_gpu_build=(N >= 12))
        E0, psi = ground_state_gpu(H)

        # Full MI matrix and effective dimension
        MI_full = mutual_information_matrix(psi, N)
        D_full = _mi_to_distance(MI_full)
        dim_full = _effective_dimension(D_full)

        for k in k_values:
            # Sample observer subsets
            all_subsets = list(combinations(range(N), k))
            rng = np.random.default_rng(seed * 100 + k)

            if N >= 12:
                n_sub = min(len(all_subsets), MAX_SUBSETS_N12)
            else:
                n_sub = min(len(all_subsets), 10)

            sub_indices = rng.choice(len(all_subsets), n_sub, replace=False)

            dim_obs_list = []
            for si in sub_indices:
                subset = list(all_subsets[si])
                MI_obs = mutual_information_matrix(psi, N, subset)
                D_obs = _mi_to_distance(MI_obs)
                dim_obs_list.append(_effective_dimension(D_obs))

            dim_obs_mean = float(np.mean(dim_obs_list))
            ratio = dim_obs_mean / dim_full if dim_full > 0 else 0.0

            data[k]["dim_full"].append(dim_full)
            data[k]["dim_obs"].append(dim_obs_mean)
            data[k]["ratio"].append(ratio)

        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s | dim_full={dim_full:.1f}", end="")
        for k in k_values:
            print(f" | k={k}: ratio={data[k]['ratio'][-1]:.3f}", end="")
        print()

    # Compute statistics
    results = {}
    print(f"\n  --- STATISTICS (N={N}) ---")
    for k in k_values:
        k_over_N = k / N
        ratios = np.array(data[k]["ratio"])
        dim_full_arr = np.array(data[k]["dim_full"])
        dim_obs_arr = np.array(data[k]["dim_obs"])

        ci = bootstrap_ci(ratios, n_bootstrap=N_BOOTSTRAP, seed=42 + N * 100 + k)
        perm = permutation_test_dim_ratio(dim_full_arr, dim_obs_arr,
                                           n_perms=N_BOOTSTRAP, seed=42 + N * 100 + k)

        results[f"k={k}"] = {
            "N": N,
            "k": k,
            "k_over_N": round(k_over_N, 4),
            "n_trials": len(ratios),
            "dim_ratio": {
                "estimate": ci["estimate"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "se": ci["se"],
            },
            "permutation_p": perm["p_value"],
            "mean_dim_full": float(np.mean(dim_full_arr)),
            "mean_dim_obs": float(np.mean(dim_obs_arr)),
            "raw_ratios": ratios.tolist(),
        }

        sig = "***" if perm["p_value"] < 0.001 else "**" if perm["p_value"] < 0.01 else "*" if perm["p_value"] < 0.05 else "ns"
        print(f"  k={k} (k/N={k_over_N:.2f}): dim_ratio = {ci['estimate']:.4f} "
              f"95%CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] "
              f"perm_p={perm['p_value']:.4f} {sig}")

    return results


def run_experiment_5_scaling(N, n_trials, seeds_base):
    """Experiment 5: Correlation decay (Pearson r) for given N."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 5 — CORRELATION DECAY — N={N}")
    print(f"  {n_trials} trials, seeds {seeds_base}+")
    print(f"{'='*60}")

    k_values = [closest_k(N, r) for r in K_RATIOS]
    k_values = sorted(set(k_values))
    print(f"  k values: {k_values} (ratios: {[k/N for k in k_values]})")

    # Storage: key = k -> list of pearson_r
    data = {k: {"real_r": [], "null_r_mean": [], "null_p": []} for k in k_values}

    for trial in range(n_trials):
        seed = seeds_base + 500 + trial  # offset from exp2 seeds
        t0 = time.time()

        print(f"\n  Trial {trial+1}/{n_trials} (seed={seed})...", end="", flush=True)

        H, couplings = random_all_to_all(N, seed=seed, use_gpu_build=(N >= 12))
        E0, psi = ground_state_gpu(H)

        for k in k_values:
            all_subsets = list(combinations(range(N), k))
            rng = np.random.default_rng(seed * 100 + k)

            if N >= 12:
                n_sub = min(len(all_subsets), MAX_SUBSETS_N12)
            else:
                n_sub = min(len(all_subsets), 10)

            sub_indices = rng.choice(len(all_subsets), n_sub, replace=False)

            for si in sub_indices:
                subset = list(all_subsets[si])
                MI_obs = mutual_information_matrix(psi, N, subset)
                C_obs = correlation_matrix(psi, N, subset)

                # Pearson r between log|C| and 1/MI
                triu_i, triu_j = np.triu_indices(len(subset), k=1)
                mi_vals = MI_obs[triu_i, triu_j]
                corr_vals = np.abs(C_obs[triu_i, triu_j])
                real_r = _pearson_r_log_corr_vs_dist(mi_vals, corr_vals)

                # Null model (shuffled MI)
                null_result = shuffled_null_pearson_r(
                    MI_obs, C_obs, n_shuffles=200,
                    seed=seed * 1000 + k * 100 + si,
                )

                data[k]["real_r"].append(real_r)
                data[k]["null_r_mean"].append(null_result["shuffled_r_mean"])
                data[k]["null_p"].append(null_result["p_value"])

        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s", end="")
        for k in k_values:
            if data[k]["real_r"]:
                latest = data[k]["real_r"][-1]
                print(f" | k={k}: r={latest:.3f}", end="")
        print()

    # Compute statistics
    results = {}
    print(f"\n  --- STATISTICS (N={N}) ---")
    for k in k_values:
        k_over_N = k / N
        r_arr = np.array(data[k]["real_r"])
        null_r_arr = np.array(data[k]["null_r_mean"])
        null_p_arr = np.array(data[k]["null_p"])

        ci = bootstrap_ci(r_arr, n_bootstrap=N_BOOTSTRAP, seed=42 + N * 100 + k)
        p_test = pvalue_r_negative(r_arr, n_bootstrap=N_BOOTSTRAP, seed=42 + N * 100 + k)

        results[f"k={k}"] = {
            "N": N,
            "k": k,
            "k_over_N": round(k_over_N, 4),
            "n_samples": len(r_arr),
            "pearson_r": {
                "estimate": ci["estimate"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "se": ci["se"],
            },
            "p_value_bootstrap": p_test["p_value_bootstrap"],
            "p_value_ttest": p_test["p_value_ttest_onesided"],
            "p_value_sign": p_test["p_value_sign_test"],
            "n_negative": p_test["n_negative"],
            "n_total": p_test["n"],
            "mean_null_r": float(np.mean(null_r_arr)),
            "frac_null_sig": float(np.mean(null_p_arr < 0.05)),
            "raw_real_r": r_arr.tolist(),
        }

        sig = "***" if p_test["p_value_ttest_onesided"] < 0.001 else "**" if p_test["p_value_ttest_onesided"] < 0.01 else "*" if p_test["p_value_ttest_onesided"] < 0.05 else "ns"
        print(f"  k={k} (k/N={k_over_N:.2f}): r = {ci['estimate']:.4f} "
              f"95%CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] "
              f"p={p_test['p_value_ttest_onesided']:.6f} {sig} "
              f"({p_test['n_negative']}/{p_test['n']} negative)")

    return results


def main():
    print("\n" + "=" * 70)
    print("  PLC SCALING STUDY: N=10 and N=12")
    print("  Emergent Locality from Partial Observation")
    print("  For Physical Review Letters")
    print("=" * 70)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("\n  GPU: not available (CUDA not found), using CPU")
    except ImportError:
        print("\n  GPU: not available (torch not installed), using CPU")
    print(f"  Trials per N: {N_TRIALS}")
    print(f"  Bootstrap resamples: {N_BOOTSTRAP}")
    print(f"  k/N ratios: {K_RATIOS}")
    print(f"  Max subsets for N=12: {MAX_SUBSETS_N12}")

    t_total = time.time()
    all_results = {}

    # --- N=10 ---
    t_n10 = time.time()
    all_results["N=10"] = {
        "exp2": run_experiment_2_scaling(10, N_TRIALS, SEEDS_BASE),
        "exp5": run_experiment_5_scaling(10, N_TRIALS, SEEDS_BASE),
    }
    print(f"\n  N=10 total: {time.time() - t_n10:.1f}s")

    # --- N=12 ---
    t_n12 = time.time()
    all_results["N=12"] = {
        "exp2": run_experiment_2_scaling(12, N_TRIALS, SEEDS_BASE + 100),
        "exp5": run_experiment_5_scaling(12, N_TRIALS, SEEDS_BASE + 100),
    }
    print(f"\n  N=12 total: {time.time() - t_n12:.1f}s")

    # --- Summary ---
    total_elapsed = time.time() - t_total
    all_results["metadata"] = {
        "n_trials": N_TRIALS,
        "n_bootstrap": N_BOOTSTRAP,
        "k_ratios_target": K_RATIOS,
        "max_subsets_n12": MAX_SUBSETS_N12,
        "seeds_base": SEEDS_BASE,
        "total_elapsed_seconds": total_elapsed,
    }

    print(f"\n\n{'='*70}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*70}")

    print(f"\n  EXPERIMENT 2 — dim_ratio (lower = more locality):")
    print(f"  {'N':>3} {'k':>3} {'k/N':>5} {'dim_ratio':>10} {'95% CI':>20} {'perm_p':>8}")
    print(f"  {'-'*55}")
    for n_key in ["N=10", "N=12"]:
        for k_key, res in all_results[n_key]["exp2"].items():
            dr = res["dim_ratio"]
            print(f"  {res['N']:>3} {res['k']:>3} {res['k_over_N']:>5.2f} "
                  f"{dr['estimate']:>10.4f} [{dr['ci_low']:.4f}, {dr['ci_high']:.4f}] "
                  f"{res['permutation_p']:>8.4f}")

    print(f"\n  EXPERIMENT 5 — Pearson r (more negative = stronger decay = more locality):")
    print(f"  {'N':>3} {'k':>3} {'k/N':>5} {'Pearson r':>10} {'95% CI':>20} {'p-value':>10} {'neg/total':>10}")
    print(f"  {'-'*65}")
    for n_key in ["N=10", "N=12"]:
        for k_key, res in all_results[n_key]["exp5"].items():
            pr = res["pearson_r"]
            print(f"  {res['N']:>3} {res['k']:>3} {res['k_over_N']:>5.2f} "
                  f"{pr['estimate']:>10.4f} [{pr['ci_low']:.4f}, {pr['ci_high']:.4f}] "
                  f"{res['p_value_ttest']:>10.6f} "
                  f"{res['n_negative']}/{res['n_total']:>3}")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s")

    # Save
    outpath = RESULTS_DIR / "scaling_N10_N12.json"
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to: {outpath}")


if __name__ == "__main__":
    main()
