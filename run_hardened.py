#!/usr/bin/env python3
"""
Hardened statistical analysis for PLC simulation.

Runs Experiments 2 and 5 with:
- 50 random Hamiltonians per system size (N=8, N=10)
- Bootstrap 95% CIs (10,000 samples)
- Permutation tests and one-sided p-values
- Shuffled null model for correlation decay

Saves everything to results/hardened_stats.json.

Usage:
    python run_hardened.py                # Full run (50 trials, N=8,10)
    python run_hardened.py --quick        # Quick test (10 trials, N=8 only)
    python run_hardened.py --no-gpu       # CPU only

Built by Opus Warrior, March 5 2026.
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.statistics import hardened_experiment_2, hardened_experiment_5


RESULTS_DIR = Path(__file__).parent / "results"


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def print_header():
    print()
    print("=" * 70)
    print("  PLC SIMULATION: HARDENED STATISTICAL ANALYSIS")
    print("  Perspectival Locality Conjecture -- Peer-Review Rigor")
    print("=" * 70)
    print()


def print_summary(exp2_results: dict, exp5_results: dict):
    print()
    print("=" * 70)
    print("  SUMMARY: STATISTICAL HARDENING RESULTS")
    print("=" * 70)

    # Experiment 2 summary
    print()
    print("  EXPERIMENT 2: Emergent Metric (Dimensionality Reduction)")
    print("  " + "-" * 66)
    print(f"  {'N':>3} {'k':>3} {'k/N':>6} {'dim_ratio':>10} {'95% CI':>22} {'perm p':>10} {'sig':>4}")
    print("  " + "-" * 66)

    for key, data in sorted(exp2_results.items()):
        ci = data["dim_ratio_bootstrap"]
        perm = data["permutation_test"]
        p = perm["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ci_str = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        print(f"  {data['N']:>3} {data['k']:>3} {data['k_over_N']:>6.2f} "
              f"{ci['estimate']:>10.4f} {ci_str:>22} {p:>10.6f} {sig:>4}")

    # Experiment 5 summary
    print()
    print("  EXPERIMENT 5: Correlation Decay (Locality Smoking Gun)")
    print("  " + "-" * 66)
    print(f"  {'N':>3} {'k':>3} {'k/N':>6} {'Pearson r':>10} {'95% CI':>22} {'p(r<0)':>10} {'sig':>4}")
    print("  " + "-" * 66)

    for key, data in sorted(exp5_results.items()):
        ci = data["pearson_r_bootstrap"]
        pt = data["decay_pvalue"]
        p = pt["p_value_ttest_onesided"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ci_str = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        print(f"  {data['N']:>3} {data['k']:>3} {data['k_over_N']:>6.2f} "
              f"{ci['estimate']:>10.4f} {ci_str:>22} {p:>10.6f} {sig:>4}")

    # Null model summary
    print()
    print("  NULL MODEL: Shuffled MI vs Real Structure")
    print("  " + "-" * 66)
    print(f"  {'N':>3} {'k':>3} {'real r':>8} {'null r':>8} {'effect d':>10} {'null p<.05':>10}")
    print("  " + "-" * 66)

    for key, data in sorted(exp5_results.items()):
        nm = data["null_model"]
        print(f"  {data['N']:>3} {data['k']:>3} "
              f"{nm['mean_real_r']:>8.4f} {nm['mean_null_r']:>8.4f} "
              f"{nm['mean_effect_size']:>10.2f} {nm['frac_null_p_lt_005']:>9.1%}")

    # Verdict
    print()
    print("  " + "=" * 66)

    # Check if all key results are significant
    all_exp2_sig = all(
        d["permutation_test"]["p_value"] < 0.05
        for d in exp2_results.values()
    )
    all_exp5_sig = all(
        d["decay_pvalue"]["p_value_ttest_onesided"] < 0.05
        for d in exp5_results.values()
    )
    null_confirms = all(
        d["null_model"]["frac_null_p_lt_005"] > 0.5
        for d in exp5_results.values()
    )

    if all_exp2_sig and all_exp5_sig and null_confirms:
        print("  VERDICT: ALL TESTS PASS. Results are statistically robust.")
        print("  - Dimensionality reduction from partiality: CONFIRMED (all p < 0.05)")
        print("  - Correlation decay with emergent distance: CONFIRMED (all p < 0.05)")
        print("  - Null model confirms effect is real, not an artifact")
    elif all_exp2_sig and all_exp5_sig:
        print("  VERDICT: PRIMARY TESTS PASS. Null model needs review.")
        print("  - Dimensionality reduction: CONFIRMED")
        print("  - Correlation decay: CONFIRMED")
        print("  - Some null model results inconclusive")
    else:
        failed = []
        if not all_exp2_sig:
            failed.append("Exp2 (dimensionality)")
        if not all_exp5_sig:
            failed.append("Exp5 (correlation decay)")
        print(f"  VERDICT: MIXED. Some tests not significant: {', '.join(failed)}")
        print("  Review individual results for details.")

    print("  " + "=" * 66)
    print()


def main():
    parser = argparse.ArgumentParser(description="Hardened PLC Statistical Analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 10 trials, N=8 only")
    parser.add_argument("--no-gpu", action="store_true",
                        help="CPU only (slower)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Override number of trials")
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="Number of bootstrap samples (default: 10000)")
    args = parser.parse_args()

    use_gpu = not args.no_gpu

    if args.quick:
        n_qubits_list = [8]
        n_trials = args.trials or 10
        n_bootstrap = min(args.bootstrap, 2000)
        n_shuffles = 50
    else:
        n_qubits_list = [8, 10]
        n_trials = args.trials or 50
        n_bootstrap = args.bootstrap
        n_shuffles = 200

    print_header()
    print(f"  Configuration:")
    print(f"    System sizes: N = {n_qubits_list}")
    print(f"    Trials per size: {n_trials}")
    print(f"    Bootstrap samples: {n_bootstrap}")
    print(f"    Null model shuffles: {n_shuffles}")
    print(f"    GPU: {'enabled' if use_gpu else 'disabled'}")
    print()

    t_total = time.time()

    # Experiment 2: Emergent Metric
    print("=" * 70)
    print("  EXPERIMENT 2: Emergent Metric (Dimensionality Reduction)")
    print("=" * 70)
    t0 = time.time()
    exp2_results = hardened_experiment_2(
        n_qubits_list=n_qubits_list,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        use_gpu=use_gpu,
    )
    t_exp2 = time.time() - t0
    print(f"\n  Experiment 2 complete: {t_exp2:.1f}s")

    # Experiment 5: Correlation Decay
    print()
    print("=" * 70)
    print("  EXPERIMENT 5: Correlation Decay + Null Model")
    print("=" * 70)
    t0 = time.time()
    exp5_results = hardened_experiment_5(
        n_qubits_list=n_qubits_list,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        n_shuffles=n_shuffles,
        use_gpu=use_gpu,
    )
    t_exp5 = time.time() - t0
    print(f"\n  Experiment 5 complete: {t_exp5:.1f}s")

    # Print summary
    print_summary(exp2_results, exp5_results)

    # Save all results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "n_qubits_list": n_qubits_list,
            "n_trials": n_trials,
            "n_bootstrap": n_bootstrap,
            "n_shuffles": n_shuffles,
            "use_gpu": use_gpu,
            "total_time_seconds": time.time() - t_total,
            "exp2_time_seconds": t_exp2,
            "exp5_time_seconds": t_exp5,
        },
        "experiment_2_emergent_metric": exp2_results,
        "experiment_5_correlation_decay": exp5_results,
    }

    outpath = RESULTS_DIR / "hardened_stats.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"  Results saved to: {outpath}")
    print(f"  Total time: {time.time() - t_total:.1f}s")
    print()


if __name__ == "__main__":
    main()
