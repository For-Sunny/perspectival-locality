#!/usr/bin/env python3
"""
PLC Paper 2: Emergent Curvature from Partial Observation.

Runner script for curvature experiments. Computes Ollivier-Ricci and
Forman-Ricci curvature on MI-weighted graphs, then tests whether
partiality induces curvature structure where none existed.

Experiments:
    partiality      - Curvature vs observer size k/N
    entanglement    - Curvature vs entanglement (sweep XXZ anisotropy)
    local-vs-nonlocal - Compare 1D chain vs all-to-all curvature
    perspective     - Perspective-dependent curvature distribution
    all             - Run all experiments

Usage:
    python run_curvature.py                              # Run all, N=8
    python run_curvature.py --experiment partiality       # Single experiment
    python run_curvature.py --n-qubits 10 --n-samples 20 # Larger system
    python run_curvature.py --output-dir results/curvature/

Built by Opus Warrior, March 5 2026.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import HAS_TORCH
from src.utils import NumpyEncoder

# Import curvature experiment functions.
# These depend on src.curvature being built.
_modules_available = False
try:
    from src.curvature_experiments import (
        experiment_curvature_vs_partiality,
        experiment_curvature_vs_entanglement,
        experiment_local_vs_nonlocal,
        experiment_perspective_dependent_curvature,
        run_all_curvature,
    )
    _modules_available = True
except ImportError as e:
    _import_error = str(e)

import numpy as np


def _check_modules():
    """Verify curvature modules are importable. Exit with clear message if not."""
    if not _modules_available:
        print(f"ERROR: Required curvature modules not available.")
        print(f"Import error: {_import_error}")
        print("Ensure src/curvature.py and src/curvature_experiments.py exist.")
        sys.exit(1)


def _save_result(data: dict, name: str, output_dir: Path):
    """Save experiment result as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{name}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {filepath}")


def _print_summary_table(results: dict):
    """Print a summary table of all experiment results to stdout."""
    print()
    print("=" * 76)
    print("  CURVATURE EXPERIMENT SUMMARY")
    print("=" * 76)
    print(f"  {'Experiment':<28} {'Metric':<22} {'Value':<12} {'Status'}")
    print("-" * 76)

    for exp_name, exp_data in results.items():
        if not isinstance(exp_data, dict):
            continue

        if exp_name == "partiality":
            for entry in exp_data.get("results_table", []):
                k = entry.get("k", "?")
                k_n = entry.get("k_over_N", 0)
                sc = entry.get("scalar_curvature_mean", float("nan"))
                status = "CURVED" if abs(sc) > 0.01 else "~FLAT"
                print(f"  {exp_name:<28} {'k=' + str(k) + ' (k/N=' + f'{k_n:.2f}' + ')':<22} {sc:<+12.4f} {status}")

        elif exp_name == "entanglement":
            for rec in exp_data.get("records", []):
                delta = rec.get("delta", "?")
                sc = rec.get("ollivier_scalar_mean", float("nan"))
                entropy = rec.get("half_system_entropy_mean", 0)
                print(f"  {exp_name:<28} {'d=' + str(delta) + ' S=' + f'{entropy:.2f}':<22} {sc:<+12.4f}")

        elif exp_name == "local_vs_nonlocal":
            a2a = exp_data.get("all_to_all", {})
            chain = exp_data.get("chain", {})
            a2a_sc = a2a.get("ollivier_mean", float("nan"))
            chain_sc = chain.get("ollivier_mean", float("nan"))
            print(f"  {exp_name:<28} {'all-to-all ORC':<22} {a2a_sc:<+12.4f}")
            sign = "MORE NEG" if chain_sc < a2a_sc else "SIMILAR"
            print(f"  {'':<28} {'1D chain ORC':<22} {chain_sc:<+12.4f} {sign}")

        elif exp_name == "perspective":
            agg = exp_data.get("aggregate", {})
            spread = agg.get("mean_spread_std", float("nan"))
            dep = agg.get("is_perspective_dependent", False)
            status = "PERSPECTIVAL" if dep else "UNIFORM"
            print(f"  {exp_name:<28} {'spread (std)':<22} {spread:<12.4f} {status}")

    print("=" * 76)


def run_partiality(n_qubits: int, n_samples: int,
                   use_gpu: bool = True) -> dict:
    """Experiment: How does curvature depend on observer partiality (k/N)?"""
    _check_modules()
    result = experiment_curvature_vs_partiality(
        n_qubits=n_qubits,
        n_samples=n_samples,
        use_gpu=use_gpu,
    )
    return result


def run_entanglement(n_qubits: int, n_samples: int,
                     use_gpu: bool = True) -> dict:
    """Experiment: How does curvature relate to entanglement (XXZ sweep)?"""
    _check_modules()
    result = experiment_curvature_vs_entanglement(
        n_qubits=n_qubits,
        n_samples=n_samples,
        use_gpu=use_gpu,
    )
    return result


def run_local_vs_nonlocal(n_qubits: int, k: int, n_samples: int,
                          use_gpu: bool = True) -> dict:
    """Experiment: Local (1D chain) vs nonlocal (all-to-all) curvature."""
    _check_modules()
    result = experiment_local_vs_nonlocal(
        n_qubits=n_qubits,
        k=k,
        n_samples=n_samples,
        use_gpu=use_gpu,
    )
    return result


def run_perspective(n_qubits: int, k: int, n_samples: int,
                    use_gpu: bool = True) -> dict:
    """Experiment: Perspective-dependent curvature distribution."""
    _check_modules()
    result = experiment_perspective_dependent_curvature(
        n_qubits=n_qubits,
        k=k,
        n_samples=n_samples,
        use_gpu=use_gpu,
    )
    return result


EXPERIMENT_MAP = {
    "partiality": "partiality",
    "entanglement": "entanglement",
    "local-vs-nonlocal": "local_vs_nonlocal",
    "perspective": "perspective",
}


def main():
    parser = argparse.ArgumentParser(
        description="PLC Paper 2: Emergent Curvature from Partial Observation"
    )
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=list(EXPERIMENT_MAP.keys()) + ["all"],
        help="Which experiment to run (default: all)"
    )
    parser.add_argument(
        "--n-qubits", type=int, default=8,
        help="Number of qubits (default: 8)"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Observer subsystem size (default: n_qubits // 2)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of random Hamiltonian samples per experiment (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/curvature/",
        help="Output directory for JSON results (default: results/curvature/)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU acceleration"
    )

    args = parser.parse_args()

    n_qubits = args.n_qubits
    k = args.k if args.k is not None else n_qubits // 2
    n_samples = args.n_samples
    use_gpu = not args.no_gpu
    output_dir = Path(args.output_dir)

    # Ensure k is valid
    if k < 3:
        k = 3
    if k >= n_qubits:
        k = n_qubits - 1

    _check_modules()

    print("\n" + "=" * 60)
    print("  PLC PAPER 2: Emergent Curvature from Partial Observation")
    print("=" * 60)
    print(f"  System:     {n_qubits} qubits (Hilbert dim = {2**n_qubits})")
    print(f"  Observer:   k = {k} qubits (k/N = {k/n_qubits:.2f})")
    print(f"  Samples:    {n_samples} random Hamiltonians per experiment")
    print(f"  GPU:        {'enabled' if use_gpu and HAS_TORCH else 'disabled'}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)

    t_total = time.time()
    all_results = {}

    if args.experiment == "all":
        experiments = list(EXPERIMENT_MAP.keys())
    else:
        experiments = [args.experiment]

    for exp_name in experiments:
        key = EXPERIMENT_MAP[exp_name]

        if key == "partiality":
            result = run_partiality(n_qubits, n_samples, use_gpu)
        elif key == "entanglement":
            result = run_entanglement(n_qubits, n_samples, use_gpu)
        elif key == "local_vs_nonlocal":
            result = run_local_vs_nonlocal(n_qubits, k, n_samples, use_gpu)
        elif key == "perspective":
            result = run_perspective(n_qubits, k, n_samples, use_gpu)

        all_results[key] = result
        _save_result(result, f"curvature_{key}", output_dir)

    # Print summary
    _print_summary_table(all_results)

    total_elapsed = time.time() - t_total
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")

    # Save combined results
    combined = {
        "paper": "PLC Paper 2: Emergent Curvature from Partial Observation",
        "n_qubits": n_qubits,
        "k": k,
        "n_samples": n_samples,
        "total_elapsed_seconds": total_elapsed,
        "experiments": all_results,
    }
    _save_result(combined, "curvature_all", output_dir)

    return all_results


if __name__ == "__main__":
    main()
