#!/usr/bin/env python3
"""
PLC Control Experiments Runner.

Three controls that make the emergent locality result publication-ready.
Answers the reviewer question: "How do you know this is special?"

Usage:
    python run_controls.py                    # Run all three controls
    python run_controls.py --control A        # Run specific control
    python run_controls.py --control B
    python run_controls.py --control C
    python run_controls.py --no-gpu           # CPU only
    python run_controls.py --trials 10        # Fewer trials (faster)

Built by Opus Warrior, March 5 2026.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.controls import (
    control_A_nearest_neighbor,
    control_B_haar_random,
    control_C_planted_partition,
    compare_all_controls,
)


def main():
    parser = argparse.ArgumentParser(description="PLC Control Experiments")
    parser.add_argument("--control", type=str, choices=["A", "B", "C", "all"],
                        default="all", help="Which control to run (default: all)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials per N")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    n_values = [8, 10]

    print("\n" + "=" * 70)
    print("  PLC CONTROL EXPERIMENTS")
    print("  Publication-grade validation of emergent locality")
    print("=" * 70)
    print(f"  N values: {n_values}")
    print(f"  Trials per N: {args.trials}")
    print(f"  GPU: {'enabled' if use_gpu else 'disabled'}")
    print("=" * 70)

    t0 = time.time()

    result_A = None
    result_B = None
    result_C = None

    if args.control in ("A", "all"):
        result_A = control_A_nearest_neighbor(n_values, args.trials, use_gpu)

    if args.control in ("B", "all"):
        result_B = control_B_haar_random(n_values, args.trials, use_gpu)

    if args.control in ("C", "all"):
        result_C = control_C_planted_partition(n_values, args.trials, use_gpu)

    # Comparative analysis
    if args.control == "all" and result_A and result_B and result_C:
        exp5_path = Path(__file__).parent / "results" / "exp5_correlation_decay.json"
        all_to_all_file = str(exp5_path) if exp5_path.exists() else None
        compare_all_controls(result_A, result_B, result_C, all_to_all_file)

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ALL CONTROLS COMPLETE. Total time: {total_time:.1f}s")
    print(f"  Results saved to results/control_A.json, control_B.json, control_C.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
