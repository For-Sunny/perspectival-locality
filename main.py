#!/usr/bin/env python3
"""
PLC Simulation: Emergent Locality from Partial Observation in Finite Quantum Systems.

Four experiments demonstrating the core result of the Perspectival Locality Conjecture:
spatial locality emerges from the act of being a partial observer.

Usage:
    python main.py                  # Run all experiments, N=8
    python main.py --n 10           # Larger system
    python main.py --exp 1          # Run specific experiment (1-4)
    python main.py --no-gpu         # CPU only

Built by Opus Warrior, March 5 2026.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments import (
    experiment_1_symmetry_breaking,
    experiment_2_emergent_metric,
    experiment_3_sheaf_convergence,
    experiment_4_scaling,
    experiment_5_correlation_decay,
    run_all,
)


def main():
    parser = argparse.ArgumentParser(description="PLC Simulation")
    parser.add_argument("--n", type=int, default=8, help="Number of qubits (default: 8)")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5], help="Run specific experiment")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    args = parser.parse_args()

    use_gpu = not args.no_gpu

    if args.exp == 1:
        experiment_1_symmetry_breaking(args.n, use_gpu)
    elif args.exp == 2:
        experiment_2_emergent_metric(args.n, use_gpu=use_gpu)
    elif args.exp == 3:
        experiment_3_sheaf_convergence(args.n, use_gpu)
    elif args.exp == 4:
        experiment_4_scaling(use_gpu=use_gpu)
    elif args.exp == 5:
        experiment_5_correlation_decay(args.n, use_gpu=use_gpu)
    else:
        run_all(args.n, use_gpu)


if __name__ == "__main__":
    main()
