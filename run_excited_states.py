#!/usr/bin/env python3
"""
Excited states test: verify PLC effect is not ground-state-specific.

Tests ground state + first 4 excited states for N=8 and N=10.
If the effect arises from partial observation (not area-law entanglement),
excited states should also show negative correlation decay.

Built by Opus Warrior, March 5 2026.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import random_all_to_all, mutual_information_matrix
from src.experiments import _mi_to_distance
from src.utils import NumpyEncoder

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def _build_pauli_op(N, positions):
    """Build tensor product with sigma_z at given positions, identity elsewhere."""
    parts = [I2] * N
    for p in positions:
        parts[p] = sz
    m = parts[0]
    for p in parts[1:]:
        m = np.kron(m, p)
    return m


def zz_correlator(psi, N, qi, qj):
    """Connected ZZ correlation between qubits qi and qj."""
    exp_zi = np.real(psi.conj() @ _build_pauli_op(N, [qi]) @ psi)
    exp_zj = np.real(psi.conj() @ _build_pauli_op(N, [qj]) @ psi)
    exp_zizj = np.real(psi.conj() @ _build_pauli_op(N, [qi, qj]) @ psi)
    return exp_zizj - exp_zi * exp_zj


def pearson_r_log_corr_vs_dist(psi, N, observer):
    """Pearson r between log|C_ij| and d_ij for observer qubits."""
    k = len(observer)
    MI = mutual_information_matrix(psi, N, observer)
    D = _mi_to_distance(MI)

    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    mi_vals = np.array([MI[a, b] for a, b in pairs])
    d_vals = np.array([D[a, b] for a, b in pairs])
    c_vals = np.abs(np.array([zz_correlator(psi, N, observer[a], observer[b]) for a, b in pairs]))

    valid = (c_vals > 1e-14) & (mi_vals > 1e-14)
    if np.sum(valid) < 3:
        return np.nan

    log_c = np.log(c_vals[valid])
    d_v = d_vals[valid]
    if np.std(d_v) < 1e-14 or np.std(log_c) < 1e-14:
        return np.nan
    return float(np.corrcoef(d_v, log_c)[0, 1])


def main():
    print("=" * 60)
    print("  EXCITED STATES TEST")
    print("  Ground + first 4 excited states")
    print("=" * 60)

    t0 = time.time()
    all_results = []

    configs = [
        (8, [3, 4], 20),
        (10, [3, 5], 20),
    ]

    for N, k_values, n_ham in configs:
        n_states = 5
        print(f"\nN={N}, {n_ham} Hamiltonians, {n_states} eigenstates...")

        for seed_idx in range(n_ham):
            seed = 1000 + seed_idx
            H, _ = random_all_to_all(N, seed=seed)
            evals, evecs = np.linalg.eigh(H)

            for si in range(n_states):
                psi = evecs[:, si]
                for k in k_values:
                    r = pearson_r_log_corr_vs_dist(psi, N, list(range(k)))
                    all_results.append({
                        "N": N, "k": k, "k_over_N": round(k / N, 3),
                        "state_index": si,
                        "energy": float(evals[si]),
                        "seed": seed,
                        "r_pearson": r if not np.isnan(r) else None,
                    })

            if (seed_idx + 1) % 10 == 0:
                print(f"  {seed_idx + 1}/{n_ham} ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s")

    # Aggregate
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)

    summary = {}
    for N, k_values, _ in configs:
        for k in k_values:
            for si in range(5):
                rs = [r["r_pearson"] for r in all_results
                      if r["N"] == N and r["k"] == k and r["state_index"] == si
                      and r["r_pearson"] is not None]
                if rs:
                    label = "ground" if si == 0 else f"excited-{si}"
                    n_neg = sum(1 for x in rs if x < 0)
                    key = f"N={N},k={k},state={si}"
                    print(f"  {key} ({label}): r={np.mean(rs):.3f} [{np.std(rs):.3f}], "
                          f"{n_neg}/{len(rs)} negative")
                    summary[key] = {
                        "N": N, "k": k, "state_index": si, "label": label,
                        "r_mean": float(np.mean(rs)), "r_std": float(np.std(rs)),
                        "n_negative": n_neg, "n_total": len(rs),
                    }

    output = {
        "experiment": "excited_states",
        "description": "PLC correlation decay for ground + first 4 excited states. "
                       "Tests whether the effect is specific to ground states or "
                       "general to any structured eigenstate.",
        "all_results": all_results,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }

    outpath = RESULTS_DIR / "excited_states.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
