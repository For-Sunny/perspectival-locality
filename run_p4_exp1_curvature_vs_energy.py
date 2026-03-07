#!/usr/bin/env python3
"""
Paper 4, Experiment 1: Curvature vs Energy Eigenstate
=====================================================

Question: Does ORC scalar curvature depend on which energy eigenstate
the system is in?

For 10 random all-to-all Heisenberg Hamiltonians (N=12, dim=4096):
  - Full diagonalization (np.linalg.eigh)
  - Lowest 20 eigenstates
  - For each eigenstate: E, MI matrix, ORC (k=12, k=6), FRC, half-chain entropy
  - Save everything, print summary with correlations.

Built by Opus Warrior, March 6 2026.
"""

import sys
import os
import json
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantum import (
    random_all_to_all,
    mutual_information_matrix,
    partial_trace,
    von_neumann_entropy,
)
from src.curvature import ollivier_ricci, forman_ricci

# ── Parameters ──────────────────────────────────────────────
N = 12
DIM = 2 ** N  # 4096
N_SEEDS = 10
SEEDS = list(range(6000, 6010))
N_STATES = 20
ORC_THRESHOLD = 0.5
ORC_ALPHA = 0.5
K_PARTIAL = 6  # partial observer subset size
HALF = N // 2  # for half-chain entropy


def half_chain_entropy(psi, n_qubits):
    """Von Neumann entropy of the reduced state on qubits 0..N/2-1."""
    keep = list(range(n_qubits // 2))
    rho_half = partial_trace(psi, keep, n_qubits)
    return von_neumann_entropy(rho_half)


def run_experiment():
    all_results = {}
    t_total = time.time()

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({seed_idx+1}/{N_SEEDS})")
        print(f"{'='*60}")

        # Build Hamiltonian
        t0 = time.time()
        H, couplings = random_all_to_all(N, seed=seed)
        print(f"  Hamiltonian built: {time.time()-t0:.1f}s")

        # Full diagonalization
        t0 = time.time()
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        print(f"  Full diag: {time.time()-t0:.1f}s, spectrum [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")

        # Pick random k=6 subset (fixed per seed for consistency)
        rng = np.random.default_rng(seed + 99999)
        partial_sites = sorted(rng.choice(N, size=K_PARTIAL, replace=False).tolist())
        print(f"  Partial observer sites (k={K_PARTIAL}): {partial_sites}")

        seed_data = {
            "seed": seed,
            "N": N,
            "partial_sites": partial_sites,
            "states": [],
        }

        for state_idx in range(N_STATES):
            t_state = time.time()
            E = float(eigenvalues[state_idx])
            psi = eigenvectors[:, state_idx]

            # Half-chain entanglement entropy
            S_half = half_chain_entropy(psi, N)

            # Full MI matrix (k=N=12, all qubits)
            MI_full = mutual_information_matrix(psi, N)

            # ORC at k=12 (full observer)
            orc_full = ollivier_ricci(MI_full, threshold=ORC_THRESHOLD, alpha=ORC_ALPHA)
            R_orc_full = orc_full["scalar_curvature"]
            n_edges_full = orc_full["n_edges"]

            # FRC at k=12
            frc_full = forman_ricci(MI_full, threshold=ORC_THRESHOLD)
            R_frc_full = frc_full["scalar_curvature"]
            R_frc_aug = frc_full["scalar_curvature_aug"]

            # ORC at k=6 (partial observer)
            MI_partial = mutual_information_matrix(psi, N, sites=partial_sites)
            orc_partial = ollivier_ricci(MI_partial, threshold=ORC_THRESHOLD, alpha=ORC_ALPHA)
            R_orc_partial = orc_partial["scalar_curvature"]
            n_edges_partial = orc_partial["n_edges"]

            elapsed = time.time() - t_state
            print(f"  State {state_idx:2d}: E={E:+9.4f}  S={S_half:.4f}  "
                  f"R_ORC(k12)={R_orc_full:+.4f}({n_edges_full}e)  "
                  f"R_ORC(k6)={R_orc_partial:+.4f}({n_edges_partial}e)  "
                  f"R_FRC={R_frc_full:+.1f}  [{elapsed:.1f}s]")

            seed_data["states"].append({
                "state_index": state_idx,
                "energy": E,
                "entropy_half_chain": S_half,
                "R_orc_k12": R_orc_full,
                "R_orc_k6": R_orc_partial,
                "R_frc": R_frc_full,
                "R_frc_aug": R_frc_aug,
                "n_edges_k12": n_edges_full,
                "n_edges_k6": n_edges_partial,
            })

        all_results[str(seed)] = seed_data

    total_time = time.time() - t_total
    print(f"\n\nTotal runtime: {total_time:.1f}s")

    # ── Save results ─────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "p4_exp1_curvature_vs_energy.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    output = {
        "experiment": "P4_Exp1_Curvature_vs_Energy",
        "parameters": {
            "N": N, "dim": DIM, "n_seeds": N_SEEDS, "seeds": SEEDS,
            "n_states": N_STATES, "orc_threshold": ORC_THRESHOLD,
            "orc_alpha": ORC_ALPHA, "k_partial": K_PARTIAL,
        },
        "total_runtime_s": total_time,
        "data": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Analysis ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS: Curvature vs Energy Eigenstate")
    print("=" * 70)

    # Collect per-state averages across seeds
    avg_E = np.zeros(N_STATES)
    avg_S = np.zeros(N_STATES)
    avg_R_orc_k12 = np.zeros(N_STATES)
    avg_R_orc_k6 = np.zeros(N_STATES)
    avg_R_frc = np.zeros(N_STATES)

    for seed_key, seed_data in all_results.items():
        for s in seed_data["states"]:
            idx = s["state_index"]
            avg_E[idx] += s["energy"]
            avg_S[idx] += s["entropy_half_chain"]
            avg_R_orc_k12[idx] += s["R_orc_k12"]
            avg_R_orc_k6[idx] += s["R_orc_k6"]
            avg_R_frc[idx] += s["R_frc"]

    avg_E /= N_SEEDS
    avg_S /= N_SEEDS
    avg_R_orc_k12 /= N_SEEDS
    avg_R_orc_k6 /= N_SEEDS
    avg_R_frc /= N_SEEDS

    # Table
    print(f"\n{'Idx':>3}  {'<E>':>10}  {'<S>':>7}  {'<R_ORC k12>':>12}  {'<R_ORC k6>':>11}  {'<R_FRC>':>8}")
    print("-" * 60)
    for i in range(N_STATES):
        print(f"{i:3d}  {avg_E[i]:+10.4f}  {avg_S[i]:7.4f}  {avg_R_orc_k12[i]:+12.6f}  "
              f"{avg_R_orc_k6[i]:+11.6f}  {avg_R_frc[i]:+8.2f}")

    # Correlations (using ALL individual data points, not just averages)
    all_E = []
    all_S = []
    all_R12 = []
    all_R6 = []
    all_Rfrc = []

    for seed_key, seed_data in all_results.items():
        for s in seed_data["states"]:
            all_E.append(s["energy"])
            all_S.append(s["entropy_half_chain"])
            all_R12.append(s["R_orc_k12"])
            all_R6.append(s["R_orc_k6"])
            all_Rfrc.append(s["R_frc"])

    all_E = np.array(all_E)
    all_S = np.array(all_S)
    all_R12 = np.array(all_R12)
    all_R6 = np.array(all_R6)
    all_Rfrc = np.array(all_Rfrc)

    print(f"\n--- Correlations (n={len(all_E)} data points) ---")

    r_E_R12, p_E_R12 = pearsonr(all_E, all_R12)
    r_E_R6, p_E_R6 = pearsonr(all_E, all_R6)
    r_S_R12, p_S_R12 = pearsonr(all_S, all_R12)
    r_S_R6, p_S_R6 = pearsonr(all_S, all_R6)
    r_E_S, p_E_S = pearsonr(all_E, all_S)
    r_E_Rfrc, p_E_Rfrc = pearsonr(all_E, all_Rfrc)
    r_S_Rfrc, p_S_Rfrc = pearsonr(all_S, all_Rfrc)

    # Spearman (rank correlation, better for monotonicity)
    rs_E_R12, ps_E_R12 = spearmanr(all_E, all_R12)
    rs_E_R6, ps_E_R6 = spearmanr(all_E, all_R6)

    print(f"  Pearson  r(E, R_ORC_k12)  = {r_E_R12:+.4f}  (p={p_E_R12:.2e})")
    print(f"  Pearson  r(E, R_ORC_k6)   = {r_E_R6:+.4f}  (p={p_E_R6:.2e})")
    print(f"  Pearson  r(S, R_ORC_k12)  = {r_S_R12:+.4f}  (p={p_S_R12:.2e})")
    print(f"  Pearson  r(S, R_ORC_k6)   = {r_S_R6:+.4f}  (p={p_S_R6:.2e})")
    print(f"  Pearson  r(E, S_half)     = {r_E_S:+.4f}  (p={p_E_S:.2e})")
    print(f"  Pearson  r(E, R_FRC)      = {r_E_Rfrc:+.4f}  (p={p_E_Rfrc:.2e})")
    print(f"  Pearson  r(S, R_FRC)      = {r_S_Rfrc:+.4f}  (p={p_S_Rfrc:.2e})")
    print(f"  Spearman r(E, R_ORC_k12)  = {rs_E_R12:+.4f}  (p={ps_E_R12:.2e})")
    print(f"  Spearman r(E, R_ORC_k6)   = {rs_E_R6:+.4f}  (p={ps_E_R6:.2e})")

    # Monotonicity check on seed-averaged data
    diffs_k12 = np.diff(avg_R_orc_k12)
    diffs_k6 = np.diff(avg_R_orc_k6)
    mono_k12 = np.all(diffs_k12 >= 0) or np.all(diffs_k12 <= 0)
    mono_k6 = np.all(diffs_k6 >= 0) or np.all(diffs_k6 <= 0)

    print(f"\n--- Monotonicity (seed-averaged) ---")
    print(f"  R_ORC(k12) vs eigenstate index: {'MONOTONIC' if mono_k12 else 'NOT monotonic'}")
    print(f"  R_ORC(k6)  vs eigenstate index: {'MONOTONIC' if mono_k6 else 'NOT monotonic'}")

    # Effect strength comparison
    print(f"\n--- Effect strength: k=12 vs k=6 ---")
    print(f"  |r(E, R_ORC_k12)| = {abs(r_E_R12):.4f}")
    print(f"  |r(E, R_ORC_k6)|  = {abs(r_E_R6):.4f}")
    stronger = "k=12 (full)" if abs(r_E_R12) > abs(r_E_R6) else "k=6 (partial)"
    print(f"  Stronger correlation at: {stronger}")

    # Range of curvature variation
    print(f"\n--- Curvature range across eigenstates (seed-averaged) ---")
    print(f"  R_ORC(k12): [{avg_R_orc_k12.min():+.6f}, {avg_R_orc_k12.max():+.6f}] "
          f"range={avg_R_orc_k12.max()-avg_R_orc_k12.min():.6f}")
    print(f"  R_ORC(k6):  [{avg_R_orc_k6.min():+.6f}, {avg_R_orc_k6.max():+.6f}] "
          f"range={avg_R_orc_k6.max()-avg_R_orc_k6.min():.6f}")
    print(f"  R_FRC:      [{avg_R_frc.min():+.2f}, {avg_R_frc.max():+.2f}] "
          f"range={avg_R_frc.max()-avg_R_frc.min():.2f}")

    print("\nDone.")


if __name__ == "__main__":
    run_experiment()
