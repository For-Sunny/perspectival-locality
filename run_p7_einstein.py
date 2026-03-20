#!/usr/bin/env python3
"""
Paper 7: Einstein's Equations from Perspectival Crystallization
Computational Experiments

Bridge Theorem: delta_TC(B)|_V = -delta_S_ent(B)|_V  (under UV stability A_UV)
Therefore: crystallization equilibrium <=> entanglement equilibrium => Einstein equations

Four experiments:
  1. Bridge Theorem Verification (delta_TC = -delta_S_ent)
  2. UV Stability (s_1 invariance under IR perturbations)
  3. Area Law Scaling (S(A) vs |boundary A| in MI geometry)
  4. Entanglement-Curvature Correlation (Ollivier-Ricci vs TC density)

Built by Opus Warrior, March 18 2026.
"""

import sys
import os
import json
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import (
    random_all_to_all, random_all_to_all_sparse,
    ground_state, ground_state_sparse,
    partial_trace, von_neumann_entropy,
    mutual_information_matrix,
    single_site_op, SIGMA_Z, SIGMA_X,
    heisenberg_all_to_all, heisenberg_all_to_all_sparse,
)
from src.experiments import _mi_to_distance, _effective_dimension
from src.utils import NumpyEncoder
import scipy.sparse as sp

# Auto-select dense vs sparse based on N
SPARSE_THRESHOLD = 14  # N >= 14 uses sparse (dense needs 64GB+ at N=16)

def auto_hamiltonian(N, couplings):
    """Build Hamiltonian: sparse for large N, dense otherwise."""
    if N >= SPARSE_THRESHOLD:
        return heisenberg_all_to_all_sparse(N, couplings)
    else:
        return heisenberg_all_to_all(N, couplings)

def auto_ground_state(H):
    """Find ground state: sparse Lanczos for large systems, dense otherwise."""
    if sp.issparse(H):
        return ground_state_sparse(H)
    else:
        return ground_state(H)

def sparse_field_perturbation(N, h_fields):
    """Build field perturbation as sparse matrix: sum_i h_i Z_i."""
    dim = 2 ** N
    bit_masks = [1 << (N - 1 - q) for q in range(N)]
    indices = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim, dtype=np.float64)
    for i in range(N):
        mask = bit_masks[i]
        signs = 1.0 - 2.0 * ((indices & mask) != 0).astype(np.float64)
        diag += h_fields[i] * signs
    return sp.diags(diag, format='csr')

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════

def total_correlation(psi, N, k):
    """
    TC(k) = sum_{i=0}^{k-1} S(rho_i) - S(rho_{0..k-1})
    Uses complement trick for k > N/2.
    Returns (TC, S_joint, sum_S_i).
    """
    observed = list(range(k))

    sum_S_i = 0.0
    for i in observed:
        rho_i = partial_trace(psi, [i], N)
        sum_S_i += von_neumann_entropy(rho_i)

    complement_size = N - k
    if complement_size < k and complement_size > 0:
        complement = list(range(k, N))
        rho_complement = partial_trace(psi, complement, N)
        S_joint = von_neumann_entropy(rho_complement)
    elif k == N:
        S_joint = 0.0
    elif k == 0:
        S_joint = 0.0
    else:
        rho_observed = partial_trace(psi, observed, N)
        S_joint = von_neumann_entropy(rho_observed)

    TC = sum_S_i - S_joint
    return TC, S_joint, sum_S_i


def total_correlation_region(psi, N, region):
    """
    TC for an arbitrary region (list of qubit indices).
    TC(A) = sum_{i in A} S(rho_i) - S(rho_A)
    """
    sum_S_i = 0.0
    for i in region:
        rho_i = partial_trace(psi, [i], N)
        sum_S_i += von_neumann_entropy(rho_i)

    if len(region) == N:
        S_joint = 0.0  # pure state
    else:
        # Use complement trick if cheaper
        complement = sorted(set(range(N)) - set(region))
        if len(complement) < len(region) and len(complement) > 0:
            rho_c = partial_trace(psi, complement, N)
            S_joint = von_neumann_entropy(rho_c)
        else:
            rho_A = partial_trace(psi, sorted(region), N)
            S_joint = von_neumann_entropy(rho_A)

    return sum_S_i - S_joint


def entanglement_entropy_region(psi, N, region):
    """S(rho_A) for a region A. Uses complement trick."""
    region = sorted(region)
    if len(region) == N:
        return 0.0
    if len(region) == 0:
        return 0.0
    complement = sorted(set(range(N)) - set(region))
    if len(complement) < len(region):
        rho = partial_trace(psi, complement, N)
    else:
        rho = partial_trace(psi, region, N)
    return von_neumann_entropy(rho)


def build_perturbed_hamiltonian(N, couplings, epsilon, seed, mode="random_field"):
    """
    Add a small perturbation to an existing Hamiltonian.

    Modes:
    - "random_field": delta_H = sum_i h_i * Z_i, h_i ~ N(0,1)
      Breaks symmetry, changes eigenstates. Good for Bridge Theorem.
    - "uniform_field": delta_H = sum_i Z_i
      Commutes with total Z -- shifts energies but not eigenstates in
      non-degenerate sectors. BAD for testing state changes.
    - "random_coupling": delta_H = sum_{i<j} delta_J_{ij} (XX+YY+ZZ)
      Changes coupling strengths. Good IR perturbation.
    - "staggered_field": delta_H = sum_i (-1)^i * Z_i
      Anti-ferromagnetic bias. Good IR perturbation that changes state.

    Returns the perturbed H (dense for N < SPARSE_THRESHOLD, sparse otherwise).
    """
    rng = np.random.default_rng(seed)
    H_base = auto_hamiltonian(N, couplings)
    dim = 2 ** N
    use_sparse = N >= SPARSE_THRESHOLD

    if use_sparse:
        delta_H = sp.csr_matrix((dim, dim), dtype=np.float64)
    else:
        delta_H = np.zeros((dim, dim), dtype=np.complex128)

    if mode == "random_field":
        h_fields = rng.standard_normal(N)
        if use_sparse:
            delta_H = sparse_field_perturbation(N, h_fields)
        else:
            for i in range(N):
                delta_H += h_fields[i] * single_site_op(SIGMA_Z, i, N)

    elif mode == "uniform_field":
        h_fields = np.ones(N, dtype=np.float64)
        if use_sparse:
            delta_H = sparse_field_perturbation(N, h_fields)
        else:
            for i in range(N):
                delta_H += single_site_op(SIGMA_Z, i, N)

    elif mode == "random_coupling":
        n_pairs = N * (N - 1) // 2
        delta_J = rng.standard_normal(n_pairs)
        if use_sparse:
            delta_H = heisenberg_all_to_all_sparse(N, delta_J)
        else:
            delta_H = heisenberg_all_to_all(N, delta_J)

    elif mode == "staggered_field":
        stag_fields = np.array([(-1)**i for i in range(N)], dtype=np.float64)
        if use_sparse:
            delta_H = sparse_field_perturbation(N, stag_fields)
        else:
            for i in range(N):
                sign = (-1) ** i
                delta_H += sign * single_site_op(SIGMA_Z, i, N)

    elif mode == "random_XZ":
        # Mix of X and Z fields -- maximally symmetry-breaking
        hx = rng.standard_normal(N)
        hz = rng.standard_normal(N)
        for i in range(N):
            delta_H += hx[i] * single_site_op(SIGMA_X, i, N)
            delta_H += hz[i] * single_site_op(SIGMA_Z, i, N)

    return H_base + epsilon * delta_H


# ══════════════════════════════════════════════════════════════
# Experiment 1: Bridge Theorem Verification
# delta_TC = -delta_S_ent for each observation fraction k
# ══════════════════════════════════════════════════════════════

def experiment_1_bridge_theorem(N_values=None, n_seeds=5, epsilons=None):
    """
    Bridge Theorem: delta_TC(B)|_V = -delta_S_ent(B)|_V  (under UV stability A_UV)

    The identity TC(k) = sum(S_i) - S_ent(k) always holds exactly.
    Therefore delta_TC = delta(sum_S_i) - delta(S_ent).
    The Bridge Theorem says: when UV stability holds (sum_S_i ~ constant),
    delta_TC approx= -delta_S_ent.

    We verify three things:
    (a) The exact identity: delta_TC + delta_S_ent = delta(sum_S_i) always
    (b) UV stability: delta(sum_S_i) is small relative to delta_TC, delta_S_ent
    (c) The bridge: correlation between delta_TC and -delta_S_ent approaches 1
        as the UV residual delta(sum_S_i) becomes negligible
    """
    if N_values is None:
        N_values = [8, 12]
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1]

    print("=" * 70)
    print("  EXPERIMENT 1: Bridge Theorem Verification")
    print("  delta_TC(B)|_V = -delta_S_ent(B)|_V  (under UV stability)")
    print("  Identity: delta_TC + delta_S_ent = delta(sum_S_i) [exact]")
    print("  Bridge holds when delta(sum_S_i) << delta_TC, delta_S_ent")
    print("=" * 70)

    t0 = time.time()
    all_results = []

    for N in N_values:
        print(f"\n  N = {N} (dim = {2**N})")

        for seed_idx in range(n_seeds):
            seed = 50000 + N * 100 + seed_idx
            rng = np.random.default_rng(seed)
            n_pairs = N * (N - 1) // 2
            couplings = rng.standard_normal(n_pairs)

            # Build base Hamiltonian (auto sparse/dense)
            H_base = auto_hamiltonian(N, couplings)
            E0, psi_0 = auto_ground_state(H_base)

            # Compute base quantities for each k
            base_TC = []
            base_S_ent = []
            base_sum_Si = []
            for k in range(1, N):
                tc, s_j, sum_si = total_correlation(psi_0, N, k)
                base_TC.append(tc)
                base_S_ent.append(s_j)  # S_ent = S(rho_{0..k-1}) = S_joint
                base_sum_Si.append(sum_si)

            # Build perturbation operator ONCE, reuse across epsilons
            perturb_seed = seed + 7000
            perturb_rng = np.random.default_rng(perturb_seed)
            h_fields = perturb_rng.standard_normal(N)
            if N >= SPARSE_THRESHOLD:
                delta_H = sparse_field_perturbation(N, h_fields)
            else:
                dim = 2 ** N
                delta_H = np.zeros((dim, dim), dtype=np.complex128)
                for i in range(N):
                    delta_H += h_fields[i] * single_site_op(SIGMA_Z, i, N)

            for eps in epsilons:
                H_pert = H_base + eps * delta_H
                E0_p, psi_p = auto_ground_state(H_pert)

                delta_TC_list = []
                delta_S_ent_list = []
                delta_sum_Si_list = []
                k_values = []

                for k_idx, k in enumerate(range(1, N)):
                    tc_p, s_j_p, sum_si_p = total_correlation(psi_p, N, k)

                    d_tc = tc_p - base_TC[k_idx]
                    d_s_ent = s_j_p - base_S_ent[k_idx]
                    d_sum_si = sum_si_p - base_sum_Si[k_idx]

                    delta_TC_list.append(d_tc)
                    delta_S_ent_list.append(d_s_ent)
                    delta_sum_Si_list.append(d_sum_si)
                    k_values.append(k)

                delta_TC_arr = np.array(delta_TC_list)
                delta_S_ent_arr = np.array(delta_S_ent_list)
                delta_sum_Si_arr = np.array(delta_sum_Si_list)

                # (a) Exact identity check: delta_TC + delta_S_ent = delta(sum_S_i)
                identity_residual = delta_TC_arr + delta_S_ent_arr - delta_sum_Si_arr
                identity_max_err = float(np.max(np.abs(identity_residual)))

                # (b) UV stability: how small is delta(sum_S_i) relative to signals?
                signal_scale = max(
                    np.max(np.abs(delta_TC_arr)),
                    np.max(np.abs(delta_S_ent_arr)),
                    1e-14
                )
                uv_residual_ratio = float(np.mean(np.abs(delta_sum_Si_arr)) / signal_scale)

                # (c) Bridge correlation: r(delta_TC, -delta_S_ent)
                neg_delta_S = -delta_S_ent_arr
                if np.std(delta_TC_arr) > 1e-14 and np.std(neg_delta_S) > 1e-14:
                    r_bridge = float(np.corrcoef(delta_TC_arr, neg_delta_S)[0, 1])
                else:
                    r_bridge = 0.0

                # Also: direct MAE between delta_TC and -delta_S_ent
                mae_bridge = float(np.mean(np.abs(delta_TC_arr - neg_delta_S)))

                # Corrected bridge: subtract UV residual
                # delta_TC_corrected = delta_TC - delta(sum_S_i) should = -delta_S_ent exactly
                delta_TC_corrected = delta_TC_arr - delta_sum_Si_arr
                mae_corrected = float(np.mean(np.abs(delta_TC_corrected - neg_delta_S)))

                entry = {
                    "N": N,
                    "seed": seed,
                    "epsilon": eps,
                    "r_bridge": r_bridge,
                    "mae_bridge": mae_bridge,
                    "mae_corrected": mae_corrected,
                    "identity_max_error": identity_max_err,
                    "uv_residual_ratio": uv_residual_ratio,
                    "mean_abs_delta_sum_Si": float(np.mean(np.abs(delta_sum_Si_arr))),
                    "mean_abs_delta_TC": float(np.mean(np.abs(delta_TC_arr))),
                    "mean_abs_delta_S_ent": float(np.mean(np.abs(delta_S_ent_arr))),
                    "k_values": k_values,
                    "delta_TC": delta_TC_list,
                    "delta_S_ent": delta_S_ent_list,
                    "delta_sum_Si": delta_sum_Si_list,
                    "E0_base": float(E0),
                    "E0_perturbed": float(E0_p),
                }
                all_results.append(entry)

                print(f"    seed={seed_idx}, eps={eps}: r_bridge={r_bridge:.4f}, "
                      f"UV_resid={uv_residual_ratio:.4f}, "
                      f"identity_err={identity_max_err:.2e}")

            del psi_0
            gc.collect()

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"  BRIDGE THEOREM SUMMARY")
    print(f"{'='*70}")

    summary = {}
    for N in N_values:
        for eps in epsilons:
            matching = [r for r in all_results if r['N'] == N and r['epsilon'] == eps]
            r_vals = [r['r_bridge'] for r in matching]
            uv_vals = [r['uv_residual_ratio'] for r in matching]
            id_vals = [r['identity_max_error'] for r in matching]
            mean_r = float(np.mean(r_vals))
            std_r = float(np.std(r_vals))
            mean_uv = float(np.mean(uv_vals))
            max_id = float(np.max(id_vals))
            key = f"N={N}_eps={eps}"
            summary[key] = {
                "N": N, "epsilon": eps,
                "mean_r_bridge": mean_r, "std_r_bridge": std_r,
                "mean_uv_residual": mean_uv,
                "max_identity_error": max_id,
                "n_seeds": len(matching),
            }
            # Bridge strength depends on UV stability
            if mean_uv < 0.1:
                bridge_quality = "STRONG (UV stable)"
            elif mean_uv < 0.3:
                bridge_quality = "MODERATE (UV partially stable)"
            else:
                bridge_quality = f"WEAK (UV residual dominates: {mean_uv:.2f})"
            print(f"  N={N}, eps={eps}: r={mean_r:.4f}+/-{std_r:.4f}, "
                  f"UV_resid={mean_uv:.4f}, identity_err<{max_id:.2e} [{bridge_quality}]")

    # Key insight: the bridge holds EXACTLY when UV is stable
    print(f"\n  KEY: r_bridge approaches 1.0 as UV_residual approaches 0.")
    print(f"  The Bridge Theorem IS the UV stability condition.")
    print(f"  delta_TC = -delta_S_ent + delta(sum_S_i)  [exact identity]")
    print(f"  Bridge holds <=> delta(sum_S_i) ~ 0 <=> UV stability")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    output = {
        "experiment": "bridge_theorem",
        "description": (
            "Verify Bridge Theorem: delta_TC = -delta_S_ent under UV stability. "
            "Exact identity: delta_TC + delta_S_ent = delta(sum_S_i). "
            "Bridge holds when delta(sum_S_i) is negligible (UV stable). "
            "Reports r_bridge (correlation of delta_TC with -delta_S_ent) and "
            "UV residual ratio (how much sum_S_i changes relative to signal)."
        ),
        "all_results": all_results,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }

    outpath = RESULTS_DIR / "paper7_bridge_theorem.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved to {outpath}")
    return output


# ══════════════════════════════════════════════════════════════
# Experiment 2: UV Stability
# s_1(i) = S(rho_i) invariance under IR perturbations
# ══════════════════════════════════════════════════════════════

def experiment_2_uv_stability(N_values=None, n_seeds=5, epsilons=None):
    """
    For each N, compute single-site entropies s_1(i).
    Apply IR perturbations (uniform field), measure stability.
    Show stability improves with N.
    """
    if N_values is None:
        N_values = [8, 12]
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]

    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: UV Stability")
    print("  s_1(i) = S(rho_i) invariance under IR perturbations")
    print("=" * 70)

    t0 = time.time()
    all_results = []

    for N in N_values:
        print(f"\n  N = {N} (dim = {2**N})")

        for seed_idx in range(n_seeds):
            seed = 60000 + N * 100 + seed_idx
            rng = np.random.default_rng(seed)
            n_pairs = N * (N - 1) // 2
            couplings = rng.standard_normal(n_pairs)

            # Base ground state
            H_base = auto_hamiltonian(N, couplings)
            E0, psi_0 = auto_ground_state(H_base)

            # Base single-site entropies
            s1_base = []
            for i in range(N):
                rho_i = partial_trace(psi_0, [i], N)
                s1_base.append(von_neumann_entropy(rho_i))
            s1_base = np.array(s1_base)
            mean_s1_base = float(np.mean(s1_base))

            # Build staggered field perturbation ONCE (reuse across epsilons)
            stag_fields = np.array([(-1)**i for i in range(N)], dtype=np.float64)
            if N >= SPARSE_THRESHOLD:
                delta_H_stag = sparse_field_perturbation(N, stag_fields)
            else:
                dim = 2 ** N
                delta_H_stag = np.zeros((dim, dim), dtype=np.complex128)
                for i in range(N):
                    sign = (-1) ** i
                    delta_H_stag += sign * single_site_op(SIGMA_Z, i, N)

            for eps in epsilons:
                # IR perturbation: staggered field (breaks symmetry, changes state)
                H_pert = H_base + eps * delta_H_stag
                E0_p, psi_p = auto_ground_state(H_pert)

                # Perturbed single-site entropies
                s1_pert = []
                for i in range(N):
                    rho_i = partial_trace(psi_p, [i], N)
                    s1_pert.append(von_neumann_entropy(rho_i))
                s1_pert = np.array(s1_pert)

                # Stability metrics
                delta_s1 = s1_pert - s1_base
                mean_delta = float(np.mean(np.abs(delta_s1)))
                max_delta = float(np.max(np.abs(delta_s1)))
                # Coefficient of variation of the CHANGE
                cv_delta = float(np.std(delta_s1) / max(np.mean(np.abs(delta_s1)), 1e-30))
                # Fractional change relative to base
                frac_change = float(np.mean(np.abs(delta_s1)) / max(mean_s1_base, 1e-30))
                # RMS fractional change per site
                rms_frac = float(np.sqrt(np.mean((delta_s1 / np.maximum(s1_base, 1e-14))**2)))

                entry = {
                    "N": N,
                    "seed": seed,
                    "epsilon": eps,
                    "mean_s1_base": mean_s1_base,
                    "mean_s1_pert": float(np.mean(s1_pert)),
                    "mean_abs_delta_s1": mean_delta,
                    "max_abs_delta_s1": max_delta,
                    "fractional_change": frac_change,
                    "rms_fractional_change": rms_frac,
                    "cv_delta": cv_delta,
                    "s1_base": s1_base.tolist(),
                    "s1_pert": s1_pert.tolist(),
                    "delta_s1": delta_s1.tolist(),
                }
                all_results.append(entry)

                print(f"    seed={seed_idx}, eps={eps}: frac={frac_change:.8f}, "
                      f"|delta_s1|={mean_delta:.2e}, max={max_delta:.2e}")

            del psi_0
            gc.collect()

    # Summary: stability vs N
    print(f"\n{'='*70}")
    print(f"  UV STABILITY SUMMARY")
    print(f"  Key claim: s_1 stability should IMPROVE with N")
    print(f"{'='*70}")

    summary = {}
    for N in N_values:
        for eps in epsilons:
            matching = [r for r in all_results if r['N'] == N and r['epsilon'] == eps]
            frac_changes = [r['fractional_change'] for r in matching]
            rms_fracs = [r['rms_fractional_change'] for r in matching]
            mean_abs_deltas = [r['mean_abs_delta_s1'] for r in matching]
            mean_frac = float(np.mean(frac_changes))
            std_frac = float(np.std(frac_changes))
            mean_rms = float(np.mean(rms_fracs))
            mean_abs = float(np.mean(mean_abs_deltas))
            key = f"N={N}_eps={eps}"
            summary[key] = {
                "N": N, "epsilon": eps,
                "mean_fractional_change": mean_frac,
                "std_fractional_change": std_frac,
                "mean_rms_fractional": mean_rms,
                "mean_abs_delta_s1": mean_abs,
                "n_seeds": len(matching),
            }
            print(f"  N={N}, eps={eps}: frac={mean_frac:.8f}, rms_frac={mean_rms:.8f}, "
                  f"|delta_s1|={mean_abs:.2e}")

    # Cross-N comparison at each epsilon
    print(f"\n  SCALING CHECK (fractional change should decrease with N):")
    for eps in epsilons:
        fracs_by_N = {}
        for N in N_values:
            matching = [r for r in all_results if r['N'] == N and r['epsilon'] == eps]
            fracs_by_N[N] = float(np.mean([r['fractional_change'] for r in matching]))
        N_list = sorted(fracs_by_N.keys())
        vals = [fracs_by_N[n] for n in N_list]
        monotonic = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        print(f"    eps={eps}: {dict(zip(N_list, [f'{v:.8f}' for v in vals]))} "
              f"{'DECREASING' if monotonic else 'non-monotonic'}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    output = {
        "experiment": "uv_stability",
        "description": "s_1(i) invariance under IR perturbations (A_UV claim)",
        "all_results": all_results,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }

    outpath = RESULTS_DIR / "paper7_uv_stability.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved to {outpath}")
    return output


# ══════════════════════════════════════════════════════════════
# Experiment 3: Area Law Scaling
# S(A) vs |boundary A| in MI-derived geometry
# ══════════════════════════════════════════════════════════════

def experiment_3_area_law(N_values=None, n_seeds=5):
    """
    For each N, compute S(A) for regions of size k=1..N-1.
    Build MI geometry, compute boundary size |dA|.
    Verify S(A) ~ |dA| (area law, not volume law).
    """
    if N_values is None:
        N_values = [8, 12]

    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Area Law Scaling")
    print("  S(A) vs |boundary(A)| in MI-derived geometry")
    print("=" * 70)

    t0 = time.time()
    all_results = []

    for N in N_values:
        print(f"\n  N = {N} (dim = {2**N})")

        for seed_idx in range(n_seeds):
            seed = 70000 + N * 100 + seed_idx

            # Build ground state (auto sparse/dense)
            if N >= SPARSE_THRESHOLD:
                H, couplings = random_all_to_all_sparse(N, seed=seed)
            else:
                H, couplings = random_all_to_all(N, seed=seed)
            E0, psi = auto_ground_state(H)

            # Full MI matrix
            MI = mutual_information_matrix(psi, N)

            # Convert to distance matrix using -log(I/I_max)
            max_mi = np.max(MI[MI > 0]) if np.any(MI > 0) else 1.0
            D = np.zeros_like(MI)
            for i in range(N):
                for j in range(i + 1, N):
                    if MI[i, j] > 1e-14:
                        D[i, j] = -np.log(MI[i, j] / max_mi + 1e-14)
                    else:
                        D[i, j] = 10.0  # far
                    D[j, i] = D[i, j]

            # Determine distance threshold for "boundary"
            # Use median distance as threshold
            upper_tri = D[np.triu_indices(N, k=1)]
            threshold = float(np.median(upper_tri[upper_tri > 0]))

            # For each region size k, use multiple random regions
            entropy_data = []
            boundary_data = []
            volume_data = []

            for k in range(1, N):
                # Sample random regions of size k
                all_subsets = list(combinations(range(N), k))
                rng = np.random.default_rng(seed + k * 10)
                n_sample = min(len(all_subsets), 20)
                sample_indices = rng.choice(len(all_subsets), n_sample, replace=False)

                for si in sample_indices:
                    region = list(all_subsets[si])
                    complement = sorted(set(range(N)) - set(region))

                    # Entanglement entropy S(A)
                    s_a = entanglement_entropy_region(psi, N, region)

                    # Boundary size: number of (i in A, j not in A) pairs
                    # with d(i,j) < threshold (nearby in MI geometry)
                    boundary_count = 0
                    for i_a in region:
                        for j_b in complement:
                            if D[i_a, j_b] < threshold:
                                boundary_count += 1

                    # Alternative: sum of MI across boundary (continuous boundary measure)
                    boundary_mi_sum = 0.0
                    for i_a in region:
                        for j_b in complement:
                            boundary_mi_sum += MI[i_a, j_b]

                    entropy_data.append(s_a)
                    boundary_data.append(boundary_count)
                    volume_data.append(k)

            # Linear fits
            entropy_arr = np.array(entropy_data)
            boundary_arr = np.array(boundary_data)
            volume_arr = np.array(volume_data)

            # S vs |dA| (area law fit)
            valid_b = boundary_arr > 0
            if np.sum(valid_b) > 2 and np.std(boundary_arr[valid_b]) > 0:
                r_area = float(np.corrcoef(boundary_arr[valid_b], entropy_arr[valid_b])[0, 1])
                # Linear fit: S = a * |dA| + b
                coeffs_area = np.polyfit(boundary_arr[valid_b], entropy_arr[valid_b], 1)
                r2_area = r_area ** 2
            else:
                r_area = 0.0
                r2_area = 0.0
                coeffs_area = [0.0, 0.0]

            # S vs volume (volume law fit)
            if np.std(volume_arr) > 0:
                r_volume = float(np.corrcoef(volume_arr, entropy_arr)[0, 1])
                r2_volume = r_volume ** 2
            else:
                r_volume = 0.0
                r2_volume = 0.0

            entry = {
                "N": N,
                "seed": seed,
                "r_area": r_area,
                "r2_area": r2_area,
                "r_volume": r_volume,
                "r2_volume": r2_volume,
                "area_law_slope": float(coeffs_area[0]),
                "area_law_intercept": float(coeffs_area[1]),
                "threshold": threshold,
                "n_points": len(entropy_data),
                "entropy_values": entropy_data,
                "boundary_values": [int(b) for b in boundary_data],
                "volume_values": volume_data,
            }
            all_results.append(entry)

            area_better = r2_area > r2_volume
            print(f"    seed={seed_idx}: R2_area={r2_area:.4f}, R2_volume={r2_volume:.4f} "
                  f"{'AREA LAW' if area_better else 'volume law'}")

            del psi
            gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"  AREA LAW SUMMARY")
    print(f"  Area law: S ~ |dA|. Volume law: S ~ |A|.")
    print(f"  R2_area > R2_volume => area law holds in MI geometry")
    print(f"{'='*70}")

    summary = {}
    for N in N_values:
        matching = [r for r in all_results if r['N'] == N]
        r2_areas = [r['r2_area'] for r in matching]
        r2_volumes = [r['r2_volume'] for r in matching]
        area_wins = sum(1 for a, v in zip(r2_areas, r2_volumes) if a > v)
        key = f"N={N}"
        summary[key] = {
            "N": N,
            "mean_r2_area": float(np.mean(r2_areas)),
            "std_r2_area": float(np.std(r2_areas)),
            "mean_r2_volume": float(np.mean(r2_volumes)),
            "std_r2_volume": float(np.std(r2_volumes)),
            "area_law_wins": area_wins,
            "total_trials": len(matching),
        }
        print(f"  N={N}: R2_area = {np.mean(r2_areas):.4f} +/- {np.std(r2_areas):.4f}, "
              f"R2_volume = {np.mean(r2_volumes):.4f} +/- {np.std(r2_volumes):.4f}, "
              f"area wins {area_wins}/{len(matching)}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    output = {
        "experiment": "area_law",
        "description": "S(A) vs |boundary(A)| in MI geometry: area law test",
        "all_results": all_results,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }

    outpath = RESULTS_DIR / "paper7_area_law.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved to {outpath}")
    return output


# ══════════════════════════════════════════════════════════════
# Experiment 4: Entanglement-Curvature Correlation
# Ollivier-Ricci curvature vs TC density
# ══════════════════════════════════════════════════════════════

def ollivier_ricci_curvature(MI, D, i, j):
    """
    Compute Ollivier-Ricci curvature between nodes i and j.

    kappa(i,j) = 1 - W_1(mu_i, mu_j) / d(i,j)

    where mu_i is the uniform distribution over neighbors of i,
    W_1 is the Wasserstein-1 (earth mover's) distance,
    and d(i,j) is the MI-derived distance.

    For simplicity we use the "lazy random walk" measure:
    mu_i(k) = MI(i,k) / sum_k MI(i,k) for k != i, mu_i(i) = 0
    (MI-weighted neighbor distribution)

    Returns curvature value. Positive = positively curved (sphere-like),
    negative = negatively curved (hyperbolic).
    """
    N = MI.shape[0]
    d_ij = D[i, j]
    if d_ij < 1e-14:
        return 0.0

    # Build MI-weighted probability distributions
    mi_i = MI[i, :].copy()
    mi_i[i] = 0.0
    sum_i = np.sum(mi_i)
    if sum_i < 1e-14:
        return 0.0
    mu_i = mi_i / sum_i

    mi_j = MI[j, :].copy()
    mi_j[j] = 0.0
    sum_j = np.sum(mi_j)
    if sum_j < 1e-14:
        return 0.0
    mu_j = mi_j / sum_j

    # Wasserstein-1 distance via linear programming
    # For small N, we can use the EMD directly
    try:
        from scipy.optimize import linprog
        # Transport plan: T[a,b] = mass moved from a to b
        # Minimize sum_{a,b} T[a,b] * D[a,b]
        # Subject to: sum_b T[a,b] = mu_i[a] for all a
        #             sum_a T[a,b] = mu_j[b] for all b
        #             T[a,b] >= 0

        # Flatten transport plan: variable x has N*N entries
        c = D.flatten()  # cost vector

        # Equality constraints
        # Row sums = mu_i
        A_eq_rows = []
        b_eq = []
        for a in range(N):
            row = np.zeros(N * N)
            for b in range(N):
                row[a * N + b] = 1.0
            A_eq_rows.append(row)
            b_eq.append(mu_i[a])
        # Column sums = mu_j
        for b in range(N):
            row = np.zeros(N * N)
            for a in range(N):
                row[a * N + b] = 1.0
            A_eq_rows.append(row)
            b_eq.append(mu_j[b])

        A_eq = np.array(A_eq_rows)
        b_eq = np.array(b_eq)

        result = linprog(c, A_eq=A_eq, b_eq=b_eq,
                        bounds=[(0, None)] * (N * N),
                        method='highs', options={'presolve': True})

        if result.success:
            W1 = result.fun
            kappa = 1.0 - W1 / d_ij
            return float(kappa)
        else:
            return 0.0

    except Exception:
        # Fallback: simple upper bound using triangle inequality
        # W1 <= sum_a mu_i(a) * d(a, j) vs sum_b mu_j(b) * d(i, b)
        W1_upper = min(
            float(np.sum(mu_i * D[:, j])),
            float(np.sum(mu_j * D[i, :]))
        )
        kappa = 1.0 - W1_upper / d_ij
        return float(kappa)


def tc_density_per_site(psi, N):
    """
    Compute local TC density for each site.
    TC_density(i) = sum_{j != i} MI(i, j) / 2
    This measures how much site i contributes to total correlations.
    """
    MI = mutual_information_matrix(psi, N)
    # TC_density(i) = S(rho_i) - (1/N) sum_j S(rho_{ij}) ... approximately
    # Simpler: use sum of MI as proxy for local TC contribution
    tc_density = np.sum(MI, axis=1) / 2.0  # divide by 2 to avoid double counting
    return tc_density, MI


def experiment_4_curvature_correlation(N_values=None, n_seeds=5):
    """
    Compute Ollivier-Ricci curvature in MI geometry and correlate
    with local TC density. Connects to Paper 2 (Ricci from partial observation).
    """
    if N_values is None:
        N_values = [8, 12]

    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Entanglement-Curvature Correlation")
    print("  Ollivier-Ricci curvature vs TC density")
    print("=" * 70)

    t0 = time.time()
    all_results = []

    for N in N_values:
        print(f"\n  N = {N} (dim = {2**N})")

        for seed_idx in range(n_seeds):
            seed = 80000 + N * 100 + seed_idx

            if N >= SPARSE_THRESHOLD:
                H, couplings = random_all_to_all_sparse(N, seed=seed)
            else:
                H, couplings = random_all_to_all(N, seed=seed)
            E0, psi = auto_ground_state(H)

            # TC density and MI matrix
            tc_dens, MI = tc_density_per_site(psi, N)

            # Distance matrix (log-MI)
            max_mi = np.max(MI[MI > 0]) if np.any(MI > 0) else 1.0
            D = np.zeros_like(MI)
            for i in range(N):
                for j in range(i + 1, N):
                    if MI[i, j] > 1e-14:
                        D[i, j] = -np.log(MI[i, j] / max_mi + 1e-14)
                    else:
                        D[i, j] = 10.0
                    D[j, i] = D[i, j]

            # Compute Ollivier-Ricci curvature for each edge
            curvatures = []
            edge_tc_density = []  # average TC density of endpoints
            edge_mi = []

            for i in range(N):
                for j in range(i + 1, N):
                    kappa = ollivier_ricci_curvature(MI, D, i, j)
                    curvatures.append(kappa)
                    edge_tc_density.append((tc_dens[i] + tc_dens[j]) / 2.0)
                    edge_mi.append(MI[i, j])

            curvatures = np.array(curvatures)
            edge_tc_density = np.array(edge_tc_density)
            edge_mi = np.array(edge_mi)

            # Per-site curvature (mean of incident edge curvatures)
            site_curvature = np.zeros(N)
            for i in range(N):
                site_curv = []
                idx = 0
                for a in range(N):
                    for b in range(a + 1, N):
                        if a == i or b == i:
                            site_curv.append(curvatures[idx])
                        idx += 1
                site_curvature[i] = np.mean(site_curv) if site_curv else 0.0

            # Correlations
            # 1. Edge-level: curvature vs edge TC density
            if np.std(curvatures) > 1e-14 and np.std(edge_tc_density) > 1e-14:
                r_edge = float(np.corrcoef(curvatures, edge_tc_density)[0, 1])
            else:
                r_edge = 0.0

            # 2. Site-level: curvature vs TC density
            if np.std(site_curvature) > 1e-14 and np.std(tc_dens) > 1e-14:
                r_site = float(np.corrcoef(site_curvature, tc_dens)[0, 1])
            else:
                r_site = 0.0

            # 3. Edge-level: curvature vs MI (high MI = short distance = more curved?)
            if np.std(curvatures) > 1e-14 and np.std(edge_mi) > 1e-14:
                r_curv_mi = float(np.corrcoef(curvatures, edge_mi)[0, 1])
            else:
                r_curv_mi = 0.0

            entry = {
                "N": N,
                "seed": seed,
                "r_edge_tc": r_edge,
                "r_site_tc": r_site,
                "r_curv_mi": r_curv_mi,
                "mean_curvature": float(np.mean(curvatures)),
                "std_curvature": float(np.std(curvatures)),
                "mean_tc_density": float(np.mean(tc_dens)),
                "scalar_curvature": float(np.sum(curvatures)),
                "site_curvatures": site_curvature.tolist(),
                "tc_density": tc_dens.tolist(),
                "edge_curvatures": curvatures.tolist(),
            }
            all_results.append(entry)

            print(f"    seed={seed_idx}: r_edge={r_edge:.4f}, r_site={r_site:.4f}, "
                  f"r_curv_mi={r_curv_mi:.4f}, mean_kappa={np.mean(curvatures):.4f}")

            del psi
            gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"  CURVATURE-ENTANGLEMENT SUMMARY")
    print(f"  Positive r = curvature correlates with TC density")
    print(f"{'='*70}")

    summary = {}
    for N in N_values:
        matching = [r for r in all_results if r['N'] == N]
        r_edges = [r['r_edge_tc'] for r in matching]
        r_sites = [r['r_site_tc'] for r in matching]
        r_mis = [r['r_curv_mi'] for r in matching]
        mean_kappas = [r['mean_curvature'] for r in matching]

        key = f"N={N}"
        summary[key] = {
            "N": N,
            "mean_r_edge": float(np.mean(r_edges)),
            "std_r_edge": float(np.std(r_edges)),
            "mean_r_site": float(np.mean(r_sites)),
            "std_r_site": float(np.std(r_sites)),
            "mean_r_curv_mi": float(np.mean(r_mis)),
            "mean_kappa": float(np.mean(mean_kappas)),
            "n_seeds": len(matching),
        }
        print(f"  N={N}: r_edge={np.mean(r_edges):.4f} +/- {np.std(r_edges):.4f}, "
              f"r_site={np.mean(r_sites):.4f} +/- {np.std(r_sites):.4f}, "
              f"mean_kappa={np.mean(mean_kappas):.4f}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    output = {
        "experiment": "curvature_correlation",
        "description": "Ollivier-Ricci curvature vs TC density in MI geometry",
        "all_results": all_results,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }

    outpath = RESULTS_DIR / "paper7_curvature_correlation.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved to {outpath}")
    return output


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Paper 7: Einstein's Equations from Perspectival Crystallization")
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4],
                        help='Run specific experiment (1-4)')
    parser.add_argument('--N', type=int, nargs='+', default=None,
                        help='System sizes (default: 8,12)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds per N (default: 5)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: N=8 only, 3 seeds')
    args = parser.parse_args()

    N_values = args.N
    n_seeds = args.seeds

    if args.quick:
        N_values = [8]
        n_seeds = 3

    if N_values is None:
        N_values = [8, 12]

    print("=" * 70)
    print("  PAPER 7: Einstein's Equations from Perspectival Crystallization")
    print("  Bridge Theorem: delta_TC = -delta_S_ent")
    print(f"  N = {N_values}, seeds = {n_seeds}")
    print("=" * 70)

    global_t0 = time.time()

    if args.exp is None or args.exp == 1:
        experiment_1_bridge_theorem(N_values=N_values, n_seeds=n_seeds)

    if args.exp is None or args.exp == 2:
        experiment_2_uv_stability(N_values=N_values, n_seeds=n_seeds)

    if args.exp is None or args.exp == 3:
        experiment_3_area_law(N_values=N_values, n_seeds=n_seeds)

    if args.exp is None or args.exp == 4:
        experiment_4_curvature_correlation(N_values=N_values, n_seeds=n_seeds)

    total = time.time() - global_t0
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total elapsed: {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
