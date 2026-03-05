"""
Circularity-Breaking Experiment for PLC.

The deepest objection: using MI-derived distance and showing MI-related correlations
decay with that distance is "correlating correlations with correlations."
Pinsker inequality guarantees the sign. This is potentially fatal.

TWO independent distance metrics that break circularity:

1. COUPLING DISTANCE: d_J(i,j) = 1/|J_ij| where J_ij are Hamiltonian couplings.
   Completely independent of the quantum state. Comes from the Hamiltonian, not MI.
   For random all-to-all: NO expected correlation decay (no spatial structure).
   For 1D chain: SHOULD see decay (coupling distance = chain distance).

2. CROSS-OBSERVER MI DISTANCE: Observer A defines distance from their MI matrix.
   Observer B's correlations are measured on a DIFFERENT subset.
   If A's distance predicts B's correlations (on overlapping pairs),
   the metric captures genuine quantum structure, not tautological correlation.

Built by Opus Warrior, March 5 2026.
"""

import numpy as np
from itertools import combinations
from typing import Optional
from scipy.stats import pearsonr

from .quantum import (
    random_all_to_all, nearest_neighbor_chain,
    ground_state, ground_state_gpu,
    mutual_information_matrix, correlation_matrix,
    connected_correlation,
)
from .statistics import bootstrap_ci


def coupling_distance_matrix(n_qubits: int, couplings: np.ndarray) -> np.ndarray:
    """
    Build coupling distance matrix: d_J(i,j) = 1/|J_ij|.

    couplings: flat array of length n_qubits*(n_qubits-1)/2,
    ordered as (0,1), (0,2), ..., (0,N-1), (1,2), ..., (N-2,N-1).

    Returns NxN symmetric matrix. Diagonal = 0. If |J_ij| < epsilon, d = large sentinel.
    """
    D = np.zeros((n_qubits, n_qubits))
    pair_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J_abs = abs(couplings[pair_idx])
            if J_abs > 1e-12:
                D[i, j] = 1.0 / J_abs
            else:
                D[i, j] = 1e6  # effectively infinite distance
            D[j, i] = D[i, j]
            pair_idx += 1
    return D


def chain_coupling_distance(n_qubits: int, couplings: np.ndarray,
                             periodic: bool = False) -> np.ndarray:
    """
    Coupling distance for 1D chain. Only nearest-neighbor couplings exist.

    d_J(i,j) = sum of 1/|J_k| along the shortest path from i to j on the chain.
    This is the natural "resistance distance" on the chain graph.
    """
    n_bonds = n_qubits if periodic else n_qubits - 1
    # Bond "lengths" = 1/|J_k|
    bond_lengths = np.array([1.0 / abs(couplings[k]) if abs(couplings[k]) > 1e-12 else 1e6
                             for k in range(n_bonds)])

    D = np.zeros((n_qubits, n_qubits))
    if not periodic:
        # Simple: cumulative sum of bond lengths
        cum = np.zeros(n_qubits)
        for k in range(1, n_qubits):
            cum[k] = cum[k - 1] + bond_lengths[k - 1]
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                D[i, j] = cum[j] - cum[i]
                D[j, i] = D[i, j]
    else:
        # Periodic: shortest of clockwise and counterclockwise
        total = np.sum(bond_lengths)
        cum = np.zeros(n_qubits)
        for k in range(1, n_qubits):
            cum[k] = cum[k - 1] + bond_lengths[k - 1]
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                d_cw = cum[j] - cum[i]
                d_ccw = total - d_cw
                D[i, j] = min(d_cw, d_ccw)
                D[j, i] = D[i, j]
    return D


def pearson_r_dist_vs_corr(D: np.ndarray, C: np.ndarray,
                            sites: Optional[list[int]] = None) -> tuple:
    """
    Pearson r between distance d(i,j) and |C(i,j)| for all pairs.

    If sites is provided, D and C are indexed into those sites.
    Returns (r, p_value, n_pairs, distances, abs_correlations).
    """
    n = D.shape[0]
    dists = []
    corrs = []

    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            c = abs(C[i, j])
            if d < 1e5 and c > 1e-15:  # skip infinite-distance pairs
                dists.append(d)
                corrs.append(c)

    if len(dists) < 3:
        return 0.0, 1.0, len(dists), np.array([]), np.array([])

    dists = np.array(dists)
    corrs = np.array(corrs)

    # Pearson r between distance and |C| (expect negative if decay exists)
    r, p = pearsonr(dists, corrs)
    return float(r), float(p), len(dists), dists, corrs


def pearson_r_dist_vs_log_corr(D: np.ndarray, C: np.ndarray) -> tuple:
    """
    Pearson r between distance and log|C| for exponential decay detection.
    Returns (r, p_value, n_pairs).
    """
    n = D.shape[0]
    dists = []
    log_corrs = []

    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            c = abs(C[i, j])
            if d < 1e5 and c > 1e-14:
                dists.append(d)
                log_corrs.append(np.log(c))

    if len(dists) < 3:
        return 0.0, 1.0, len(dists)

    r, p = pearsonr(dists, log_corrs)
    return float(r), float(p), len(dists)


def cross_observer_experiment(psi: np.ndarray, n_qubits: int, k: int,
                               seed: int = 0) -> dict:
    """
    Cross-observer circularity breaker.

    1. Draw random k-subset A (observer A).
    2. Draw random k-subset B (observer B), ensuring some overlap with A.
    3. Compute MI-distance from observer A's MI matrix.
    4. Compute |C_ij| from observer B's correlation matrix.
    5. On the OVERLAPPING pairs (i,j in A AND in B),
       compute Pearson r between A's MI-distance and B's |C_ij|.

    If r < 0 on overlapping pairs: A's distance predicts B's correlations.
    This is NOT circular because the distance and correlations come from
    different observers (different partial traces).

    Args:
        psi: ground state vector
        n_qubits: total qubits
        k: observer subset size
        seed: RNG seed

    Returns: dict with r_cross, p_value, n_overlap_pairs, subsets_A, subsets_B
    """
    rng = np.random.default_rng(seed)

    all_qubits = list(range(n_qubits))

    # Draw observer A
    subset_A = sorted(rng.choice(all_qubits, size=k, replace=False).tolist())

    # Draw observer B: ensure overlap but not identical
    # Strategy: keep some of A, replace others
    max_attempts = 50
    for attempt in range(max_attempts):
        subset_B = sorted(rng.choice(all_qubits, size=k, replace=False).tolist())
        overlap = set(subset_A) & set(subset_B)
        # Need at least 3 overlapping qubits for meaningful pairs
        n_overlap_pairs = len(overlap) * (len(overlap) - 1) // 2
        if len(overlap) >= 3 and subset_A != subset_B:
            break
    else:
        # Fallback: force overlap
        subset_B = subset_A[:k // 2] + sorted(rng.choice(
            [q for q in all_qubits if q not in subset_A[:k // 2]],
            size=k - k // 2, replace=False
        ).tolist())
        overlap = set(subset_A) & set(subset_B)

    overlap_list = sorted(overlap)

    # Observer A: MI matrix on subset_A
    MI_A = mutual_information_matrix(psi, n_qubits, subset_A)

    # Observer B: correlation matrix on subset_B
    C_B = correlation_matrix(psi, n_qubits, subset_B)

    # Build MI-distance from A for overlapping pairs
    # Map global qubit indices to local indices in each observer's matrix
    A_local = {q: i for i, q in enumerate(subset_A)}
    B_local = {q: i for i, q in enumerate(subset_B)}

    dists_A = []
    corrs_B = []
    pair_labels = []

    for i_idx, qi in enumerate(overlap_list):
        for j_idx, qj in enumerate(overlap_list):
            if qi >= qj:
                continue
            # A's MI-distance for this pair
            ai, aj = A_local[qi], A_local[qj]
            mi_val = MI_A[ai, aj]
            if mi_val > 1e-14:
                d_A = 1.0 / mi_val
            else:
                continue

            # B's correlation for this pair
            bi, bj = B_local[qi], B_local[qj]
            c_B = abs(C_B[bi, bj])
            if c_B < 1e-15:
                continue

            dists_A.append(d_A)
            corrs_B.append(c_B)
            pair_labels.append((qi, qj))

    if len(dists_A) < 3:
        return {
            "r_cross": 0.0,
            "p_value": 1.0,
            "n_overlap_pairs": len(dists_A),
            "subset_A": subset_A,
            "subset_B": subset_B,
            "overlap": overlap_list,
            "sufficient_data": False,
        }

    dists_A = np.array(dists_A)
    corrs_B = np.array(corrs_B)

    # Pearson r: A's distance vs B's |C|
    r, p = pearsonr(dists_A, corrs_B)

    # Also log-space
    log_corrs_B = np.log(corrs_B)
    r_log, p_log = pearsonr(dists_A, log_corrs_B)

    return {
        "r_cross": float(r),
        "r_cross_log": float(r_log),
        "p_value": float(p),
        "p_value_log": float(p_log),
        "n_overlap_pairs": len(dists_A),
        "subset_A": subset_A,
        "subset_B": subset_B,
        "overlap": overlap_list,
        "sufficient_data": True,
    }


def run_circularity_breaking(n_values: list[int] = None,
                              n_hamiltonians: int = 20,
                              use_gpu: bool = True,
                              verbose: bool = True) -> dict:
    """
    Main circularity-breaking experiment.

    For each N in n_values, for each of n_hamiltonians random Hamiltonians:

    TEST 1 - Coupling distance vs |C|:
        d_J(i,j) = 1/|J_ij|. Completely state-independent.
        For random all-to-all: expect r ~ 0 (no structure).
        This is the NEGATIVE CONTROL.

    TEST 2 - Cross-observer MI distance:
        Observer A's MI-distance vs Observer B's |C| on overlapping pairs.
        If r < 0: the emergent metric captures genuine structure, not tautology.
        This is the POSITIVE TEST.

    TEST 3 - 1D Chain control (coupling distance SHOULD work):
        For a nearest-neighbor chain, coupling distance = physical distance.
        Correlations SHOULD decay with coupling distance.
        This validates that coupling distance CAN detect decay when structure exists.
    """
    if n_values is None:
        n_values = [8, 10]

    diag_fn = ground_state_gpu if use_gpu else ground_state

    results = {
        "experiment": "circularity_breaking",
        "n_values": n_values,
        "n_hamiltonians": n_hamiltonians,
        "tests": {},
    }

    for N in n_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  CIRCULARITY BREAKING: N = {N}")
            print(f"{'='*60}")

        test1_rs = []       # coupling dist vs |C| (all-to-all, negative control)
        test2_rs = []       # cross-observer MI-dist vs |C| (positive test)
        test2_rs_log = []   # same but log-space
        test3_rs = []       # coupling dist vs |C| (1D chain, positive control)

        k = max(4, N // 2)  # observer subset size

        # ── TEST 1 + TEST 2: Random all-to-all ──
        if verbose:
            print(f"\n  Tests 1+2: Random all-to-all Hamiltonians")
        for h in range(n_hamiltonians):
            seed = 10000 + N * 1000 + h

            # Build random all-to-all Hamiltonian
            H, couplings = random_all_to_all(N, seed=seed)
            E0, psi = diag_fn(H)

            # TEST 1: Coupling distance vs full correlations
            D_coupling = coupling_distance_matrix(N, couplings)
            C_full = correlation_matrix(psi, N)
            r1, p1, n1, _, _ = pearson_r_dist_vs_corr(D_coupling, C_full)
            test1_rs.append(r1)

            # TEST 2: Cross-observer
            # Run multiple random A/B splits for this Hamiltonian
            n_splits = 5
            for s in range(n_splits):
                cross = cross_observer_experiment(psi, N, k, seed=seed + s * 100)
                if cross["sufficient_data"]:
                    test2_rs.append(cross["r_cross"])
                    test2_rs_log.append(cross["r_cross_log"])

            if verbose and (h + 1) % 5 == 0:
                print(f"    Hamiltonian {h+1}/{n_hamiltonians}: "
                      f"test1_r={r1:.3f}, "
                      f"test2 samples={len(test2_rs)}")

        # ── TEST 3: 1D Chain positive control ──
        if verbose:
            print(f"\n  Test 3: 1D nearest-neighbor chain (positive control)")
        for h in range(n_hamiltonians):
            seed = 20000 + N * 1000 + h
            rng_chain = np.random.default_rng(seed)

            # Build 1D chain with random couplings
            chain_couplings = rng_chain.standard_normal(N - 1)
            H_chain, used_couplings = nearest_neighbor_chain(N, chain_couplings)
            E0_chain, psi_chain = diag_fn(H_chain)

            # Coupling distance on chain (path distance)
            D_chain = chain_coupling_distance(N, used_couplings)
            C_chain = correlation_matrix(psi_chain, N)
            r3, p3, n3, _, _ = pearson_r_dist_vs_corr(D_chain, C_chain)
            test3_rs.append(r3)

            if verbose and (h + 1) % 5 == 0:
                print(f"    Chain {h+1}/{n_hamiltonians}: r={r3:.3f}")

        # ── Aggregate statistics ──
        test1_rs = np.array(test1_rs)
        test2_rs = np.array(test2_rs)
        test2_rs_log = np.array(test2_rs_log)
        test3_rs = np.array(test3_rs)

        # Bootstrap CIs
        ci_test1 = bootstrap_ci(test1_rs, seed=42 + N)
        ci_test2 = bootstrap_ci(test2_rs, seed=43 + N) if len(test2_rs) >= 3 else None
        ci_test2_log = bootstrap_ci(test2_rs_log, seed=44 + N) if len(test2_rs_log) >= 3 else None
        ci_test3 = bootstrap_ci(test3_rs, seed=45 + N)

        # Store results for this N
        results["tests"][f"N={N}"] = {
            "N": N,
            "k_observer": k,

            "test1_coupling_vs_corr_alltoall": {
                "description": "Coupling distance d_J=1/|J_ij| vs |C_ij|, random all-to-all. NEGATIVE CONTROL.",
                "expected": "r ~ 0 (random couplings, no spatial structure)",
                "n_hamiltonians": n_hamiltonians,
                "r_values": test1_rs.tolist(),
                "mean_r": float(np.mean(test1_rs)),
                "std_r": float(np.std(test1_rs)),
                "bootstrap_ci": ci_test1,
            },

            "test2_cross_observer": {
                "description": "Observer A MI-distance vs Observer B |C_ij| on overlapping pairs. CIRCULARITY BREAKER.",
                "expected": "r < 0 if emergent metric is real (not circular)",
                "n_samples": len(test2_rs),
                "r_values": test2_rs.tolist(),
                "r_log_values": test2_rs_log.tolist(),
                "mean_r": float(np.mean(test2_rs)) if len(test2_rs) > 0 else None,
                "std_r": float(np.std(test2_rs)) if len(test2_rs) > 0 else None,
                "mean_r_log": float(np.mean(test2_rs_log)) if len(test2_rs_log) > 0 else None,
                "bootstrap_ci": ci_test2,
                "bootstrap_ci_log": ci_test2_log,
            },

            "test3_coupling_vs_corr_chain": {
                "description": "Coupling distance vs |C_ij|, 1D chain. POSITIVE CONTROL.",
                "expected": "r < 0 (correlations decay with chain distance)",
                "n_hamiltonians": n_hamiltonians,
                "r_values": test3_rs.tolist(),
                "mean_r": float(np.mean(test3_rs)),
                "std_r": float(np.std(test3_rs)),
                "bootstrap_ci": ci_test3,
            },
        }

        if verbose:
            print(f"\n  {'─'*50}")
            print(f"  RESULTS FOR N = {N}:")
            print(f"  {'─'*50}")
            print(f"  TEST 1 (Coupling dist, all-to-all, NEGATIVE CONTROL):")
            print(f"    mean r = {np.mean(test1_rs):.4f} +/- {np.std(test1_rs):.4f}")
            print(f"    95% CI: [{ci_test1['ci_low']:.4f}, {ci_test1['ci_high']:.4f}]")
            t1_verdict = "PASS (r~0)" if abs(np.mean(test1_rs)) < 0.2 else "UNEXPECTED"
            print(f"    Verdict: {t1_verdict}")

            print(f"\n  TEST 2 (Cross-observer MI-dist, CIRCULARITY BREAKER):")
            if len(test2_rs) > 0:
                print(f"    mean r = {np.mean(test2_rs):.4f} +/- {np.std(test2_rs):.4f}")
                if ci_test2:
                    print(f"    95% CI: [{ci_test2['ci_low']:.4f}, {ci_test2['ci_high']:.4f}]")
                if len(test2_rs_log) > 0:
                    print(f"    mean r (log): {np.mean(test2_rs_log):.4f}")
                    if ci_test2_log:
                        print(f"    95% CI (log): [{ci_test2_log['ci_low']:.4f}, {ci_test2_log['ci_high']:.4f}]")
                frac_neg = np.mean(test2_rs < 0)
                t2_verdict = "STRONG" if np.mean(test2_rs) < -0.2 and frac_neg > 0.6 else \
                             "MODERATE" if np.mean(test2_rs) < -0.1 else \
                             "WEAK" if np.mean(test2_rs) < 0 else "ABSENT"
                print(f"    {frac_neg:.0%} of samples show r < 0")
                print(f"    Verdict: {t2_verdict} — emergent metric is {'NOT ' if t2_verdict == 'ABSENT' else ''}real")
            else:
                print(f"    Insufficient overlap data")

            print(f"\n  TEST 3 (Coupling dist, 1D chain, POSITIVE CONTROL):")
            print(f"    mean r = {np.mean(test3_rs):.4f} +/- {np.std(test3_rs):.4f}")
            print(f"    95% CI: [{ci_test3['ci_low']:.4f}, {ci_test3['ci_high']:.4f}]")
            t3_verdict = "PASS (r<0)" if np.mean(test3_rs) < -0.1 else "WEAK"
            print(f"    Verdict: {t3_verdict}")

    return results
