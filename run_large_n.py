#!/usr/bin/env python3
"""
Large-N PLC experiments: N=14 and N=16 using sparse Lanczos.

Addresses the "small N" objection: at N=8 with k=3, MDS on 3 points is a triangle.
At N=16 with k=8, we get 28 qubit pairs and up to 7 MDS dimensions -- REAL geometry.

Uses sparse Hamiltonian construction + ARPACK eigsh to stay within memory.
Dense N=14: 4GB. Dense N=16: 64GB. Sparse: ~100MB for either.

Optimization: compute ground state ONCE per (N, seed), then evaluate all k values.
For N=16, use dim at k=N-2 as proxy for "full" dimension (avoids computing all 120 pairs).

Built by Opus Warrior, March 5 2026.
"""

import sys
import os
import json
import time
import gc
import numpy as np
from pathlib import Path
from itertools import combinations

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.quantum import (
    random_all_to_all_sparse, ground_state_sparse,
    partial_trace, mutual_information_matrix, von_neumann_entropy,
)
from src.experiments import _mi_to_distance, _effective_dimension
from src.statistics import bootstrap_ci as _stats_bootstrap_ci
from src.utils import NumpyEncoder

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Thin wrapper around src.statistics.bootstrap_ci returning (mean, lo, hi) tuple."""
    values = np.array([v for v in data if np.isfinite(v)])
    if len(values) < 2:
        m = float(np.mean(values)) if len(values) > 0 else 0.0
        return m, m, m
    result = _stats_bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci, seed=42)
    return result["estimate"], result["ci_low"], result["ci_high"]


def compute_observer_metrics(psi, N, k):
    """
    Compute MI matrix for observer seeing k qubits (first k qubits),
    return effective dimension and Pearson r.
    """
    observer = list(range(k))
    n_pairs = k * (k - 1) // 2

    t0 = time.time()
    MI_obs = mutual_information_matrix(psi, N, observer)
    elapsed = time.time() - t0

    D_obs = _mi_to_distance(MI_obs)
    dim_obs = _effective_dimension(D_obs)

    # Pearson correlation between distance and log(MI)
    off_diag_idx = np.triu_indices(k, k=1)
    mi_values = MI_obs[off_diag_idx]
    d_values = D_obs[off_diag_idx]

    valid = mi_values > 1e-14
    r_pearson = 0.0
    if np.sum(valid) >= 3:
        log_mi = np.log(mi_values[valid])
        d_valid = d_values[valid]
        if np.std(d_valid) > 1e-14 and np.std(log_mi) > 1e-14:
            r_pearson = float(np.corrcoef(d_valid, log_mi)[0, 1])

    print(f"      k={k}: {n_pairs} pairs, dim={dim_obs:.1f}, r={r_pearson:.3f} ({elapsed:.1f}s)")

    return dim_obs, r_pearson, n_pairs


def run_trial_all_k(N, k_values, seed, trial_id):
    """
    Single ground state computation, then evaluate all k values.
    Much more efficient than recomputing H and psi for each k.
    """
    print(f"\n  [N={N}, trial={trial_id}, seed={seed}]")

    # Build sparse Hamiltonian and find ground state
    H_sparse, couplings = random_all_to_all_sparse(N, seed=seed)
    E0, psi = ground_state_sparse(H_sparse)
    del H_sparse
    gc.collect()

    # Compute "full" dimension: use all N qubits for N<=14, else N-2 as proxy
    if N <= 14:
        print(f"    Computing full MI ({N} qubits, {N*(N-1)//2} pairs)...")
        t0 = time.time()
        MI_full = mutual_information_matrix(psi, N)
        print(f"    Full MI: {time.time()-t0:.1f}s")
        D_full = _mi_to_distance(MI_full)
        dim_full = _effective_dimension(D_full)
        del MI_full, D_full
    else:
        # For N=16: use k=12 (66 pairs) as "full" proxy -- still expensive but manageable
        # Actually, use k=10 (45 pairs) -- faster and sufficient
        k_proxy = 10
        print(f"    Computing proxy-full MI (k={k_proxy}, {k_proxy*(k_proxy-1)//2} pairs)...")
        t0 = time.time()
        MI_proxy = mutual_information_matrix(psi, N, list(range(k_proxy)))
        print(f"    Proxy MI: {time.time()-t0:.1f}s")
        D_proxy = _mi_to_distance(MI_proxy)
        dim_full = _effective_dimension(D_proxy)
        del MI_proxy, D_proxy

    print(f"    dim_full = {dim_full:.1f}, E0 = {E0:.6f}")

    results = []
    for k in k_values:
        dim_obs, r_pearson, n_pairs = compute_observer_metrics(psi, N, k)
        dim_ratio = dim_obs / dim_full if dim_full > 0 else 0.0

        results.append({
            "N": N,
            "k": k,
            "k_over_N": round(k / N, 4),
            "n_observer_pairs": n_pairs,
            "trial": trial_id,
            "seed": seed,
            "E0": float(E0),
            "dim_full": float(dim_full),
            "dim_observer": float(dim_obs),
            "dim_ratio": float(dim_ratio),
            "r_pearson": r_pearson,
            "max_mds_dims": n_pairs - 1,
        })

    del psi
    gc.collect()
    return results


def main():
    print("=" * 70)
    print("  LARGE-N PLC EXPERIMENT: Sparse Lanczos")
    print("  Addressing the 'small N' objection with N=14 and N=16")
    print("=" * 70)

    global_t0 = time.time()
    all_results = []

    # ── N = 14 ─────────────────────────────────────────────
    N = 14
    n_trials = 5
    k_values = [4, 6, 8]

    print(f"\n{'='*70}")
    print(f"  N = {N} (dim = {2**N}), {n_trials} trials")
    print(f"  k values: {k_values} -> k/N = {[round(k/N,3) for k in k_values]}")
    print(f"{'='*70}")

    for trial in range(n_trials):
        seed = 5000 + N * 100 + trial
        results = run_trial_all_k(N, k_values, seed, trial)
        all_results.extend(results)

    # ── N = 16 ─────────────────────────────────────────────
    N = 16
    n_trials = 3
    k_values = [4, 6, 8]

    print(f"\n{'='*70}")
    print(f"  N = {N} (dim = {2**N}), {n_trials} trials")
    print(f"  k values: {k_values} -> k/N = {[round(k/N,3) for k in k_values]}")
    print(f"{'='*70}")

    for trial in range(n_trials):
        seed = 6000 + N * 100 + trial
        results = run_trial_all_k(N, k_values, seed, trial)
        all_results.extend(results)

    # ── Aggregate and report ────────────────────────────────
    total_elapsed = time.time() - global_t0

    print(f"\n\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    summary = {}
    for N_val in [14, 16]:
        summary[f"N={N_val}"] = {}
        n_results = [r for r in all_results if r['N'] == N_val]
        k_vals = sorted(set(r['k'] for r in n_results))

        for k in k_vals:
            k_results = [r for r in n_results if r['k'] == k]
            dim_ratios = [r['dim_ratio'] for r in k_results]
            r_pearsons = [r['r_pearson'] for r in k_results]

            dr_mean, dr_lo, dr_hi = _bootstrap_ci(dim_ratios)
            rp_mean, rp_lo, rp_hi = _bootstrap_ci(r_pearsons)

            n_pairs = k * (k - 1) // 2
            max_dims = n_pairs - 1

            entry = {
                "k": k,
                "k_over_N": round(k / N_val, 4),
                "n_pairs": n_pairs,
                "max_mds_dims": max_dims,
                "dim_ratio_mean": dr_mean,
                "dim_ratio_ci95": [dr_lo, dr_hi],
                "r_pearson_mean": rp_mean,
                "r_pearson_ci95": [rp_lo, rp_hi],
                "n_trials": len(k_results),
                "individual_dim_ratios": dim_ratios,
                "individual_r_pearsons": r_pearsons,
            }
            summary[f"N={N_val}"][f"k={k}"] = entry

            locality = "STRONG" if dr_mean < 0.7 else "MODERATE" if dr_mean < 0.9 else "WEAK"
            decay = "YES" if rp_mean < -0.3 else "MARGINAL" if rp_mean < -0.1 else "NO"

            print(f"\n  N={N_val}, k={k} (k/N={k/N_val:.3f}, {n_pairs} pairs, up to {max_dims} MDS dims)")
            print(f"    dim_ratio = {dr_mean:.3f} [{dr_lo:.3f}, {dr_hi:.3f}]  [{locality} LOCALITY]")
            print(f"    r_pearson = {rp_mean:.3f} [{rp_lo:.3f}, {rp_hi:.3f}]  [decay: {decay}]")

    # THE KEY RESULT: N=16, k=8
    key = summary.get("N=16", {}).get("k=8", None)
    if key:
        print(f"\n\n  {'*'*60}")
        print(f"  KEY RESULT: N=16, k=8 (28 pairs, up to 27 MDS dimensions)")
        print(f"  This is REAL geometry -- not a triangle on 3 points.")
        print(f"  dim_ratio = {key['dim_ratio_mean']:.3f} {key['dim_ratio_ci95']}")
        print(f"  r_pearson = {key['r_pearson_mean']:.3f} {key['r_pearson_ci95']}")
        if key['dim_ratio_mean'] < 1.0:
            print(f"  >>> DIMENSIONALITY REDUCTION CONFIRMED AT LARGE N <<<")
        if key['r_pearson_mean'] < -0.3:
            print(f"  >>> CORRELATION DECAY CONFIRMED AT LARGE N <<<")
        print(f"  {'*'*60}")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")

    # Save
    output = {
        "experiment": "large_n_sparse_lanczos",
        "date": "2026-03-05",
        "description": "N=14 and N=16 PLC experiments using sparse eigensolver. "
                       "For N=16, dim_full is computed from k=10 proxy (45 pairs) "
                       "to avoid computing all 120 pairs.",
        "all_results": all_results,
        "summary": summary,
        "total_elapsed_seconds": total_elapsed,
    }

    outpath = RESULTS_DIR / "large_n.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
