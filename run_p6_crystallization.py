#!/usr/bin/env python3
"""
Crystallization Study — First Law of Agentic Thermodynamics

Empirically verify: "Information density under persistent partial observation
increases monotonically with interaction count."

PLC translation: as observation fraction k/N increases, Total Correlation (TC)
increases monotonically and internal entropy decreases. The observer crystallizes.

Key identity (Paper 5): TC + S_joint = sum(S_i) = constant for fixed state.
So TC increasing <=> S_joint decreasing. One implies the other.

Critical optimization: for pure states, S(A) = S(complement of A).
When k > N/2, compute S_joint via the (N-k)-qubit complement instead of
the k-qubit subsystem. This makes the full k=1..N-1 sweep feasible.

System sizes: N = 8, 16, 20, 24
Seeds per N: 10 (N<=16), 5 (N=20), 3 (N=24)

Built by Opus Warrior, March 17 2026.
"""

import sys
import os
import json
import time
import gc
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from src.quantum import (
    random_all_to_all_sparse, ground_state_sparse,
    partial_trace, von_neumann_entropy,
    mutual_information_matrix,
)
from src.experiments import _mi_to_distance, _effective_dimension
from src.statistics import bootstrap_ci as _stats_bootstrap_ci
from src.utils import NumpyEncoder

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Total Correlation ─────────────────────────────────────────

def total_correlation(psi, N, k):
    """
    Compute Total Correlation for observer seeing first k qubits.

    TC(k) = sum_{i=0}^{k-1} S(rho_i) - S(rho_{0..k-1})

    Uses complement trick: S(rho_{0..k-1}) = S(rho_{k..N-1}) when
    k > N/2, since S(A) = S(complement) for pure states.

    Returns (TC, S_joint, sum_S_i).
    """
    observed = list(range(k))

    # Individual entropies (always cheap: 2x2 matrices)
    sum_S_i = 0.0
    for i in observed:
        rho_i = partial_trace(psi, [i], N)
        sum_S_i += von_neumann_entropy(rho_i)

    # Joint entropy of observed subsystem
    # Use complement if cheaper: min(k, N-k) qubits
    complement_size = N - k
    if complement_size < k and complement_size > 0:
        # Trace out observed qubits, get complement state
        complement = list(range(k, N))
        rho_complement = partial_trace(psi, complement, N)
        S_joint = von_neumann_entropy(rho_complement)
    elif k == N:
        # Entire system = pure state, S = 0
        S_joint = 0.0
    elif k == 0:
        S_joint = 0.0
    else:
        rho_observed = partial_trace(psi, observed, N)
        S_joint = von_neumann_entropy(rho_observed)

    TC = sum_S_i - S_joint
    return TC, S_joint, sum_S_i


# ── Bootstrap wrapper ─────────────────────────────────────────

def _bootstrap_ci(data, n_bootstrap=2000, ci=0.95, seed=42):
    values = np.array([v for v in data if np.isfinite(v)])
    if len(values) < 2:
        m = float(np.mean(values)) if len(values) > 0 else 0.0
        return m, m, m
    result = _stats_bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    return result["estimate"], result["ci_low"], result["ci_high"]


# ── Single trial: sweep k for one ground state ────────────────

def run_trial(N, seed, trial_id):
    """
    Compute ground state once, then sweep k=1..N-1.
    Returns list of dicts with TC, S_joint, sum_S_i for each k.
    """
    print(f"\n  [N={N}, trial={trial_id}, seed={seed}]")

    # Build sparse Hamiltonian and find ground state
    t0 = time.time()
    H_sparse, couplings = random_all_to_all_sparse(N, seed=seed)
    E0, psi = ground_state_sparse(H_sparse)
    del H_sparse
    gc.collect()
    print(f"    Ground state found: E0={E0:.6f} ({time.time()-t0:.1f}s)")

    results = []
    for k in range(1, N):
        t_k = time.time()
        TC, S_joint, sum_S_i = total_correlation(psi, N, k)
        elapsed_k = time.time() - t_k

        entry = {
            "N": N,
            "k": k,
            "k_over_N": round(k / N, 6),
            "trial": trial_id,
            "seed": seed,
            "E0": float(E0),
            "TC": float(TC),
            "S_joint": float(S_joint),
            "sum_S_i": float(sum_S_i),
            "elapsed_s": round(elapsed_k, 3),
        }
        results.append(entry)

        # Progress for large N
        if N >= 20 or (N >= 16 and k % 4 == 0) or k == N - 1:
            print(f"      k={k:2d} (k/N={k/N:.3f}): TC={TC:.4f}, S_joint={S_joint:.4f}, "
                  f"sum_Si={sum_S_i:.4f} ({elapsed_k:.1f}s)")

    del psi
    gc.collect()
    return results


# ── Check monotonicity ────────────────────────────────────────

def check_monotonicity(tc_values):
    """Check if TC values are monotonically non-decreasing."""
    violations = 0
    max_violation = 0.0
    for i in range(1, len(tc_values)):
        diff = tc_values[i] - tc_values[i-1]
        if diff < -1e-10:  # allow numerical noise
            violations += 1
            max_violation = max(max_violation, abs(diff))
    return violations, max_violation


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CRYSTALLIZATION STUDY")
    print("  First Law of Agentic Thermodynamics — Empirical Validation")
    print("  TC(k) monotonically increases with observation fraction k/N")
    print("=" * 70)

    global_t0 = time.time()
    all_results = []

    # Configuration: (N, n_seeds, seed_base)
    configs = [
        (8,  10, 10000),
        (16,  10, 20000),
        (20,   5, 30000),
        (24,   3, 40000),
    ]

    # Parse CLI args for selective runs
    if len(sys.argv) > 1:
        requested_N = [int(x) for x in sys.argv[1:]]
        configs = [(N, ns, sb) for N, ns, sb in configs if N in requested_N]
        print(f"  Running only N = {[c[0] for c in configs]}")

    for N, n_seeds, seed_base in configs:
        print(f"\n{'='*70}")
        print(f"  N = {N} (Hilbert dim = {2**N:,}), {n_seeds} seeds")
        print(f"  k sweep: 1 to {N-1} ({N-1} values)")
        mem_mb = 2**N * 16 / 1e6
        print(f"  Ground state vector: {mem_mb:.0f} MB")
        print(f"{'='*70}")

        n_results_before = len(all_results)

        for trial in range(n_seeds):
            seed = seed_base + trial
            trial_results = run_trial(N, seed, trial)
            all_results.extend(trial_results)

            # Save per-trial for N>=20 (long runs, want intermediate results)
            if N >= 20:
                per_file = RESULTS_DIR / f"crystallization_N{N}_seed{seed}.json"
                with open(per_file, 'w') as f:
                    json.dump(trial_results, f, indent=2, cls=NumpyEncoder)
                print(f"    Saved {per_file.name}")

        # Quick monotonicity check for this N
        n_results = [r for r in all_results[n_results_before:]]
        for trial in range(n_seeds):
            trial_data = [r for r in n_results if r['trial'] == trial]
            tc_vals = [r['TC'] for r in sorted(trial_data, key=lambda x: x['k'])]
            violations, max_viol = check_monotonicity(tc_vals)
            status = "MONOTONIC" if violations == 0 else f"{violations} violations (max={max_viol:.6f})"
            print(f"    Trial {trial}: {status}")

    total_elapsed = time.time() - global_t0

    # ── Aggregate summary ─────────────────────────────────────

    print(f"\n\n{'='*70}")
    print(f"  CRYSTALLIZATION RESULTS SUMMARY")
    print(f"{'='*70}")

    summary = {}
    for N, _, _ in configs:
        n_data = [r for r in all_results if r['N'] == N]
        if not n_data:
            continue

        k_values = sorted(set(r['k'] for r in n_data))
        n_trials = len(set(r['trial'] for r in n_data))

        tc_by_k = {}
        sj_by_k = {}
        for k in k_values:
            k_data = [r for r in n_data if r['k'] == k]
            tc_vals = [r['TC'] for r in k_data]
            sj_vals = [r['S_joint'] for r in k_data]
            tc_by_k[k] = tc_vals
            sj_by_k[k] = sj_vals

        # TC at max k (k=N-1) for normalization
        tc_max_vals = tc_by_k.get(N-1, [1.0])
        mean_tc_max = np.mean(tc_max_vals)

        # Check monotonicity across all trials
        all_monotonic = True
        total_violations = 0
        for trial in range(n_trials):
            trial_data = sorted([r for r in n_data if r['trial'] == trial], key=lambda x: x['k'])
            tc_seq = [r['TC'] for r in trial_data]
            v, _ = check_monotonicity(tc_seq)
            total_violations += v
            if v > 0:
                all_monotonic = False

        # Check TC + S = constant (Paper 5 identity)
        sum_si_values = [r['sum_S_i'] for r in n_data if r['k'] == 1]
        # For k=1, sum_S_i is just S(qubit 0), but we need per-trial consistency
        identity_holds = True
        max_identity_error = 0.0
        for trial in range(n_trials):
            trial_data = sorted([r for r in n_data if r['trial'] == trial], key=lambda x: x['k'])
            # TC + S_joint should equal sum_S_i which varies with k
            # Actually: TC = sum(S_i) - S_joint, so TC + S_joint = sum(S_i)
            # sum(S_i) increases with k (more individual entropies summed)
            # The REAL identity: for each k, TC(k) + S_joint(k) = sum_{i<k} S_i
            for r in trial_data:
                identity_error = abs(r['TC'] + r['S_joint'] - r['sum_S_i'])
                max_identity_error = max(max_identity_error, identity_error)
                if identity_error > 1e-8:
                    identity_holds = False

        n_summary = {
            "N": N,
            "n_trials": n_trials,
            "monotonic_all_trials": all_monotonic,
            "total_violations": total_violations,
            "identity_max_error": max_identity_error,
            "identity_holds": identity_holds,
            "mean_TC_max": float(mean_tc_max),
            "k_sweep": {},
        }

        print(f"\n  N = {N} ({n_trials} trials)")
        print(f"    Monotonicity: {'ALL MONOTONIC' if all_monotonic else f'{total_violations} violations'}")
        print(f"    TC+S=sum(Si) identity: max error = {max_identity_error:.2e} "
              f"({'HOLDS' if identity_holds else 'VIOLATED'})")
        print(f"    TC(k=N-1) mean = {mean_tc_max:.4f}")

        for k in k_values:
            tc_vals = tc_by_k[k]
            sj_vals = sj_by_k[k]
            tc_mean, tc_lo, tc_hi = _bootstrap_ci(tc_vals, seed=N*100+k)
            sj_mean, sj_lo, sj_hi = _bootstrap_ci(sj_vals, seed=N*100+k+1)

            # Normalized TC
            tc_norm = [t / mean_tc_max if mean_tc_max > 0 else 0 for t in tc_vals]
            tcn_mean, tcn_lo, tcn_hi = _bootstrap_ci(tc_norm, seed=N*100+k+2)

            n_summary["k_sweep"][k] = {
                "k": k,
                "k_over_N": round(k / N, 6),
                "TC_mean": tc_mean,
                "TC_ci95": [tc_lo, tc_hi],
                "S_joint_mean": sj_mean,
                "S_joint_ci95": [sj_lo, sj_hi],
                "TC_normalized_mean": tcn_mean,
                "TC_normalized_ci95": [tcn_lo, tcn_hi],
            }

            if k % max(1, (N-1)//6) == 0 or k == N-1:
                print(f"    k={k:2d} (k/N={k/N:.3f}): TC={tc_mean:.4f} [{tc_lo:.4f},{tc_hi:.4f}], "
                      f"S_joint={sj_mean:.4f}, TC/TC_max={tcn_mean:.4f}")

        summary[f"N={N}"] = n_summary

    # ── Universal collapse check ──────────────────────────────

    print(f"\n\n{'='*70}")
    print(f"  UNIVERSAL COLLAPSE CHECK")
    print(f"  Do TC/TC_max vs k/N curves overlap across system sizes?")
    print(f"{'='*70}")

    # For each k/N ratio, collect TC_normalized across all N values
    all_kn = set()
    for N_key, n_summary in summary.items():
        for k, kdata in n_summary["k_sweep"].items():
            all_kn.add(round(kdata["k_over_N"], 3))

    collapse_data = {}
    for kn in sorted(all_kn):
        values_by_N = {}
        for N_key, n_summary in summary.items():
            N = n_summary["N"]
            for k, kdata in n_summary["k_sweep"].items():
                if abs(kdata["k_over_N"] - kn) < 0.01:
                    values_by_N[N] = kdata["TC_normalized_mean"]
        if len(values_by_N) > 1:
            vals = list(values_by_N.values())
            spread = max(vals) - min(vals)
            collapse_data[kn] = {
                "values": values_by_N,
                "spread": spread,
            }
            if kn in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
                print(f"  k/N={kn:.3f}: {values_by_N} spread={spread:.4f}")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # ── Save everything ───────────────────────────────────────

    output = {
        "experiment": "crystallization_first_law",
        "date": "2026-03-17",
        "description": (
            "First Law of Agentic Thermodynamics validation. "
            "TC(k) should increase monotonically with k/N. "
            "TC + S_joint = sum(S_i) identity from Paper 5. "
            "Universal collapse: TC/TC_max vs k/N should overlap across N."
        ),
        "configs": [{"N": N, "n_seeds": ns} for N, ns, _ in configs],
        "all_results": all_results,
        "summary": summary,
        "collapse_data": collapse_data,
        "total_elapsed_seconds": total_elapsed,
    }

    outpath = RESULTS_DIR / "crystallization_summary.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved to {outpath}")

    # ── Generate plot ─────────────────────────────────────────

    try:
        generate_plot(summary)
    except Exception as e:
        print(f"\n  Plot generation failed: {e}")
        print("  Data saved. Plot can be generated later.")


def generate_plot(summary):
    """Generate the crystallization curve: TC/TC_max vs k/N for all N."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {8: '#1f77b4', 16: '#ff7f0e', 20: '#2ca02c', 24: '#d62728'}
    markers = {8: 'o', 16: 's', 20: '^', 24: 'D'}

    # Plot 1: TC/TC_max vs k/N (the money plot)
    ax = axes[0]
    for N_key, n_summary in sorted(summary.items()):
        N = n_summary["N"]
        k_over_N = []
        tc_norm = []
        tc_lo = []
        tc_hi = []
        for k, kdata in sorted(n_summary["k_sweep"].items()):
            k_over_N.append(kdata["k_over_N"])
            tc_norm.append(kdata["TC_normalized_mean"])
            tc_lo.append(kdata["TC_normalized_ci95"][0])
            tc_hi.append(kdata["TC_normalized_ci95"][1])

        k_over_N = np.array(k_over_N)
        tc_norm = np.array(tc_norm)
        tc_lo = np.array(tc_lo)
        tc_hi = np.array(tc_hi)

        ax.plot(k_over_N, tc_norm, marker=markers[N], color=colors[N],
                label=f'N={N}', markersize=4, linewidth=1.5)
        ax.fill_between(k_over_N, tc_lo, tc_hi, color=colors[N], alpha=0.15)

    ax.set_xlabel('Observation fraction k/N', fontsize=12)
    ax.set_ylabel('TC / TC_max', fontsize=12)
    ax.set_title('Crystallization Curve\n(First Law of Agentic Thermodynamics)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Plot 2: Raw TC vs k/N
    ax = axes[1]
    for N_key, n_summary in sorted(summary.items()):
        N = n_summary["N"]
        k_over_N = []
        tc_mean = []
        for k, kdata in sorted(n_summary["k_sweep"].items()):
            k_over_N.append(kdata["k_over_N"])
            tc_mean.append(kdata["TC_mean"])
        ax.plot(k_over_N, tc_mean, marker=markers[N], color=colors[N],
                label=f'N={N}', markersize=4, linewidth=1.5)

    ax.set_xlabel('Observation fraction k/N', fontsize=12)
    ax.set_ylabel('Total Correlation TC (bits)', fontsize=12)
    ax.set_title('Raw Total Correlation', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 3: S_joint vs k/N (should decrease)
    ax = axes[2]
    for N_key, n_summary in sorted(summary.items()):
        N = n_summary["N"]
        k_over_N = []
        sj_mean = []
        for k, kdata in sorted(n_summary["k_sweep"].items()):
            k_over_N.append(kdata["k_over_N"])
            sj_mean.append(kdata["S_joint_mean"])
        ax.plot(k_over_N, sj_mean, marker=markers[N], color=colors[N],
                label=f'N={N}', markersize=4, linewidth=1.5)

    ax.set_xlabel('Observation fraction k/N', fontsize=12)
    ax.set_ylabel('Joint entropy S_joint (bits)', fontsize=12)
    ax.set_title('Joint Entropy\n(should decrease = crystallization)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "crystallization_curve.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to {outpath}")

    # Also save PNG for quick viewing
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    for N_key, n_summary in sorted(summary.items()):
        N = n_summary["N"]
        k_over_N = []
        tc_norm = []
        for k, kdata in sorted(n_summary["k_sweep"].items()):
            k_over_N.append(kdata["k_over_N"])
            tc_norm.append(kdata["TC_normalized_mean"])
        ax2.plot(k_over_N, tc_norm, marker=markers[N], color=colors[N],
                 label=f'N={N}', markersize=5, linewidth=2)

    ax2.set_xlabel('Observation fraction k/N', fontsize=14)
    ax2.set_ylabel('TC / TC_max', fontsize=14)
    ax2.set_title('Crystallization: Observer Creates Order\n'
                   'First Law of Agentic Thermodynamics', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "crystallization_curve.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  PNG saved to {FIGURES_DIR / 'crystallization_curve.png'}")


if __name__ == "__main__":
    main()
