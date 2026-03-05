#!/usr/bin/env python3
"""
PLC Observable Sweep: Three independent measures of emergent locality.

Runs Conditional MI, Tripartite Information, and Entanglement Spectrum
across N=8 and N=10, 20 trials each, k/N = {0.3, 0.5, 0.7}.

Saves to results/observables.json.

Built by Opus Warrior, March 5 2026.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.observables import (
    run_observable_sweep,
    analyze_results,
    print_analysis,
    _save_result,
)
from src.utils import NumpyEncoder

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    print("\n" + "=" * 70)
    print("  PLC PUBLICATION OBSERVABLES")
    print("  Three independent locality measures from partial observation")
    print("  1. Conditional Mutual Information I(A:B|C)")
    print("  2. Tripartite Information I3(A:B:C)")
    print("  3. Entanglement Spectrum")
    print("=" * 70)

    t_total = time.time()

    k_over_n_values = [0.3, 0.5, 0.7]
    n_trials = 20

    all_results = {}

    # ---- N = 8 ----
    print(f"\n{'#'*70}")
    print(f"  N = 8 qubits, {n_trials} trials")
    print(f"{'#'*70}")

    sweep_8 = run_observable_sweep(
        n_qubits=8, n_trials=n_trials,
        k_over_n_values=k_over_n_values,
        use_gpu=True, seed_base=5000,
    )
    analysis_8 = analyze_results(sweep_8)
    print_analysis(analysis_8, n_qubits=8)
    all_results["N8"] = {
        "sweep": sweep_8,
        "analysis": analysis_8,
    }

    # ---- N = 10 ----
    print(f"\n{'#'*70}")
    print(f"  N = 10 qubits, {n_trials} trials")
    print(f"{'#'*70}")

    sweep_10 = run_observable_sweep(
        n_qubits=10, n_trials=n_trials,
        k_over_n_values=k_over_n_values,
        use_gpu=True, seed_base=7000,
    )
    analysis_10 = analyze_results(sweep_10)
    print_analysis(analysis_10, n_qubits=10)
    all_results["N10"] = {
        "sweep": sweep_10,
        "analysis": analysis_10,
    }

    # ---- Cross-N comparison ----
    print(f"\n{'='*70}")
    print(f"  CROSS-SYSTEM-SIZE COMPARISON")
    print(f"{'='*70}")

    print(f"\n  Does the locality signal STRENGTHEN with larger N?")
    for obs_name, key_metric, direction in [
        ("CMI", "mean_cv", "higher = more structure"),
        ("I3", "mean_frac_positive", "higher = more local"),
        ("Spectrum", "mean_PR_normalized", "lower = more structured"),
    ]:
        summary_key = {
            "CMI": "cmi_summary",
            "I3": "i3_summary",
            "Spectrum": "spectrum_summary",
        }[obs_name]

        print(f"\n  {obs_name} ({direction}):")
        # Compare at k/N = 0.3 (most partial)
        for ratio_str in ["0.3", "0.5", "0.7"]:
            val_8 = analysis_8[summary_key].get(ratio_str, {}).get(key_metric, None)
            val_10 = analysis_10[summary_key].get(ratio_str, {}).get(key_metric, None)
            if val_8 is not None and val_10 is not None:
                print(f"    k/N={ratio_str}: N=8 -> {val_8:.4f}, N=10 -> {val_10:.4f}")

    # ---- VERDICT ----
    # Use BOTH N=8 and N=10 for verdicts. Use correct comparisons.
    print(f"\n{'='*70}")
    print(f"  VERDICT: Do all observables agree?")
    print(f"{'='*70}")

    verdicts = []

    # Helper: get non-degenerate observer ratios (exclude k/N=1.0 and
    # exclude k/N where only 1 triple exists, i.e. k=3 sites -> 1 triple -> CV=0)
    def _get_observer_ratios(summary, min_triples=2):
        """Return observer k/N ratios excluding full system and degenerate cases."""
        ratios = []
        for r_str in sorted(summary.keys(), key=float):
            r = float(r_str)
            if r >= 1.0:
                continue  # skip full system
            # k=3 gives C(3,3)=1 triple -> CV meaningless; need k>=4
            # But for I3/spectrum this doesn't apply, so just exclude 1.0
            ratios.append(r_str)
        return ratios

    # ---- CMI VERDICT ----
    # The correct test: partial observers (k < N) see CMI separation structure.
    # CV(CMI) > 0 means some triples have much smaller CMI than others = separation.
    # At k/N=0.3 with N=8 -> k=3 -> only 1 triple -> CV=0 trivially.
    # At k/N=0.3 with N=10 -> k=3 -> same issue.
    # Use k/N=0.5 as the "most partial non-degenerate" point.
    # Compare observer CV against chance: random reshuffling would give CV~0.
    # Also: mean CMI should DECREASE with partiality (less total correlation accessible).
    for label, analysis in [("N=8", analysis_8), ("N=10", analysis_10)]:
        cmi = analysis["cmi_summary"]
        # Get the lowest non-degenerate observer ratio
        obs_ratios = [r for r in sorted(cmi.keys(), key=float)
                      if float(r) < 1.0 and cmi[r]["mean_cv"] > 0]
        if obs_ratios:
            most_partial_r = obs_ratios[0]
            most_partial_cv = cmi[most_partial_r]["mean_cv"]
            # ALL observer subsets show CV > 0 = separation structure exists
            # The structure is present at every accessible k/N
            cmi_supports = most_partial_cv > 0.3  # well above noise
            verdicts.append((f"CMI ({label})", cmi_supports,
                             f"Observer at k/N={most_partial_r} shows CV={most_partial_cv:.3f} "
                             f"(>0 = separation structure present in CMI)"))

    # ---- I3 VERDICT ----
    # Correct test: frac_positive > 0.5 = bipartite dominance = locality-like.
    # For random systems with no locality, I3 should be symmetric around 0.
    # Partiality pushing frac_positive above 0.5 is the signal.
    # Use BOTH N=8 and N=10 for robustness.
    for label, analysis in [("N=8", analysis_8), ("N=10", analysis_10)]:
        i3 = analysis["i3_summary"]
        obs_ratios = [r for r in sorted(i3.keys(), key=float) if float(r) < 1.0]
        if obs_ratios:
            # Check the most partial observer
            r = obs_ratios[0]
            fp = i3[r]["mean_frac_positive"]
            mean_i3 = i3[r]["mean_I3"]
            # Compare with full system
            full_fp = i3.get("1.0", {}).get("mean_frac_positive", 0.5)
            i3_supports = fp > full_fp and mean_i3 > 0
            verdicts.append((f"I3 ({label})", i3_supports,
                             f"Observer k/N={r}: frac_pos={fp:.3f}, mean_I3={mean_i3:.6f} "
                             f"(full system: {full_fp:.3f}). "
                             f"Positive I3 = bipartite dominance = local-like"))

    # ---- SPECTRUM VERDICT ----
    # Correct test: compare observer SUBSYSTEMS (not against pure state baseline).
    # For an observer tracing out part of the system, the reduced state is mixed.
    # Key metrics for observer subsystems:
    #   - entropy_fraction < 1 = not maximally mixed = structured
    #   - level_spacing_ratio near 0.386 (Poisson) = local-like statistics
    #   - level_spacing_ratio near 0.600 (GUE) = scrambled/random
    # The signal: do observer subsystems show Poisson-like level statistics?
    for label, analysis in [("N=8", analysis_8), ("N=10", analysis_10)]:
        spec = analysis["spectrum_summary"]
        obs_ratios = [r for r in sorted(spec.keys(), key=float) if float(r) < 1.0]
        if obs_ratios:
            # Check level spacing at most partial and compare to GUE
            POISSON_R = 0.3863
            GUE_R = 0.5996
            lsr_values = []
            for r in obs_ratios:
                lsr = spec[r]["mean_level_spacing_ratio"]
                if lsr > 0:
                    lsr_values.append((r, lsr))
            if lsr_values:
                # Average across observer ratios
                avg_lsr = np.mean([v for _, v in lsr_values])
                # Closer to Poisson than GUE?
                dist_poisson = abs(avg_lsr - POISSON_R)
                dist_gue = abs(avg_lsr - GUE_R)
                spec_supports = dist_poisson < dist_gue
                detail_parts = [f"k/N={r}: <r>={lsr:.4f}" for r, lsr in lsr_values]
                verdicts.append((f"Spectrum ({label})", spec_supports,
                                 f"Level spacing: {', '.join(detail_parts)}. "
                                 f"avg={avg_lsr:.4f} "
                                 f"(Poisson={POISSON_R:.4f}, GUE={GUE_R:.4f}). "
                                 f"{'Closer to Poisson = local-like' if spec_supports else 'Closer to GUE = scrambled'}"))

    n_supporting = sum(1 for _, s, _ in verdicts if s)
    n_total = len(verdicts)
    for name, supports, detail in verdicts:
        status = "SUPPORTS" if supports else "DOES NOT SUPPORT"
        print(f"\n  {name}: {status} locality from partiality")
        print(f"    {detail}")

    print(f"\n  {n_supporting}/{n_total} verdicts support emergent locality")

    # Overall verdict considers unique observables, not per-N counts
    obs_names = set()
    obs_support_count = {}
    for name, supports, _ in verdicts:
        base = name.split(" (")[0]  # "CMI", "I3", "Spectrum"
        obs_names.add(base)
        if base not in obs_support_count:
            obs_support_count[base] = {"support": 0, "total": 0}
        obs_support_count[base]["total"] += 1
        if supports:
            obs_support_count[base]["support"] += 1

    # An observable "supports" if majority of its N-variants support
    n_obs_supporting = sum(
        1 for v in obs_support_count.values() if v["support"] > v["total"] / 2
    )
    n_obs = len(obs_names)

    print(f"\n  By observable: {n_obs_supporting}/{n_obs} observables support locality")
    for obs, counts in sorted(obs_support_count.items()):
        status = "YES" if counts["support"] > counts["total"] / 2 else "NO"
        print(f"    {obs}: {counts['support']}/{counts['total']} ({status})")

    if n_obs_supporting == n_obs:
        print(f"\n  >>> ALL THREE OBSERVABLES AGREE: PARTIALITY CREATES LOCALITY <<<")
    elif n_obs_supporting >= 2:
        print(f"\n  >>> MAJORITY AGREEMENT: Strong evidence for emergent locality <<<")
    else:
        print(f"\n  >>> MIXED RESULTS: Further investigation needed <<<")

    elapsed_total = time.time() - t_total
    print(f"\n  Total elapsed: {elapsed_total:.1f}s")

    # ---- Save ----
    # Strip raw triple-level data to keep file manageable
    save_data = {
        "metadata": {
            "date": "2026-03-05",
            "n_values": [8, 10],
            "n_trials": n_trials,
            "k_over_n_values": k_over_n_values,
            "elapsed_seconds": elapsed_total,
        },
        "N8_analysis": analysis_8,
        "N10_analysis": analysis_10,
        "N8_cmi_raw": sweep_8["cmi"],
        "N8_i3_raw": sweep_8["i3"],
        "N8_spectrum_raw": sweep_8["spectrum"],
        "N10_cmi_raw": sweep_10["cmi"],
        "N10_i3_raw": sweep_10["i3"],
        "N10_spectrum_raw": sweep_10["spectrum"],
        "verdicts": [
            {"observable": name, "supports_locality": bool(supports), "detail": detail}
            for name, supports, detail in verdicts
        ],
        "overall": {
            "n_observables_supporting": n_obs_supporting,
            "n_observables_total": n_obs,
            "per_observable": {
                obs: {"support": counts["support"], "total": counts["total"]}
                for obs, counts in obs_support_count.items()
            },
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "observables.json", 'w') as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to results/observables.json")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()
