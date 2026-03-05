"""
Extended observables for PLC publication.

Three independent measures of emergent locality from partial observation:
1. Conditional Mutual Information I(A:B|C)
2. Tripartite Information I3(A:B:C)
3. Entanglement Spectrum Analysis

All built on top of quantum.py primitives.
Built by Opus Warrior, March 5 2026.
"""

import numpy as np
from itertools import combinations
from typing import Optional
import json
import time
from pathlib import Path

from .quantum import (
    random_all_to_all,
    ground_state, ground_state_gpu,
    partial_trace, von_neumann_entropy,
    mutual_information, mutual_information_matrix,
)
from .utils import NumpyEncoder

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _save_result(name: str, data: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


# ---------------------------------------------------------------
# Observable 1: Conditional Mutual Information I(A:B|C)
# ---------------------------------------------------------------

def conditional_mutual_information(psi: np.ndarray,
                                    sites_a: list[int],
                                    sites_b: list[int],
                                    sites_c: list[int],
                                    n_qubits: int) -> float:
    """
    Conditional mutual information I(A:B|C) = S(AC) + S(BC) - S(ABC) - S(C).

    Measures correlations between A and B that cannot be explained by C.
    For local systems: I(A:B|C) is small when C "separates" A from B.
    """
    ac = sorted(set(sites_a) | set(sites_c))
    bc = sorted(set(sites_b) | set(sites_c))
    abc = sorted(set(sites_a) | set(sites_b) | set(sites_c))
    c = sorted(set(sites_c))

    rho_ac = partial_trace(psi, ac, n_qubits)
    rho_bc = partial_trace(psi, bc, n_qubits)
    rho_abc = partial_trace(psi, abc, n_qubits)
    rho_c = partial_trace(psi, c, n_qubits)

    S_ac = von_neumann_entropy(rho_ac)
    S_bc = von_neumann_entropy(rho_bc)
    S_abc = von_neumann_entropy(rho_abc)
    S_c = von_neumann_entropy(rho_c)

    return S_ac + S_bc - S_abc - S_c


def cmi_separation_structure(psi: np.ndarray,
                              observer_sites: list[int],
                              n_qubits: int) -> dict:
    """
    For all triples (A, B, C) within observer's subset, compute I(A:B|C).

    Measures whether there is "separation structure": do some triples
    have much smaller CMI than others? If yes, C acts as a separator,
    which is a hallmark of locality.

    Returns statistics on the distribution of CMI values.
    """
    sites = sorted(observer_sites)
    n = len(sites)
    if n < 3:
        return {"error": "need at least 3 sites"}

    cmi_values = []
    triples_data = []

    # All ordered triples (A, B, C) where A, B, C are single qubits
    for a_idx, b_idx, c_idx in combinations(range(n), 3):
        a = [sites[a_idx]]
        b = [sites[b_idx]]
        c = [sites[c_idx]]

        cmi = conditional_mutual_information(psi, a, b, c, n_qubits)
        cmi_values.append(cmi)
        triples_data.append({
            "A": a[0], "B": b[0], "C": c[0],
            "I_AB_given_C": float(cmi),
        })

    cmi_arr = np.array(cmi_values)

    # Key metric: coefficient of variation of CMI
    # High CV = some triples have very different CMI = separation structure
    mean_cmi = float(np.mean(cmi_arr))
    std_cmi = float(np.std(cmi_arr))
    cv_cmi = std_cmi / mean_cmi if mean_cmi > 1e-14 else 0.0

    # Also: what fraction of CMI values are "small" (below median/2)?
    median_cmi = float(np.median(cmi_arr))
    frac_small = float(np.mean(cmi_arr < median_cmi / 2)) if median_cmi > 1e-14 else 0.0

    # Range ratio: max/min indicates separation dynamic range
    min_cmi = float(np.min(cmi_arr[cmi_arr > 1e-14])) if np.any(cmi_arr > 1e-14) else 0.0
    max_cmi = float(np.max(cmi_arr))
    range_ratio = max_cmi / min_cmi if min_cmi > 1e-14 else float('inf')

    return {
        "n_triples": len(cmi_values),
        "mean_cmi": mean_cmi,
        "std_cmi": std_cmi,
        "cv_cmi": float(cv_cmi),
        "median_cmi": median_cmi,
        "min_cmi": min_cmi,
        "max_cmi": max_cmi,
        "range_ratio": float(range_ratio) if np.isfinite(range_ratio) else 999.0,
        "frac_small": frac_small,
        "triples": triples_data,
    }


# ---------------------------------------------------------------
# Observable 2: Tripartite Information I3(A:B:C)
# ---------------------------------------------------------------

def tripartite_information(psi: np.ndarray,
                            sites_a: list[int],
                            sites_b: list[int],
                            sites_c: list[int],
                            n_qubits: int) -> float:
    """
    Tripartite information I3(A:B:C) = I(A:B) + I(A:C) - I(A:BC).

    Equivalently: I3 = S(A) + S(B) + S(C) - S(AB) - S(AC) - S(BC) + S(ABC).

    Sign interpretation:
    - Negative I3 = multipartite entanglement (scrambling, holographic)
    - Positive I3 = bipartite structure (locality-like, correlations are pairwise)

    Key PLC test: does I3 become more positive for more partial observers?
    That would mean partiality pushes correlations toward pairwise (local) structure.
    """
    a = sorted(set(sites_a))
    b = sorted(set(sites_b))
    c = sorted(set(sites_c))
    ab = sorted(set(a) | set(b))
    ac = sorted(set(a) | set(c))
    bc = sorted(set(b) | set(c))
    abc = sorted(set(a) | set(b) | set(c))

    S_a = von_neumann_entropy(partial_trace(psi, a, n_qubits))
    S_b = von_neumann_entropy(partial_trace(psi, b, n_qubits))
    S_c = von_neumann_entropy(partial_trace(psi, c, n_qubits))
    S_ab = von_neumann_entropy(partial_trace(psi, ab, n_qubits))
    S_ac = von_neumann_entropy(partial_trace(psi, ac, n_qubits))
    S_bc = von_neumann_entropy(partial_trace(psi, bc, n_qubits))
    S_abc = von_neumann_entropy(partial_trace(psi, abc, n_qubits))

    return S_a + S_b + S_c - S_ab - S_ac - S_bc + S_abc


def tripartite_structure(psi: np.ndarray,
                          observer_sites: list[int],
                          n_qubits: int) -> dict:
    """
    Compute I3 for all triples of single qubits within the observer's subset.

    Key metrics:
    - Mean I3: positive = bipartite dominance = locality-like
    - Fraction positive: what share of triples show local structure
    - Comparison with full system I3 to measure partiality effect
    """
    sites = sorted(observer_sites)
    n = len(sites)
    if n < 3:
        return {"error": "need at least 3 sites"}

    i3_values = []
    triples_data = []

    for a_idx, b_idx, c_idx in combinations(range(n), 3):
        a = [sites[a_idx]]
        b = [sites[b_idx]]
        c = [sites[c_idx]]

        i3 = tripartite_information(psi, a, b, c, n_qubits)
        i3_values.append(i3)
        triples_data.append({
            "A": a[0], "B": b[0], "C": c[0],
            "I3": float(i3),
        })

    i3_arr = np.array(i3_values)

    mean_i3 = float(np.mean(i3_arr))
    std_i3 = float(np.std(i3_arr))
    frac_positive = float(np.mean(i3_arr > 0))
    frac_negative = float(np.mean(i3_arr < 0))
    median_i3 = float(np.median(i3_arr))

    return {
        "n_triples": len(i3_values),
        "mean_I3": mean_i3,
        "std_I3": std_i3,
        "median_I3": median_i3,
        "min_I3": float(np.min(i3_arr)),
        "max_I3": float(np.max(i3_arr)),
        "frac_positive": frac_positive,
        "frac_negative": frac_negative,
        "bipartite_dominance": frac_positive > 0.5,
        "triples": triples_data,
    }


# ---------------------------------------------------------------
# Observable 3: Entanglement Spectrum
# ---------------------------------------------------------------

def entanglement_spectrum(psi: np.ndarray,
                           observer_sites: list[int],
                           n_qubits: int) -> dict:
    """
    Compute the entanglement spectrum of the observer's reduced density matrix.

    The entanglement spectrum {xi_i} is defined by rho = exp(-H_ent) where
    xi_i = -log(lambda_i) and lambda_i are eigenvalues of rho.

    For local systems (Li-Haldane conjecture), the entanglement spectrum
    has specific structure: level spacing follows statistics connected to
    the universality class of the boundary theory.

    Key metrics:
    - Spectral entropy: H_spec = -sum p_i log p_i where p_i = lambda_i / sum(lambda)
      (this is just von Neumann entropy, but we also compute it from the spectrum)
    - Participation ratio: PR = 1 / sum(p_i^2)
      High PR = flat spectrum (scrambled). Low PR = structured (local-like).
    - Spectral gap: gap between largest and second-largest eigenvalue
      Large gap = dominant sector = local-like structure.
    - Level spacing ratio: r = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
      where s_i = xi_{i+1} - xi_i. For random matrices r ~ 0.53 (GUE),
      for integrable/local systems r ~ 0.39 (Poisson).
    """
    sites = sorted(observer_sites)
    rho = partial_trace(psi, sites, n_qubits)

    # Eigenvalues of the reduced density matrix
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    # Filter numerical noise
    eigenvalues_clean = eigenvalues[eigenvalues > 1e-14]

    if len(eigenvalues_clean) == 0:
        return {"error": "degenerate density matrix"}

    # Normalize (should already sum to 1 but enforce)
    eigenvalues_clean = eigenvalues_clean / np.sum(eigenvalues_clean)

    # Entanglement energies: xi = -log(lambda)
    xi = -np.log(eigenvalues_clean)

    # Von Neumann entropy (spectral entropy)
    S_vn = float(-np.sum(eigenvalues_clean * np.log2(eigenvalues_clean)))

    # Participation ratio
    PR = float(1.0 / np.sum(eigenvalues_clean ** 2))

    # Normalized PR: PR / d where d = dimension of reduced space
    dim_obs = 2 ** len(sites)
    PR_normalized = PR / dim_obs

    # Spectral gap: ratio of two largest eigenvalues
    if len(eigenvalues_clean) >= 2:
        spectral_gap = float(eigenvalues_clean[0] - eigenvalues_clean[1])
        gap_ratio = float(eigenvalues_clean[0] / eigenvalues_clean[1])
    else:
        spectral_gap = 0.0
        gap_ratio = 1.0

    # Level spacing ratio of entanglement energies
    level_spacing_ratio = _level_spacing_ratio(xi)

    # Maximum entropy for this dimension (fully mixed state)
    S_max = np.log2(dim_obs)
    entropy_fraction = S_vn / S_max if S_max > 0 else 0.0

    return {
        "n_sites": len(sites),
        "dim_reduced": dim_obs,
        "n_nonzero_eigenvalues": len(eigenvalues_clean),
        "eigenvalues": eigenvalues_clean.tolist(),
        "entanglement_energies": xi.tolist(),
        "von_neumann_entropy": S_vn,
        "max_entropy": float(S_max),
        "entropy_fraction": float(entropy_fraction),
        "participation_ratio": PR,
        "PR_normalized": float(PR_normalized),
        "spectral_gap": spectral_gap,
        "gap_ratio": gap_ratio,
        "level_spacing_ratio": float(level_spacing_ratio),
    }


def _level_spacing_ratio(energies: np.ndarray) -> float:
    """
    Average level spacing ratio <r> for entanglement energies.

    r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    where s_i = E_{i+1} - E_i.

    GUE (random, scrambled): <r> ~ 0.5996
    GOE (time-reversal symmetric random): <r> ~ 0.5307
    Poisson (integrable/local): <r> ~ 0.3863

    Values closer to Poisson indicate more local-like structure.
    """
    if len(energies) < 3:
        return 0.0

    sorted_e = np.sort(energies)
    spacings = np.diff(sorted_e)

    # Filter zero spacings
    spacings = spacings[spacings > 1e-14]

    if len(spacings) < 2:
        return 0.0

    ratios = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i + 1]
        r = min(s1, s2) / max(s1, s2)
        ratios.append(r)

    return float(np.mean(ratios)) if ratios else 0.0


# ---------------------------------------------------------------
# Sweep: run all observables across k/N ratios
# ---------------------------------------------------------------

def run_observable_sweep(n_qubits: int, n_trials: int,
                          k_over_n_values: list[float],
                          use_gpu: bool = True, seed_base: int = 5000) -> dict:
    """
    Run all three observables across multiple k/N ratios.

    For each trial:
    - Generate random all-to-all Hamiltonian (no spatial structure)
    - Find ground state
    - For each k/N ratio, pick random observer subsets and compute:
      1. CMI separation structure
      2. Tripartite information structure
      3. Entanglement spectrum
    - Also compute full-system baselines

    Returns comprehensive results dict.
    """
    diag_fn = ground_state_gpu if use_gpu else ground_state

    print(f"\n{'='*70}")
    print(f"  OBSERVABLE SWEEP: N={n_qubits}, {n_trials} trials")
    print(f"  k/N ratios: {k_over_n_values}")
    print(f"  Observables: CMI, I3, Entanglement Spectrum")
    print(f"{'='*70}\n")

    t0 = time.time()

    # Results containers
    cmi_results = []
    i3_results = []
    spectrum_results = []

    for trial in range(n_trials):
        seed = seed_base + trial
        H, couplings = random_all_to_all(n_qubits, seed=seed)
        E0, psi = diag_fn(H)

        # Full system baselines
        all_sites = list(range(n_qubits))

        # Full system I3
        full_i3 = tripartite_structure(psi, all_sites, n_qubits)
        cmi_results.append({
            "trial": trial, "N": n_qubits,
            "k": n_qubits, "k_over_N": 1.0,
            "label": "full",
            **{k: v for k, v in cmi_separation_structure(psi, all_sites, n_qubits).items()
               if k != "triples"},
        })
        i3_results.append({
            "trial": trial, "N": n_qubits,
            "k": n_qubits, "k_over_N": 1.0,
            "label": "full",
            **{k: v for k, v in full_i3.items() if k != "triples"},
        })
        spectrum_results.append({
            "trial": trial, "N": n_qubits,
            "k": n_qubits, "k_over_N": 1.0,
            "label": "full",
            **{k: v for k, v in entanglement_spectrum(psi, all_sites, n_qubits).items()
               if k != "eigenvalues" and k != "entanglement_energies"},
        })

        # Observer subsets at each k/N ratio
        for k_ratio in k_over_n_values:
            k = max(3, round(k_ratio * n_qubits))
            if k >= n_qubits:
                continue

            # Sample multiple random observer subsets
            all_subsets = list(combinations(range(n_qubits), k))
            rng = np.random.default_rng(seed * 100 + int(k_ratio * 1000))
            n_sample = min(len(all_subsets), 5)
            indices = rng.choice(len(all_subsets), n_sample, replace=False)

            for idx in indices:
                subset = list(all_subsets[idx])

                # Observable 1: CMI
                cmi_data = cmi_separation_structure(psi, subset, n_qubits)
                cmi_results.append({
                    "trial": trial, "N": n_qubits,
                    "k": k, "k_over_N": float(k / n_qubits),
                    "label": f"k={k}",
                    **{key: val for key, val in cmi_data.items() if key != "triples"},
                })

                # Observable 2: Tripartite Information
                i3_data = tripartite_structure(psi, subset, n_qubits)
                i3_results.append({
                    "trial": trial, "N": n_qubits,
                    "k": k, "k_over_N": float(k / n_qubits),
                    "label": f"k={k}",
                    **{key: val for key, val in i3_data.items() if key != "triples"},
                })

                # Observable 3: Entanglement Spectrum
                spec_data = entanglement_spectrum(psi, subset, n_qubits)
                spectrum_results.append({
                    "trial": trial, "N": n_qubits,
                    "k": k, "k_over_N": float(k / n_qubits),
                    "label": f"k={k}",
                    **{key: val for key, val in spec_data.items()
                       if key != "eigenvalues" and key != "entanglement_energies"},
                })

        if (trial + 1) % 5 == 0 or trial == 0:
            elapsed_so_far = time.time() - t0
            rate = (trial + 1) / elapsed_so_far
            eta = (n_trials - trial - 1) / rate if rate > 0 else 0
            print(f"  Trial {trial+1}/{n_trials} "
                  f"({elapsed_so_far:.1f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\n  Sweep complete in {elapsed:.1f}s")

    return {
        "cmi": cmi_results,
        "i3": i3_results,
        "spectrum": spectrum_results,
        "elapsed_seconds": elapsed,
    }


def analyze_results(sweep: dict) -> dict:
    """
    Aggregate and analyze sweep results. Produce summary statistics
    grouped by k/N ratio for each observable.
    """
    analysis = {}

    # ---- CMI Analysis ----
    cmi = sweep["cmi"]
    cmi_by_ratio = {}
    for entry in cmi:
        ratio = round(entry["k_over_N"], 2)
        cmi_by_ratio.setdefault(ratio, []).append(entry)

    cmi_summary = {}
    for ratio in sorted(cmi_by_ratio.keys()):
        entries = cmi_by_ratio[ratio]
        cv_values = [e["cv_cmi"] for e in entries if "cv_cmi" in e]
        range_values = [e["range_ratio"] for e in entries
                        if "range_ratio" in e and e["range_ratio"] < 999]
        mean_cmi_values = [e["mean_cmi"] for e in entries if "mean_cmi" in e]

        cmi_summary[str(ratio)] = {
            "n_samples": len(entries),
            "mean_cv": float(np.mean(cv_values)) if cv_values else 0,
            "std_cv": float(np.std(cv_values)) if cv_values else 0,
            "mean_range_ratio": float(np.mean(range_values)) if range_values else 0,
            "mean_cmi": float(np.mean(mean_cmi_values)) if mean_cmi_values else 0,
        }
    analysis["cmi_summary"] = cmi_summary

    # ---- I3 Analysis ----
    i3 = sweep["i3"]
    i3_by_ratio = {}
    for entry in i3:
        ratio = round(entry["k_over_N"], 2)
        i3_by_ratio.setdefault(ratio, []).append(entry)

    i3_summary = {}
    for ratio in sorted(i3_by_ratio.keys()):
        entries = i3_by_ratio[ratio]
        mean_i3_vals = [e["mean_I3"] for e in entries if "mean_I3" in e]
        frac_pos_vals = [e["frac_positive"] for e in entries if "frac_positive" in e]

        i3_summary[str(ratio)] = {
            "n_samples": len(entries),
            "mean_I3": float(np.mean(mean_i3_vals)) if mean_i3_vals else 0,
            "std_I3": float(np.std(mean_i3_vals)) if mean_i3_vals else 0,
            "mean_frac_positive": float(np.mean(frac_pos_vals)) if frac_pos_vals else 0,
            "std_frac_positive": float(np.std(frac_pos_vals)) if frac_pos_vals else 0,
        }
    analysis["i3_summary"] = i3_summary

    # ---- Spectrum Analysis ----
    spec = sweep["spectrum"]
    spec_by_ratio = {}
    for entry in spec:
        ratio = round(entry["k_over_N"], 2)
        spec_by_ratio.setdefault(ratio, []).append(entry)

    spec_summary = {}
    for ratio in sorted(spec_by_ratio.keys()):
        entries = spec_by_ratio[ratio]
        pr_norm = [e["PR_normalized"] for e in entries if "PR_normalized" in e]
        ent_frac = [e["entropy_fraction"] for e in entries if "entropy_fraction" in e]
        gap_ratios = [e["gap_ratio"] for e in entries if "gap_ratio" in e]
        lsr = [e["level_spacing_ratio"] for e in entries if "level_spacing_ratio" in e]

        spec_summary[str(ratio)] = {
            "n_samples": len(entries),
            "mean_PR_normalized": float(np.mean(pr_norm)) if pr_norm else 0,
            "std_PR_normalized": float(np.std(pr_norm)) if pr_norm else 0,
            "mean_entropy_fraction": float(np.mean(ent_frac)) if ent_frac else 0,
            "std_entropy_fraction": float(np.std(ent_frac)) if ent_frac else 0,
            "mean_gap_ratio": float(np.mean(gap_ratios)) if gap_ratios else 0,
            "mean_level_spacing_ratio": float(np.mean(lsr)) if lsr else 0,
        }
    analysis["spectrum_summary"] = spec_summary

    return analysis


def print_analysis(analysis: dict, n_qubits: int):
    """Print formatted analysis of all three observables."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS: N = {n_qubits}")
    print(f"{'='*70}")

    # CMI
    print(f"\n  --- Observable 1: Conditional Mutual Information ---")
    print(f"  High CV = strong separation structure = locality-like")
    print(f"  {'k/N':>6}  {'CV(CMI)':>12}  {'range_ratio':>14}  {'mean_CMI':>12}  {'n':>4}")
    for ratio, stats in sorted(analysis["cmi_summary"].items(), key=lambda x: float(x[0])):
        print(f"  {ratio:>6}  {stats['mean_cv']:>12.4f}  "
              f"{stats['mean_range_ratio']:>14.2f}  "
              f"{stats['mean_cmi']:>12.6f}  {stats['n_samples']:>4}")

    # I3
    print(f"\n  --- Observable 2: Tripartite Information ---")
    print(f"  Positive I3 = bipartite structure = locality-like")
    print(f"  Negative I3 = multipartite entanglement = scrambled")
    print(f"  {'k/N':>6}  {'mean_I3':>12}  {'frac_pos':>12}  {'n':>4}")
    for ratio, stats in sorted(analysis["i3_summary"].items(), key=lambda x: float(x[0])):
        sign = "+" if stats["mean_I3"] > 0 else "-"
        label = "LOCAL" if stats["mean_frac_positive"] > 0.5 else "SCRAMBLED"
        print(f"  {ratio:>6}  {sign}{abs(stats['mean_I3']):>11.6f}  "
              f"{stats['mean_frac_positive']:>12.3f}  {stats['n_samples']:>4}  [{label}]")

    # Spectrum
    print(f"\n  --- Observable 3: Entanglement Spectrum ---")
    print(f"  Low PR_norm = structured spectrum = local-like")
    print(f"  Low entropy_frac = far from maximally mixed = structured")
    print(f"  Level spacing: Poisson=0.386 (local), GUE=0.600 (scrambled)")
    print(f"  {'k/N':>6}  {'PR_norm':>10}  {'S/S_max':>10}  {'gap_ratio':>12}  {'<r>':>8}  {'n':>4}")
    for ratio, stats in sorted(analysis["spectrum_summary"].items(), key=lambda x: float(x[0])):
        print(f"  {ratio:>6}  {stats['mean_PR_normalized']:>10.4f}  "
              f"{stats['mean_entropy_fraction']:>10.4f}  "
              f"{stats['mean_gap_ratio']:>12.3f}  "
              f"{stats['mean_level_spacing_ratio']:>8.4f}  {stats['n_samples']:>4}")
