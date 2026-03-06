"""
PLC Paper 2 Experiments: Emergent Curvature from Partial Observation.

Four experiments demonstrating that discrete Ricci curvature on the
mutual-information graph depends on observer partiality:

1. Curvature vs Partiality    - scalar curvature changes with subsystem size k
2. Curvature vs Entanglement  - curvature tracks entanglement across XXZ anisotropy
3. Local vs Non-Local          - chain geometry yields negative (AdS-like) curvature
4. Perspective-Dependent       - different k-subsystems see different curvature

Depends on:
    curvature.py  (ollivier_ricci, forman_ricci, mi_to_graph)
    quantum.py    (extended: build_xxz_hamiltonian, build_local_hamiltonian, entanglement_entropy)

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
    mutual_information_matrix,
    partial_trace, von_neumann_entropy,
)
from .curvature import ollivier_ricci, forman_ricci, mi_to_graph
from .utils import NumpyEncoder

# Extended quantum functions from hamiltonians module
try:
    from .hamiltonians import (
        build_xxz_hamiltonian,
        build_local_hamiltonian,
        entanglement_entropy,
    )
except ImportError:
    # Stubs for development
    build_xxz_hamiltonian = None
    build_local_hamiltonian = None
    entanglement_entropy = None


RESULTS_DIR = Path(__file__).parent.parent / "results"


def _save_result(name: str, data: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def _curvature_stats(edge_curvatures: dict) -> dict:
    """Summary statistics for a dict of edge curvatures."""
    if not edge_curvatures:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "n_positive": 0, "n_negative": 0, "n_edges": 0}
    vals = np.array(list(edge_curvatures.values()))
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n_positive": int(np.sum(vals > 0)),
        "n_negative": int(np.sum(vals < 0)),
        "n_edges": len(vals),
    }


# -----------------------------------------------------------------
# Experiment 1: Curvature vs Partiality
# -----------------------------------------------------------------

def experiment_curvature_vs_partiality(
    n_qubits: int = 8,
    n_samples: int = 10,
    use_gpu: bool = True,
    alpha: float = 0.5,
) -> dict:
    """
    How does scalar curvature change with observer subsystem size k?

    For a fixed random all-to-all Hamiltonian ground state, compute
    the MI graph restricted to k observed qubits and measure its
    Ollivier-Ricci scalar curvature.  Sweep k from 3 to n_qubits.

    Averages over n_samples random Hamiltonians.
    For each (sample, k), averages over up to 15 random k-subsets.

    Returns list of records: (k, scalar_curvature_mean, scalar_curvature_std,
    curvature_stats, k_over_N).
    """
    print(f"\n{'='*60}")
    print(f"  CURVATURE EXP 1: Curvature vs Partiality")
    print(f"  N = {n_qubits}, {n_samples} samples, alpha = {alpha}")
    print(f"{'='*60}\n")

    t0 = time.time()
    diag_fn = ground_state_gpu if use_gpu else ground_state

    # Collect per-k aggregated data
    k_records = {}  # k -> list of scalar curvatures

    for s in range(n_samples):
        seed = 5000 + s
        H, _ = random_all_to_all(n_qubits, seed=seed)
        E0, psi = diag_fn(H)

        if s % max(1, n_samples // 5) == 0:
            print(f"  Sample {s+1}/{n_samples}  (E0 = {E0:.4f})")

        # Full MI matrix once per sample
        MI_full = mutual_information_matrix(psi, n_qubits)

        # Full system curvature (k = N)
        oll_full = ollivier_ricci(MI_full, threshold=0.5, alpha=alpha)
        k_records.setdefault(n_qubits, []).append(oll_full['scalar_curvature'])

        # Sweep k from 3 to N-1
        for k in range(3, n_qubits):
            all_subsets = list(combinations(range(n_qubits), k))
            rng = np.random.default_rng(42 + s * 100 + k)
            n_sub = min(len(all_subsets), 15)
            chosen = rng.choice(len(all_subsets), n_sub, replace=False)

            for idx in chosen:
                subset = list(all_subsets[idx])
                MI_obs = mutual_information_matrix(psi, n_qubits, subset)
                oll = ollivier_ricci(MI_obs, threshold=0.5, alpha=alpha)
                k_records.setdefault(k, []).append(oll['scalar_curvature'])

    # Build output table
    results_table = []
    for k in sorted(k_records.keys()):
        vals = np.array(k_records[k])
        rec = {
            "k": k,
            "k_over_N": float(k / n_qubits),
            "scalar_curvature_mean": float(np.mean(vals)),
            "scalar_curvature_std": float(np.std(vals)),
            "scalar_curvature_median": float(np.median(vals)),
            "n_observations": len(vals),
        }
        results_table.append(rec)
        print(f"  k={k:2d} (k/N={rec['k_over_N']:.2f}): "
              f"R = {rec['scalar_curvature_mean']:+.4f} +/- {rec['scalar_curvature_std']:.4f}  "
              f"[n={rec['n_observations']}]")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    result = {
        "experiment": "curvature_vs_partiality",
        "n_qubits": n_qubits,
        "n_samples": n_samples,
        "alpha": alpha,
        "results_table": results_table,
        "elapsed_seconds": elapsed,
    }

    _save_result("curv_exp1_partiality", result)
    return result


# -----------------------------------------------------------------
# Experiment 2: Curvature vs Entanglement
# -----------------------------------------------------------------

def experiment_curvature_vs_entanglement(
    n_qubits: int = 8,
    n_samples: int = 5,
    delta_values: Optional[list[float]] = None,
    use_gpu: bool = True,
    alpha: float = 0.5,
    threshold_percentile: float = 0.5,
) -> dict:
    """
    Sweep XXZ anisotropy Delta from product-like (large Delta) to
    highly entangled (small Delta).  At each Delta, compute:
      - ground state
      - half-system entanglement entropy
      - MI matrix -> Ollivier-Ricci and Forman-Ricci scalar curvatures

    Prediction: curvature magnitude tracks entanglement.

    Requires build_xxz_hamiltonian from quantum.py (built by other lead).
    Falls back to xxz_all_to_all if the extended function is unavailable.
    """
    if delta_values is None:
        delta_values = [10.0, 5.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.1]

    print(f"\n{'='*60}")
    print(f"  CURVATURE EXP 2: Curvature vs Entanglement (XXZ sweep)")
    print(f"  N = {n_qubits}, {n_samples} samples, {len(delta_values)} Delta values")
    print(f"{'='*60}\n")

    t0 = time.time()
    diag_fn = ground_state_gpu if use_gpu else ground_state

    # Fallback: use existing xxz_all_to_all if extended builder unavailable
    if build_xxz_hamiltonian is None:
        from .quantum import xxz_all_to_all as _xxz_fallback
        print("  [Using xxz_all_to_all fallback — build_xxz_hamiltonian not yet available]")
    else:
        _xxz_fallback = None

    records = []

    for delta in delta_values:
        ollivier_scalars = []
        forman_scalars = []
        entropies = []

        for s in range(n_samples):
            seed = 6000 + s

            # Build XXZ Hamiltonian
            if _xxz_fallback is not None:
                H, _ = _xxz_fallback(n_qubits, delta=delta, seed=seed)
            else:
                H, _ = build_xxz_hamiltonian(
                    n_qubits, delta=delta, coupling='all_to_all', seed=6000 + s
                )
                # hamiltonians module returns sparse; convert for ground_state_gpu
                import scipy.sparse as sp
                if sp.issparse(H):
                    H = H.toarray()

            E0, psi = diag_fn(H)

            # Half-system entanglement entropy
            half = n_qubits // 2
            if entanglement_entropy is not None:
                S_half = entanglement_entropy(psi, list(range(half)), n_qubits)
            else:
                rho_half = partial_trace(psi, list(range(half)), n_qubits)
                S_half = von_neumann_entropy(rho_half)
            entropies.append(S_half)

            # MI matrix
            MI = mutual_information_matrix(psi, n_qubits)

            # Ollivier-Ricci (threshold to avoid trivially complete graph)
            oll = ollivier_ricci(MI, threshold=threshold_percentile, alpha=alpha)
            ollivier_scalars.append(oll['scalar_curvature'])

            # Forman-Ricci (basic — degree-based, robust correlation)
            form = forman_ricci(MI, threshold=threshold_percentile)
            forman_scalars.append(form['scalar_curvature'])

        rec = {
            "delta": float(delta),
            "half_system_entropy_mean": float(np.mean(entropies)),
            "half_system_entropy_std": float(np.std(entropies)),
            "ollivier_scalar_mean": float(np.mean(ollivier_scalars)),
            "ollivier_scalar_std": float(np.std(ollivier_scalars)),
            "forman_scalar_mean": float(np.mean(forman_scalars)),
            "forman_scalar_std": float(np.std(forman_scalars)),
            "n_samples": n_samples,
        }
        records.append(rec)

        print(f"  Delta={delta:6.2f}  S_half={rec['half_system_entropy_mean']:.4f}  "
              f"R_oll={rec['ollivier_scalar_mean']:+.4f}  "
              f"R_for={rec['forman_scalar_mean']:+.4f}")

    # Compute correlation between entropy and curvature across Delta values
    S_vals = np.array([r['half_system_entropy_mean'] for r in records])
    R_oll_vals = np.array([r['ollivier_scalar_mean'] for r in records])
    R_for_vals = np.array([r['forman_scalar_mean'] for r in records])

    corr_oll = float(np.corrcoef(S_vals, R_oll_vals)[0, 1]) if len(S_vals) > 2 else 0.0
    corr_for = float(np.corrcoef(S_vals, R_for_vals)[0, 1]) if len(S_vals) > 2 else 0.0

    print(f"\n  Correlation (entropy, Ollivier curvature):  r = {corr_oll:.4f}")
    print(f"  Correlation (entropy, Forman curvature):    r = {corr_for:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    result = {
        "experiment": "curvature_vs_entanglement",
        "n_qubits": n_qubits,
        "n_samples": n_samples,
        "alpha": alpha,
        "delta_values": delta_values,
        "records": records,
        "correlation_entropy_ollivier": corr_oll,
        "correlation_entropy_forman": corr_for,
        "elapsed_seconds": elapsed,
    }

    _save_result("curv_exp2_entanglement", result)
    return result


# -----------------------------------------------------------------
# Experiment 3: Local vs Non-Local Geometry
# -----------------------------------------------------------------

def experiment_local_vs_nonlocal(
    n_qubits: int = 8,
    k: Optional[int] = None,
    n_samples: int = 10,
    delta: float = 1.0,
    use_gpu: bool = True,
    alpha: float = 0.5,
) -> dict:
    """
    Compare curvature from two coupling geometries:

    (A) All-to-all coupling (Paper 1 style) -- no built-in spatial structure.
        Prediction: positive or near-zero scalar curvature (sphere-like / flat).

    (B) 1D nearest-neighbour chain -- has built-in locality.
        Prediction: NEGATIVE scalar curvature (AdS-like / hyperbolic).

    Same n_qubits, same observer size k, same interaction strength.
    Only the coupling topology differs.

    Requires build_local_hamiltonian from quantum.py.
    Falls back to nearest_neighbor_chain if unavailable.
    """
    if k is None:
        k = max(3, n_qubits // 2)

    print(f"\n{'='*60}")
    print(f"  CURVATURE EXP 3: Local vs Non-Local Geometry")
    print(f"  N = {n_qubits}, k = {k}, {n_samples} samples, delta = {delta}")
    print(f"{'='*60}\n")

    t0 = time.time()
    diag_fn = ground_state_gpu if use_gpu else ground_state

    # Fallback imports
    if build_local_hamiltonian is None:
        from .quantum import nearest_neighbor_chain as _chain_fallback
        print("  [Using nearest_neighbor_chain fallback]")
    else:
        _chain_fallback = None

    a2a_ollivier = []
    a2a_forman = []
    chain_ollivier = []
    chain_forman = []

    for s in range(n_samples):
        seed = 7000 + s
        rng = np.random.default_rng(seed + 999)

        # --- All-to-all ---
        H_a2a, _ = random_all_to_all(n_qubits, seed=seed)
        _, psi_a2a = diag_fn(H_a2a)

        # Pick random k-subset
        subset = sorted(rng.choice(n_qubits, k, replace=False).tolist())
        MI_a2a = mutual_information_matrix(psi_a2a, n_qubits, subset)

        oll_a2a = ollivier_ricci(MI_a2a, threshold=0.5, alpha=alpha)
        for_a2a = forman_ricci(MI_a2a, threshold=0.5)
        a2a_ollivier.append(oll_a2a['scalar_curvature'])
        a2a_forman.append(for_a2a['scalar_curvature'])

        # --- 1D Chain ---
        if _chain_fallback is not None:
            H_chain, _ = _chain_fallback(n_qubits, seed=seed)
        else:
            H_chain, _ = build_local_hamiltonian(
                n_qubits, geometry='chain', delta=delta
            )
            import scipy.sparse as sp
            if sp.issparse(H_chain):
                H_chain = H_chain.toarray()
        _, psi_chain = diag_fn(H_chain)

        MI_chain = mutual_information_matrix(psi_chain, n_qubits, subset)

        oll_chain = ollivier_ricci(MI_chain, threshold=0.5, alpha=alpha)
        for_chain = forman_ricci(MI_chain, threshold=0.5)
        chain_ollivier.append(oll_chain['scalar_curvature'])
        chain_forman.append(for_chain['scalar_curvature'])

        if s % max(1, n_samples // 5) == 0:
            print(f"  Sample {s+1}/{n_samples}: "
                  f"A2A R_oll={oll_a2a['scalar_curvature']:+.4f}  "
                  f"Chain R_oll={oll_chain['scalar_curvature']:+.4f}")

    # Summarize
    a2a_oll_mean = float(np.mean(a2a_ollivier))
    a2a_oll_std = float(np.std(a2a_ollivier))
    chain_oll_mean = float(np.mean(chain_ollivier))
    chain_oll_std = float(np.std(chain_ollivier))
    a2a_for_mean = float(np.mean(a2a_forman))
    chain_for_mean = float(np.mean(chain_forman))

    print(f"\n  ALL-TO-ALL:  R_oll = {a2a_oll_mean:+.4f} +/- {a2a_oll_std:.4f}"
          f"   R_for = {a2a_for_mean:+.4f}")
    print(f"  1D CHAIN:    R_oll = {chain_oll_mean:+.4f} +/- {chain_oll_std:.4f}"
          f"   R_for = {chain_for_mean:+.4f}")

    sign_match = (chain_oll_mean < a2a_oll_mean)
    print(f"\n  Chain curvature < A2A curvature: {sign_match}")
    if sign_match:
        print(f"  >>> LOCAL GEOMETRY PRODUCES MORE NEGATIVE CURVATURE (AdS-like) <<<")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    result = {
        "experiment": "local_vs_nonlocal",
        "n_qubits": n_qubits,
        "k": k,
        "n_samples": n_samples,
        "delta": delta,
        "alpha": alpha,
        "all_to_all": {
            "ollivier_scalars": a2a_ollivier,
            "forman_scalars": a2a_forman,
            "ollivier_mean": a2a_oll_mean,
            "ollivier_std": a2a_oll_std,
            "forman_mean": a2a_for_mean,
        },
        "chain": {
            "ollivier_scalars": chain_ollivier,
            "forman_scalars": chain_forman,
            "ollivier_mean": chain_oll_mean,
            "ollivier_std": chain_oll_std,
            "forman_mean": chain_for_mean,
        },
        "chain_more_negative": bool(sign_match),
        "elapsed_seconds": elapsed,
    }

    _save_result("curv_exp3_local_nonlocal", result)
    return result


# -----------------------------------------------------------------
# Experiment 4: Perspective-Dependent Curvature
# -----------------------------------------------------------------

def experiment_perspective_dependent_curvature(
    n_qubits: int = 8,
    k: Optional[int] = None,
    n_samples: int = 10,
    max_subsets: int = 50,
    use_gpu: bool = True,
    alpha: float = 0.5,
) -> dict:
    """
    Different observers (different k-subsystems of the SAME state) see
    different scalar curvature.  This is the genuinely novel PLC result:
    curvature is perspectival, not absolute.

    For a fixed ground state, enumerate (or sample) all k-subsets.
    Compute Ollivier-Ricci scalar curvature for each observer.
    Report: mean, std, min, max, full distribution.

    The WIDTH of this distribution is the key observable.
    Wider distribution = curvature is more observer-dependent.
    We also check whether the spread depends on entanglement structure.
    """
    if k is None:
        k = max(3, n_qubits // 2)

    print(f"\n{'='*60}")
    print(f"  CURVATURE EXP 4: Perspective-Dependent Curvature")
    print(f"  N = {n_qubits}, k = {k}, {n_samples} samples")
    print(f"  up to {max_subsets} k-subsets per sample")
    print(f"{'='*60}\n")

    t0 = time.time()
    diag_fn = ground_state_gpu if use_gpu else ground_state

    all_subsets_enum = list(combinations(range(n_qubits), k))
    n_sub = min(len(all_subsets_enum), max_subsets)

    sample_records = []

    for s in range(n_samples):
        seed = 8000 + s
        H, _ = random_all_to_all(n_qubits, seed=seed)
        E0, psi = diag_fn(H)

        # Choose subsets
        rng = np.random.default_rng(seed + 777)
        if len(all_subsets_enum) <= max_subsets:
            chosen_subsets = all_subsets_enum
        else:
            indices = rng.choice(len(all_subsets_enum), n_sub, replace=False)
            chosen_subsets = [all_subsets_enum[i] for i in indices]

        curvatures = []
        edge_curvature_stats = []

        for subset in chosen_subsets:
            subset = list(subset)
            MI_obs = mutual_information_matrix(psi, n_qubits, subset)
            oll = ollivier_ricci(MI_obs, threshold=0.5, alpha=alpha)
            curvatures.append(oll['scalar_curvature'])
            edge_curvature_stats.append(_curvature_stats(oll['edge_curvatures']))

        curvatures = np.array(curvatures)

        rec = {
            "sample": s,
            "E0": float(E0),
            "n_subsets": len(chosen_subsets),
            "curvature_mean": float(np.mean(curvatures)),
            "curvature_std": float(np.std(curvatures)),
            "curvature_min": float(np.min(curvatures)),
            "curvature_max": float(np.max(curvatures)),
            "curvature_range": float(np.max(curvatures) - np.min(curvatures)),
            "curvature_iqr": float(np.percentile(curvatures, 75) - np.percentile(curvatures, 25)),
            "curvatures": curvatures.tolist(),
        }
        sample_records.append(rec)

        print(f"  Sample {s+1}/{n_samples}: "
              f"R_mean = {rec['curvature_mean']:+.4f}, "
              f"R_std = {rec['curvature_std']:.4f}, "
              f"range = [{rec['curvature_min']:+.4f}, {rec['curvature_max']:+.4f}]")

    # Aggregate across samples
    all_spreads = [r['curvature_std'] for r in sample_records]
    all_ranges = [r['curvature_range'] for r in sample_records]
    all_iqrs = [r['curvature_iqr'] for r in sample_records]

    print(f"\n  AGGREGATE over {n_samples} samples:")
    print(f"  Mean perspective spread (std):   {np.mean(all_spreads):.4f} +/- {np.std(all_spreads):.4f}")
    print(f"  Mean perspective range:          {np.mean(all_ranges):.4f} +/- {np.std(all_ranges):.4f}")
    print(f"  Mean perspective IQR:            {np.mean(all_iqrs):.4f} +/- {np.std(all_iqrs):.4f}")

    significant = np.mean(all_spreads) > 0.01
    print(f"\n  Curvature is perspective-dependent: {significant}")
    if significant:
        print(f"  >>> CURVATURE IS NOT ABSOLUTE -- IT DEPENDS ON WHICH QUBITS THE OBSERVER SEES <<<")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    result = {
        "experiment": "perspective_dependent_curvature",
        "n_qubits": n_qubits,
        "k": k,
        "n_samples": n_samples,
        "max_subsets": max_subsets,
        "alpha": alpha,
        "sample_records": sample_records,
        "aggregate": {
            "mean_spread_std": float(np.mean(all_spreads)),
            "std_spread_std": float(np.std(all_spreads)),
            "mean_range": float(np.mean(all_ranges)),
            "mean_iqr": float(np.mean(all_iqrs)),
            "is_perspective_dependent": bool(significant),
        },
        "elapsed_seconds": elapsed,
    }

    _save_result("curv_exp4_perspective", result)
    return result


# -----------------------------------------------------------------
# Run all curvature experiments
# -----------------------------------------------------------------

def run_all_curvature(n_qubits: int = 8, use_gpu: bool = True) -> dict:
    """Run all four curvature experiments and save results."""
    print("\n" + "=" * 60)
    print("  PLC PAPER 2: Emergent Curvature from Partial Observation")
    print("  Perspectival Locality Conjecture -- Curvature Experiments")
    print("=" * 60)
    print(f"  System: {n_qubits} qubits")
    print(f"  Hilbert space: 2^{n_qubits} = {2**n_qubits} dimensions")
    print(f"  GPU: {'enabled' if use_gpu else 'disabled'}")
    print("=" * 60)

    results = {}

    results['exp1'] = experiment_curvature_vs_partiality(
        n_qubits=n_qubits, use_gpu=use_gpu
    )
    results['exp2'] = experiment_curvature_vs_entanglement(
        n_qubits=n_qubits, use_gpu=use_gpu
    )
    results['exp3'] = experiment_local_vs_nonlocal(
        n_qubits=n_qubits, use_gpu=use_gpu
    )
    results['exp4'] = experiment_perspective_dependent_curvature(
        n_qubits=n_qubits, use_gpu=use_gpu
    )

    print("\n" + "=" * 60)
    print("  ALL CURVATURE EXPERIMENTS COMPLETE")
    print("=" * 60)

    return results
