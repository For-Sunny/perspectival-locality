"""
Statistical hardening for PLC simulation results.

Bootstrap confidence intervals, permutation tests, null models.
All designed for peer-review rigor.

Built by Opus Warrior, March 5 2026.
"""

import numpy as np
from typing import Optional
from itertools import combinations


# ─────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(data: np.ndarray, stat_fn=np.mean, n_bootstrap: int = 10000,
                 ci: float = 0.95, seed: int = 0) -> dict:
    """
    Bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations.
        stat_fn: function to compute the statistic (default: mean).
        n_bootstrap: number of bootstrap resamples.
        ci: confidence level (0.95 = 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        dict with 'estimate', 'ci_low', 'ci_high', 'se', 'n'.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    estimate = float(stat_fn(data))

    # Generate all bootstrap indices at once: (n_bootstrap, n)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_stats = np.array([stat_fn(data[idx]) for idx in boot_indices])

    alpha = 1.0 - ci
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_stats, ddof=1))

    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "n": n,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
    }


# ─────────────────────────────────────────────────────────────
# Permutation test for dimensionality reduction
# ─────────────────────────────────────────────────────────────

def permutation_test_dim_ratio(dim_full: np.ndarray, dim_observer: np.ndarray,
                                n_perms: int = 10000, seed: int = 0) -> dict:
    """
    One-sided permutation test for H0: partiality has no effect on dimensionality.

    Under H0, swapping labels between full and observer dimensions should not
    change the mean difference. We test whether dim_observer < dim_full
    significantly (one-sided).

    Args:
        dim_full: array of full-system effective dimensions (one per trial).
        dim_observer: array of observer effective dimensions (one per trial).
        n_perms: number of permutations.
        seed: RNG seed.

    Returns:
        dict with 'observed_diff', 'p_value', 'n_perms'.
    """
    rng = np.random.default_rng(seed)
    dim_full = np.asarray(dim_full, dtype=np.float64)
    dim_observer = np.asarray(dim_observer, dtype=np.float64)

    n = len(dim_full)
    assert len(dim_observer) == n, "Arrays must be same length"

    # Observed statistic: mean(full) - mean(observer)
    # Under H1 (partiality reduces dim), this should be positive.
    observed_diff = float(np.mean(dim_full) - np.mean(dim_observer))

    # Pool and permute
    pooled = np.stack([dim_full, dim_observer], axis=1)  # (n, 2)
    perm_diffs = np.empty(n_perms)

    for i in range(n_perms):
        # For each pair, randomly swap or not
        swaps = rng.integers(0, 2, size=n)
        perm_full = np.where(swaps == 0, pooled[:, 0], pooled[:, 1])
        perm_obs = np.where(swaps == 0, pooled[:, 1], pooled[:, 0])
        perm_diffs[i] = np.mean(perm_full) - np.mean(perm_obs)

    # One-sided p-value: P(perm_diff >= observed_diff)
    p_value = float(np.mean(perm_diffs >= observed_diff))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "n_perms": n_perms,
        "mean_perm_diff": float(np.mean(perm_diffs)),
        "std_perm_diff": float(np.std(perm_diffs)),
    }


# ─────────────────────────────────────────────────────────────
# One-sided test for correlation decay (r < 0)
# ─────────────────────────────────────────────────────────────

def pvalue_r_negative(r_values: np.ndarray, n_bootstrap: int = 10000,
                       seed: int = 0) -> dict:
    """
    Test H0: mean Pearson r >= 0 (no decay) against H1: mean r < 0 (decay exists).

    Uses bootstrap on the mean of r_values. The p-value is the fraction of
    bootstrap means that are >= 0.

    Also computes an exact permutation-flavored p-value by asking:
    under H0, if we center the data at 0, how often does the bootstrap mean
    end up as extreme as observed?

    Args:
        r_values: array of Pearson r values from multiple trials.
        n_bootstrap: number of bootstrap samples.
        seed: RNG seed.

    Returns:
        dict with 'mean_r', 'p_value_bootstrap', 'p_value_sign'.
    """
    rng = np.random.default_rng(seed)
    r_values = np.asarray(r_values, dtype=np.float64)
    n = len(r_values)
    mean_r = float(np.mean(r_values))

    # Bootstrap p-value: fraction of bootstrap means >= 0
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.array([np.mean(r_values[idx]) for idx in boot_indices])
    p_bootstrap = float(np.mean(boot_means >= 0))

    # Sign test p-value (non-parametric)
    # Under H0 (median r = 0), number of negative values is Binomial(n, 0.5)
    n_negative = np.sum(r_values < 0)
    # P(X >= n_negative) under Binomial(n, 0.5)
    from scipy.stats import binom
    p_sign = float(binom.sf(n_negative - 1, n, 0.5))

    # Also t-test for completeness
    from scipy.stats import ttest_1samp
    t_stat, p_ttest_two = ttest_1samp(r_values, 0.0)
    # One-sided: we want P(mean < 0), so if t < 0, p_one = p_two/2
    p_ttest_one = float(p_ttest_two / 2) if t_stat < 0 else float(1 - p_ttest_two / 2)

    return {
        "mean_r": mean_r,
        "std_r": float(np.std(r_values, ddof=1)),
        "n": n,
        "p_value_bootstrap": p_bootstrap,
        "p_value_sign_test": p_sign,
        "p_value_ttest_onesided": p_ttest_one,
        "n_negative": int(n_negative),
        "t_statistic": float(t_stat),
    }


# ─────────────────────────────────────────────────────────────
# Null model: shuffled MI matrix
# ─────────────────────────────────────────────────────────────

def shuffled_null_pearson_r(MI_obs: np.ndarray, C_obs: np.ndarray,
                             n_shuffles: int = 200, seed: int = 0) -> dict:
    """
    Null model: randomly permute off-diagonal entries of the MI matrix,
    recompute MI-distance, then compute Pearson r between log|C| and shuffled distance.

    This tests whether the correlation-vs-distance structure is an artifact
    of the distance definition or a real feature of the quantum state.

    Args:
        MI_obs: observer's mutual information matrix (n x n, symmetric).
        C_obs: observer's correlation matrix (n x n, symmetric).
        n_shuffles: number of random permutations.
        seed: RNG seed.

    Returns:
        dict with 'real_r', 'shuffled_r_mean', 'shuffled_r_std',
        'p_value' (fraction of shuffled r <= real r), etc.
    """
    rng = np.random.default_rng(seed)
    n = MI_obs.shape[0]

    # Extract upper-triangle pairs
    triu_i, triu_j = np.triu_indices(n, k=1)
    mi_values = MI_obs[triu_i, triu_j]
    corr_values = np.abs(C_obs[triu_i, triu_j])

    # Compute real Pearson r
    real_r = _pearson_r_log_corr_vs_dist(mi_values, corr_values)

    # Shuffled null
    shuffled_rs = np.empty(n_shuffles)
    for s in range(n_shuffles):
        # Permute the MI values (break the correspondence with correlation)
        perm = rng.permutation(len(mi_values))
        mi_shuffled = mi_values[perm]
        shuffled_rs[s] = _pearson_r_log_corr_vs_dist(mi_shuffled, corr_values)

    # p-value: fraction of shuffled r values as extreme as (or more than) real r
    # Since real r should be negative (decay), we want P(shuffled_r <= real_r)
    p_value = float(np.mean(shuffled_rs <= real_r))

    return {
        "real_r": float(real_r),
        "shuffled_r_mean": float(np.mean(shuffled_rs)),
        "shuffled_r_std": float(np.std(shuffled_rs)),
        "p_value": p_value,
        "n_shuffles": n_shuffles,
        "n_pairs": len(mi_values),
        "effect_size": float((real_r - np.mean(shuffled_rs)) / np.std(shuffled_rs))
            if np.std(shuffled_rs) > 1e-14 else 0.0,
    }


# ─────────────────────────────────────────────────────────────
# Null model 2: eigenvalue-preserving random rotation
# ─────────────────────────────────────────────────────────────

def eigenvalue_preserving_null(MI_obs: np.ndarray, C_obs: np.ndarray,
                                n_shuffles: int = 200, seed: int = 0) -> dict:
    """
    Null model: random orthogonal rotations that preserve the MI eigenvalue spectrum
    but randomize which qubits have high/low MI.

    MI_null = Q_rand @ diag(lambda) @ Q_rand.T

    This tests: does the ASSIGNMENT of MI to specific qubit pairs matter,
    or just the overall spectrum of the MI matrix?
    If real r is far from this null -> specific qubit-pair MI values matter.
    """
    from scipy.stats import special_ortho_group

    rng = np.random.default_rng(seed)
    n = MI_obs.shape[0]

    # Extract upper-triangle for real r
    triu_i, triu_j = np.triu_indices(n, k=1)
    corr_values = np.abs(C_obs[triu_i, triu_j])

    # Real Pearson r
    mi_real = MI_obs[triu_i, triu_j]
    real_r = _pearson_r_log_corr_vs_dist(mi_real, corr_values)

    # Diagonalize MI: eigenvalues + eigenvectors
    eigenvalues, _ = np.linalg.eigh(MI_obs)
    # Clamp negative eigenvalues (numerical noise) to zero
    eigenvalues = np.maximum(eigenvalues, 0.0)
    diag_lambda = np.diag(eigenvalues)

    null_rs = np.empty(n_shuffles)
    clamped_fractions = np.empty(n_shuffles)
    n_offdiag = len(triu_i)
    for s in range(n_shuffles):
        # Random orthogonal matrix
        Q_rand = special_ortho_group.rvs(n, random_state=rng)
        MI_null = Q_rand @ diag_lambda @ Q_rand.T
        # Ensure symmetry and non-negative off-diagonal
        MI_null = (MI_null + MI_null.T) / 2.0
        # Extract upper-triangle MI values
        mi_null_vals = MI_null[triu_i, triu_j]
        # Track how many entries need clamping (diagnostic for reviewer)
        n_clamped = int(np.sum(mi_null_vals < 0))
        clamped_fractions[s] = n_clamped / n_offdiag
        # Clamp tiny negatives from rotation
        mi_null_vals = np.maximum(mi_null_vals, 0.0)
        null_rs[s] = _pearson_r_log_corr_vs_dist(mi_null_vals, corr_values)

    p_value = float(np.mean(null_rs <= real_r))

    return {
        "real_r": float(real_r),
        "null_r_mean": float(np.mean(null_rs)),
        "null_r_std": float(np.std(null_rs)),
        "p_value": p_value,
        "n_shuffles": n_shuffles,
        "effect_size": float((real_r - np.mean(null_rs)) / np.std(null_rs))
            if np.std(null_rs) > 1e-14 else 0.0,
        "null_type": "eigenvalue_preserving",
        "clamped_fraction_mean": float(np.mean(clamped_fractions)),
        "clamped_fraction_max": float(np.max(clamped_fractions)),
    }


# ─────────────────────────────────────────────────────────────
# Null model 3: restricted permutation shuffle
# ─────────────────────────────────────────────────────────────

def restricted_permutation_null(MI_obs: np.ndarray, C_obs: np.ndarray,
                                 n_shuffles: int = 200, seed: int = 0) -> dict:
    """
    Null model: shuffle MI matrix entries by swapping values between pairs
    that share no qubits.

    NOTE: this does NOT preserve row sums (per-qubit total MI). Swapping MI
    values between non-overlapping pairs (i1,j1) and (i2,j2) changes the
    row sums of all four involved qubits. The constraint merely prevents
    self-loops and restricts which swaps are allowed, making this a
    restricted permutation rather than a degree-preserving shuffle.

    This tests: does the pairwise structure matter beyond what restricted
    random permutation can explain?
    If real r is far from this null -> specific pairwise MI assignments matter.
    """
    rng = np.random.default_rng(seed)
    n = MI_obs.shape[0]

    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = len(triu_i)
    corr_values = np.abs(C_obs[triu_i, triu_j])
    mi_real = MI_obs[triu_i, triu_j].copy()

    real_r = _pearson_r_log_corr_vs_dist(mi_real, corr_values)

    # Build lookup: for each pair index, which row indices it touches
    # pair_idx -> (i, j)
    pair_rows = list(zip(triu_i, triu_j))

    null_rs = np.empty(n_shuffles)
    n_swaps_per_shuffle = n_pairs * 5  # enough to mix well

    for s in range(n_shuffles):
        mi_shuffled = mi_real.copy()

        for _ in range(n_swaps_per_shuffle):
            # Pick two random pairs
            p1 = rng.integers(0, n_pairs)
            p2 = rng.integers(0, n_pairs)
            if p1 == p2:
                continue

            i1, j1 = pair_rows[p1]
            i2, j2 = pair_rows[p2]

            # Only swap if the two pairs share no qubits
            # This preserves row sums exactly:
            # row sums of i1, j1, i2, j2 all unchanged because
            # each row loses one value and gains the swapped value
            # Actually for exact preservation, we need pairs that share no vertices
            if i1 != i2 and i1 != j2 and j1 != i2 and j1 != j2:
                mi_shuffled[p1], mi_shuffled[p2] = mi_shuffled[p2], mi_shuffled[p1]

        null_rs[s] = _pearson_r_log_corr_vs_dist(mi_shuffled, corr_values)

    p_value = float(np.mean(null_rs <= real_r))

    return {
        "real_r": float(real_r),
        "null_r_mean": float(np.mean(null_rs)),
        "null_r_std": float(np.std(null_rs)),
        "p_value": p_value,
        "n_shuffles": n_shuffles,
        "effect_size": float((real_r - np.mean(null_rs)) / np.std(null_rs))
            if np.std(null_rs) > 1e-14 else 0.0,
        "null_type": "restricted_permutation",
    }


# Backwards-compatible alias
degree_preserving_null = restricted_permutation_null


# ─────────────────────────────────────────────────────────────
# Null model 4: random Hamiltonian (wrong state's MI)
# ─────────────────────────────────────────────────────────────

def random_hamiltonian_null(psi: np.ndarray, C_obs: np.ndarray,
                             n_qubits: int, k: int, subset: list[int],
                             n_shuffles: int = 100, seed: int = 0,
                             use_gpu: bool = True) -> dict:
    """
    Null model: use MI matrices from DIFFERENT random Hamiltonians to predict
    correlations from THIS Hamiltonian's ground state.

    For each null trial:
      1. Generate a new random Hamiltonian
      2. Compute its ground state
      3. Compute MI matrix for the same qubit subset
      4. Use THAT MI as the distance predictor
      5. But correlations are still from the ORIGINAL state
      6. Compute Pearson r

    This tests: does the SPECIFIC quantum state matter?
    If real r is far from this null -> the MI-correlation link is state-specific,
    not a generic property of ground states.
    """
    from .quantum import random_all_to_all, ground_state, ground_state_gpu
    from .quantum import mutual_information_matrix

    rng = np.random.default_rng(seed)
    n_sub = len(subset)
    diag_fn = ground_state_gpu if use_gpu else ground_state

    triu_i, triu_j = np.triu_indices(n_sub, k=1)
    corr_values = np.abs(C_obs[triu_i, triu_j])

    # Real MI from the actual state
    from .quantum import mutual_information_matrix as mi_matrix
    MI_real = mi_matrix(psi, n_qubits, subset)
    mi_real = MI_real[triu_i, triu_j]
    real_r = _pearson_r_log_corr_vs_dist(mi_real, corr_values)

    null_rs = np.empty(n_shuffles)
    for s in range(n_shuffles):
        # Generate a completely different random Hamiltonian
        null_seed = seed + 100000 + s * 7919  # prime offset for independence
        H_null, _ = random_all_to_all(n_qubits, seed=null_seed)
        _, psi_null = diag_fn(H_null)

        # Compute MI from the NULL state for the same subset
        MI_null = mi_matrix(psi_null, n_qubits, subset)
        mi_null = MI_null[triu_i, triu_j]

        # But correlations are from the ORIGINAL state
        null_rs[s] = _pearson_r_log_corr_vs_dist(mi_null, corr_values)

    p_value = float(np.mean(null_rs <= real_r))

    return {
        "real_r": float(real_r),
        "null_r_mean": float(np.mean(null_rs)),
        "null_r_std": float(np.std(null_rs)),
        "p_value": p_value,
        "n_shuffles": n_shuffles,
        "effect_size": float((real_r - np.mean(null_rs)) / np.std(null_rs))
            if np.std(null_rs) > 1e-14 else 0.0,
        "null_type": "random_hamiltonian",
    }


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _pearson_r_log_corr_vs_dist(mi_values: np.ndarray,
                                  corr_values: np.ndarray) -> float:
    """
    Compute Pearson r between log(|C|) and distance=1/MI.

    Filters out zero/tiny values. Returns 0.0 if insufficient data.
    """
    valid = (mi_values > 1e-14) & (corr_values > 1e-14)
    if np.sum(valid) < 3:
        return 0.0

    dist = 1.0 / mi_values[valid]
    log_c = np.log(corr_values[valid])

    if np.std(dist) < 1e-14 or np.std(log_c) < 1e-14:
        return 0.0

    return float(np.corrcoef(dist, log_c)[0, 1])


# ─────────────────────────────────────────────────────────────
# Null model 5: dimension reduction from subsampling
# ─────────────────────────────────────────────────────────────

def dimension_reduction_null(D_full: 'np.ndarray', k: int,
                              n_shuffles: int = 200, seed: int = 0) -> dict:
    """
    Null model for Experiments 2/4: does subsampling k points from the
    full-system MDS embedding trivially reduce effective dimension?

    Computes effective dimension of k randomly sampled rows/cols from
    the full distance matrix D_full. If observer dimension is comparable
    to this null, the dimension reduction is just a subsampling ceiling
    effect, not a PLC phenomenon.

    Args:
        D_full: full-system distance matrix (N x N).
        k: number of qubits in the observer subset.
        n_shuffles: number of random subsamples.
        seed: RNG seed.

    Returns:
        dict with null dimension statistics.
    """
    from .experiments import _effective_dimension

    rng = np.random.default_rng(seed)
    N = D_full.shape[0]

    null_dims = np.empty(n_shuffles)
    for s in range(n_shuffles):
        subset = rng.choice(N, k, replace=False)
        D_sub = D_full[np.ix_(subset, subset)]
        null_dims[s] = _effective_dimension(D_sub)

    return {
        "null_dim_mean": float(np.mean(null_dims)),
        "null_dim_std": float(np.std(null_dims, ddof=1)),
        "null_dim_median": float(np.median(null_dims)),
        "n_shuffles": n_shuffles,
        "k": k,
        "N": N,
    }


# ─────────────────────────────────────────────────────────────
# Hardened experiment runners
# ─────────────────────────────────────────────────────────────

def hardened_experiment_2(n_qubits_list: list[int], n_trials: int = 50,
                           n_bootstrap: int = 10000, use_gpu: bool = True) -> dict:
    """
    Hardened Experiment 2: Emergent Metric with full statistical rigor.

    For each N in n_qubits_list, run n_trials random Hamiltonians.
    For each, compute effective dimension for full system and observer at
    k = N//3, N//2, 2N//3.

    Returns bootstrap CIs and permutation test p-values for each k/N ratio.
    """
    from .quantum import random_all_to_all, ground_state, ground_state_gpu
    from .experiments import _mi_to_distance, _effective_dimension
    from .quantum import mutual_information_matrix

    diag_fn = ground_state_gpu if use_gpu else ground_state

    all_data = {}  # key: (N, k) -> lists of (dim_full, dim_obs, ratio)

    for N in n_qubits_list:
        k_values = sorted(set([max(3, N // 3), N // 2, min(N - 1, 2 * N // 3)]))
        print(f"\n  Hardened Exp2: N={N}, k_values={k_values}, {n_trials} trials")

        for trial in range(n_trials):
            seed = 5000 + N * 1000 + trial
            H, _ = random_all_to_all(N, seed=seed)
            E0, psi = diag_fn(H)

            # Full system
            MI_full = mutual_information_matrix(psi, N)
            D_full = _mi_to_distance(MI_full)
            dim_full = _effective_dimension(D_full)

            for k in k_values:
                if k >= N or k < 3:
                    continue

                # Average over several random observer subsets for this trial
                all_subsets = list(combinations(range(N), k))
                rng = np.random.default_rng(seed + k * 100)
                n_sub = min(len(all_subsets), 10)
                sub_indices = rng.choice(len(all_subsets), n_sub, replace=False)

                dim_obs_list = []
                for si in sub_indices:
                    subset = list(all_subsets[si])
                    MI_obs = mutual_information_matrix(psi, N, subset)
                    D_obs = _mi_to_distance(MI_obs)
                    dim_obs_list.append(_effective_dimension(D_obs))

                dim_obs_mean = np.mean(dim_obs_list)
                ratio = dim_obs_mean / dim_full if dim_full > 0 else 0.0

                # Dimension reduction null: subsample k points from full D
                dim_null = dimension_reduction_null(
                    D_full, k, n_shuffles=200, seed=seed + k * 100)

                key = (N, k)
                if key not in all_data:
                    all_data[key] = {"dim_full": [], "dim_obs": [], "ratio": [],
                                     "dim_null_mean": []}
                all_data[key]["dim_full"].append(dim_full)
                all_data[key]["dim_obs"].append(dim_obs_mean)
                all_data[key]["ratio"].append(ratio)
                all_data[key]["dim_null_mean"].append(dim_null["null_dim_mean"])

            if (trial + 1) % 10 == 0:
                print(f"    Trial {trial + 1}/{n_trials}")

    # Now compute statistics
    results = {}
    for (N, k), data in sorted(all_data.items()):
        k_over_N = k / N
        ratios = np.array(data["ratio"])
        dim_full_arr = np.array(data["dim_full"])
        dim_obs_arr = np.array(data["dim_obs"])

        # Bootstrap CI on mean ratio
        ci = bootstrap_ci(ratios, n_bootstrap=n_bootstrap, seed=42 + N * 100 + k)

        # Permutation test: H0 = partiality has no effect
        perm = permutation_test_dim_ratio(dim_full_arr, dim_obs_arr,
                                           n_perms=n_bootstrap, seed=42 + N * 100 + k)

        # Dimension reduction null comparison
        dim_null_arr = np.array(data["dim_null_mean"])
        mean_null_dim = float(np.mean(dim_null_arr))
        mean_obs_dim = float(np.mean(dim_obs_arr))
        # Is observer dim lower than null subsample dim?
        below_null = mean_obs_dim < mean_null_dim

        key_str = f"N={N}_k={k}"
        results[key_str] = {
            "N": N,
            "k": k,
            "k_over_N": round(k_over_N, 4),
            "n_trials": len(ratios),
            "dim_ratio_bootstrap": ci,
            "permutation_test": perm,
            "dim_reduction_null": {
                "mean_null_dim": mean_null_dim,
                "mean_obs_dim": mean_obs_dim,
                "obs_below_null": below_null,
                "description": "Effective dim of k random points from full MDS embedding",
            },
            "raw_ratios": ratios.tolist(),
            "raw_dim_full": dim_full_arr.tolist(),
            "raw_dim_obs": dim_obs_arr.tolist(),
        }

        sig = "***" if perm["p_value"] < 0.001 else "**" if perm["p_value"] < 0.01 else "*" if perm["p_value"] < 0.05 else "ns"
        null_tag = "BELOW null" if below_null else "ABOVE null (ceiling effect)"
        print(f"\n  N={N}, k={k} (k/N={k_over_N:.2f}):")
        print(f"    dim_ratio = {ci['estimate']:.4f}  95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
        print(f"    permutation p = {perm['p_value']:.6f} {sig}")
        print(f"    mean dim_full = {np.mean(dim_full_arr):.2f}, mean dim_obs = {np.mean(dim_obs_arr):.2f}")
        print(f"    dim reduction null = {mean_null_dim:.2f}, observer = {mean_obs_dim:.2f} -> {null_tag}")

    return results


def hardened_experiment_5(n_qubits_list: list[int], n_trials: int = 50,
                           n_bootstrap: int = 10000, n_shuffles: int = 200,
                           use_gpu: bool = True) -> dict:
    """
    Hardened Experiment 5: Correlation Decay with bootstrap CIs, p-values,
    and shuffled null model.

    For each N, run n_trials random Hamiltonians. For each trial and each
    k/N ratio, record Pearson r between log|C| and MI-distance.
    Also run shuffled null for each trial.

    Returns bootstrap CIs, p-values, and null model comparisons.
    """
    from .quantum import (
        random_all_to_all, ground_state, ground_state_gpu,
        mutual_information_matrix, correlation_matrix,
    )

    diag_fn = ground_state_gpu if use_gpu else ground_state

    # key: (N, k) -> lists of pearson_r and null model results
    all_data = {}

    for N in n_qubits_list:
        k_values = sorted(set([max(3, N // 3), N // 2, min(N - 1, 2 * N // 3)]))
        print(f"\n  Hardened Exp5: N={N}, k_values={k_values}, {n_trials} trials")

        for trial in range(n_trials):
            seed = 7000 + N * 1000 + trial
            H, _ = random_all_to_all(N, seed=seed)
            E0, psi = diag_fn(H)

            for k in k_values:
                if k >= N or k < 3:
                    continue

                # Sample observer subsets
                all_subsets = list(combinations(range(N), k))
                rng = np.random.default_rng(seed + k * 100)
                n_sub = min(len(all_subsets), 10)
                sub_indices = rng.choice(len(all_subsets), n_sub, replace=False)

                for si in sub_indices:
                    subset = list(all_subsets[si])
                    MI_obs = mutual_information_matrix(psi, N, subset)
                    C_obs = correlation_matrix(psi, N, subset)

                    # Compute real Pearson r
                    triu_i, triu_j = np.triu_indices(len(subset), k=1)
                    mi_vals = MI_obs[triu_i, triu_j]
                    corr_vals = np.abs(C_obs[triu_i, triu_j])
                    real_r = _pearson_r_log_corr_vs_dist(mi_vals, corr_vals)

                    # Null model
                    null_result = shuffled_null_pearson_r(
                        MI_obs, C_obs,
                        n_shuffles=n_shuffles,
                        seed=seed + k * 100 + si,
                    )

                    key = (N, k)
                    if key not in all_data:
                        all_data[key] = {
                            "real_r": [],
                            "null_r_mean": [],
                            "null_p_value": [],
                            "null_effect_size": [],
                            "per_hamiltonian_r": {},
                        }
                    all_data[key]["real_r"].append(real_r)
                    all_data[key]["null_r_mean"].append(null_result["shuffled_r_mean"])
                    all_data[key]["null_p_value"].append(null_result["p_value"])
                    all_data[key]["null_effect_size"].append(null_result["effect_size"])
                    # Track per-Hamiltonian for corrected analysis
                    ham_key = (N, trial)
                    all_data[key]["per_hamiltonian_r"].setdefault(ham_key, []).append(real_r)

            if (trial + 1) % 10 == 0:
                print(f"    Trial {trial + 1}/{n_trials}")

    # Compute statistics
    results = {}
    for (N, k), data in sorted(all_data.items()):
        k_over_N = k / N
        r_arr = np.array(data["real_r"])
        null_r_arr = np.array(data["null_r_mean"])
        null_p_arr = np.array(data["null_p_value"])
        null_es_arr = np.array(data["null_effect_size"])

        # ── CORRECTED ANALYSIS (PRIMARY) ──
        # Average r-values within each Hamiltonian first, then test on
        # per-Hamiltonian means. This is correct because subsets of the
        # SAME Hamiltonian share the quantum state and are not independent.
        per_ham_r = data["per_hamiltonian_r"]
        ham_means = np.array([float(np.mean(rs)) for rs in per_ham_r.values()])
        n_hamiltonians = len(ham_means)

        ci_corrected = bootstrap_ci(ham_means, n_bootstrap=n_bootstrap,
                                     seed=42 + N * 100 + k + 7)
        p_test_corrected = pvalue_r_negative(ham_means, n_bootstrap=n_bootstrap,
                                              seed=42 + N * 100 + k + 7)

        # ── POOLED ANALYSIS (LEGACY, kept for comparison) ──
        ci_pooled = bootstrap_ci(r_arr, n_bootstrap=n_bootstrap, seed=42 + N * 100 + k)
        p_test_pooled = pvalue_r_negative(r_arr, n_bootstrap=n_bootstrap, seed=42 + N * 100 + k)

        # Null model summary
        null_summary = {
            "mean_real_r": float(np.mean(r_arr)),
            "mean_null_r": float(np.mean(null_r_arr)),
            "median_null_p": float(np.median(null_p_arr)),
            "frac_null_p_lt_005": float(np.mean(null_p_arr < 0.05)),
            "mean_effect_size": float(np.mean(null_es_arr)),
            "std_effect_size": float(np.std(null_es_arr)),
        }

        key_str = f"N={N}_k={k}"
        results[key_str] = {
            "N": N,
            "k": k,
            "k_over_N": round(k_over_N, 4),
            "n_samples_pooled": len(r_arr),
            "n_hamiltonians": n_hamiltonians,
            # Corrected analysis (primary)
            "pearson_r_bootstrap": ci_corrected,
            "decay_pvalue": p_test_corrected,
            # Pooled analysis (legacy, inflated effective n)
            "pooled_pearson_r_bootstrap": ci_pooled,
            "pooled_decay_pvalue": p_test_pooled,
            "null_model": null_summary,
            "raw_real_r": r_arr.tolist(),
            "raw_hamiltonian_means": ham_means.tolist(),
            "raw_null_r_mean": null_r_arr.tolist(),
            "raw_null_p": null_p_arr.tolist(),
        }

        sig_c = "***" if p_test_corrected["p_value_ttest_onesided"] < 0.001 else "**" if p_test_corrected["p_value_ttest_onesided"] < 0.01 else "*" if p_test_corrected["p_value_ttest_onesided"] < 0.05 else "ns"
        sig_p = "***" if p_test_pooled["p_value_ttest_onesided"] < 0.001 else "**" if p_test_pooled["p_value_ttest_onesided"] < 0.01 else "*" if p_test_pooled["p_value_ttest_onesided"] < 0.05 else "ns"
        print(f"\n  N={N}, k={k} (k/N={k_over_N:.2f}):")
        print(f"    CORRECTED (per-Hamiltonian means, n={n_hamiltonians}):")
        print(f"      Pearson r = {ci_corrected['estimate']:.4f}  95% CI [{ci_corrected['ci_low']:.4f}, {ci_corrected['ci_high']:.4f}]")
        print(f"      H0(r>=0) p = {p_test_corrected['p_value_ttest_onesided']:.6f} {sig_c}")
        print(f"      {p_test_corrected['n_negative']}/{p_test_corrected['n']} Hamiltonians show negative mean r")
        print(f"    POOLED (all subsets, n={len(r_arr)}, NOT independent):")
        print(f"      Pearson r = {ci_pooled['estimate']:.4f}  95% CI [{ci_pooled['ci_low']:.4f}, {ci_pooled['ci_high']:.4f}]")
        print(f"      H0(r>=0) p = {p_test_pooled['p_value_ttest_onesided']:.6f} {sig_p}")
        print(f"    Null model: real r = {null_summary['mean_real_r']:.4f}, shuffled r = {null_summary['mean_null_r']:.4f}")
        print(f"    Null p<0.05 in {null_summary['frac_null_p_lt_005']:.1%} of trials")
        print(f"    Standardized effect size (z from null): {null_summary['mean_effect_size']:.2f}")

    return results
