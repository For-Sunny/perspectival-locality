"""
Stub tests for curvature_experiments.py.

Uses N=4 systems for speed. Verifies function signatures, return
structure, and basic sanity (curvatures are finite, records are populated).

These tests mock curvature.py since it is being built by another lead.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Mock the curvature module that another lead is building
# ---------------------------------------------------------------------------

def _mock_ollivier_ricci(mi_matrix, threshold=None, alpha=0.5):
    """Deterministic mock: scalar curvature = mean of MI off-diagonal."""
    n = mi_matrix.shape[0]
    off_diag = mi_matrix[np.triu_indices(n, k=1)]
    sc = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0
    edges = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges[(i, j)] = float(mi_matrix[i, j])
            idx += 1
    return {"edge_curvatures": edges, "scalar_curvature": sc}


def _mock_forman_ricci(mi_matrix, threshold=None):
    """Deterministic mock: negative of Ollivier for variety."""
    result = _mock_ollivier_ricci(mi_matrix, threshold=threshold)
    result["scalar_curvature"] = -result["scalar_curvature"]
    result["edge_curvatures"] = {k: -v for k, v in result["edge_curvatures"].items()}
    return result


def _mock_mi_to_graph(mi_matrix, threshold=None):
    """Return a mock networkx-like object (not used in experiments directly)."""
    return MagicMock()


@pytest.fixture(autouse=True)
def mock_curvature_module():
    """Patch curvature functions for all tests."""
    with patch("src.curvature_experiments.ollivier_ricci", side_effect=_mock_ollivier_ricci), \
         patch("src.curvature_experiments.forman_ricci", side_effect=_mock_forman_ricci), \
         patch("src.curvature_experiments.mi_to_graph", side_effect=_mock_mi_to_graph):
        yield


# ---------------------------------------------------------------------------
# Test Experiment 1: Curvature vs Partiality
# ---------------------------------------------------------------------------

class TestCurvatureVsPartiality:

    def test_returns_dict_with_expected_keys(self):
        from src.curvature_experiments import experiment_curvature_vs_partiality
        result = experiment_curvature_vs_partiality(
            n_qubits=4, n_samples=2, use_gpu=False
        )
        assert isinstance(result, dict)
        assert result["experiment"] == "curvature_vs_partiality"
        assert result["n_qubits"] == 4
        assert "results_table" in result
        assert "elapsed_seconds" in result

    def test_results_table_has_records(self):
        from src.curvature_experiments import experiment_curvature_vs_partiality
        result = experiment_curvature_vs_partiality(
            n_qubits=4, n_samples=1, use_gpu=False
        )
        table = result["results_table"]
        assert len(table) > 0
        # Should have k=3 and k=4 at minimum
        ks = [r["k"] for r in table]
        assert 3 in ks
        assert 4 in ks  # full system

    def test_curvatures_are_finite(self):
        from src.curvature_experiments import experiment_curvature_vs_partiality
        result = experiment_curvature_vs_partiality(
            n_qubits=4, n_samples=1, use_gpu=False
        )
        for rec in result["results_table"]:
            assert np.isfinite(rec["scalar_curvature_mean"])
            assert np.isfinite(rec["scalar_curvature_std"])
            assert rec["k_over_N"] > 0
            assert rec["k_over_N"] <= 1.0


# ---------------------------------------------------------------------------
# Test Experiment 2: Curvature vs Entanglement
# ---------------------------------------------------------------------------

class TestCurvatureVsEntanglement:

    def test_returns_dict_with_expected_keys(self):
        from src.curvature_experiments import experiment_curvature_vs_entanglement
        result = experiment_curvature_vs_entanglement(
            n_qubits=4, n_samples=1, delta_values=[1.0, 0.5], use_gpu=False
        )
        assert isinstance(result, dict)
        assert result["experiment"] == "curvature_vs_entanglement"
        assert "records" in result
        assert "correlation_entropy_ollivier" in result
        assert "correlation_entropy_forman" in result

    def test_records_match_delta_values(self):
        from src.curvature_experiments import experiment_curvature_vs_entanglement
        deltas = [2.0, 1.0, 0.5]
        result = experiment_curvature_vs_entanglement(
            n_qubits=4, n_samples=1, delta_values=deltas, use_gpu=False
        )
        assert len(result["records"]) == len(deltas)
        for rec, d in zip(result["records"], deltas):
            assert rec["delta"] == d

    def test_entropy_and_curvature_are_finite(self):
        from src.curvature_experiments import experiment_curvature_vs_entanglement
        result = experiment_curvature_vs_entanglement(
            n_qubits=4, n_samples=1, delta_values=[1.0], use_gpu=False
        )
        for rec in result["records"]:
            assert np.isfinite(rec["half_system_entropy_mean"])
            assert np.isfinite(rec["ollivier_scalar_mean"])
            assert np.isfinite(rec["forman_scalar_mean"])


# ---------------------------------------------------------------------------
# Test Experiment 3: Local vs Non-Local
# ---------------------------------------------------------------------------

class TestLocalVsNonLocal:

    def test_returns_dict_with_expected_keys(self):
        from src.curvature_experiments import experiment_local_vs_nonlocal
        result = experiment_local_vs_nonlocal(
            n_qubits=4, k=3, n_samples=2, use_gpu=False
        )
        assert isinstance(result, dict)
        assert result["experiment"] == "local_vs_nonlocal"
        assert "all_to_all" in result
        assert "chain" in result
        assert "chain_more_negative" in result

    def test_both_geometries_have_data(self):
        from src.curvature_experiments import experiment_local_vs_nonlocal
        result = experiment_local_vs_nonlocal(
            n_qubits=4, k=3, n_samples=3, use_gpu=False
        )
        assert len(result["all_to_all"]["ollivier_scalars"]) == 3
        assert len(result["chain"]["ollivier_scalars"]) == 3
        assert len(result["all_to_all"]["forman_scalars"]) == 3
        assert len(result["chain"]["forman_scalars"]) == 3

    def test_curvatures_are_finite(self):
        from src.curvature_experiments import experiment_local_vs_nonlocal
        result = experiment_local_vs_nonlocal(
            n_qubits=4, k=3, n_samples=1, use_gpu=False
        )
        for key in ["all_to_all", "chain"]:
            assert np.isfinite(result[key]["ollivier_mean"])
            assert np.isfinite(result[key]["forman_mean"])


# ---------------------------------------------------------------------------
# Test Experiment 4: Perspective-Dependent Curvature
# ---------------------------------------------------------------------------

class TestPerspectiveDependentCurvature:

    def test_returns_dict_with_expected_keys(self):
        from src.curvature_experiments import experiment_perspective_dependent_curvature
        result = experiment_perspective_dependent_curvature(
            n_qubits=4, k=3, n_samples=2, max_subsets=4, use_gpu=False
        )
        assert isinstance(result, dict)
        assert result["experiment"] == "perspective_dependent_curvature"
        assert "sample_records" in result
        assert "aggregate" in result

    def test_sample_records_populated(self):
        from src.curvature_experiments import experiment_perspective_dependent_curvature
        result = experiment_perspective_dependent_curvature(
            n_qubits=4, k=3, n_samples=2, max_subsets=4, use_gpu=False
        )
        assert len(result["sample_records"]) == 2
        for rec in result["sample_records"]:
            assert "curvature_mean" in rec
            assert "curvature_std" in rec
            assert "curvature_range" in rec
            assert "curvatures" in rec
            assert len(rec["curvatures"]) > 0

    def test_multiple_perspectives_give_distribution(self):
        from src.curvature_experiments import experiment_perspective_dependent_curvature
        # N=4, k=3 -> C(4,3)=4 subsets, all enumerated
        result = experiment_perspective_dependent_curvature(
            n_qubits=4, k=3, n_samples=1, max_subsets=10, use_gpu=False
        )
        rec = result["sample_records"][0]
        assert rec["n_subsets"] == 4  # C(4,3) = 4
        assert len(rec["curvatures"]) == 4

    def test_aggregate_has_spread_info(self):
        from src.curvature_experiments import experiment_perspective_dependent_curvature
        result = experiment_perspective_dependent_curvature(
            n_qubits=4, k=3, n_samples=2, max_subsets=4, use_gpu=False
        )
        agg = result["aggregate"]
        assert "mean_spread_std" in agg
        assert "mean_range" in agg
        assert "mean_iqr" in agg
        assert "is_perspective_dependent" in agg


# ---------------------------------------------------------------------------
# Test run_all_curvature
# ---------------------------------------------------------------------------

class TestRunAll:

    def test_runs_and_returns_all_four(self):
        from src.curvature_experiments import run_all_curvature
        results = run_all_curvature(n_qubits=4, use_gpu=False)
        assert "exp1" in results
        assert "exp2" in results
        assert "exp3" in results
        assert "exp4" in results


# ---------------------------------------------------------------------------
# Test helper: _curvature_stats
# ---------------------------------------------------------------------------

class TestCurvatureStats:

    def test_empty_dict(self):
        from src.curvature_experiments import _curvature_stats
        stats = _curvature_stats({})
        assert stats["n_edges"] == 0
        assert stats["mean"] == 0.0

    def test_populated_dict(self):
        from src.curvature_experiments import _curvature_stats
        edges = {(0, 1): 0.5, (1, 2): -0.3, (0, 2): 0.1}
        stats = _curvature_stats(edges)
        assert stats["n_edges"] == 3
        assert stats["n_positive"] == 2
        assert stats["n_negative"] == 1
        assert np.isclose(stats["mean"], 0.1, atol=1e-10)
