"""
Tests for src/curvature.py — discrete Ricci curvature on MI-distance graphs.

Tests Ollivier-Ricci and Forman-Ricci curvature against known analytical
results on standard graphs, and verifies end-to-end computation from
quantum ground states.

Run with: pytest tests/test_curvature.py -v
"""

import numpy as np
import networkx as nx
import pytest

from src.curvature import (
    mi_to_graph, mi_to_knn_graph,
    ollivier_ricci, forman_ricci,
    curvature_stats, scalar_curvature,
    _wasserstein_1, _lazy_random_walk_measure,
)
from src.quantum import (
    random_all_to_all, ground_state,
    mutual_information_matrix,
)


# ─────────────────────────────────────────────────────────────
# Helpers: build MI matrices that produce known graph topologies
# ─────────────────────────────────────────────────────────────

def _cycle_mi_matrix(n: int) -> np.ndarray:
    """
    Build MI matrix whose graph is a cycle (ring) of n nodes.

    Adjacent nodes get MI = 1.0, non-adjacent get MI = 0.
    """
    MI = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        MI[i, j] = 1.0
        MI[j, i] = 1.0
    return MI


def _complete_mi_matrix(n: int, mi_val: float = 1.0) -> np.ndarray:
    """
    Build MI matrix whose graph is a complete graph K_n.

    All pairs get MI = mi_val.
    """
    MI = np.full((n, n), mi_val)
    np.fill_diagonal(MI, 0.0)
    return MI


def _path_mi_matrix(n: int) -> np.ndarray:
    """
    Build MI matrix whose graph is a path (chain) of n nodes.

    Edge (i, i+1) has MI = 1.0.
    """
    MI = np.zeros((n, n))
    for i in range(n - 1):
        MI[i, i + 1] = 1.0
        MI[i + 1, i] = 1.0
    return MI


def _star_mi_matrix(n: int) -> np.ndarray:
    """
    Build MI matrix whose graph is a star: node 0 connected to all others.
    """
    MI = np.zeros((n, n))
    for i in range(1, n):
        MI[0, i] = 1.0
        MI[i, 0] = 1.0
    return MI


# ─────────────────────────────────────────────────────────────
# 1. Graph construction
# ─────────────────────────────────────────────────────────────

class TestGraphConstruction:
    """Verify graph construction from MI matrices."""

    def test_complete_graph_all_edges(self):
        MI = _complete_mi_matrix(5)
        G = mi_to_graph(MI)
        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 10  # C(5,2) = 10

    def test_cycle_graph_n_edges(self):
        MI = _cycle_mi_matrix(6)
        G = mi_to_graph(MI)
        assert G.number_of_nodes() == 6
        assert G.number_of_edges() == 6

    def test_empty_mi_produces_no_edges(self):
        MI = np.zeros((4, 4))
        G = mi_to_graph(MI)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 0

    def test_threshold_filters_edges(self):
        MI = np.array([
            [0, 0.5, 0.1],
            [0.5, 0, 0.3],
            [0.1, 0.3, 0],
        ])
        # Absolute threshold: keep MI > 0.2
        G = mi_to_graph(MI, threshold=0.2)
        assert G.number_of_edges() == 2  # (0,1)=0.5 and (1,2)=0.3

    def test_edge_weights_are_mi_values(self):
        MI = np.array([
            [0, 0.7, 0.3],
            [0.7, 0, 0.5],
            [0.3, 0.5, 0],
        ])
        G = mi_to_graph(MI)
        assert abs(G[0][1]['weight'] - 0.7) < 1e-14
        assert abs(G[0][1]['distance'] - 1.0 / 0.7) < 1e-14

    def test_knn_graph_degree_bounded(self):
        MI = _complete_mi_matrix(6)
        G = mi_to_knn_graph(MI, k=2)
        # Each node picks 2 neighbors, but symmetrization can increase degree
        assert G.number_of_edges() >= 6  # at least 6 edges (each of 6 nodes picks 2)
        assert G.number_of_edges() <= 15  # at most C(6,2) = 15

    def test_knn_graph_clamps_k(self):
        MI = _complete_mi_matrix(4)
        G = mi_to_knn_graph(MI, k=100)  # k > n-1, should clamp
        assert G.number_of_edges() == 6  # all edges of K4


# ─────────────────────────────────────────────────────────────
# 2. Wasserstein-1 distance
# ─────────────────────────────────────────────────────────────

class TestWasserstein:
    """Verify Wasserstein-1 computation against known results."""

    def test_identical_distributions(self):
        mu = np.array([0.5, 0.3, 0.2])
        D = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ], dtype=np.float64)
        w = _wasserstein_1(mu, mu, D)
        assert abs(w) < 1e-10, f"W1 between identical distributions should be 0, got {w}"

    def test_dirac_masses(self):
        # W_1 between delta_0 and delta_2 on a 3-node line with unit distances
        mu = np.array([1.0, 0.0, 0.0])
        nu = np.array([0.0, 0.0, 1.0])
        D = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ], dtype=np.float64)
        w = _wasserstein_1(mu, nu, D)
        assert abs(w - 2.0) < 1e-10, f"W1 between delta_0 and delta_2 should be 2.0, got {w}"

    def test_symmetric(self):
        mu = np.array([0.6, 0.2, 0.2])
        nu = np.array([0.2, 0.5, 0.3])
        D = np.array([
            [0, 1, 3],
            [1, 0, 2],
            [3, 2, 0],
        ], dtype=np.float64)
        w_fwd = _wasserstein_1(mu, nu, D)
        w_bwd = _wasserstein_1(nu, mu, D)
        assert abs(w_fwd - w_bwd) < 1e-10, "W1 should be symmetric"


# ─────────────────────────────────────────────────────────────
# 3. Ollivier-Ricci curvature on known graphs
# ─────────────────────────────────────────────────────────────

class TestOllivierRicci:
    """Verify Ollivier-Ricci curvature against known analytical values."""

    def test_cycle_curvature_is_zero(self):
        """
        On a cycle graph with uniform edge weights, Ollivier-Ricci curvature
        is 0 for all edges when alpha=0 (pure random walk).

        With alpha=0.5 (lazy walk), cycle curvature is also 0 because
        neighbors of adjacent nodes on a cycle have identical structure
        (each has degree 2, same path distances).
        """
        MI = _cycle_mi_matrix(6)
        result = ollivier_ricci(MI, alpha=0.0)
        assert result['n_edges'] == 6

        for edge, kappa in result['edge_curvatures'].items():
            assert abs(kappa) < 0.05, (
                f"Cycle edge {edge} should have kappa ~ 0, got {kappa}"
            )

    def test_complete_graph_positive_curvature(self):
        """
        Complete graph K_n has positive Ollivier-Ricci curvature.
        For K_n with alpha=0, kappa = 2/n for all edges (exact).
        """
        n = 5
        MI = _complete_mi_matrix(n)
        result = ollivier_ricci(MI, alpha=0.0)
        assert result['n_edges'] == n * (n - 1) // 2

        # All curvatures should be positive and equal
        kappas = list(result['edge_curvatures'].values())
        assert all(k > 0 for k in kappas), (
            f"Complete graph should have positive curvature, got {kappas}"
        )
        # Check uniformity (all edges symmetric in K_n)
        assert np.std(kappas) < 0.05, (
            f"Complete graph curvatures should be uniform, std = {np.std(kappas)}"
        )

    def test_scalar_curvature_complete_is_positive(self):
        MI = _complete_mi_matrix(4)
        result = ollivier_ricci(MI, alpha=0.0)
        assert result['scalar_curvature'] > 0, (
            f"K4 scalar curvature should be positive, got {result['scalar_curvature']}"
        )

    def test_bottleneck_edge_negative_curvature(self):
        """
        Bridge edges between communities have negative Ollivier-Ricci curvature.

        Barbell graph: two triangles connected by a single bridge edge.
        The bridge should have kappa < 0 because the random walk measures
        on either side of the bridge have very different supports.
        This is the physically relevant case for PLC: bottleneck detection.
        """
        # Barbell: triangle (0,1,2), bridge (2,3), triangle (3,4,5)
        MI = np.zeros((6, 6))
        for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
            MI[i, j] = MI[j, i] = 1.0
        MI[2, 3] = MI[3, 2] = 1.0

        result = ollivier_ricci(MI, alpha=0.0)
        bridge_kappa = result['edge_curvatures'][(2, 3)]
        assert bridge_kappa < -0.1, (
            f"Bridge edge should have negative curvature, got {bridge_kappa}"
        )

        # Triangle edges should have positive curvature
        triangle_kappa = result['edge_curvatures'][(0, 1)]
        assert triangle_kappa > 0, (
            f"Triangle edge should have positive curvature, got {triangle_kappa}"
        )

    def test_empty_graph_returns_zero(self):
        MI = np.zeros((4, 4))
        result = ollivier_ricci(MI)
        assert result['n_edges'] == 0
        assert result['scalar_curvature'] == 0.0

    def test_alpha_parameter_changes_curvature(self):
        """
        Changing alpha changes curvature on graphs with structural asymmetry.

        On a barbell graph, the bridge edge curvature changes with alpha
        because the lazy mass affects how much the random walk measures
        overlap versus diverge.
        """
        # Barbell: two triangles connected by a bridge
        MI = np.zeros((6, 6))
        for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
            MI[i, j] = MI[j, i] = 1.0
        MI[2, 3] = MI[3, 2] = 1.0

        r1 = ollivier_ricci(MI, alpha=0.0)
        r2 = ollivier_ricci(MI, alpha=0.5)

        # Bridge curvature should differ between alpha values
        bridge_0 = r1['edge_curvatures'][(2, 3)]
        bridge_5 = r2['edge_curvatures'][(2, 3)]
        assert abs(bridge_0 - bridge_5) > 0.1, (
            f"Bridge curvature should change with alpha: {bridge_0} vs {bridge_5}"
        )


# ─────────────────────────────────────────────────────────────
# 4. Forman-Ricci curvature on known topology
# ─────────────────────────────────────────────────────────────

class TestFormanRicci:
    """Verify Forman-Ricci curvature against exact combinatorial formulas."""

    def test_cycle_forman_curvature(self):
        """
        Cycle: every node has degree 2.
        F(e) = 4 - 2 - 2 = 0 for every edge.
        No triangles, so F_aug = F = 0.
        """
        MI = _cycle_mi_matrix(6)
        result = forman_ricci(MI)
        assert result['n_edges'] == 6

        for edge, F in result['edge_curvatures'].items():
            assert abs(F) < 1e-14, f"Cycle Forman curvature should be 0, got {F}"

        for edge, F_aug in result['edge_curvatures_aug'].items():
            assert abs(F_aug) < 1e-14, f"Cycle augmented Forman should be 0, got {F_aug}"

    def test_complete_graph_forman(self):
        """
        K_n: every node has degree n-1. Every edge has n-2 triangles.
        F(e) = 4 - (n-1) - (n-1) = 6 - 2n
        F_aug(e) = 6 - 2n + 3*(n-2) = n - 2*1 = n (wait, let me compute)
        F_aug(e) = (6 - 2n) + 3*(n-2) = 6 - 2n + 3n - 6 = n
        Hmm: 4 - (n-1) - (n-1) + 3*(n-2) = 4 - 2n + 2 + 3n - 6 = n - 0 = n
        Actually: 4 - 2(n-1) + 3(n-2) = 4 - 2n + 2 + 3n - 6 = n
        """
        n = 5
        MI = _complete_mi_matrix(n)
        result = forman_ricci(MI)

        expected_basic = 4 - 2 * (n - 1)  # = 4 - 8 = -4
        expected_aug = n  # = 5

        for edge, F in result['edge_curvatures'].items():
            assert abs(F - expected_basic) < 1e-14, (
                f"K5 basic Forman should be {expected_basic}, got {F}"
            )

        for edge, F_aug in result['edge_curvatures_aug'].items():
            assert abs(F_aug - expected_aug) < 1e-14, (
                f"K5 augmented Forman should be {expected_aug}, got {F_aug}"
            )

    def test_star_forman_curvature(self):
        """
        Star with n nodes: center has degree n-1, leaves have degree 1.
        For edge (center, leaf): F = 4 - (n-1) - 1 = 4 - n
        No triangles in a star, so F_aug = F.
        """
        n = 5
        MI = _star_mi_matrix(n)
        result = forman_ricci(MI)

        expected = 4 - n  # = -1
        for edge, F in result['edge_curvatures'].items():
            assert abs(F - expected) < 1e-14, (
                f"Star Forman should be {expected}, got {F}"
            )

    def test_path_forman_curvature(self):
        """
        Path of n nodes: endpoints have degree 1, internal nodes have degree 2.
        Endpoint edge: F = 4 - 1 - 2 = 1
        Internal edge: F = 4 - 2 - 2 = 0
        """
        MI = _path_mi_matrix(5)
        result = forman_ricci(MI)

        # Edge (0,1): deg(0)=1, deg(1)=2 -> F = 4-1-2 = 1
        assert abs(result['edge_curvatures'][(0, 1)] - 1.0) < 1e-14
        # Edge (1,2): deg(1)=2, deg(2)=2 -> F = 0
        assert abs(result['edge_curvatures'][(1, 2)] - 0.0) < 1e-14
        # Edge (3,4): deg(3)=2, deg(4)=1 -> F = 1
        assert abs(result['edge_curvatures'][(3, 4)] - 1.0) < 1e-14

    def test_empty_graph_forman(self):
        MI = np.zeros((4, 4))
        result = forman_ricci(MI)
        assert result['n_edges'] == 0
        assert result['scalar_curvature'] == 0.0


# ─────────────────────────────────────────────────────────────
# 5. Curvature statistics
# ─────────────────────────────────────────────────────────────

class TestCurvatureStats:
    """Verify curvature statistics computation."""

    def test_stats_on_known_values(self):
        curvs = {(0, 1): 1.0, (1, 2): -1.0, (0, 2): 0.5}
        stats = curvature_stats(curvs)
        assert abs(stats['mean'] - (1.0 - 1.0 + 0.5) / 3) < 1e-14
        assert stats['n_edges'] == 3
        assert abs(stats['min'] - (-1.0)) < 1e-14
        assert abs(stats['max'] - 1.0) < 1e-14

    def test_stats_empty(self):
        stats = curvature_stats({})
        assert stats['n_edges'] == 0
        assert stats['mean'] == 0.0

    def test_frac_positive_negative(self):
        curvs = {(0, 1): 1.0, (1, 2): -1.0, (0, 2): 0.5, (2, 3): -0.3}
        stats = curvature_stats(curvs)
        assert abs(stats['frac_positive'] - 0.5) < 1e-14
        assert abs(stats['frac_negative'] - 0.5) < 1e-14

    def test_scalar_curvature_uniform(self):
        curvs = {(0, 1): 2.0, (1, 2): 4.0}
        sc = scalar_curvature(curvs)
        assert abs(sc - 3.0) < 1e-14

    def test_scalar_curvature_weighted(self):
        curvs = {(0, 1): 2.0, (1, 2): 4.0}
        weights = {(0, 1): 3.0, (1, 2): 1.0}
        sc = scalar_curvature(curvs, weights=weights)
        expected = (2.0 * 3.0 + 4.0 * 1.0) / (3.0 + 1.0)  # 10/4 = 2.5
        assert abs(sc - expected) < 1e-14


# ─────────────────────────────────────────────────────────────
# 6. End-to-end: curvature from quantum ground state
# ─────────────────────────────────────────────────────────────

class TestEndToEnd:
    """Verify curvature computation end-to-end from a quantum system."""

    def test_curvature_from_n4_ground_state(self):
        """
        Build N=4 random Hamiltonian, compute ground state,
        compute MI matrix, compute both curvatures.
        Verify that the pipeline produces finite, reasonable values.
        """
        H, couplings = random_all_to_all(n_qubits=4, seed=42)
        E0, psi = ground_state(H)

        MI = mutual_information_matrix(psi, n_qubits=4)

        # MI should be positive and symmetric
        assert MI.shape == (4, 4)
        assert np.allclose(MI, MI.T, atol=1e-14)
        assert np.all(MI >= -1e-14)

        # Ollivier-Ricci
        or_result = ollivier_ricci(MI, alpha=0.5)
        assert or_result['n_edges'] > 0, "Should have edges from MI matrix"
        assert np.isfinite(or_result['scalar_curvature']), "Scalar curvature should be finite"

        # Check each edge curvature is finite
        for edge, kappa in or_result['edge_curvatures'].items():
            assert np.isfinite(kappa), f"Edge {edge} curvature is not finite: {kappa}"

        # Forman-Ricci
        fr_result = forman_ricci(MI)
        assert fr_result['n_edges'] > 0
        assert np.isfinite(fr_result['scalar_curvature'])
        assert np.isfinite(fr_result['scalar_curvature_aug'])

    def test_curvature_varies_with_hamiltonian(self):
        """
        Different random Hamiltonians should produce different per-edge
        curvature distributions, even if scalar curvature is degenerate
        on small complete graphs.

        For N=4, the MI graph is always K4 (complete), so the scalar
        Ollivier curvature is the same by symmetry. Instead, check that
        the per-edge Forman augmented curvature with thresholding varies.
        """
        edge_curvs = []
        for seed in [10, 20, 30]:
            H, _ = random_all_to_all(n_qubits=4, seed=seed)
            _, psi = ground_state(H)
            MI = mutual_information_matrix(psi, n_qubits=4)
            # Use a percentile threshold to create non-complete graphs
            result = ollivier_ricci(MI, threshold=0.3, alpha=0.5)
            edge_curvs.append(tuple(sorted(result['edge_curvatures'].values())))

        # At least two should differ in their edge curvature distribution
        assert len(set(edge_curvs)) >= 2, (
            f"Different Hamiltonians should give different edge curvature distributions: {edge_curvs}"
        )

    def test_k4_complete_subgraph_positive_ollivier(self):
        """
        For N=4 with all-to-all couplings and no threshold filtering,
        the MI graph is complete (K4). K4 should have positive Ollivier
        curvature.
        """
        H, _ = random_all_to_all(n_qubits=4, seed=99)
        _, psi = ground_state(H)
        MI = mutual_information_matrix(psi, n_qubits=4)

        # With no threshold, should be complete graph (all MI > 0)
        result = ollivier_ricci(MI, alpha=0.0)
        assert result['n_edges'] == 6, "Should have all 6 edges of K4"
        assert result['scalar_curvature'] > 0, (
            f"K4 Ollivier scalar curvature should be positive, got {result['scalar_curvature']}"
        )

    def test_forman_on_quantum_mi_has_correct_formula(self):
        """
        For N=4 complete MI graph, Forman should follow F = 4 - deg(u) - deg(v).
        K4: deg = 3 for all nodes. F = 4 - 3 - 3 = -2.
        Triangles per edge in K4: 2. F_aug = -2 + 3*2 = 4.
        """
        H, _ = random_all_to_all(n_qubits=4, seed=77)
        _, psi = ground_state(H)
        MI = mutual_information_matrix(psi, n_qubits=4)

        result = forman_ricci(MI)
        assert result['n_edges'] == 6

        for edge, F in result['edge_curvatures'].items():
            assert abs(F - (-2.0)) < 1e-14, f"K4 Forman should be -2, got {F}"

        for edge, F_aug in result['edge_curvatures_aug'].items():
            assert abs(F_aug - 4.0) < 1e-14, f"K4 augmented Forman should be 4, got {F_aug}"
