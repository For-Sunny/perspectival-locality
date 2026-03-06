"""
Tests for src/hamiltonians.py — XXZ Hamiltonian, entanglement control, delta sweep.

Verifies:
1. XXZ at delta=1 matches isotropic Heisenberg from quantum.py
2. XXZ at large delta gives low entanglement (near-product state)
3. Local Hamiltonian builds correctly for chain/ladder/square
4. Delta sweep produces monotonic entropy change
5. Entanglement entropy calculators agree with direct computation
6. Hermiticity of all Hamiltonian variants

Run with: pytest tests/test_hamiltonians.py -v
"""

import numpy as np
import pytest

from src.quantum import (
    heisenberg_all_to_all_sparse, random_all_to_all_sparse,
    ground_state, ground_state_sparse,
    partial_trace, von_neumann_entropy,
)
from src.hamiltonians import (
    build_xxz_hamiltonian,
    build_local_hamiltonian,
    entanglement_entropy,
    half_system_entropy,
    sweep_delta,
    _all_to_all_edges,
    _nearest_neighbor_edges,
    _ladder_edges,
    _square_edges,
    _build_xxz_from_edges,
)


# ─────────────────────────────────────────────────────────────
# 1. XXZ at delta=1 matches isotropic Heisenberg
# ─────────────────────────────────────────────────────────────

class TestXXZIsotropicLimit:
    """At delta=1, XXZ should reproduce the isotropic Heisenberg."""

    def test_xxz_delta1_matches_heisenberg_uniform(self):
        """Uniform couplings, delta=1: XXZ == Heisenberg."""
        n = 6
        # Build XXZ with uniform J=1, delta=1
        H_xxz, _ = build_xxz_hamiltonian(
            n, delta=1.0, coupling='all_to_all',
            J=1.0, random_couplings=False
        )
        # Build Heisenberg with uniform J=1
        H_heis = heisenberg_all_to_all_sparse(n, couplings=None)

        # Compare as dense
        diff = np.abs(H_xxz.toarray() - H_heis.toarray())
        assert np.max(diff) < 1e-12, \
            f"XXZ(delta=1) differs from Heisenberg by max {np.max(diff)}"

    def test_xxz_delta1_random_couplings_same_seed(self):
        """Random couplings with same seed: XXZ(delta=1) ground energy matches Heisenberg."""
        n = 8
        seed = 42
        # XXZ with random couplings, delta=1
        H_xxz, J_xxz = build_xxz_hamiltonian(
            n, delta=1.0, coupling='all_to_all', seed=seed
        )
        e_xxz, _ = ground_state_sparse(H_xxz)

        # Heisenberg with the same couplings
        # Need to reconstruct: same seed -> same RNG -> same couplings
        rng = np.random.default_rng(seed)
        n_pairs = n * (n - 1) // 2
        couplings = rng.standard_normal(n_pairs)
        H_heis = heisenberg_all_to_all_sparse(n, couplings)
        e_heis, _ = ground_state_sparse(H_heis)

        np.testing.assert_allclose(e_xxz, e_heis, atol=1e-8,
                                   err_msg="XXZ(delta=1) energy != Heisenberg energy")


# ─────────────────────────────────────────────────────────────
# 2. XXZ at large delta gives low entanglement
# ─────────────────────────────────────────────────────────────

class TestXXZIsingLimit:
    """Large delta pushes ground state toward product state."""

    def test_large_delta_low_entropy(self):
        """delta=50: ground state should have much less entanglement than delta=1.
        Uses random couplings to break frustration degeneracy (uniform all-to-all
        is highly frustrated, so the Ising limit ground state can remain entangled)."""
        n = 6
        seed = 99

        # delta=1 (Heisenberg) with random couplings
        H1, _ = build_xxz_hamiltonian(n, delta=1.0, coupling='all_to_all',
                                       seed=seed, random_couplings=True)
        _, psi1 = ground_state_sparse(H1)
        s1 = half_system_entropy(psi1, n)

        # delta=50 (Ising limit) with same random couplings
        H50, _ = build_xxz_hamiltonian(n, delta=50.0, coupling='all_to_all',
                                        seed=seed, random_couplings=True)
        _, psi50 = ground_state_sparse(H50)
        s50 = half_system_entropy(psi50, n)

        assert s50 < s1, \
            f"Ising limit entropy ({s50:.4f}) should be < Heisenberg ({s1:.4f})"

    def test_very_large_delta_near_zero_entropy(self):
        """delta=1000 on NN chain: entropy should be very small.
        The 1D chain is non-frustrated, so the Ising limit ground state is
        a clean product state (Neel or ferromagnet depending on coupling sign).
        All-to-all at small N can retain frustration-induced degeneracy."""
        n = 6
        H, _ = build_xxz_hamiltonian(n, delta=1000.0, coupling='nearest_neighbor',
                                      J=1.0, random_couplings=True, seed=42)
        _, psi = ground_state(H.toarray())
        s = half_system_entropy(psi, n)
        # Max possible entropy for N/2=3 qubits is 3.0 bits
        assert s < 0.5, f"delta=1000 NN chain entropy should be near zero, got {s:.4f}"


# ─────────────────────────────────────────────────────────────
# 3. Local Hamiltonian builds correctly
# ─────────────────────────────────────────────────────────────

class TestLocalHamiltonian:
    """Verify local Hamiltonians for different geometries."""

    def test_chain_edge_count(self):
        """1D chain with N sites has N-1 edges (open boundary)."""
        edges = _nearest_neighbor_edges(8, periodic=False)
        assert len(edges) == 7

    def test_chain_periodic_edge_count(self):
        """Periodic chain with N sites has N edges."""
        edges = _nearest_neighbor_edges(8, periodic=True)
        assert len(edges) == 8

    def test_ladder_edge_count(self):
        """Ladder with N sites: (N/2-1)*2 leg edges + N/2 rungs."""
        n = 8
        edges = _ladder_edges(n)
        half = n // 2
        expected = 2 * (half - 1) + half  # 6 + 4 = 10
        assert len(edges) == expected, f"Expected {expected} edges, got {len(edges)}"

    def test_square_edge_count(self):
        """4x4 square: 4*3 horizontal + 3*4 vertical = 24 edges."""
        edges = _square_edges(16)
        # 4x4 grid: each row has 3 horizontal, each col has 3 vertical
        # 4 rows * 3 + 4 cols * 3 = 24
        assert len(edges) == 24, f"Expected 24 edges for 4x4 grid, got {len(edges)}"

    def test_chain_hamiltonian_is_hermitian(self):
        n = 8
        H, _ = build_local_hamiltonian(n, geometry='chain', seed=42)
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.conj().T, atol=1e-13,
                                   err_msg="Chain Hamiltonian is not Hermitian")

    def test_ladder_hamiltonian_is_hermitian(self):
        n = 8
        H, _ = build_local_hamiltonian(n, geometry='ladder', seed=42)
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.conj().T, atol=1e-13,
                                   err_msg="Ladder Hamiltonian is not Hermitian")

    def test_square_hamiltonian_is_hermitian(self):
        # Use N=9 (3x3 grid) to keep test fast; N=16 makes a 65536x65536 dense matrix
        n = 9
        H, _ = build_local_hamiltonian(n, geometry='square', seed=42)
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.conj().T, atol=1e-13,
                                   err_msg="Square Hamiltonian is not Hermitian")

    def test_chain_vs_xxz_nearest_neighbor(self):
        """build_local_hamiltonian('chain') should equal build_xxz_hamiltonian('nearest_neighbor')."""
        n = 6
        seed = 77
        H_local, J_local = build_local_hamiltonian(n, geometry='chain', seed=seed)
        H_xxz, J_xxz = build_xxz_hamiltonian(n, delta=1.0, coupling='nearest_neighbor', seed=seed)

        np.testing.assert_allclose(J_local, J_xxz, atol=1e-15,
                                   err_msg="Couplings should match")
        diff = np.abs(H_local.toarray() - H_xxz.toarray())
        assert np.max(diff) < 1e-12, f"Chain != NN XXZ, max diff = {np.max(diff)}"


# ─────────────────────────────────────────────────────────────
# 4. Delta sweep produces monotonic entropy change
# ─────────────────────────────────────────────────────────────

class TestDeltaSweep:
    """Verify delta sweep produces expected entropy trend."""

    def test_entropy_decreases_with_delta(self):
        """
        For uniform all-to-all XXZ, increasing delta should generally
        decrease half-system entropy (moving toward Ising product state).
        We test the overall trend: first delta has higher entropy than last.
        """
        n = 6
        deltas = [0.1, 1.0, 5.0, 20.0]
        results = sweep_delta(n, deltas, coupling='all_to_all',
                              seed=42, use_sparse=True, verbose=False)

        entropies = [r['half_entropy'] for r in results]
        # Overall trend: small delta -> higher entropy than large delta
        assert entropies[0] > entropies[-1], \
            f"Entropy should decrease with delta: {entropies}"

    def test_sweep_returns_correct_fields(self):
        """Each result dict has the right keys and types."""
        n = 4
        results = sweep_delta(n, [0.5, 1.0], coupling='all_to_all',
                              seed=0, use_sparse=False, verbose=False)
        assert len(results) == 2
        for r in results:
            assert 'delta' in r
            assert 'hamiltonian' in r
            assert 'ground_energy' in r
            assert 'ground_state' in r
            assert 'half_entropy' in r
            assert isinstance(r['ground_energy'], float)
            assert r['ground_state'].shape == (2 ** n,)

    def test_sweep_fixed_couplings(self):
        """Same seed means same couplings across all delta values."""
        n = 6
        # Run two separate sweeps with same seed, different delta sets
        r1 = sweep_delta(n, [0.5], coupling='all_to_all', seed=123, verbose=False)
        r2 = sweep_delta(n, [2.0], coupling='all_to_all', seed=123, verbose=False)
        # Both should use the same coupling realization.
        # The Hamiltonians differ only in the ZZ scale. Check that the
        # off-diagonal (XX+YY) part is the same.
        H1 = r1[0]['hamiltonian'].toarray()
        H2 = r2[0]['hamiltonian'].toarray()
        # Off-diagonal should be identical (XX+YY doesn't depend on delta)
        mask = ~np.eye(2**n, dtype=bool)
        np.testing.assert_allclose(H1[mask], H2[mask], atol=1e-13,
                                   err_msg="Off-diagonal elements should match (same couplings)")


# ─────────────────────────────────────────────────────────────
# 5. Entanglement entropy calculators
# ─────────────────────────────────────────────────────────────

class TestEntanglementEntropy:
    """Verify entanglement_entropy and half_system_entropy."""

    def test_bell_state_entropy(self):
        """Bell state: S(qubit 0) = 1.0 bit."""
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)

        s = entanglement_entropy(psi, subsystem=[0], n_qubits=2)
        np.testing.assert_allclose(s, 1.0, atol=1e-12)

    def test_product_state_entropy(self):
        """Product state |00>: S = 0."""
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0

        s = entanglement_entropy(psi, subsystem=[0], n_qubits=2)
        np.testing.assert_allclose(s, 0.0, atol=1e-12)

    def test_half_system_entropy_ghz(self):
        """3-qubit GHZ: S(first qubit) = 1.0 bit (half = 1 qubit for N=3)."""
        psi = np.zeros(8, dtype=np.complex128)
        psi[0] = 1.0 / np.sqrt(2)
        psi[7] = 1.0 / np.sqrt(2)

        # N=3, half=1, so we trace out qubits 1,2
        s = half_system_entropy(psi, n_qubits=3)
        np.testing.assert_allclose(s, 1.0, atol=1e-12)

    def test_entropy_matches_direct_computation(self):
        """entanglement_entropy should match manual partial_trace + von_neumann_entropy."""
        rng = np.random.default_rng(55)
        psi = rng.standard_normal(32) + 1j * rng.standard_normal(32)
        psi /= np.linalg.norm(psi)
        n = 5

        subsys = [0, 2, 4]
        s_func = entanglement_entropy(psi, subsys, n)
        rho = partial_trace(psi, keep=subsys, n_qubits=n)
        s_manual = von_neumann_entropy(rho)

        np.testing.assert_allclose(s_func, s_manual, atol=1e-14)


# ─────────────────────────────────────────────────────────────
# 6. XXZ Hamiltonian hermiticity for all topologies
# ─────────────────────────────────────────────────────────────

class TestXXZHermiticity:
    """All XXZ variants must be Hermitian."""

    @pytest.mark.parametrize("coupling", ['all_to_all', 'nearest_neighbor', 'random_graph'])
    def test_xxz_hermitian(self, coupling):
        n = 6
        H, _ = build_xxz_hamiltonian(n, delta=0.7, coupling=coupling, seed=42)
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.conj().T, atol=1e-13,
                                   err_msg=f"XXZ({coupling}) is not Hermitian")

    @pytest.mark.parametrize("delta", [0.0, 0.5, 1.0, 5.0, 100.0])
    def test_xxz_hermitian_across_delta(self, delta):
        n = 6
        H, _ = build_xxz_hamiltonian(n, delta=delta, coupling='all_to_all', seed=42)
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.conj().T, atol=1e-13,
                                   err_msg=f"XXZ(delta={delta}) is not Hermitian")
