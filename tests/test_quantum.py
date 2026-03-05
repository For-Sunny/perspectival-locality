"""
Minimal test suite for src/quantum.py — physics layer verification.

Tests Pauli algebra, entanglement measures, partial traces, and
Hamiltonian construction against exact analytical benchmarks.

Run with: pytest tests/test_quantum.py -v
"""

import numpy as np
import pytest

from src.quantum import (
    SIGMA_X, SIGMA_Y, SIGMA_Z, IDENTITY_2,
    single_site_op, two_site_op,
    heisenberg_all_to_all, heisenberg_all_to_all_sparse,
    random_all_to_all, random_all_to_all_sparse,
    ground_state, ground_state_sparse,
    partial_trace, density_matrix,
    von_neumann_entropy, mutual_information,
    connected_correlation,
)


# ─────────────────────────────────────────────────────────────
# 1. Pauli algebra
# ─────────────────────────────────────────────────────────────

class TestPauliAlgebra:
    """Verify fundamental Pauli matrix identities."""

    def test_sigma_x_squared_is_identity(self):
        result = SIGMA_X @ SIGMA_X
        np.testing.assert_allclose(result, IDENTITY_2, atol=1e-15)

    def test_sigma_y_squared_is_identity(self):
        result = SIGMA_Y @ SIGMA_Y
        np.testing.assert_allclose(result, IDENTITY_2, atol=1e-15)

    def test_sigma_z_squared_is_identity(self):
        result = SIGMA_Z @ SIGMA_Z
        np.testing.assert_allclose(result, IDENTITY_2, atol=1e-15)

    def test_commutator_xy_equals_2i_sigma_z(self):
        # [sigma_x, sigma_y] = sigma_x sigma_y - sigma_y sigma_x = 2i sigma_z
        commutator = SIGMA_X @ SIGMA_Y - SIGMA_Y @ SIGMA_X
        expected = 2j * SIGMA_Z
        np.testing.assert_allclose(commutator, expected, atol=1e-15)

    def test_commutator_yz_equals_2i_sigma_x(self):
        commutator = SIGMA_Y @ SIGMA_Z - SIGMA_Z @ SIGMA_Y
        expected = 2j * SIGMA_X
        np.testing.assert_allclose(commutator, expected, atol=1e-15)

    def test_commutator_zx_equals_2i_sigma_y(self):
        commutator = SIGMA_Z @ SIGMA_X - SIGMA_X @ SIGMA_Z
        expected = 2j * SIGMA_Y
        np.testing.assert_allclose(commutator, expected, atol=1e-15)

    def test_anticommutator_xy_vanishes(self):
        # {sigma_x, sigma_y} = 0
        anticommutator = SIGMA_X @ SIGMA_Y + SIGMA_Y @ SIGMA_X
        np.testing.assert_allclose(anticommutator, np.zeros((2, 2)), atol=1e-15)

    def test_paulis_are_hermitian(self):
        for name, sigma in [("X", SIGMA_X), ("Y", SIGMA_Y), ("Z", SIGMA_Z)]:
            np.testing.assert_allclose(
                sigma, sigma.conj().T, atol=1e-15,
                err_msg=f"sigma_{name} is not Hermitian"
            )

    def test_paulis_are_traceless(self):
        for name, sigma in [("X", SIGMA_X), ("Y", SIGMA_Y), ("Z", SIGMA_Z)]:
            assert abs(np.trace(sigma)) < 1e-15, f"sigma_{name} is not traceless"


# ─────────────────────────────────────────────────────────────
# 2-3. Mutual information: Bell state and product state
# ─────────────────────────────────────────────────────────────

class TestMutualInformation:
    """Verify MI against exact analytical values."""

    def test_bell_state_mi_equals_2(self):
        # Bell state |00> + |11> / sqrt(2) has MI = 2.0 bits (log2)
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0 / np.sqrt(2)  # |00>
        psi[3] = 1.0 / np.sqrt(2)  # |11>

        mi = mutual_information(psi, sites_a=[0], sites_b=[1], n_qubits=2)
        np.testing.assert_allclose(mi, 2.0, atol=1e-12,
                                   err_msg="Bell state MI should be exactly 2.0 bits")

    def test_product_state_mi_equals_0(self):
        # Product state |0>|0> has MI = 0.0
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0  # |00>

        mi = mutual_information(psi, sites_a=[0], sites_b=[1], n_qubits=2)
        np.testing.assert_allclose(mi, 0.0, atol=1e-12,
                                   err_msg="Product state MI should be exactly 0.0")


# ─────────────────────────────────────────────────────────────
# 4. GHZ partial trace
# ─────────────────────────────────────────────────────────────

class TestPartialTrace:
    """Verify partial trace against analytical results."""

    def test_ghz_trace_out_qubit_2(self):
        # 3-qubit GHZ: (|000> + |111>) / sqrt(2)
        # Tracing out qubit 2 gives rho = (|00><00| + |11><11|) / 2
        psi = np.zeros(8, dtype=np.complex128)
        psi[0] = 1.0 / np.sqrt(2)  # |000>
        psi[7] = 1.0 / np.sqrt(2)  # |111>

        rho_01 = partial_trace(psi, keep=[0, 1], n_qubits=3)

        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = 0.5  # |00><00|
        expected[3, 3] = 0.5  # |11><11|

        np.testing.assert_allclose(rho_01, expected, atol=1e-14,
                                   err_msg="GHZ partial trace over qubit 2 is wrong")

    def test_bell_state_single_qubit_trace(self):
        # Bell state traced to single qubit should give I/2
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)

        rho_0 = partial_trace(psi, keep=[0], n_qubits=2)
        expected = 0.5 * IDENTITY_2
        np.testing.assert_allclose(rho_0, expected, atol=1e-14)

    def test_partial_trace_preserves_trace(self):
        # Tr(rho_reduced) = 1 for any pure state
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)

        rho = partial_trace(psi, keep=[0, 1], n_qubits=4)
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-13)


# ─────────────────────────────────────────────────────────────
# 5-6. Von Neumann entropy
# ─────────────────────────────────────────────────────────────

class TestEntropy:
    """Verify von Neumann entropy against analytical values."""

    def test_maximally_mixed_state_entropy(self):
        # S(I/2) = 1.0 bit (log2)
        rho = 0.5 * IDENTITY_2
        s = von_neumann_entropy(rho)
        np.testing.assert_allclose(s, 1.0, atol=1e-14,
                                   err_msg="Entropy of I/2 should be 1.0 bit")

    def test_pure_state_entropy_is_zero(self):
        # S(|0><0|) = 0.0
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        s = von_neumann_entropy(rho)
        np.testing.assert_allclose(s, 0.0, atol=1e-14,
                                   err_msg="Entropy of pure state should be 0.0")

    def test_maximally_mixed_2qubit_entropy(self):
        # S(I/4) = 2.0 bits
        rho = 0.25 * np.eye(4, dtype=np.complex128)
        s = von_neumann_entropy(rho)
        np.testing.assert_allclose(s, 2.0, atol=1e-14)

    def test_entropy_is_non_negative(self):
        rng = np.random.default_rng(99)
        psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        rho = partial_trace(psi, keep=[0], n_qubits=3)
        s = von_neumann_entropy(rho)
        assert s >= -1e-14, f"Entropy should be non-negative, got {s}"


# ─────────────────────────────────────────────────────────────
# 7. Connected correlator
# ─────────────────────────────────────────────────────────────

class TestConnectedCorrelation:
    """Verify connected correlation C_ZZ = <ZZ> - <Z><Z>."""

    def test_product_state_czz_is_zero(self):
        # For |0>|0>, <ZZ> = 1, <Z> = 1, <Z> = 1, so C_ZZ = 1 - 1*1 = 0
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0

        czz = connected_correlation(psi, site_a=0, site_b=1, n_qubits=2, pauli='Z')
        np.testing.assert_allclose(czz, 0.0, atol=1e-14,
                                   err_msg="C_ZZ for product state should be 0")

    def test_bell_state_czz_is_nonzero(self):
        # Bell state (|00> + |11>)/sqrt(2):
        # <ZZ> = (1*1 + (-1)*(-1))/2 = 1
        # <Z_0> = (1 + (-1))/2 = 0, <Z_1> = 0
        # C_ZZ = 1 - 0*0 = 1
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)

        czz = connected_correlation(psi, site_a=0, site_b=1, n_qubits=2, pauli='Z')
        np.testing.assert_allclose(czz, 1.0, atol=1e-14,
                                   err_msg="C_ZZ for Bell state should be 1.0")

    def test_antibell_state_czz(self):
        # |01> + |10> / sqrt(2): <ZZ> = -1, <Z> = 0 each => C_ZZ = -1
        psi = np.zeros(4, dtype=np.complex128)
        psi[1] = 1.0 / np.sqrt(2)  # |01>
        psi[2] = 1.0 / np.sqrt(2)  # |10>

        czz = connected_correlation(psi, site_a=0, site_b=1, n_qubits=2, pauli='Z')
        np.testing.assert_allclose(czz, -1.0, atol=1e-14)


# ─────────────────────────────────────────────────────────────
# 8. Hamiltonian hermiticity
# ─────────────────────────────────────────────────────────────

class TestHamiltonianHermiticity:
    """Verify H = H-dagger for random Hamiltonians."""

    def test_random_all_to_all_is_hermitian_n6(self):
        H, _ = random_all_to_all(n_qubits=6, seed=42)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-13,
                                   err_msg="random_all_to_all(N=6) is not Hermitian")

    def test_uniform_coupling_is_hermitian(self):
        H = heisenberg_all_to_all(n_qubits=4, couplings=None)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-13,
                                   err_msg="heisenberg_all_to_all(N=4) is not Hermitian")

    def test_random_couplings_hermitian_n8(self):
        H, _ = random_all_to_all(n_qubits=8, seed=77)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-13,
                                   err_msg="random_all_to_all(N=8) is not Hermitian")


# ─────────────────────────────────────────────────────────────
# 9. Sparse vs dense agreement
# ─────────────────────────────────────────────────────────────

class TestSparseVsDense:
    """Verify sparse and dense Hamiltonians produce identical ground states."""

    def test_ground_state_energy_agrees_n8(self):
        seed = 123
        H_dense, couplings_dense = random_all_to_all(n_qubits=8, seed=seed)
        H_sparse, couplings_sparse = random_all_to_all_sparse(n_qubits=8, seed=seed)

        # Couplings should be identical (same seed, same RNG sequence)
        np.testing.assert_allclose(couplings_dense, couplings_sparse, atol=1e-15,
                                   err_msg="Couplings differ between dense and sparse")

        e_dense, psi_dense = ground_state(H_dense)
        e_sparse, psi_sparse = ground_state_sparse(H_sparse)

        np.testing.assert_allclose(e_dense, e_sparse, atol=1e-10,
                                   err_msg="Ground state energies differ between dense and sparse")

    def test_matrix_elements_agree_n6(self):
        seed = 456
        H_dense, _ = random_all_to_all(n_qubits=6, seed=seed)
        H_sparse, _ = random_all_to_all_sparse(n_qubits=6, seed=seed)

        # Convert sparse to dense for element-wise comparison
        H_sparse_dense = H_sparse.toarray()
        np.testing.assert_allclose(H_dense.real, H_sparse_dense, atol=1e-13,
                                   err_msg="Matrix elements differ between dense and sparse (N=6)")
