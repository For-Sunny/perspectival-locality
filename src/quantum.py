"""
Core quantum mechanics for PLC simulation.

Partial trace, density matrices, entanglement measures.
All exact (no approximations). GPU-accelerated where possible.

Built by Opus Warrior, March 5 2026.
Optimized for N>=12 using bit-manipulation Hamiltonian construction.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from itertools import combinations
from functools import lru_cache
from typing import Optional
import warnings
import time

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────
# Pauli matrices and Hamiltonian construction
# ─────────────────────────────────────────────────────────────

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
IDENTITY_2 = np.eye(2, dtype=np.complex128)


def _kron_chain(ops: list[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def single_site_op(op: np.ndarray, site: int, n_qubits: int) -> np.ndarray:
    """Embed a single-site operator into the full Hilbert space."""
    ops = [IDENTITY_2] * n_qubits
    ops[site] = op
    return _kron_chain(ops)


def two_site_op(op_a: np.ndarray, op_b: np.ndarray,
                site_a: int, site_b: int, n_qubits: int) -> np.ndarray:
    """Embed a two-site operator into the full Hilbert space."""
    ops = [IDENTITY_2] * n_qubits
    ops[site_a] = op_a
    ops[site_b] = op_b
    return _kron_chain(ops)


def heisenberg_all_to_all(n_qubits: int, couplings: Optional[np.ndarray] = None) -> np.ndarray:
    """
    All-to-all Heisenberg Hamiltonian via direct bit-manipulation.
    Orders of magnitude faster than kronecker product construction for N >= 10.

    H = sum_{i<j} J_ij (X_i X_j + Y_i Y_j + Z_i Z_j)

    For the Heisenberg coupling XX + YY + ZZ between qubits i,j:
    - ZZ: diagonal, element (a,a) gets sign_i(a) * sign_j(a)
    - XX + YY = 2*(|01><10| + |10><01|) on qubits (i,j)
      i.e. flip both qubits i and j, with coefficient +2J if bits differ, 0 if same
    Actually: XX + YY on qubits i,j:
      If bit_i(a) != bit_j(a): H[a, a ^ bit_i ^ bit_j] += 2*J
      If bit_i(a) == bit_j(a): no off-diagonal contribution
    And ZZ on qubits i,j:
      H[a, a] += J * sign_i(a) * sign_j(a)
      where sign_k(a) = +1 if bit_k(a)=0, -1 if bit_k(a)=1
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)

    t0 = time.time()

    # Precompute bit masks for each qubit
    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]

    # All basis state indices
    indices = np.arange(dim, dtype=np.int64)

    # Precompute signs for Z operator: +1 if bit=0, -1 if bit=1
    # Fully vectorized: signs[q] is a length-dim array
    signs = np.zeros((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        mask = bit_masks[q]
        signs[q] = 1.0 - 2.0 * ((indices & mask) != 0).astype(np.float64)

    # Diagonal index for ZZ accumulation
    diag_idx = np.arange(dim)

    pair_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J = 1.0 if couplings is None else couplings[pair_idx]
            mask_i = bit_masks[i]
            mask_j = bit_masks[j]
            flip_mask = mask_i | mask_j

            # ZZ contribution (diagonal) - fully vectorized
            H[diag_idx, diag_idx] += J * signs[i] * signs[j]

            # XX + YY contribution (off-diagonal) - fully vectorized
            # Find all basis states where bits i and j differ
            bit_i_vals = (indices & mask_i) != 0
            bit_j_vals = (indices & mask_j) != 0
            differ = bit_i_vals != bit_j_vals  # boolean mask

            a_vals = indices[differ]
            b_vals = a_vals ^ flip_mask  # flip both qubits
            H[a_vals, b_vals] += 2.0 * J

            pair_idx += 1

    elapsed = time.time() - t0
    if elapsed > 1.0:
        print(f"    Hamiltonian construction: {elapsed:.1f}s ({n_qubits} qubits, {dim}x{dim})")

    return H


def heisenberg_all_to_all_gpu(n_qubits: int, couplings: Optional[np.ndarray] = None) -> np.ndarray:
    """
    GPU-accelerated Hamiltonian construction using PyTorch vectorization.
    Dramatically faster for N >= 12.
    """
    if not HAS_TORCH:
        return heisenberg_all_to_all(n_qubits, couplings)

    dim = 2 ** n_qubits
    t0 = time.time()

    # Work in float64 on GPU
    H = torch.zeros((dim, dim), dtype=torch.complex128, device='cuda')
    indices = torch.arange(dim, device='cuda')

    # Precompute bit masks
    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]

    # Precompute Z signs on GPU
    signs = torch.zeros((n_qubits, dim), dtype=torch.float64, device='cuda')
    for q in range(n_qubits):
        mask = bit_masks[q]
        # bit_q = (indices & mask) >> shift
        bit_q = ((indices & mask) != 0).to(torch.float64)
        signs[q] = 1.0 - 2.0 * bit_q  # +1 if bit=0, -1 if bit=1

    pair_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J = 1.0 if couplings is None else float(couplings[pair_idx])
            mask_i = bit_masks[i]
            mask_j = bit_masks[j]
            flip_mask = mask_i | mask_j

            # ZZ (diagonal) - fully vectorized
            H[indices, indices] += J * (signs[i] * signs[j]).to(torch.complex128)

            # XX + YY: flip both qubits when bits i,j differ
            bit_i = (indices & mask_i) != 0
            bit_j = (indices & mask_j) != 0
            differ_mask = bit_i != bit_j  # boolean mask where bits differ
            a_differ = indices[differ_mask]
            b_differ = a_differ ^ flip_mask
            H[a_differ, b_differ] += 2.0 * J

            pair_idx += 1

    elapsed = time.time() - t0
    print(f"    GPU Hamiltonian construction: {elapsed:.1f}s ({n_qubits} qubits, {dim}x{dim})")

    return H.cpu().numpy()


def xxz_all_to_all(n_qubits: int, delta: float = 0.5, seed: Optional[int] = None) -> tuple:
    """
    All-to-all XXZ Hamiltonian with anisotropy parameter delta.

    H = sum_{i<j} J_ij (X_iX_j + Y_iY_j + delta * Z_iZ_j)

    Breaks SU(2) down to U(1) when delta != 1.
    At delta=1, recovers isotropic Heisenberg.
    At delta=0.5, Z-Z coupling is half strength of XX/YY.

    Returns (H, couplings).
    """
    rng = np.random.default_rng(seed)
    n_pairs = n_qubits * (n_qubits - 1) // 2
    couplings = rng.standard_normal(n_pairs)

    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)

    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]
    indices = np.arange(dim, dtype=np.int64)

    # Precompute Z signs
    signs = np.zeros((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        mask = bit_masks[q]
        signs[q] = 1.0 - 2.0 * ((indices & mask) != 0).astype(np.float64)

    diag_idx = np.arange(dim)
    pair_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J = couplings[pair_idx]
            mask_i = bit_masks[i]
            mask_j = bit_masks[j]
            flip_mask = mask_i | mask_j

            # ZZ contribution scaled by delta
            H[diag_idx, diag_idx] += (J * delta) * signs[i] * signs[j]

            # XX + YY contribution (unchanged)
            bit_i_vals = (indices & mask_i) != 0
            bit_j_vals = (indices & mask_j) != 0
            differ = bit_i_vals != bit_j_vals
            a_vals = indices[differ]
            b_vals = a_vals ^ flip_mask
            H[a_vals, b_vals] += 2.0 * J

            pair_idx += 1

    return H, couplings


def random_pauli_all_to_all(n_qubits: int, seed: Optional[int] = None) -> tuple:
    """
    All-to-all Hamiltonian with INDEPENDENT random couplings per Pauli channel.

    H = sum_{i<j} (Jx_ij X_iX_j + Jy_ij Y_iY_j + Jz_ij Z_iZ_j)

    Jx, Jy, Jz drawn independently from N(0,1).
    NO continuous symmetry at all (neither SU(2) nor U(1)).

    Returns (H, couplings_dict) where couplings_dict has keys 'Jx', 'Jy', 'Jz'.
    """
    rng = np.random.default_rng(seed)
    n_pairs = n_qubits * (n_qubits - 1) // 2
    Jx = rng.standard_normal(n_pairs)
    Jy = rng.standard_normal(n_pairs)
    Jz = rng.standard_normal(n_pairs)

    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)

    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]
    indices = np.arange(dim, dtype=np.int64)

    # Precompute Z signs
    signs = np.zeros((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        mask = bit_masks[q]
        signs[q] = 1.0 - 2.0 * ((indices & mask) != 0).astype(np.float64)

    diag_idx = np.arange(dim)
    pair_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            jx = Jx[pair_idx]
            jy = Jy[pair_idx]
            jz = Jz[pair_idx]
            mask_i = bit_masks[i]
            mask_j = bit_masks[j]
            flip_mask = mask_i | mask_j

            # ZZ contribution (diagonal)
            H[diag_idx, diag_idx] += jz * signs[i] * signs[j]

            # XX and YY contributions (off-diagonal)
            # For qubits i,j with bits differing:
            #   XX flips both: coefficient +1 per pair where bits differ
            #   YY flips both: coefficient +1 per pair where bits differ
            #     (XX: |01><10| + |10><01|, YY: -(-1)|01><10| + ... = same real part)
            # Actually need to be careful:
            #   X_i X_j on state |a>: flips bits i,j. coefficient = 1.
            #   Y_i Y_j on state |a>: flips bits i,j. coefficient = (-i)^2 * (-1)^(b_i XOR b_j)
            #     where b_i, b_j are the bit values.
            #   If b_i != b_j: Y_i Y_j gives -1 * flip  (since i*(-i) or (-i)*i = +1... let me be precise)
            #
            # Y = [[0,-i],[i,0]]. Y|0> = i|1>, Y|1> = -i|0>.
            # Y_i Y_j |b_i b_j>:
            #   |00> -> (i|1>)(i|1>) = -|11>  ... wait, i*i = -1
            #   |01> -> (i|1>)(-i|0>) = +|10>
            #   |10> -> (-i|0>)(i|1>) = +|01>
            #   |11> -> (-i|0>)(-i|0>) = -|00>
            # So YY: same-bits -> -1 * flip, diff-bits -> +1 * flip
            #
            # X_i X_j |b_i b_j>: always +1 * flip (XX is real, all +1)
            #
            # Combined: Jx*XX + Jy*YY on diff-bits: (Jx + Jy) * flip
            #           Jx*XX + Jy*YY on same-bits: (Jx - Jy) * flip

            bit_i_vals = (indices & mask_i) != 0
            bit_j_vals = (indices & mask_j) != 0
            differ = bit_i_vals != bit_j_vals
            same = ~differ

            # Bits differ: XX gives +1, YY gives +1 -> coefficient = Jx + Jy
            a_differ = indices[differ]
            b_differ = a_differ ^ flip_mask
            H[a_differ, b_differ] += (jx + jy)

            # Bits same: XX gives +1, YY gives -1 -> coefficient = Jx - Jy
            a_same = indices[same]
            b_same = a_same ^ flip_mask
            H[a_same, b_same] += (jx - jy)

            pair_idx += 1

    return H, {'Jx': Jx, 'Jy': Jy, 'Jz': Jz}


def nearest_neighbor_chain(n_qubits: int, couplings: Optional[np.ndarray] = None,
                           periodic: bool = False) -> np.ndarray:
    """
    1D nearest-neighbor Heisenberg chain.

    H = sum_i J_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

    Has built-in spatial structure: locality is geometric, not emergent.
    This is the POSITIVE CONTROL for PLC experiments.

    Args:
        n_qubits: number of sites
        couplings: array of length n_bonds. If None, random Gaussian J_i.
        periodic: if True, adds bond between site N-1 and site 0.
    """
    n_bonds = n_qubits if periodic else n_qubits - 1
    dim = 2 ** n_qubits

    if couplings is None:
        couplings = np.random.standard_normal(n_bonds)

    H = np.zeros((dim, dim), dtype=np.complex128)

    for bond in range(n_bonds):
        i = bond
        j = (bond + 1) % n_qubits
        J = couplings[bond]
        H += J * two_site_op(SIGMA_X, SIGMA_X, i, j, n_qubits)
        H += J * two_site_op(SIGMA_Y, SIGMA_Y, i, j, n_qubits)
        H += J * two_site_op(SIGMA_Z, SIGMA_Z, i, j, n_qubits)

    return H, couplings


def planted_partition_hamiltonian(n_qubits: int, seed: Optional[int] = None) -> tuple:
    """
    All-to-all Heisenberg with a PLANTED PARTITION.

    Split N qubits into two groups of N/2.
    - Within-group couplings: J ~ N(1, 0.1)  (strong, positive mean)
    - Between-group couplings: J ~ N(0, 0.1)  (weak, zero mean)

    Has hidden cluster structure but no spatial structure.
    CONTROL C for PLC: partial observers should discover the partition.
    """
    rng = np.random.default_rng(seed)
    half = n_qubits // 2
    group_A = list(range(half))
    group_B = list(range(half, n_qubits))

    n_pairs = n_qubits * (n_qubits - 1) // 2
    couplings = np.zeros(n_pairs)

    pair_idx = 0
    pair_labels = []  # 'AA', 'BB', or 'AB'
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            in_A = (i in group_A) and (j in group_A)
            in_B = (i in group_B) and (j in group_B)
            if in_A or in_B:
                # Within-group: strong coupling
                couplings[pair_idx] = rng.normal(1.0, 0.1)
                pair_labels.append('within')
            else:
                # Between-group: weak coupling
                couplings[pair_idx] = rng.normal(0.0, 0.1)
                pair_labels.append('between')
            pair_idx += 1

    H = heisenberg_all_to_all(n_qubits, couplings)
    return H, couplings, group_A, group_B, pair_labels


def random_all_to_all(n_qubits: int, seed: Optional[int] = None, use_gpu_build: bool = False) -> np.ndarray:
    """
    Random all-to-all Heisenberg with Gaussian couplings.
    NO spatial structure. Every qubit coupled to every other randomly.
    """
    rng = np.random.default_rng(seed)
    n_pairs = n_qubits * (n_qubits - 1) // 2
    couplings = rng.standard_normal(n_pairs)
    build_fn = heisenberg_all_to_all_gpu if (use_gpu_build and HAS_TORCH) else heisenberg_all_to_all
    return build_fn(n_qubits, couplings), couplings


# ─────────────────────────────────────────────────────────────
# Sparse Hamiltonian construction and eigensolver (N >= 14)
# ─────────────────────────────────────────────────────────────

def heisenberg_all_to_all_sparse(n_qubits: int, couplings: Optional[np.ndarray] = None) -> sp.csr_matrix:
    """
    Build all-to-all Heisenberg Hamiltonian as sparse CSR matrix.
    Same physics as heisenberg_all_to_all but O(N^2 * 2^N) nonzeros
    instead of O(2^(2N)) dense storage.

    For N=14: ~1.8M nonzeros vs 2.1B dense entries (1000x savings).
    For N=16: ~8.4M nonzeros vs 34B dense entries (4000x savings).
    """
    dim = 2 ** n_qubits
    t0 = time.time()

    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]
    indices = np.arange(dim, dtype=np.int64)

    # Precompute Z signs: +1 if bit=0, -1 if bit=1
    signs = np.zeros((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        mask = bit_masks[q]
        signs[q] = 1.0 - 2.0 * ((indices & mask) != 0).astype(np.float64)

    # Accumulate diagonal (ZZ) contributions
    diag = np.zeros(dim, dtype=np.float64)

    # Collect off-diagonal COO entries
    rows_list = []
    cols_list = []
    vals_list = []

    pair_idx = 0
    n_pairs = n_qubits * (n_qubits - 1) // 2
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J = 1.0 if couplings is None else float(couplings[pair_idx])
            mask_i = bit_masks[i]
            mask_j = bit_masks[j]
            flip_mask = mask_i | mask_j

            # ZZ diagonal
            diag += J * signs[i] * signs[j]

            # XX + YY off-diagonal: flip both qubits where bits differ
            bit_i_vals = (indices & mask_i) != 0
            bit_j_vals = (indices & mask_j) != 0
            differ = bit_i_vals != bit_j_vals

            a_vals = indices[differ]
            b_vals = a_vals ^ flip_mask

            rows_list.append(a_vals)
            cols_list.append(b_vals)
            vals_list.append(np.full(len(a_vals), 2.0 * J))

            pair_idx += 1

    # Add diagonal entries
    rows_list.append(np.arange(dim, dtype=np.int64))
    cols_list.append(np.arange(dim, dtype=np.int64))
    vals_list.append(diag)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    H_sparse = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))

    elapsed = time.time() - t0
    nnz = H_sparse.nnz
    print(f"    Sparse Hamiltonian: {elapsed:.1f}s, {n_qubits} qubits, "
          f"{dim}x{dim}, {nnz:,} nonzeros ({nnz/(dim*dim)*100:.4f}% fill)")

    return H_sparse


def random_all_to_all_sparse(n_qubits: int, seed: Optional[int] = None) -> tuple:
    """
    Random all-to-all Heisenberg as sparse matrix.
    Returns (H_sparse, couplings).
    """
    rng = np.random.default_rng(seed)
    n_pairs = n_qubits * (n_qubits - 1) // 2
    couplings = rng.standard_normal(n_pairs)
    return heisenberg_all_to_all_sparse(n_qubits, couplings), couplings


def ground_state_sparse(H_sparse: sp.csr_matrix, sigma: float = None) -> tuple[float, np.ndarray]:
    """
    Find ground state of sparse Hamiltonian via Lanczos (ARPACK).
    Returns (energy, state_vector).

    Uses shift-invert mode with sigma for better convergence on
    interior eigenvalues, but for the ground state (smallest algebraic),
    which='SA' with no shift is typically fine.
    """
    dim = H_sparse.shape[0]
    t0 = time.time()

    # For real-valued Hamiltonians (Heisenberg), eigsh works directly
    # k=1: just the ground state. which='SA' = smallest algebraic
    # maxiter scaled for large systems; ncv (Lanczos vectors) helps convergence
    ncv = min(40, dim - 1)
    maxiter = max(1000, dim // 10)

    if sigma is not None:
        eigenvalues, eigenvectors = spla.eigsh(
            H_sparse, k=1, sigma=sigma, which='LM',
            ncv=ncv, maxiter=maxiter
        )
    else:
        eigenvalues, eigenvectors = spla.eigsh(
            H_sparse, k=1, which='SA',
            ncv=ncv, maxiter=maxiter
        )

    elapsed = time.time() - t0
    print(f"    Sparse eigensolver: {elapsed:.1f}s, dim={dim}, E0={eigenvalues[0]:.6f}")

    return float(eigenvalues[0]), eigenvectors[:, 0]


def ground_state(H: np.ndarray) -> tuple[float, np.ndarray]:
    """Find ground state via exact diagonalization. Returns (energy, state_vector)."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = np.argmin(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def ground_state_gpu(H: np.ndarray) -> tuple[float, np.ndarray]:
    """GPU-accelerated exact diagonalization."""
    if not HAS_TORCH:
        return ground_state(H)

    H_t = torch.from_numpy(H).to('cuda')
    eigenvalues, eigenvectors = torch.linalg.eigh(H_t)
    idx = torch.argmin(eigenvalues)

    e0 = eigenvalues[idx].cpu().item()
    psi = eigenvectors[:, idx].cpu().numpy()

    del H_t, eigenvalues, eigenvectors
    torch.cuda.empty_cache()

    return e0, psi


# ─────────────────────────────────────────────────────────────
# Partial trace and reduced density matrices
# ─────────────────────────────────────────────────────────────

def partial_trace(psi: np.ndarray, keep: list[int], n_qubits: int) -> np.ndarray:
    """
    Compute reduced density matrix by tracing out all qubits NOT in 'keep'.

    psi: state vector of dimension 2^n_qubits
    keep: list of qubit indices to keep (0-indexed)
    n_qubits: total number of qubits

    Returns: reduced density matrix of dimension 2^len(keep) x 2^len(keep)
    """
    # Reshape state vector into tensor with one index per qubit
    psi_tensor = psi.reshape([2] * n_qubits)

    # Compute full density matrix as outer product in tensor form
    rho_tensor = np.einsum(
        psi_tensor, range(n_qubits),
        np.conj(psi_tensor), range(n_qubits, 2 * n_qubits)
    )
    # rho_tensor has indices [0, 1, ..., n-1, n, n+1, ..., 2n-1]
    # where first n are ket indices, last n are bra indices

    # Trace out qubits not in keep
    trace_out = sorted(set(range(n_qubits)) - set(keep))

    # Contract ket and bra indices for traced-out qubits
    # We need to trace over pairs (i, i+n_qubits) for i in trace_out
    # Do this iteratively from highest index to preserve ordering
    result = rho_tensor
    offset = 0
    for q in sorted(trace_out, reverse=True):
        ket_idx = q - offset
        bra_idx = ket_idx + (n_qubits - offset)  # NOT RIGHT
        # Actually, let's use a simpler approach

    # Simpler: use numpy trace over axes
    # Reshape into density matrix, then trace

    # Better approach: direct computation
    keep_sorted = sorted(keep)
    n_keep = len(keep_sorted)
    n_trace = n_qubits - n_keep
    dim_keep = 2 ** n_keep
    dim_trace = 2 ** n_trace

    # Reorder qubit indices: keep qubits first, then traced qubits
    trace_out_sorted = sorted(set(range(n_qubits)) - set(keep_sorted))
    perm = keep_sorted + trace_out_sorted

    psi_tensor = psi.reshape([2] * n_qubits)
    psi_reordered = np.transpose(psi_tensor, perm)

    # Reshape: (dim_keep, dim_trace)
    psi_matrix = psi_reordered.reshape(dim_keep, dim_trace)

    # Reduced density matrix = psi_matrix @ psi_matrix^dagger
    rho_reduced = psi_matrix @ psi_matrix.conj().T

    return rho_reduced


def density_matrix(psi: np.ndarray) -> np.ndarray:
    """Full density matrix from pure state."""
    return np.outer(psi, np.conj(psi))


# ─────────────────────────────────────────────────────────────
# Entanglement and correlation measures
# ─────────────────────────────────────────────────────────────

def von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy S(rho) = -Tr(rho log rho)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out zero/negative eigenvalues (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-14]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def mutual_information(psi: np.ndarray, sites_a: list[int], sites_b: list[int],
                       n_qubits: int) -> float:
    """
    Quantum mutual information I(A:B) = S(A) + S(B) - S(AB).

    Measures total (quantum + classical) correlations between subsystems A and B.
    """
    rho_a = partial_trace(psi, sites_a, n_qubits)
    rho_b = partial_trace(psi, sites_b, n_qubits)
    rho_ab = partial_trace(psi, sites_a + sites_b, n_qubits)

    S_a = von_neumann_entropy(rho_a)
    S_b = von_neumann_entropy(rho_b)
    S_ab = von_neumann_entropy(rho_ab)

    return S_a + S_b - S_ab


def mutual_information_matrix(psi: np.ndarray, n_qubits: int,
                              sites: Optional[list[int]] = None) -> np.ndarray:
    """
    Compute pairwise mutual information matrix for all pairs in 'sites'.

    If sites is None, uses all qubits.
    Returns symmetric matrix where M[i,j] = I(site_i : site_j).
    """
    if sites is None:
        sites = list(range(n_qubits))

    n = len(sites)
    MI = np.zeros((n, n))

    # Precompute single-site entropies
    S_single = {}
    for i, s in enumerate(sites):
        rho = partial_trace(psi, [s], n_qubits)
        S_single[s] = von_neumann_entropy(rho)

    for i in range(n):
        for j in range(i + 1, n):
            rho_pair = partial_trace(psi, [sites[i], sites[j]], n_qubits)
            S_pair = von_neumann_entropy(rho_pair)
            mi = S_single[sites[i]] + S_single[sites[j]] - S_pair
            MI[i, j] = mi
            MI[j, i] = mi

    return MI


def connected_correlation(psi: np.ndarray, site_a: int, site_b: int,
                          n_qubits: int, pauli: str = 'Z') -> float:
    """
    Connected correlation function <O_a O_b> - <O_a><O_b>.

    Uses the specified Pauli operator (X, Y, or Z).
    """
    op = {'X': SIGMA_X, 'Y': SIGMA_Y, 'Z': SIGMA_Z}[pauli.upper()]

    # <O_a O_b>
    O_ab = two_site_op(op, op, site_a, site_b, n_qubits)
    expect_ab = np.real(np.conj(psi) @ O_ab @ psi)

    # <O_a>
    O_a = single_site_op(op, site_a, n_qubits)
    expect_a = np.real(np.conj(psi) @ O_a @ psi)

    # <O_b>
    O_b = single_site_op(op, site_b, n_qubits)
    expect_b = np.real(np.conj(psi) @ O_b @ psi)

    return expect_ab - expect_a * expect_b


def correlation_matrix(psi: np.ndarray, n_qubits: int,
                       sites: Optional[list[int]] = None,
                       pauli: str = 'Z') -> np.ndarray:
    """
    Connected correlation matrix C[i,j] = <Z_i Z_j> - <Z_i><Z_j>.
    """
    if sites is None:
        sites = list(range(n_qubits))

    n = len(sites)
    C = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            c = connected_correlation(psi, sites[i], sites[j], n_qubits, pauli)
            C[i, j] = c
            C[j, i] = c

    return C
