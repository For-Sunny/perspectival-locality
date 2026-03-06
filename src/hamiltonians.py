"""
XXZ Hamiltonian variants and entanglement control for PLC Paper 2.

Extends quantum.py with:
- XXZ Hamiltonian builder (multiple coupling topologies)
- Entanglement entropy calculators
- Local Hamiltonian builder (chain, ladder, square)
- Delta sweep helper for anisotropy scans

All matrices built sparse (scipy.sparse CSR). Dense fallback via .toarray().
Bit-manipulation construction matches quantum.py conventions.

Built by Opus Warrior, March 5 2026. Paper 2 module.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional
import time

from src.quantum import (
    partial_trace, von_neumann_entropy, ground_state, ground_state_sparse,
)


# ─────────────────────────────────────────────────────────────
# Core sparse builder: XXZ with arbitrary adjacency
# ─────────────────────────────────────────────────────────────

def _build_xxz_from_edges(n_qubits: int, edges: list[tuple[int, int]],
                           couplings: np.ndarray, delta: float) -> sp.csr_matrix:
    """
    Build XXZ Hamiltonian as sparse CSR from an explicit edge list.

    H = sum_{(i,j)} J_{ij} [ X_i X_j + Y_i Y_j + delta * Z_i Z_j ]

    Internal workhorse. All public builders call this.
    """
    dim = 2 ** n_qubits
    bit_masks = [1 << (n_qubits - 1 - q) for q in range(n_qubits)]
    indices = np.arange(dim, dtype=np.int64)

    # Precompute Z signs: +1 if bit=0, -1 if bit=1
    signs = np.zeros((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        mask = bit_masks[q]
        signs[q] = 1.0 - 2.0 * ((indices & mask) != 0).astype(np.float64)

    # Accumulate diagonal (ZZ) and collect off-diagonal COO entries
    diag = np.zeros(dim, dtype=np.float64)
    rows_list = []
    cols_list = []
    vals_list = []

    for edge_idx, (i, j) in enumerate(edges):
        J = float(couplings[edge_idx])
        mask_i = bit_masks[i]
        mask_j = bit_masks[j]
        flip_mask = mask_i | mask_j

        # ZZ diagonal scaled by delta
        diag += (J * delta) * signs[i] * signs[j]

        # XX + YY off-diagonal: flip both qubits where bits differ
        bit_i_vals = (indices & mask_i) != 0
        bit_j_vals = (indices & mask_j) != 0
        differ = bit_i_vals != bit_j_vals

        a_vals = indices[differ]
        b_vals = a_vals ^ flip_mask

        rows_list.append(a_vals)
        cols_list.append(b_vals)
        vals_list.append(np.full(len(a_vals), 2.0 * J))

    # Add diagonal
    rows_list.append(np.arange(dim, dtype=np.int64))
    cols_list.append(np.arange(dim, dtype=np.int64))
    vals_list.append(diag)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


# ─────────────────────────────────────────────────────────────
# Edge list generators for different topologies
# ─────────────────────────────────────────────────────────────

def _all_to_all_edges(n_qubits: int) -> list[tuple[int, int]]:
    """All pairs (i, j) with i < j."""
    return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]


def _nearest_neighbor_edges(n_qubits: int, periodic: bool = False) -> list[tuple[int, int]]:
    """1D chain: (0,1), (1,2), ..., optionally (N-1, 0)."""
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    if periodic:
        edges.append((n_qubits - 1, 0))
    return edges


def _ladder_edges(n_qubits: int) -> list[tuple[int, int]]:
    """
    2-leg ladder: sites 0..N/2-1 on top rung, N/2..N-1 on bottom.
    Rungs: (i, i+N/2). Legs: nearest-neighbor along each leg.
    Requires even N.
    """
    assert n_qubits % 2 == 0, f"Ladder requires even N, got {n_qubits}"
    half = n_qubits // 2
    edges = []
    # Top leg
    for i in range(half - 1):
        edges.append((i, i + 1))
    # Bottom leg
    for i in range(half, n_qubits - 1):
        edges.append((i, i + 1))
    # Rungs
    for i in range(half):
        edges.append((i, i + half))
    return edges


def _square_edges(n_qubits: int) -> list[tuple[int, int]]:
    """
    2D square lattice. Finds the most square arrangement of N sites.
    Sites indexed row-major. Nearest-neighbor on the grid.
    """
    # Find best rectangular layout
    nrow = int(np.sqrt(n_qubits))
    while n_qubits % nrow != 0 and nrow > 1:
        nrow -= 1
    ncol = n_qubits // nrow

    edges = []
    for r in range(nrow):
        for c in range(ncol):
            idx = r * ncol + c
            # Right neighbor
            if c + 1 < ncol:
                edges.append((idx, idx + 1))
            # Down neighbor
            if r + 1 < nrow:
                edges.append((idx, idx + ncol))
    return edges


def _random_graph_edges(n_qubits: int, p: float = 0.5,
                         rng: np.random.Generator = None) -> list[tuple[int, int]]:
    """Erdos-Renyi random graph: each edge present with probability p."""
    if rng is None:
        rng = np.random.default_rng()
    edges = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if rng.random() < p:
                edges.append((i, j))
    # Guarantee connectivity: if isolated node, add one edge
    connected = set()
    for i, j in edges:
        connected.add(i)
        connected.add(j)
    for q in range(n_qubits):
        if q not in connected:
            partner = (q + 1) % n_qubits
            edges.append((min(q, partner), max(q, partner)))
    return edges


# ─────────────────────────────────────────────────────────────
# Public API: XXZ Hamiltonian builder
# ─────────────────────────────────────────────────────────────

def build_xxz_hamiltonian(n_qubits: int, delta: float = 1.0,
                           coupling: str = 'all_to_all',
                           J: float = 1.0,
                           random_couplings: bool = True,
                           seed: Optional[int] = None,
                           periodic: bool = False,
                           edge_prob: float = 0.5) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    XXZ Hamiltonian with configurable topology and anisotropy.

    H = sum_{(i,j) in graph} J_ij [ X_i X_j + Y_i Y_j + delta * Z_i Z_j ]

    Args:
        n_qubits: number of spin-1/2 sites
        delta: anisotropy parameter
            delta >> 1: Ising limit (low entanglement, near-product ground state)
            delta = 1:  isotropic Heisenberg (Paper 1 case)
            delta -> 0: XX model (intermediate entanglement)
        coupling: topology — 'all_to_all', 'nearest_neighbor', 'random_graph'
        J: overall coupling scale (multiplied into random or uniform couplings)
        random_couplings: if True, J_ij ~ J * Normal(0,1). If False, J_ij = J (uniform).
        seed: RNG seed for reproducibility
        periodic: for nearest_neighbor, whether to close the chain into a ring
        edge_prob: for random_graph, Erdos-Renyi edge probability

    Returns:
        (H_sparse, couplings_array)
    """
    rng = np.random.default_rng(seed)

    if coupling == 'all_to_all':
        edges = _all_to_all_edges(n_qubits)
    elif coupling == 'nearest_neighbor':
        edges = _nearest_neighbor_edges(n_qubits, periodic=periodic)
    elif coupling == 'random_graph':
        edges = _random_graph_edges(n_qubits, p=edge_prob, rng=rng)
    else:
        raise ValueError(f"Unknown coupling topology: {coupling}")

    n_edges = len(edges)
    if random_couplings:
        couplings = J * rng.standard_normal(n_edges)
    else:
        couplings = np.full(n_edges, J)

    H = _build_xxz_from_edges(n_qubits, edges, couplings, delta)
    return H, couplings


# ─────────────────────────────────────────────────────────────
# Public API: Local Hamiltonian builder
# ─────────────────────────────────────────────────────────────

def build_local_hamiltonian(n_qubits: int, geometry: str = 'chain',
                             delta: float = 1.0,
                             J: float = 1.0,
                             random_couplings: bool = True,
                             seed: Optional[int] = None,
                             periodic: bool = False) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Local (spatially structured) XXZ Hamiltonian for comparison experiments.

    Geometry controls the graph structure:
        'chain':  1D nearest-neighbor (with optional periodic boundary)
        'ladder': 2-leg ladder (requires even N)
        'square': 2D square lattice (N should factor into a reasonable grid)

    Local Hamiltonians give the INTERESTING case for Paper 2:
    negative curvature (spatial structure) vs all-to-all (null case).

    Returns:
        (H_sparse, couplings_array)
    """
    rng = np.random.default_rng(seed)

    if geometry == 'chain':
        edges = _nearest_neighbor_edges(n_qubits, periodic=periodic)
    elif geometry == 'ladder':
        edges = _ladder_edges(n_qubits)
    elif geometry == 'square':
        edges = _square_edges(n_qubits)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    n_edges = len(edges)
    if random_couplings:
        couplings = J * rng.standard_normal(n_edges)
    else:
        couplings = np.full(n_edges, J)

    H = _build_xxz_from_edges(n_qubits, edges, couplings, delta)
    return H, couplings


# ─────────────────────────────────────────────────────────────
# Entanglement entropy calculators
# ─────────────────────────────────────────────────────────────

def entanglement_entropy(state: np.ndarray, subsystem: list[int],
                          n_qubits: int) -> float:
    """
    Von Neumann entropy of the reduced state on 'subsystem'.

    S(rho_A) = -Tr(rho_A log2 rho_A)

    where rho_A = Tr_B(|psi><psi|).

    Args:
        state: pure state vector of dimension 2^n_qubits
        subsystem: list of qubit indices to keep
        n_qubits: total number of qubits

    Returns:
        Entanglement entropy in bits (log base 2)
    """
    rho = partial_trace(state, keep=subsystem, n_qubits=n_qubits)
    return von_neumann_entropy(rho)


def half_system_entropy(state: np.ndarray, n_qubits: int) -> float:
    """
    Entanglement entropy of the first N/2 qubits (half-cut entropy).

    Standard measure for quantifying total entanglement in the ground state.
    For area-law states (gapped local Hamiltonians), S ~ boundary size.
    For volume-law states (thermal/random), S ~ system size.
    """
    half = n_qubits // 2
    subsystem = list(range(half))
    return entanglement_entropy(state, subsystem, n_qubits)


# ─────────────────────────────────────────────────────────────
# Delta sweep helper
# ─────────────────────────────────────────────────────────────

def sweep_delta(n_qubits: int, deltas: list[float],
                coupling: str = 'all_to_all',
                seed: Optional[int] = None,
                use_sparse: bool = True,
                verbose: bool = True) -> list[dict]:
    """
    Compute ground states across a range of delta values.

    Fixes the coupling realization (same J_ij for all delta) and varies
    only the anisotropy. This isolates the effect of entanglement structure.

    Args:
        n_qubits: number of qubits
        deltas: list of anisotropy values to sweep
        coupling: topology passed to build_xxz_hamiltonian
        seed: RNG seed (same couplings for all delta)
        use_sparse: if True, use sparse eigensolver (recommended for N >= 12)
        verbose: print progress

    Returns:
        List of dicts, each containing:
            'delta': anisotropy value
            'hamiltonian': sparse matrix
            'ground_energy': float
            'ground_state': state vector
            'half_entropy': half-system entanglement entropy
    """
    # Generate fixed coupling realization
    rng = np.random.default_rng(seed)

    if coupling == 'all_to_all':
        edges = _all_to_all_edges(n_qubits)
    elif coupling == 'nearest_neighbor':
        edges = _nearest_neighbor_edges(n_qubits)
    elif coupling == 'random_graph':
        edges = _random_graph_edges(n_qubits, rng=rng)
    else:
        raise ValueError(f"Unknown coupling: {coupling}")

    n_edges = len(edges)
    couplings = rng.standard_normal(n_edges)

    results = []
    t0 = time.time()

    for i, delta in enumerate(deltas):
        H = _build_xxz_from_edges(n_qubits, edges, couplings, delta)

        if use_sparse:
            energy, psi = ground_state_sparse(H)
        else:
            H_dense = H.toarray()
            energy, psi = ground_state(H_dense)

        entropy = half_system_entropy(psi, n_qubits)

        results.append({
            'delta': delta,
            'hamiltonian': H,
            'ground_energy': energy,
            'ground_state': psi,
            'half_entropy': entropy,
        })

        if verbose:
            elapsed = time.time() - t0
            print(f"    delta={delta:.3f}: E0={energy:.6f}, S_half={entropy:.4f} "
                  f"[{i+1}/{len(deltas)}, {elapsed:.1f}s]")

    return results
