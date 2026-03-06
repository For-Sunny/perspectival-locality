"""
Discrete Ricci curvature on MI-distance graphs.

Two curvature notions for the emergent metric space defined by mutual information:
1. Ollivier-Ricci curvature (optimal transport, captures metric geometry)
2. Forman-Ricci curvature (combinatorial, captures topology)

Both computed from scratch — no external graph-Ricci packages.
Wasserstein-1 distance solved via scipy.optimize.linprog.

Built by Opus Warrior, March 5 2026.
Paper 2: "Emergent Curvature from Partial Observation."
"""

import numpy as np
import networkx as nx
from scipy.optimize import linprog
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Graph construction from MI matrix
# ─────────────────────────────────────────────────────────────

def mi_to_graph(mi_matrix: np.ndarray, threshold: Optional[float] = None,
                min_distance: Optional[float] = None) -> nx.Graph:
    """
    Build weighted graph from mutual information matrix.

    Edge weight = MI value (higher MI = stronger connection).
    Edge distance = max(1/MI, min_distance) (used for shortest-path computations).

    Args:
        mi_matrix: NxN symmetric MI matrix (diagonal ignored).
        threshold: if float in (0,1), treat as percentile of nonzero MI values
                   and keep only edges above that percentile.
                   If float >= 1, treat as absolute MI threshold.
                   If None, keep all edges with MI > 0.
        min_distance: floor on edge distances. Prevents 1/MI from approaching
                      zero when MI is very large, which causes ORC instability.
                      If None, no floor is applied (backward compatible).
                      If a float > 0, all edge distances are clipped to
                      max(1/MI, min_distance).

    Returns:
        nx.Graph with 'weight' (MI) and 'distance' (1/MI, floored) edge attributes.
    """
    n = mi_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Collect all positive MI values for thresholding
    mi_values = []
    for i in range(n):
        for j in range(i + 1, n):
            if mi_matrix[i, j] > 1e-14:
                mi_values.append(mi_matrix[i, j])

    if not mi_values:
        return G

    # Determine cutoff
    cutoff = 0.0
    if threshold is not None:
        if 0 < threshold < 1:
            cutoff = np.percentile(mi_values, threshold * 100)
        else:
            cutoff = threshold

    # Distance floor
    d_floor = float(min_distance) if min_distance is not None and min_distance > 0 else 0.0

    # Add edges
    for i in range(n):
        for j in range(i + 1, n):
            mi = mi_matrix[i, j]
            if mi > max(cutoff, 1e-14):
                d = 1.0 / float(mi)
                if d_floor > 0:
                    d = max(d, d_floor)
                G.add_edge(i, j, weight=float(mi), distance=d)

    return G


def mi_to_knn_graph(mi_matrix: np.ndarray, k: int = 3) -> nx.Graph:
    """
    Build k-nearest-neighbor graph from MI matrix.

    For each node, keep edges to the k nodes with highest MI.
    Symmetrized: edge exists if EITHER endpoint considers the other a neighbor.

    Args:
        mi_matrix: NxN symmetric MI matrix.
        k: number of nearest neighbors per node.

    Returns:
        nx.Graph with 'weight' and 'distance' edge attributes.
    """
    n = mi_matrix.shape[0]
    k = min(k, n - 1)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        # Get MI values for node i (exclude self)
        mi_row = mi_matrix[i].copy()
        mi_row[i] = -1  # exclude self
        # Find k largest MI neighbors
        neighbors = np.argsort(mi_row)[-k:]

        for j in neighbors:
            mi = mi_matrix[i, j]
            if mi > 1e-14 and not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(mi), distance=1.0 / float(mi))

    return G


# ─────────────────────────────────────────────────────────────
# Wasserstein-1 distance via linear programming
# ─────────────────────────────────────────────────────────────

def _wasserstein_1(mu: np.ndarray, nu: np.ndarray, dist_matrix: np.ndarray) -> float:
    """
    Wasserstein-1 (earth mover's) distance between discrete distributions.

    Solved as a linear program:
        min  sum_{i,j} d(i,j) * T(i,j)
        s.t. sum_j T(i,j) = mu(i)   for all i
             sum_i T(i,j) = nu(j)   for all j
             T(i,j) >= 0

    Args:
        mu: source distribution, length n.
        nu: target distribution, length n.
        dist_matrix: n x n pairwise distance matrix.

    Returns:
        W_1(mu, nu).
    """
    n = len(mu)

    # Flatten the transport plan T into a vector of length n*n
    # T[i,j] -> x[i*n + j]
    c = dist_matrix.flatten().astype(np.float64)

    # Equality constraints: A_eq @ x = b_eq
    # Row marginal: sum_j T(i,j) = mu(i) for each i  (n constraints)
    # Col marginal: sum_i T(i,j) = nu(j) for each j  (n constraints)
    A_eq = np.zeros((2 * n, n * n), dtype=np.float64)
    b_eq = np.zeros(2 * n, dtype=np.float64)

    for i in range(n):
        # Row marginal: sum_j T(i,j) = mu(i)
        A_eq[i, i * n:(i + 1) * n] = 1.0
        b_eq[i] = mu[i]

    for j in range(n):
        # Col marginal: sum_i T(i,j) = nu(j)
        for i in range(n):
            A_eq[n + j, i * n + j] = 1.0
        b_eq[n + j] = nu[j]

    # Bounds: T(i,j) >= 0
    bounds = [(0, None)] * (n * n)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        return float(result.fun)
    else:
        # Fallback: if LP fails, return a large value
        return float('inf')


# ─────────────────────────────────────────────────────────────
# Ollivier-Ricci curvature
# ─────────────────────────────────────────────────────────────

def _lazy_random_walk_measure(G: nx.Graph, node: int, alpha: float,
                               node_list: list[int]) -> np.ndarray:
    """
    Lazy random walk probability measure centered at node.

    m_x(z) = alpha              if z = x
    m_x(z) = (1-alpha) / deg(x) if z in N(x)
    m_x(z) = 0                  otherwise

    Args:
        G: graph.
        node: center node x.
        alpha: laziness parameter (probability of staying).
        node_list: ordered list of all nodes (for indexing).

    Returns:
        probability vector of length len(node_list).
    """
    n = len(node_list)
    node_to_idx = {v: i for i, v in enumerate(node_list)}
    mu = np.zeros(n, dtype=np.float64)

    neighbors = list(G.neighbors(node))
    deg = len(neighbors)

    idx_center = node_to_idx[node]
    mu[idx_center] = alpha

    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbors:
            mu[node_to_idx[nb]] = spread

    return mu


def ollivier_ricci(mi_matrix: np.ndarray, threshold: Optional[float] = None,
                   alpha: float = 0.5, min_distance: Optional[float] = None) -> dict:
    """
    Ollivier-Ricci curvature for the MI-distance graph.

    kappa(x, y) = 1 - W_1(m_x, m_y) / d(x, y)

    where m_x is the lazy random walk measure centered at x,
    W_1 is the Wasserstein-1 distance, and d(x,y) is the graph
    shortest-path distance (using 1/MI as edge lengths).

    Args:
        mi_matrix: NxN symmetric mutual information matrix.
        threshold: passed to mi_to_graph for edge filtering.
        alpha: laziness parameter for random walk measure. Default 0.5.

    Returns:
        dict with:
            'edge_curvatures': dict mapping (i,j) -> kappa(i,j)
            'scalar_curvature': mean of all edge curvatures
            'edges': list of (i, j, kappa) tuples
            'n_edges': number of edges
            'min_distance_used': the distance floor applied
    """
    G = mi_to_graph(mi_matrix, threshold=threshold, min_distance=min_distance)

    if G.number_of_edges() == 0:
        return {
            'edge_curvatures': {},
            'scalar_curvature': 0.0,
            'edges': [],
            'n_edges': 0,
        }

    node_list = sorted(G.nodes())
    n = len(node_list)

    # Compute all-pairs shortest path distances using 1/MI edge weights
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='distance'))

    # Build full distance matrix for Wasserstein computation
    dist_matrix = np.full((n, n), float('inf'), dtype=np.float64)
    node_to_idx = {v: i for i, v in enumerate(node_list)}

    for u in node_list:
        for v, d in shortest_paths[u].items():
            dist_matrix[node_to_idx[u], node_to_idx[v]] = d
    # Self-distance = 0
    np.fill_diagonal(dist_matrix, 0.0)

    # Replace inf with a large finite value for LP stability
    max_finite = np.max(dist_matrix[np.isfinite(dist_matrix)])
    dist_matrix[~np.isfinite(dist_matrix)] = max_finite * 10

    # Compute curvature for each edge
    edge_curvatures = {}
    edges_list = []

    for u, v in G.edges():
        d_uv = dist_matrix[node_to_idx[u], node_to_idx[v]]
        if d_uv < 1e-14:
            continue

        mu_u = _lazy_random_walk_measure(G, u, alpha, node_list)
        mu_v = _lazy_random_walk_measure(G, v, alpha, node_list)

        w1 = _wasserstein_1(mu_u, mu_v, dist_matrix)
        kappa = 1.0 - w1 / d_uv

        edge_key = (min(u, v), max(u, v))
        edge_curvatures[edge_key] = float(kappa)
        edges_list.append((edge_key[0], edge_key[1], float(kappa)))

    # Scalar curvature: mean over all edges
    if edge_curvatures:
        sc = float(np.mean(list(edge_curvatures.values())))
    else:
        sc = 0.0

    return {
        'edge_curvatures': edge_curvatures,
        'scalar_curvature': sc,
        'edges': edges_list,
        'n_edges': len(edge_curvatures),
    }


# ─────────────────────────────────────────────────────────────
# Forman-Ricci curvature
# ─────────────────────────────────────────────────────────────

def forman_ricci(mi_matrix: np.ndarray, threshold: Optional[float] = None) -> dict:
    """
    Forman-Ricci curvature for the MI-distance graph.

    Basic Forman curvature for an edge e = (v1, v2):
        F(e) = 4 - deg(v1) - deg(v2)

    Augmented Forman curvature (counts triangles):
        F_aug(e) = 4 - deg(v1) - deg(v2) + 3 * triangles(e)

    where triangles(e) is the number of triangles containing edge e.

    Args:
        mi_matrix: NxN symmetric mutual information matrix.
        threshold: passed to mi_to_graph for edge filtering.

    Returns:
        dict with:
            'edge_curvatures': dict mapping (i,j) -> F(i,j) (basic)
            'edge_curvatures_aug': dict mapping (i,j) -> F_aug(i,j)
            'scalar_curvature': mean of basic curvatures
            'scalar_curvature_aug': mean of augmented curvatures
            'edges': list of (i, j, F, F_aug) tuples
            'n_edges': number of edges
    """
    G = mi_to_graph(mi_matrix, threshold=threshold)

    if G.number_of_edges() == 0:
        return {
            'edge_curvatures': {},
            'edge_curvatures_aug': {},
            'scalar_curvature': 0.0,
            'scalar_curvature_aug': 0.0,
            'edges': [],
            'n_edges': 0,
        }

    edge_curvatures = {}
    edge_curvatures_aug = {}
    edges_list = []

    for u, v in G.edges():
        deg_u = G.degree(u)
        deg_v = G.degree(v)

        # Basic Forman
        F_basic = 4 - deg_u - deg_v

        # Count triangles containing edge (u, v)
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common = neighbors_u & neighbors_v
        n_triangles = len(common)

        # Augmented Forman
        F_aug = F_basic + 3 * n_triangles

        edge_key = (min(u, v), max(u, v))
        edge_curvatures[edge_key] = float(F_basic)
        edge_curvatures_aug[edge_key] = float(F_aug)
        edges_list.append((edge_key[0], edge_key[1], float(F_basic), float(F_aug)))

    sc = float(np.mean(list(edge_curvatures.values()))) if edge_curvatures else 0.0
    sc_aug = float(np.mean(list(edge_curvatures_aug.values()))) if edge_curvatures_aug else 0.0

    return {
        'edge_curvatures': edge_curvatures,
        'edge_curvatures_aug': edge_curvatures_aug,
        'scalar_curvature': sc,
        'scalar_curvature_aug': sc_aug,
        'edges': edges_list,
        'n_edges': len(edge_curvatures),
    }


# ─────────────────────────────────────────────────────────────
# Curvature statistics
# ─────────────────────────────────────────────────────────────

def curvature_stats(curvatures: dict) -> dict:
    """
    Summary statistics for a set of edge curvatures.

    Args:
        curvatures: dict mapping (i,j) -> curvature value.

    Returns:
        dict with mean, std, min, max, median, and histogram bin data.
    """
    if not curvatures:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'median': 0.0, 'n_edges': 0,
            'frac_positive': 0.0, 'frac_negative': 0.0,
            'hist_counts': [], 'hist_edges': [],
        }

    vals = np.array(list(curvatures.values()))

    # Histogram: 20 bins
    n_bins = min(20, max(3, len(vals) // 2))
    counts, bin_edges = np.histogram(vals, bins=n_bins)

    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'median': float(np.median(vals)),
        'n_edges': len(vals),
        'frac_positive': float(np.mean(vals > 0)),
        'frac_negative': float(np.mean(vals < 0)),
        'hist_counts': counts.tolist(),
        'hist_edges': bin_edges.tolist(),
    }


def scalar_curvature(curvatures: dict, weights: Optional[dict] = None) -> float:
    """
    Scalar curvature: weighted mean of edge curvatures.

    If weights is None, uses uniform weighting (simple mean).
    If weights is provided, it should map (i,j) -> weight for each edge.

    Args:
        curvatures: dict mapping (i,j) -> curvature value.
        weights: optional dict mapping (i,j) -> edge weight.

    Returns:
        weighted mean curvature.
    """
    if not curvatures:
        return 0.0

    if weights is None:
        return float(np.mean(list(curvatures.values())))

    total_weight = 0.0
    weighted_sum = 0.0
    for edge, kappa in curvatures.items():
        w = weights.get(edge, 1.0)
        weighted_sum += kappa * w
        total_weight += w

    if total_weight < 1e-14:
        return 0.0

    return float(weighted_sum / total_weight)
