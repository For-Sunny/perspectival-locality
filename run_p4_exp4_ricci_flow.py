#!/usr/bin/env python3
"""
Paper 4, Experiment 4: Discrete Ricci Flow Convergence

Does iterating discrete Ricci flow on an MI graph converge to the
ground-state MI geometry from a random or uniform starting point?

If YES: the MI geometry is a dynamical attractor -- Ricci flow "finds"
the ground state geometry, meaning the geometry is the endpoint of a
dynamical process analogous to Einstein's equations driving spacetime
toward equilibrium.

Ricci flow (Hamilton convention):
  w_{t+1}(i,j) = w_t(i,j) - eps * kappa_t(i,j) * w_t(i,j)

Positive curvature -> edge contracts. Negative curvature -> edge expands.

We use the SAME thresholded graph construction (50th percentile) as the
true MI graph, so each step has ~33 edges and ORC is tractable.

Protocol:
  For 10 seeds (N=12): compute M_true, then run Ricci flow from:
    UNIFORM+noise: all edges ~ mean(M_true) + small Gaussian noise
    RANDOM: exponential random with same mean, symmetrized
    REVERSE: uniform+noise with negative epsilon (control)

Built by Opus Warrior, March 6 2026.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantum import random_all_to_all, ground_state, mutual_information_matrix
from src.curvature import ollivier_ricci

# ── Parameters ──────────────────────────────────────────────
N_QUBITS = 12
SEEDS = list(range(6000, 6010))
MAX_STEPS = 200
EPSILON = 0.01
THRESHOLD = 0.5          # percentile threshold for graph construction
ALPHA_ORC = 0.5
LOG_EVERY = 50


def compute_orc_matrix(mi_matrix, n):
    """Compute ORC with 50th-percentile threshold. Return NxN kappa matrix."""
    result = ollivier_ricci(mi_matrix, threshold=THRESHOLD, alpha=ALPHA_ORC)
    kappa_matrix = np.zeros((n, n))
    for (i, j), kappa in result['edge_curvatures'].items():
        kappa_matrix[i, j] = kappa
        kappa_matrix[j, i] = kappa
    return kappa_matrix, result['scalar_curvature'], result['n_edges']


def ricci_flow(M_true, M_init, epsilon, max_steps, n, label="", seed_label=""):
    """
    Discrete Hamilton Ricci flow:
      M_{t+1} = M_t - eps * kappa_t * M_t
    """
    M_true_norm = np.linalg.norm(M_true, 'fro')
    if M_true_norm < 1e-14:
        M_true_norm = 1.0

    M_t = M_init.copy()
    history = []

    for step in range(max_steps + 1):
        frob_dist = np.linalg.norm(M_t - M_true, 'fro') / M_true_norm

        try:
            kappa_matrix, scalar_curv, n_edges = compute_orc_matrix(M_t, n)
        except Exception as e:
            print(f"    [{seed_label} {label}] ORC failed step {step}: {e}",
                  flush=True)
            history.append({
                'step': step, 'frob_dist': float(frob_dist),
                'scalar_curvature': None, 'max_kappa_dev': None,
                'n_edges': 0, 'failed': True,
            })
            break

        max_kappa_dev = float(np.max(np.abs(kappa_matrix))) if n_edges > 0 else 0.0

        history.append({
            'step': step,
            'frob_dist': float(frob_dist),
            'scalar_curvature': float(scalar_curv),
            'max_kappa_dev': float(max_kappa_dev),
            'n_edges': int(n_edges),
        })

        if step % LOG_EVERY == 0:
            print(f"    [{seed_label} {label}] step {step:4d}: "
                  f"d={frob_dist:.4f}, R={scalar_curv:.4f}, edges={n_edges}",
                  flush=True)

        if step < max_steps:
            M_t = M_t - epsilon * kappa_matrix * M_t
            M_t = np.clip(M_t, 0, None)
            M_t = (M_t + M_t.T) / 2.0
            np.fill_diagonal(M_t, 0.0)

    return history


def run_experiment():
    all_results = {}
    t_total = time.time()

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*70}", flush=True)
        print(f"Seed {seed} ({seed_idx+1}/{len(SEEDS)})", flush=True)
        print(f"{'='*70}", flush=True)
        t_seed = time.time()

        # 1. Ground state
        H, couplings = random_all_to_all(N_QUBITS, seed=seed)
        e0, psi = ground_state(H)
        print(f"  E0 = {e0:.6f}", flush=True)

        # 2. True MI matrix
        M_true = mutual_information_matrix(psi, N_QUBITS)
        mi_pos = M_true[M_true > 1e-14]
        mi_mean = float(np.mean(mi_pos))
        mi_max = float(np.max(M_true))
        mi_std = float(np.std(mi_pos))
        print(f"  MI: mean={mi_mean:.4f}, std={mi_std:.4f}, max={mi_max:.4f}",
              flush=True)

        # True curvature
        _, true_R, true_edges = compute_orc_matrix(M_true, N_QUBITS)
        print(f"  True: R={true_R:.4f}, edges={true_edges}", flush=True)

        # 3. Initializations
        # UNIFORM + small noise (5% of std) to break degeneracy
        # This ensures the percentile thresholding creates a non-trivial graph
        rng_u = np.random.default_rng(seed + 20000)
        noise_scale = 0.05 * mi_std
        M_uniform = np.full((N_QUBITS, N_QUBITS), mi_mean)
        M_uniform += rng_u.normal(0, noise_scale, (N_QUBITS, N_QUBITS))
        M_uniform = (M_uniform + M_uniform.T) / 2.0
        M_uniform = np.clip(M_uniform, 0, None)
        np.fill_diagonal(M_uniform, 0.0)

        # RANDOM: exponential random, same mean, symmetrized
        rng_r = np.random.default_rng(seed + 10000)
        M_random = rng_r.exponential(scale=mi_mean, size=(N_QUBITS, N_QUBITS))
        M_random = (M_random + M_random.T) / 2.0
        np.fill_diagonal(M_random, 0.0)

        seed_label = f"s{seed}"

        # 4. Hamilton Ricci flow
        print(f"\n  --- UNIFORM+noise (Hamilton, eps={EPSILON}) ---", flush=True)
        hist_uniform = ricci_flow(M_true, M_uniform, EPSILON, MAX_STEPS,
                                   N_QUBITS, label="UNI", seed_label=seed_label)

        print(f"\n  --- RANDOM (Hamilton, eps={EPSILON}) ---", flush=True)
        hist_random = ricci_flow(M_true, M_random, EPSILON, MAX_STEPS,
                                  N_QUBITS, label="RND", seed_label=seed_label)

        # 5. Reverse flow control
        print(f"\n  --- UNIFORM+noise (Reverse, eps={-EPSILON}) ---", flush=True)
        hist_reverse = ricci_flow(M_true, M_uniform.copy(), -EPSILON, MAX_STEPS,
                                   N_QUBITS, label="REV", seed_label=seed_label)

        elapsed = time.time() - t_seed
        print(f"\n  Seed {seed}: {elapsed:.1f}s", flush=True)

        all_results[str(seed)] = {
            'e0': float(e0),
            'mi_mean': mi_mean, 'mi_std': mi_std, 'mi_max': mi_max,
            'true_scalar_curvature': float(true_R),
            'true_n_edges': int(true_edges),
            'uniform_flow': hist_uniform,
            'random_flow': hist_random,
            'reverse_flow': hist_reverse,
        }

    total_elapsed = time.time() - t_total

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY (total: {total_elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)

    true_Rs = [r['true_scalar_curvature'] for r in all_results.values()]
    print(f"\nTrue MI scalar curvature: R = {np.mean(true_Rs):.4f} +/- {np.std(true_Rs):.4f}")
    print(f"  (Paper 3 reference: R ~ 0.24)")

    for flow_name, flow_key in [("UNIFORM Hamilton", "uniform_flow"),
                                  ("RANDOM Hamilton", "random_flow"),
                                  ("UNIFORM Reverse (control)", "reverse_flow")]:
        init_d = []
        final_d = []
        final_R = []
        for res in all_results.values():
            hist = res[flow_key]
            if hist:
                init_d.append(hist[0]['frob_dist'])
                final_d.append(hist[-1]['frob_dist'])
                if hist[-1].get('scalar_curvature') is not None:
                    final_R.append(hist[-1]['scalar_curvature'])

        if init_d:
            mi_val = np.mean(init_d)
            mf_val = np.mean(final_d)
            ratio = mf_val / mi_val if mi_val > 0 else float('nan')
            converged = mf_val < mi_val
            print(f"\n  {flow_name}:")
            print(f"    Initial dist: {mi_val:.4f} +/- {np.std(init_d):.4f}")
            print(f"    Final dist:   {mf_val:.4f} +/- {np.std(final_d):.4f}")
            print(f"    Ratio: {ratio:.4f}")
            tag = "CONVERGED" if converged else "DIVERGED"
            print(f"    {tag}")
            if final_R:
                print(f"    Final R: {np.mean(final_R):.4f} +/- {np.std(final_R):.4f}")

    # ── Detailed trajectory analysis ─────────────────────────
    print(f"\n--- Trajectory analysis (mean over seeds) ---")
    for flow_name, flow_key in [("UNIFORM", "uniform_flow"),
                                  ("RANDOM", "random_flow"),
                                  ("REVERSE", "reverse_flow")]:
        # Collect distances at steps 0, 50, 100, 150, 200
        checkpoints = [0, 50, 100, 150, 200]
        for cp in checkpoints:
            dists_at_cp = []
            for res in all_results.values():
                hist = res[flow_key]
                if len(hist) > cp:
                    dists_at_cp.append(hist[cp]['frob_dist'])
            if dists_at_cp:
                print(f"  {flow_name} step {cp:4d}: d = {np.mean(dists_at_cp):.4f} "
                      f"+/- {np.std(dists_at_cp):.4f}")

    # ── Save ─────────────────────────────────────────────────
    summary = {
        'experiment': 'P4_Exp4_Ricci_Flow_Convergence',
        'n_qubits': N_QUBITS, 'seeds': SEEDS,
        'max_steps': MAX_STEPS, 'epsilon': EPSILON,
        'threshold': THRESHOLD, 'alpha_orc': ALPHA_ORC,
        'flow_convention': 'Hamilton: M_{t+1} = M_t - eps*kappa*M_t',
        'total_time_s': total_elapsed,
        'true_R_mean': float(np.mean(true_Rs)),
        'true_R_std': float(np.std(true_Rs)),
    }

    for fname, fkey in [("uniform", "uniform_flow"),
                         ("random", "random_flow"),
                         ("reverse", "reverse_flow")]:
        id_list = [all_results[s][fkey][0]['frob_dist']
                   for s in all_results if all_results[s][fkey]]
        fd_list = [all_results[s][fkey][-1]['frob_dist']
                   for s in all_results if all_results[s][fkey]]
        fR_list = [all_results[s][fkey][-1]['scalar_curvature']
                   for s in all_results if all_results[s][fkey]
                   and all_results[s][fkey][-1].get('scalar_curvature') is not None]
        if id_list:
            mi_v = np.mean(id_list)
            mf_v = np.mean(fd_list)
            summary[f'{fname}_init_dist'] = float(mi_v)
            summary[f'{fname}_final_dist'] = float(mf_v)
            summary[f'{fname}_ratio'] = float(mf_v / mi_v) if mi_v > 0 else None
            summary[f'{fname}_converged'] = bool(mf_v < mi_v)
            if fR_list:
                summary[f'{fname}_final_R_mean'] = float(np.mean(fR_list))

    output = {'summary': summary, 'per_seed': all_results}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'p4_exp4_ricci_flow.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == '__main__':
    run_experiment()
