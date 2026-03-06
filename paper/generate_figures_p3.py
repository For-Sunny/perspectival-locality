#!/usr/bin/env python3
"""
Generate publication-quality figures for Paper 3:
"Perspectival Locality at Scale: Partiality, Topology, and Entanglement
 in N=8 to N=20 Quantum Systems"

Produces 4 PDF figures in paper/figures/ matching revtex4-2 PRA style.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Global style: revtex4-2 / PRA conventions
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif',
                   'Times New Roman', 'Times'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.8,
    'ytick.minor.size': 1.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'text.usetex': False,
})

OUTDIR = Path(__file__).parent / 'figures'
OUTDIR.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (Okabe-Ito)
CB_BLUE = '#0072B2'
CB_RED = '#D55E00'
CB_GREEN = '#009E73'
CB_ORANGE = '#E69F00'
CB_PURPLE = '#CC79A7'
CB_GRAY = '#999999'

COL_W = 3.375   # single-column width in inches
TWO_W = 7.0     # two-column width in inches


# ===== FIGURE 1: Partiality scaling across system sizes =====
def fig1_partiality_scaling():
    # Data per system size
    data = {
        8: {
            'kN': np.array([0.38, 0.50, 0.63, 0.75, 0.88, 1.00]),
            'R':  np.array([1.00, 0.35, 0.15, 0.10, 0.08, 0.07]),
        },
        12: {
            'kN': np.array([0.25, 0.33, 0.50, 0.67, 0.83, 1.00]),
            'R':  np.array([1.00, 0.40, 0.28, 0.24, 0.23, 0.23]),
        },
        16: {
            'kN': np.array([0.19, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88, 1.00]),
            'R':  np.array([1.000, 0.399, 0.303, 0.268, 0.268, 0.252, 0.265, 0.262]),
        },
        20: {
            'kN': np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]),
            'R':  np.array([0.262, 0.255, 0.249, 0.229, 0.232, 0.229, 0.238, 0.238, 0.240]),
        },
    }

    colors = {8: CB_BLUE, 12: CB_RED, 16: CB_GREEN, 20: CB_ORANGE}
    markers = {8: 'o', 12: 's', 16: 'D', 20: '^'}

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    for N in [8, 12, 16, 20]:
        d = data[N]
        ax.plot(d['kN'], d['R'], marker=markers[N], color=colors[N],
                markersize=4, markeredgewidth=0.5, markeredgecolor='white',
                label=f'$N={N}$', zorder=10)

    ax.set_xlabel(r'Partiality $k/N$')
    ax.set_ylabel(r'Scalar Curvature $\mathcal{R}_{\mathrm{ORC}}$')
    ax.set_xlim(0.12, 1.05)
    ax.set_ylim(-0.05, 1.15)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, loc='upper right')

    fig.savefig(OUTDIR / 'fig1_p3_partiality_scaling.pdf', format='pdf')
    plt.close(fig)
    print('  fig1_p3_partiality_scaling.pdf')


# ===== FIGURE 2: Topology gap vs partiality (per-trial scatter + medians) =====
def fig2_topology_gap():
    import json

    # Load per-trial data from extended 200-trial JSON
    json_path = Path(__file__).parent.parent / 'results' / 'n20_exp3_topology_k_sweep_extended.json'
    with open(json_path) as f:
        data = json.load(f)

    trials = data['trials']
    k_vals = [4, 5, 6, 7, 8, 10]
    kN_vals = [k / 20 for k in k_vals]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))

    medians = []
    for i, k in enumerate(k_vals):
        kt = trials[str(k)]
        gaps = np.array([t['chain_ORC'] - t['all2all_ORC'] for t in kt])
        med = np.median(gaps)
        medians.append(med)

        # Per-trial scatter (jittered)
        jitter = np.random.default_rng(42).uniform(-0.008, 0.008, len(gaps))
        # Clip display for readability but mark outlier count
        clipped = np.abs(gaps) > 8
        display_gaps = np.clip(gaps, -8, 8)

        ax.scatter(kN_vals[i] + jitter[~clipped], display_gaps[~clipped],
                   s=6, color=CB_GRAY, alpha=0.3, zorder=5, linewidths=0)
        # Show clipped points at the boundary
        if clipped.any():
            n_neg = int(np.sum(gaps < -8))
            n_pos = int(np.sum(gaps > 8))
            if n_neg > 0:
                word = 'outlier' if n_neg == 1 else 'outliers'
                ax.annotate(f'{n_neg} {word}\n(min {min(gaps[gaps < -8]):.0f})',
                            xy=(kN_vals[i], -7.8), fontsize=5, color=CB_RED,
                            ha='center', va='top')
                ax.plot(kN_vals[i], -7.5, 'v', color=CB_RED, markersize=4)
            if n_pos > 0:
                word = 'outlier' if n_pos == 1 else 'outliers'
                ax.annotate(f'{n_pos} {word}\n(max +{max(gaps[gaps > 8]):.0f})',
                            xy=(kN_vals[i], 7.8), fontsize=5, color=CB_RED,
                            ha='center', va='bottom')
                ax.plot(kN_vals[i], 7.5, '^', color=CB_RED, markersize=4)

    # Median line
    ax.plot(kN_vals, medians, 's-', color=CB_BLUE, markersize=5,
            markeredgewidth=0.5, markeredgecolor='white', zorder=10,
            label='Median')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', zorder=3)

    ax.set_xlabel(r'Partiality $k/N$')
    ax.set_ylabel(r'ORC Gap (chain $-$ all-to-all)')
    ax.set_xlim(0.15, 0.55)
    ax.set_ylim(-8.5, 8.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, loc='upper right')

    fig.savefig(OUTDIR / 'fig2_p3_topology_gap.pdf', format='pdf')
    plt.close(fig)
    print('  fig2_p3_topology_gap.pdf')


# ===== FIGURE 3: Perspectival convergence across N =====
def fig3_perspective_convergence():
    N_vals = np.array([8, 12, 16, 20])
    sigma = np.array([0.81, 0.31, 0.15, 0.09])

    # Fit A / sqrt(N)
    def inv_sqrt(x, A):
        return A / np.sqrt(x)

    popt, _ = curve_fit(inv_sqrt, N_vals, sigma)
    A_fit = popt[0]

    N_fit = np.linspace(6, 24, 100)

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    ax.plot(N_vals, sigma, 'o', color=CB_BLUE, markersize=5,
            markeredgewidth=0.5, markeredgecolor='white', zorder=10,
            label='Data')
    ax.plot(N_vals, sigma, '-', color=CB_BLUE, linewidth=0.8, zorder=5)
    ax.plot(N_fit, inv_sqrt(N_fit, A_fit), '--', color=CB_GRAY, linewidth=0.8,
            label=rf'${A_fit:.2f}/\sqrt{{N}}$', zorder=3)
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':', zorder=1)

    ax.set_xlabel(r'System size $N$')
    ax.set_ylabel(r'Perspective spread $\sigma_{\mathcal{R}}$')
    ax.set_xlim(5, 23)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(N_vals)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, loc='upper right')

    fig.savefig(OUTDIR / 'fig3_p3_perspective_convergence.pdf', format='pdf')
    plt.close(fig)
    print('  fig3_p3_perspective_convergence.pdf')


# ===== FIGURE 4: Entanglement correlation across N =====
def fig4_entanglement_scaling():
    N_vals = np.array([8, 12, 16, 20])
    r_vals = np.array([0.98, 0.94, 0.94, 0.48])

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    ax.plot(N_vals, r_vals, 'o-', color=CB_RED, markersize=5,
            markeredgewidth=0.5, markeredgecolor='white', zorder=10)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--', zorder=1)

    ax.set_xlabel(r'System size $N$')
    ax.set_ylabel(r'Pearson $r$ (FRC vs entropy)')
    ax.set_xlim(5, 23)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(N_vals)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.savefig(OUTDIR / 'fig4_p3_entanglement_scaling.pdf', format='pdf')
    plt.close(fig)
    print('  fig4_p3_entanglement_scaling.pdf')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating Paper 3 figures...')
    fig1_partiality_scaling()
    fig2_topology_gap()
    fig3_perspective_convergence()
    fig4_entanglement_scaling()
    print('Done. All figures saved to', OUTDIR.resolve())
