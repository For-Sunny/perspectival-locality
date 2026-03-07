#!/usr/bin/env python3
"""
Generate publication-quality figures for Paper 4:
"Matter-Geometry Complementarity from Partial Observation"

Produces 4 PDF figures in paper/figures/ matching revtex4-2 PRA style.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Global style: revtex4-2 / PRA conventions (matching Paper 3)
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
RESULTS = Path(__file__).parent.parent / 'results'

# Colorblind-friendly palette (Okabe-Ito)
CB_BLUE = '#0072B2'
CB_RED = '#D55E00'
CB_GREEN = '#009E73'
CB_ORANGE = '#E69F00'
CB_PURPLE = '#CC79A7'
CB_GRAY = '#999999'

COL_W = 3.375   # single-column width in inches
TWO_W = 7.0     # two-column width in inches


# ===== FIGURE 1: Edge-level kappa vs T_ZZ across seeds =====
def fig1_edge_kappa_vs_tzz():
    """
    Per-seed edge-level Pearson r(kappa, T_ZZ) for each eigenstate,
    shown as a strip plot across 20 disorder realizations.  Each dot is
    one seed's within-graph correlation; the horizontal bar is the mean.
    Eigenstate indices: ground (0), first excited (1), mid-spectrum (5),
    high-energy (10).
    """
    with open(RESULTS / 'p4_exp2_edge_stress_energy.json') as f:
        data = json.load(f)

    per_seed = data['per_seed']
    state_indices = ['0', '1', '5', '10']
    state_labels = ['Ground\n($n=0$)', '$n=1$', '$n=5$', '$n=10$']

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))

    rng = np.random.default_rng(42)
    for i, (si, label) in enumerate(zip(state_indices, state_labels)):
        r_vals = []
        for entry in per_seed:
            r_vals.append(entry['states'][si]['r_kappa_tzz'])
        r_vals = np.array(r_vals)

        # Jittered strip
        jitter = rng.uniform(-0.12, 0.12, len(r_vals))
        ax.scatter(np.full_like(r_vals, i) + jitter, r_vals,
                   s=12, color=CB_GRAY, alpha=0.4, zorder=5,
                   linewidths=0)

        # Mean marker
        mean_r = np.mean(r_vals)
        ax.plot([i - 0.22, i + 0.22], [mean_r, mean_r],
                '-', color=CB_BLUE, linewidth=1.8, zorder=10)

        # Annotate mean
        ax.text(i + 0.28, mean_r, f'{mean_r:.2f}', fontsize=7,
                color=CB_BLUE, va='center', ha='left')

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', zorder=3)

    ax.set_xticks(range(len(state_indices)))
    ax.set_xticklabels(state_labels)
    ax.set_ylabel(
        r'Edge-level $r(\kappa, T_{ZZ})$'
    )
    ax.set_xlim(-0.5, len(state_indices) - 0.2)
    ax.set_ylim(-1.05, 0.55)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Overall annotation
    summary = data['summary']
    ground_mean = summary['0']['mean_r_kappa_tzz']
    ground_frac = summary['0']['frac_sig_tzz']
    ax.text(0.97, 0.05,
            f'Ground: $\\bar{{r}} = {ground_mean:.2f}$\n'
            f'{ground_frac*100:.0f}% seeds $p < 0.05$',
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=CB_GRAY, alpha=0.8, linewidth=0.5))

    fig.savefig(OUTDIR / 'fig1_p4_edge_kappa_vs_tzz.pdf', format='pdf')
    plt.close(fig)
    print('  fig1_p4_edge_kappa_vs_tzz.pdf')


# ===== FIGURE 2: Matter-geometry complementarity scatter =====
def fig2_complementarity_scatter():
    """
    Individual r(T_obs, S_obs) values for N=14, k=7 across 10 seeds,
    showing the universality of anti-correlation.
    """
    with open(RESULTS / 'p4_complementarity.json') as f:
        data = json.load(f)

    # Find N=14, k=7 in universality
    target = None
    for entry in data['universality']:
        if entry['N'] == 14 and entry['k'] == 7:
            target = entry
            break

    if target is None:
        print('  WARNING: N=14,k=7 not found; skipping fig2')
        return

    individual_r = np.array(target['individual_r'])
    seeds = np.arange(1, len(individual_r) + 1)

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))

    # Individual seed dots
    ax.scatter(seeds, individual_r, s=25, color=CB_GRAY, alpha=0.6,
               zorder=5, linewidths=0.3, edgecolors='white')

    # Mean and std band
    mean_r = target['mean_r_TS']
    std_r = target['std_r_TS']
    ax.axhline(mean_r, color=CB_RED, linewidth=1.2, zorder=10,
               label=f'Mean $r = {mean_r:.2f}$')
    ax.axhspan(mean_r - std_r, mean_r + std_r, color=CB_RED, alpha=0.12,
               zorder=1)

    # Reference line at r=0
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', zorder=3)

    ax.set_xlabel('Seed index')
    ax.set_ylabel(
        r'Pearson $r(T_{\mathrm{obs}}, S_{\mathrm{obs}})$'
    )
    ax.set_xlim(0, len(individual_r) + 1)
    ax.set_ylim(-1.05, 0.2)
    ax.set_xticks(seeds)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, loc='upper right')

    # Title annotation
    ax.set_title(f'$N = {target["N"]}$, $k = {target["k"]}$', fontsize=9)

    fig.savefig(OUTDIR / 'fig2_p4_complementarity_scatter.pdf', format='pdf')
    plt.close(fig)
    print('  fig2_p4_complementarity_scatter.pdf')


# ===== FIGURE 3: Complementarity vs partiality (k/N) =====
def fig3_complementarity_vs_partiality():
    """
    Mean r(T_obs, S_obs) vs k/N with error bars from std across seeds.
    """
    with open(RESULTS / 'p4_complementarity.json') as f:
        data = json.load(f)

    partiality = data['partiality']

    k_over_N = np.array([p['k_over_N'] for p in partiality])
    mean_r = np.array([p['mean_r_TS'] for p in partiality])
    std_r = np.array([p['std_r_TS'] for p in partiality])

    # Separate the degenerate k/N=1 point (full observation, r=0 by definition)
    partial_mask = k_over_N < 0.99
    full_mask = ~partial_mask

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    ax.errorbar(k_over_N[partial_mask], mean_r[partial_mask],
                yerr=std_r[partial_mask], fmt='o-', color=CB_BLUE,
                markersize=5, markeredgewidth=0.5, markeredgecolor='white',
                capsize=2.5, capthick=0.8, linewidth=1.0, zorder=10)

    # Degenerate point: open marker
    if full_mask.any():
        ax.plot(k_over_N[full_mask], mean_r[full_mask], 'o',
                color='white', markersize=5, markeredgewidth=1.0,
                markeredgecolor=CB_BLUE, zorder=10)
        # Connect with dashed line to last partial point
        last_partial = np.where(partial_mask)[0][-1]
        ax.plot([k_over_N[last_partial], k_over_N[full_mask][0]],
                [mean_r[last_partial], mean_r[full_mask][0]],
                '--', color=CB_BLUE, linewidth=0.8, zorder=9)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', zorder=3)

    ax.set_xlabel(r'Partiality $k/N$')
    ax.set_ylabel(r'Pearson $r(T_{\mathrm{obs}}, S_{\mathrm{obs}})$')
    ax.set_xlim(0.18, 1.05)
    ax.set_ylim(-1.1, 0.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.savefig(OUTDIR / 'fig3_p4_complementarity_vs_partiality.pdf',
                format='pdf')
    plt.close(fig)
    print('  fig3_p4_complementarity_vs_partiality.pdf')


# ===== FIGURE 4: Complementarity across system sizes =====
def fig4_complementarity_vs_N():
    """
    Mean r(T_obs, S_obs) vs N for N=12, 14, 16 with error bars.
    """
    with open(RESULTS / 'p4_complementarity.json') as f:
        data = json.load(f)

    universality = data['universality']

    N_vals = np.array([u['N'] for u in universality])
    mean_r = np.array([u['mean_r_TS'] for u in universality])
    std_r = np.array([u['std_r_TS'] for u in universality])

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    ax.errorbar(N_vals, mean_r, yerr=std_r, fmt='o-', color=CB_BLUE,
                markersize=5, markeredgewidth=0.5, markeredgecolor='white',
                capsize=2.5, capthick=0.8, linewidth=1.0, zorder=10)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', zorder=3)

    ax.set_xlabel(r'System size $N$')
    ax.set_ylabel(r'Pearson $r(T_{\mathrm{obs}}, S_{\mathrm{obs}})$')
    ax.set_xlim(10, 18)
    ax.set_ylim(-1.0, 0.2)
    ax.set_xticks(N_vals)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.savefig(OUTDIR / 'fig4_p4_complementarity_vs_N.pdf', format='pdf')
    plt.close(fig)
    print('  fig4_p4_complementarity_vs_N.pdf')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating Paper 4 figures...')
    fig1_edge_kappa_vs_tzz()
    fig2_complementarity_scatter()
    fig3_complementarity_vs_partiality()
    fig4_complementarity_vs_N()
    print('Done. All figures saved to', OUTDIR.resolve())
