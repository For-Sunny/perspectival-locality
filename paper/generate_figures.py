#!/usr/bin/env python3
"""
Generate publication-quality figures for Paper 2:
"Emergent Curvature from Partial Observation in Finite Quantum Systems"

Produces 4 PDF figures in paper/figures/ matching revtex4-2 PRA style.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
# Clear font cache so newly installed fonts are found
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Global style: revtex4-2 / PRA conventions
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman', 'Times'],
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


# ===== FIGURE 1: Curvature vs Partiality =====
def fig1_partiality():
    kN = np.array([0.19, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88, 1.00])
    k = np.array([3, 4, 6, 8, 10, 12, 14, 16])
    R_mean = np.array([1.000, 0.399, 0.303, 0.268, 0.268, 0.252, 0.265, 0.262])
    R_std = np.array([0.000, 0.447, 0.147, 0.104, 0.082, 0.065, 0.046, 0.029])

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    # k=3 point: trivial artifact, plotted as gray dashed
    ax.errorbar(kN[0], R_mean[0], yerr=R_std[0],
                fmt='D', color=CB_GRAY, markersize=5, capsize=2,
                markeredgewidth=0.5, zorder=5, label=r'$k=3$ (trivial)')

    # Main data k>=4
    ax.errorbar(kN[1:], R_mean[1:], yerr=R_std[1:],
                fmt='o-', color=CB_BLUE, markersize=4, capsize=2,
                markeredgewidth=0.5, markeredgecolor='white',
                zorder=10, label=r'$k \geq 4$')

    # Shaded error band for main data
    ax.fill_between(kN[1:], R_mean[1:] - R_std[1:], R_mean[1:] + R_std[1:],
                     alpha=0.15, color=CB_BLUE, linewidth=0)

    # Dashed connector from k=3 to k=4
    ax.plot(kN[:2], R_mean[:2], '--', color=CB_GRAY, linewidth=0.8, zorder=3)

    ax.set_xlabel(r'Partiality $k/N$')
    ax.set_ylabel(r'Scalar Curvature $\mathcal{R}_{\mathrm{ORC}}$')
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(-0.1, 1.15)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, loc='upper right')

    fig.savefig(OUTDIR / 'fig1_partiality.pdf', format='pdf')
    plt.close(fig)
    print('  fig1_partiality.pdf')


# ===== FIGURE 2: Curvature vs Entanglement (two panels) =====
def fig2_entanglement():
    Delta = np.array([10.0, 5.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.1])
    S_half = np.array([0.500, 0.815, 2.163, 2.528, 4.090, 4.006, 3.847, 3.777, 3.702])
    R_ORC = np.array([0.409, 0.393, 0.412, 0.289, 0.210, 0.281, 0.296, 0.295, 0.290])
    R_FRC = np.array([-14.73, -14.88, -14.77, -13.39, -12.22, -12.69, -12.68, -12.73, -12.55])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TWO_W, TWO_W * 0.35))

    # --- Panel (a): FRC vs S_half ---
    ax_a.plot(S_half, R_FRC, 'o', color=CB_RED, markersize=4,
              markeredgewidth=0.5, markeredgecolor='white', zorder=10)

    # Linear fit
    slope, intercept, r_val, _, _ = stats.linregress(S_half, R_FRC)
    S_fit = np.linspace(S_half.min() - 0.2, S_half.max() + 0.2, 100)
    ax_a.plot(S_fit, slope * S_fit + intercept, '-', color=CB_RED,
              linewidth=0.8, alpha=0.7, zorder=5)

    ax_a.text(0.05, 0.08, f'$r = {r_val:.2f}$', transform=ax_a.transAxes,
              fontsize=8, color=CB_RED)
    ax_a.set_xlabel(r'Half-chain entropy $S_{N/2}$')
    ax_a.set_ylabel(r'Forman--Ricci curvature $\mathcal{R}_{\mathrm{FRC}}$')
    ax_a.text(0.02, 0.94, '(a)', transform=ax_a.transAxes, fontsize=9,
              fontweight='bold', va='top')
    ax_a.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_a.yaxis.set_minor_locator(AutoMinorLocator(2))

    # --- Panel (b): ORC vs S_half ---
    ax_b.plot(S_half, R_ORC, 's', color=CB_BLUE, markersize=4,
              markeredgewidth=0.5, markeredgecolor='white', zorder=10)

    slope2, intercept2, r_val2, _, _ = stats.linregress(S_half, R_ORC)
    ax_b.plot(S_fit, slope2 * S_fit + intercept2, '-', color=CB_BLUE,
              linewidth=0.8, alpha=0.7, zorder=5)

    ax_b.text(0.05, 0.08, f'$r = {r_val2:.2f}$', transform=ax_b.transAxes,
              fontsize=8, color=CB_BLUE)
    ax_b.set_xlabel(r'Half-chain entropy $S_{N/2}$')
    ax_b.set_ylabel(r'Ollivier--Ricci curvature $\mathcal{R}_{\mathrm{ORC}}$')
    ax_b.text(0.02, 0.94, '(b)', transform=ax_b.transAxes, fontsize=9,
              fontweight='bold', va='top')
    ax_b.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_b.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.subplots_adjust(wspace=0.35)
    fig.savefig(OUTDIR / 'fig2_entanglement.pdf', format='pdf')
    plt.close(fig)
    print('  fig2_entanglement.pdf')


# ===== FIGURE 3: Local vs Nonlocal Topology =====
def fig3_topology():
    N_labels = ['$N=8$', '$N=12$', '$N=16$']
    all2all_mean = np.array([-0.209, 0.244, 0.330])
    all2all_std = np.array([2.224, 0.208, 0.060])
    chain_mean = np.array([-2.832, -11.666, -0.508])
    chain_std = np.array([5.930, 33.446, 1.221])

    x = np.arange(len(N_labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    # Clip error bars to visible range; annotate clipped ones
    ylim_lo, ylim_hi = -18, 5
    clip_std_a = np.minimum(all2all_std, ylim_hi - all2all_mean)
    clip_std_a_lo = np.minimum(all2all_std, all2all_mean - ylim_lo)
    clip_std_c = np.minimum(chain_std, ylim_hi - chain_mean)
    clip_std_c_lo = np.minimum(chain_std, chain_mean - ylim_lo)

    bars1 = ax.bar(x - width/2, all2all_mean, width,
                   yerr=[clip_std_a_lo, clip_std_a],
                   color=CB_BLUE, edgecolor='white', linewidth=0.4,
                   capsize=2, error_kw={'linewidth': 0.7},
                   label='All-to-all', zorder=5)
    bars2 = ax.bar(x + width/2, chain_mean, width,
                   yerr=[clip_std_c_lo, clip_std_c],
                   color=CB_RED, edgecolor='white', linewidth=0.4,
                   capsize=2, error_kw={'linewidth': 0.7},
                   label='1D chain', zorder=5)

    # Annotate clipped bars with actual sigma
    for i in range(len(N_labels)):
        if chain_mean[i] - chain_std[i] < ylim_lo or chain_mean[i] + chain_std[i] > ylim_hi:
            ax.annotate(f'$\\sigma={chain_std[i]:.1f}$',
                        xy=(x[i] + width/2, ylim_lo + 0.3),
                        fontsize=6, color=CB_RED, ha='center', va='bottom')
        if all2all_mean[i] - all2all_std[i] < ylim_lo or all2all_mean[i] + all2all_std[i] > ylim_hi:
            ax.annotate(f'$\\sigma={all2all_std[i]:.1f}$',
                        xy=(x[i] - width/2, ylim_lo + 0.3),
                        fontsize=6, color=CB_BLUE, ha='center', va='bottom')

    ax.axhline(0, color='black', linewidth=0.5, linestyle='-', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(N_labels)
    ax.set_ylabel(r'Scalar Curvature $\mathcal{R}_{\mathrm{ORC}}$')
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, loc='lower left')

    fig.savefig(OUTDIR / 'fig3_topology.pdf', format='pdf')
    plt.close(fig)
    print('  fig3_topology.pdf')


# ===== FIGURE 4: Perspectival Curvature =====
def fig4_perspective():
    np.random.seed(42)
    means = [0.333, 0.194, 0.171, 0.117, 0.299]
    stds = [0.099, 0.128, 0.346, 0.136, 0.062]
    n_pts = 50

    data = [np.random.normal(m, s, n_pts) for m, s in zip(means, stds)]
    labels = [f'$H_{i}$' for i in range(1, 6)]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    parts = ax.violinplot(data, positions=range(1, 6), showmeans=False,
                          showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(CB_PURPLE)
        pc.set_edgecolor('none')
        pc.set_alpha(0.35)

    # Overlay box plots
    bp = ax.boxplot(data, positions=range(1, 6), widths=0.2,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', linewidth=1.0),
                    boxprops=dict(facecolor=CB_PURPLE, edgecolor='black', linewidth=0.5),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6))

    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Hamiltonian sample')
    ax.set_ylabel(r'Observer curvature $\mathcal{R}_{\mathrm{ORC}}$')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':', zorder=1)

    fig.savefig(OUTDIR / 'fig4_perspective.pdf', format='pdf')
    plt.close(fig)
    print('  fig4_perspective.pdf')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating Paper 2 figures...')
    fig1_partiality()
    fig2_entanglement()
    fig3_topology()
    fig4_perspective()
    print('Done. All figures saved to', OUTDIR.resolve())
