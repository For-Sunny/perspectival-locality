#!/usr/bin/env python3
"""
Publication-quality figures for PLC paper (PRA upgrade, 10+ pages).
All new experimental data incorporated: hardened stats, scaling, circularity
breaking, symmetry breaking, null models, distance robustness.

Figures 1-6, all output as PDF.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# -- PRA Style Configuration --------------------------------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Computer Modern Roman', 'Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

# Colorblind-safe palette
C_BLUE   = '#0072B2'
C_ORANGE = '#D55E00'
C_GREEN  = '#009E73'
C_PINK   = '#CC79A7'
C_YELLOW = '#F0E442'
C_SKY    = '#56B4E9'
C_BLACK  = '#000000'
C_GRAY   = '#999999'

# PRA figure widths (inches)
SINGLE_COL = 3.375
DOUBLE_COL = 7.0

# -- Data paths ----------------------------------------------------------------

BASE = Path(__file__).parent.parent
RESULTS = BASE / 'results'
FIGDIR = BASE / 'figures'
FIGDIR.mkdir(exist_ok=True)


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


# -- Load all data -------------------------------------------------------------

hardened  = load('hardened_stats.json')
scaling   = load('scaling_N10_N12.json')
circular  = load('circularity_breaking.json')
symmetry  = load('symmetry_breaking.json')
nulls     = load('null_models.json')
dist_rob  = load('distance_robustness.json')


def panel_label(ax, label, x=-0.14, y=1.08):
    """Place (a), (b) style panel label in bold."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')


# ==============================================================================
# FIGURE 1: Core Result (2 panels)
# (a) Pearson r vs k/N at N=8
# (b) dim_ratio vs k/N at N=8
# ==============================================================================

def make_figure1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    exp5 = hardened['experiment_5_correlation_decay']
    exp2 = hardened['experiment_2_emergent_metric']

    # -- Panel (a): Pearson r vs k/N at N=8 --
    k_over_n_r, r_est, r_lo, r_hi = [], [], [], []
    for key in sorted(exp5.keys()):
        d = exp5[key]
        if d['N'] != 8:
            continue
        k_over_n_r.append(d['k_over_N'])
        bs = d['pearson_r_bootstrap']
        r_est.append(bs['estimate'])
        r_lo.append(bs['ci_low'])
        r_hi.append(bs['ci_high'])

    k_over_n_r = np.array(k_over_n_r)
    r_est = np.array(r_est)
    r_lo = np.array(r_lo)
    r_hi = np.array(r_hi)
    yerr_r = np.array([r_est - r_lo, r_hi - r_est])

    ax1.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1,
                label='No correlation')
    ax1.errorbar(k_over_n_r, r_est, yerr=yerr_r, fmt='o', color=C_BLUE,
                 capsize=4, capthick=0.8, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.5, zorder=3,
                 label='Observed (N=8)')

    # Shade CI region
    ax1.fill_between(k_over_n_r, r_lo, r_hi, alpha=0.15, color=C_BLUE, zorder=1)

    ax1.set_xlabel('Observer fraction k/N')
    ax1.set_ylabel('Pearson r (MI-distance vs |C_ij|)')
    ax1.set_xlim(0.3, 0.7)
    ax1.set_ylim(-0.85, 0.15)
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    panel_label(ax1, '(a)')

    # -- Panel (b): dim_ratio vs k/N at N=8 --
    k_over_n_d, d_est, d_lo, d_hi = [], [], [], []
    for key in sorted(exp2.keys()):
        d = exp2[key]
        if d['N'] != 8:
            continue
        k_over_n_d.append(d['k_over_N'])
        bs = d['dim_ratio_bootstrap']
        d_est.append(bs['estimate'])
        d_lo.append(bs['ci_low'])
        d_hi.append(bs['ci_high'])

    k_over_n_d = np.array(k_over_n_d)
    d_est = np.array(d_est)
    d_lo = np.array(d_lo)
    d_hi = np.array(d_hi)
    yerr_d = np.array([d_est - d_lo, d_hi - d_est])

    ax2.axhline(1.0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1,
                label='No compression')
    ax2.errorbar(k_over_n_d, d_est, yerr=yerr_d, fmt='s', color=C_ORANGE,
                 capsize=4, capthick=0.8, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.5, zorder=3,
                 label='Observed (N=8)')

    ax2.fill_between(k_over_n_d, d_lo, d_hi, alpha=0.15, color=C_ORANGE, zorder=1)

    ax2.set_xlabel('Observer fraction k/N')
    ax2.set_ylabel('Effective dim. ratio d_obs / d_full')
    ax2.set_xlim(0.3, 0.7)
    ax2.set_ylim(0.4, 1.35)
    ax2.legend(loc='lower right', framealpha=0.9, edgecolor='none', fontsize=7)
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig1_core_result.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ==============================================================================
# FIGURE 2: Scaling (2 panels)
# (a) Pearson r vs k/N for N=8, 10, 12
# (b) Pearson r at k/N ~ 0.3 vs N
# ==============================================================================

def make_figure2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    exp5_h = hardened['experiment_5_correlation_decay']

    # Collect all data points: (N, k_over_N, r, ci_lo, ci_hi)
    all_points = []

    # N=8 from hardened
    for key in sorted(exp5_h.keys()):
        d = exp5_h[key]
        if d['N'] == 8:
            bs = d['pearson_r_bootstrap']
            all_points.append((8, d['k_over_N'], bs['estimate'], bs['ci_low'], bs['ci_high']))

    # N=10 from scaling (more k-values than hardened)
    for k_key, val in scaling['N=10']['exp5'].items():
        bs = val['pearson_r']
        all_points.append((10, val['k_over_N'], bs['estimate'], bs['ci_low'], bs['ci_high']))

    # N=12 from scaling
    for k_key, val in scaling['N=12']['exp5'].items():
        bs = val['pearson_r']
        all_points.append((12, val['k_over_N'], bs['estimate'], bs['ci_low'], bs['ci_high']))

    # -- Panel (a): r vs k/N for each N --
    markers = {'8': 'o', '10': 's', '12': 'D'}
    colors_n = {'8': C_BLUE, '10': C_ORANGE, '12': C_GREEN}

    for N_val in [8, 10, 12]:
        pts = sorted([p for p in all_points if p[0] == N_val], key=lambda x: x[1])
        if not pts:
            continue
        kn = np.array([p[1] for p in pts])
        r_vals = np.array([p[2] for p in pts])
        lo = np.array([p[3] for p in pts])
        hi = np.array([p[4] for p in pts])
        yerr = np.array([r_vals - lo, hi - r_vals])

        ax1.errorbar(kn, r_vals, yerr=yerr,
                     fmt=markers[str(N_val)] + '-', color=colors_n[str(N_val)],
                     capsize=3, capthick=0.7, markersize=5,
                     markeredgecolor='white', markeredgewidth=0.4,
                     label=f'N = {N_val}', zorder=3)

    ax1.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)
    ax1.set_xlabel('Observer fraction k/N')
    ax1.set_ylabel('Pearson r (MI-distance vs |C_ij|)')
    ax1.set_xlim(0.2, 0.8)
    ax1.set_ylim(-0.95, 0.1)
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    panel_label(ax1, '(a)')

    # -- Panel (b): r at k/N ~ 0.3 vs N --
    # Find the point closest to k/N=0.3 for each N
    target_kn = 0.33  # approximate target
    n_vals_plot = []
    r_at_03 = []
    r_lo_03 = []
    r_hi_03 = []

    for N_val in [8, 10, 12]:
        pts = [p for p in all_points if p[0] == N_val]
        # Pick the point with k/N closest to 0.33
        best = min(pts, key=lambda p: abs(p[1] - target_kn))
        n_vals_plot.append(N_val)
        r_at_03.append(best[2])
        r_lo_03.append(best[3])
        r_hi_03.append(best[4])

    n_vals_plot = np.array(n_vals_plot)
    r_at_03 = np.array(r_at_03)
    r_lo_03 = np.array(r_lo_03)
    r_hi_03 = np.array(r_hi_03)
    yerr_03 = np.array([r_at_03 - r_lo_03, r_hi_03 - r_at_03])

    ax2.errorbar(n_vals_plot, r_at_03, yerr=yerr_03,
                 fmt='o-', color=C_BLUE, capsize=5, capthick=0.8,
                 markersize=7, markeredgecolor='white', markeredgewidth=0.5,
                 zorder=3, linewidth=1.2)

    ax2.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)

    # Shade a band showing the effect stays strong
    ax2.fill_between(n_vals_plot, r_lo_03, r_hi_03, alpha=0.15, color=C_BLUE, zorder=1)

    ax2.set_xlabel('System size N (qubits)')
    ax2.set_ylabel('Pearson r at k/N ~ 0.3')
    ax2.set_xticks([8, 10, 12])
    ax2.set_xlim(7, 13)
    ax2.set_ylim(-0.95, 0.1)

    # Annotate: "No weakening"
    ax2.annotate('No weakening with N',
                 xy=(10, -0.55), fontsize=8, ha='center', color=C_BLUE,
                 fontstyle='italic')
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig2_scaling.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ==============================================================================
# FIGURE 3: Cross-Observer Circularity Breaker (2 panels)
# (a) Bar chart comparing coupling distance r vs cross-observer MI r vs
#     self-observer MI r (hardened stats)
# (b) Histogram of cross-observer Pearson r values
# ==============================================================================

def make_figure3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    circ_n8 = circular['tests']['N=8']

    # Test 1: coupling distance vs correlator (random all-to-all) -> weak r
    t1 = circ_n8['test1_coupling_vs_corr_alltoall']
    t1_r = t1['mean_r']
    t1_ci = t1['bootstrap_ci']

    # Test 2: cross-observer MI-distance -> strong r
    t2 = circ_n8['test2_cross_observer']
    t2_r = t2['mean_r']
    t2_ci = t2['bootstrap_ci']

    # Self-observer: use hardened_stats exp5 at k/N=0.375 (same k as cross-observer k_obs=4 -> k/N=0.5)
    # Actually use k=4 (k/N=0.5) to match the observer fraction
    exp5_self = hardened['experiment_5_correlation_decay']['N=8_k=4']
    self_r = exp5_self['pearson_r_bootstrap']['estimate']
    self_ci_lo = exp5_self['pearson_r_bootstrap']['ci_low']
    self_ci_hi = exp5_self['pearson_r_bootstrap']['ci_high']

    # -- Panel (a): Bar chart --
    labels = ['Coupling\ndistance', 'Cross-observer\nMI-distance', 'Self-observer\nMI-distance']
    means = [t1_r, t2_r, self_r]
    ci_lo = [t1_ci['ci_low'], t2_ci['ci_low'], self_ci_lo]
    ci_hi = [t1_ci['ci_high'], t2_ci['ci_high'], self_ci_hi]
    colors = [C_GRAY, C_BLUE, C_ORANGE]

    x_pos = np.arange(len(labels))
    yerr = np.array([np.array(means) - np.array(ci_lo),
                     np.array(ci_hi) - np.array(means)])

    bars = ax1.bar(x_pos, means, width=0.6, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.8, zorder=2)
    ax1.errorbar(x_pos, means, yerr=yerr, fmt='none',
                 ecolor=C_BLACK, capsize=5, capthick=0.8, zorder=3)

    ax1.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel('Mean Pearson r')
    ax1.set_ylim(-1.0, 0.15)

    # Annotate magnitudes
    for i, (m, c) in enumerate(zip(means, colors)):
        ax1.text(i, m - 0.06, f'r = {m:.2f}', ha='center', va='top',
                 fontsize=7, fontweight='bold', color='white' if abs(m) > 0.3 else C_BLACK)

    ax1.set_title('N = 8, circularity breaking', fontsize=9, pad=4)
    panel_label(ax1, '(a)')

    # -- Panel (b): Histogram of cross-observer r values --
    r_vals = np.array(t2['r_values'])
    n_neg = np.sum(r_vals < 0)
    pct_neg = 100 * n_neg / len(r_vals)

    bins = np.linspace(-1.1, 1.1, 30)
    ax2.hist(r_vals, bins=bins, color=C_BLUE, alpha=0.75, edgecolor='white',
             linewidth=0.4, zorder=2)
    ax2.axvline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)
    ax2.axvline(t2_r, color=C_ORANGE, linewidth=1.5, linestyle='-', zorder=3,
                label=f'Mean r = {t2_r:.2f}')

    ax2.set_xlabel('Pearson r (cross-observer)')
    ax2.set_ylabel('Count')
    ax2.set_xlim(-1.15, 1.15)
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='none', fontsize=7)

    # Annotate percentage negative
    ax2.annotate(f'{pct_neg:.0f}% negative',
                 xy=(-0.5, ax2.get_ylim()[1] * 0.85), fontsize=8,
                 ha='center', color=C_BLUE, fontweight='bold')

    ax2.set_title('Cross-observer MI-distance r values', fontsize=9, pad=4)
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig3_circularity_breaking.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ==============================================================================
# FIGURE 4: Symmetry Breaking (2 panels)
# (a) Grouped bar chart: dim_ratio at k/N=0.375 for each model
# (b) Grouped bar chart: Pearson r (ZZ and XX) at k/N=0.375
# ==============================================================================

def make_figure4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    sym = symmetry['summary']
    model_keys = ['heisenberg', 'xxz_0.5', 'random_pauli']
    model_labels = ['Heisenberg\nSU(2)', 'XXZ\nU(1)', 'Random Pauli\n(no symmetry)']
    model_colors = [C_BLUE, C_ORANGE, C_GREEN]

    k_target = '3'  # k=3 -> k/N = 0.375

    # -- Panel (a): dim_ratio at k/N=0.375 --
    dim_means = []
    dim_lo = []
    dim_hi = []

    for mk in model_keys:
        kr = sym[mk]['k_results'][k_target]
        dim_means.append(kr['dim_ratio']['mean'])
        dim_lo.append(kr['dim_ratio']['ci_lo'])
        dim_hi.append(kr['dim_ratio']['ci_hi'])

    dim_means = np.array(dim_means)
    dim_lo = np.array(dim_lo)
    dim_hi = np.array(dim_hi)
    yerr_dim = np.array([dim_means - dim_lo, dim_hi - dim_means])

    x_pos = np.arange(len(model_keys))
    bars1 = ax1.bar(x_pos, dim_means, width=0.55, color=model_colors, alpha=0.85,
                    edgecolor='white', linewidth=0.8, zorder=2)
    ax1.errorbar(x_pos, dim_means, yerr=yerr_dim, fmt='none',
                 ecolor=C_BLACK, capsize=5, capthick=0.8, zorder=3)

    ax1.axhline(1.0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1,
                label='No compression')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_labels, fontsize=7.5)
    ax1.set_ylabel('Dim. ratio d_obs / d_full')
    ax1.set_ylim(0.4, 1.2)
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax1.set_title('k/N = 0.375, N = 8', fontsize=9, pad=4)

    # Annotate: strongest without symmetry
    ax1.annotate('Strongest\ncompression',
                 xy=(2, dim_means[2] - 0.02), fontsize=7, ha='center', va='top',
                 color=C_GREEN, fontweight='bold')
    panel_label(ax1, '(a)')

    # -- Panel (b): Pearson r (ZZ and XX) at k/N=0.375 --
    bar_width = 0.35
    x_pos2 = np.arange(len(model_keys))

    zz_means, zz_lo, zz_hi = [], [], []
    xx_means, xx_lo, xx_hi = [], [], []

    for mk in model_keys:
        kr = sym[mk]['k_results'][k_target]
        zz_means.append(kr['pearson_zz']['mean'])
        zz_lo.append(kr['pearson_zz']['ci_lo'])
        zz_hi.append(kr['pearson_zz']['ci_hi'])
        xx_means.append(kr['pearson_xx']['mean'])
        xx_lo.append(kr['pearson_xx']['ci_lo'])
        xx_hi.append(kr['pearson_xx']['ci_hi'])

    zz_means = np.array(zz_means)
    zz_lo = np.array(zz_lo)
    zz_hi = np.array(zz_hi)
    xx_means = np.array(xx_means)
    xx_lo = np.array(xx_lo)
    xx_hi = np.array(xx_hi)

    yerr_zz = np.array([zz_means - zz_lo, zz_hi - zz_means])
    yerr_xx = np.array([xx_means - xx_lo, xx_hi - xx_means])

    bars_zz = ax2.bar(x_pos2 - bar_width/2, zz_means, bar_width,
                      color=C_BLUE, alpha=0.85, edgecolor='white',
                      linewidth=0.8, zorder=2, label='ZZ correlator')
    bars_xx = ax2.bar(x_pos2 + bar_width/2, xx_means, bar_width,
                      color=C_PINK, alpha=0.85, edgecolor='white',
                      linewidth=0.8, zorder=2, label='XX correlator')

    ax2.errorbar(x_pos2 - bar_width/2, zz_means, yerr=yerr_zz, fmt='none',
                 ecolor=C_BLACK, capsize=3, capthick=0.7, zorder=3)
    ax2.errorbar(x_pos2 + bar_width/2, xx_means, yerr=yerr_xx, fmt='none',
                 ecolor=C_BLACK, capsize=3, capthick=0.7, zorder=3)

    ax2.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(model_labels, fontsize=7.5)
    ax2.set_ylabel('Pearson r')
    ax2.set_ylim(-1.1, 0.1)
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax2.set_title('k/N = 0.375, N = 8', fontsize=9, pad=4)
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig4_symmetry_breaking.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ==============================================================================
# FIGURE 5: Null Models (2 panels)
# (a) Real r vs null model means at k/N=0.375
# (b) Effect sizes (Cohen's d) at k/N=0.375 and k/N=0.5
# ==============================================================================

def make_figure5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    null_data = nulls['results']

    # k/N=0.375 data
    r375 = null_data['N=8_k=3_ratio=0.375']['summary']
    # k/N=0.5 data
    r500 = null_data['N=8_k=4_ratio=0.5']['summary']

    null_types = ['shuffle', 'eigenvalue', 'degree', 'hamiltonian']
    null_labels = ['Shuffle', 'Eigenvalue-\npreserving', 'Degree-\npreserving', 'Random H']

    # -- Panel (a): Real r vs null means at k/N=0.375 --
    real_r = r375['real_r_mean']
    real_ci = r375['real_r_ci']

    null_means_375 = [r375[nt]['null_r_mean'] for nt in null_types]
    # Use effect_size_ci to derive approximate null CIs
    # Actually use null_r_std for error bars
    null_stds_375 = [r375[nt]['null_r_std'] for nt in null_types]

    x_pos = np.arange(len(null_types))

    # Plot real as horizontal band
    ax1.axhspan(real_ci['ci_low'], real_ci['ci_high'], alpha=0.2, color=C_BLUE, zorder=1)
    ax1.axhline(real_r, color=C_BLUE, linewidth=1.2, linestyle='-', zorder=2,
                label=f'Real r = {real_r:.2f}')

    # Plot null means with error bars
    ax1.errorbar(x_pos, null_means_375, yerr=null_stds_375,
                 fmt='D', color=C_ORANGE, capsize=5, capthick=0.8,
                 markersize=7, markeredgecolor='white', markeredgewidth=0.5,
                 zorder=3, label='Null model mean')

    ax1.axhline(0, color=C_GRAY, linestyle=':', linewidth=0.5, zorder=0)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(null_labels, fontsize=7)
    ax1.set_ylabel('Pearson r')
    ax1.set_ylim(-1.15, 0.5)
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax1.set_title('k/N = 0.375, N = 8', fontsize=9, pad=4)
    panel_label(ax1, '(a)')

    # -- Panel (b): Cohen's d at both k/N values --
    bar_width = 0.35

    d_375 = [abs(r375[nt]['mean_effect_size']) for nt in null_types]
    d_500 = [abs(r500[nt]['mean_effect_size']) for nt in null_types]

    # CIs on effect sizes
    d_375_lo = [abs(r375[nt]['effect_size_ci']['ci_low']) for nt in null_types]
    d_375_hi = [abs(r375[nt]['effect_size_ci']['ci_high']) for nt in null_types]
    d_500_lo = [abs(r500[nt]['effect_size_ci']['ci_low']) for nt in null_types]
    d_500_hi = [abs(r500[nt]['effect_size_ci']['ci_high']) for nt in null_types]

    # Compute proper error bars (distance from mean)
    d_375_arr = np.array(d_375)
    d_500_arr = np.array(d_500)

    # For effect sizes, use se from CI
    d_375_se = [r375[nt]['effect_size_ci']['se'] for nt in null_types]
    d_500_se = [r500[nt]['effect_size_ci']['se'] for nt in null_types]

    bars1 = ax2.bar(x_pos - bar_width/2, d_375_arr, bar_width,
                    color=C_BLUE, alpha=0.85, edgecolor='white',
                    linewidth=0.8, zorder=2, label='k/N = 0.375')
    bars2 = ax2.bar(x_pos + bar_width/2, d_500_arr, bar_width,
                    color=C_ORANGE, alpha=0.85, edgecolor='white',
                    linewidth=0.8, zorder=2, label='k/N = 0.5')

    ax2.errorbar(x_pos - bar_width/2, d_375_arr, yerr=d_375_se, fmt='none',
                 ecolor=C_BLACK, capsize=3, capthick=0.7, zorder=3)
    ax2.errorbar(x_pos + bar_width/2, d_500_arr, yerr=d_500_se, fmt='none',
                 ecolor=C_BLACK, capsize=3, capthick=0.7, zorder=3)

    # Reference lines for effect size interpretation
    ax2.axhline(0.8, color=C_GRAY, linestyle=':', linewidth=0.6, zorder=0)
    ax2.text(len(null_types) - 0.5, 0.85, 'Large effect', fontsize=6,
             color=C_GRAY, ha='right', va='bottom')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(null_labels, fontsize=7)
    ax2.set_ylabel("|Cohen's d|")
    ax2.set_ylim(0, 3.2)
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='none', fontsize=7)
    ax2.set_title('Effect sizes vs null models', fontsize=9, pad=4)
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig5_null_models.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ==============================================================================
# FIGURE 6: Distance Robustness (1 panel, single-column)
# Pearson r at k/N=0.375 for all 5 distance metrics
# ==============================================================================

def make_figure6():
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL, 2.8))

    metric_keys = ['subtract', 'inverse', 'neglog', 'normalized', 'sqrt']
    metric_labels = ['Subtract\n(1 - MI)', 'Inverse\n(1/MI)', 'Neg-log\n(-log MI)',
                     'Normalized\n(1 - MI/H)', 'Sqrt\n(sqrt(1-MI))']

    k_target = '0.375'

    r_vals = []
    r_lo = []
    r_hi = []

    for mk in metric_keys:
        d = dist_rob[mk][k_target]['pearson_r']
        r_vals.append(d['estimate'])
        r_lo.append(d['ci_low'])
        r_hi.append(d['ci_high'])

    r_vals = np.array(r_vals)
    r_lo = np.array(r_lo)
    r_hi = np.array(r_hi)
    yerr = np.array([r_vals - r_lo, r_hi - r_vals])

    x_pos = np.arange(len(metric_keys))
    colors_met = [C_BLUE, C_ORANGE, C_GREEN, C_PINK, C_SKY]

    ax.bar(x_pos, r_vals, width=0.6, color=colors_met, alpha=0.85,
           edgecolor='white', linewidth=0.8, zorder=2)
    ax.errorbar(x_pos, r_vals, yerr=yerr, fmt='none',
                ecolor=C_BLACK, capsize=5, capthick=0.8, zorder=3)

    ax.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)

    # Shade the band showing they all cluster
    r_mean_all = np.mean(r_vals)
    r_std_all = np.std(r_vals)
    ax.axhspan(r_mean_all - r_std_all, r_mean_all + r_std_all,
               alpha=0.1, color=C_BLUE, zorder=0)
    ax.axhline(r_mean_all, color=C_BLUE, linestyle=':', linewidth=0.8, zorder=1,
               label=f'Mean r = {r_mean_all:.2f}')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_labels, fontsize=7)
    ax.set_ylabel('Pearson r')
    ax.set_ylim(-1.0, 0.1)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax.set_title('k/N = 0.375, N = 8: all distance metrics', fontsize=9, pad=4)
    panel_label(ax, '(a)')

    fig.tight_layout()
    outpath = FIGDIR / 'fig6_distance_robustness.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ==============================================================================
# Generate all figures
# ==============================================================================

if __name__ == '__main__':
    print('Generating PLC publication figures (PRA v2)...')
    print(f'Output directory: {FIGDIR}')
    print()

    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    make_figure6()

    print()
    print('All 6 figures generated successfully.')
