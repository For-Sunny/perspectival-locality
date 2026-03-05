#!/usr/bin/env python3
"""
Publication-quality figures for PLC (Perspectival Locality Conjecture) paper.
Target: Physical Review Letters format.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os
from pathlib import Path

# ── PRL Style Configuration ──────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
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

# Colorblind-safe palette (Wong 2011 / Okabe-Ito)
C_BLUE   = '#0072B2'
C_ORANGE = '#D55E00'
C_GREEN  = '#009E73'
C_PINK   = '#CC79A7'
C_SKY    = '#56B4E9'
C_YELLOW = '#E69F00'
C_BLACK  = '#000000'
C_GRAY   = '#999999'

# PRL figure widths (inches)
SINGLE_COL = 3.375
DOUBLE_COL = 7.0

# ── Data paths ────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent
RESULTS = BASE / 'results'
FIGDIR = BASE / 'figures'
FIGDIR.mkdir(exist_ok=True)


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


# ── Load all data ─────────────────────────────────────────────────────────────

hardened = load('hardened_stats.json')
exp4     = load('exp4_scaling.json')
ctrl_a   = load('control_A.json')
ctrl_b   = load('control_B.json')
ctrl_c   = load('control_C.json')
obs      = load('observables.json')


def panel_label(ax, label, x=-0.12, y=1.08):
    """Place (a), (b) style panel label."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Emergent Metric (the money figure)
# ═════════════════════════════════════════════════════════════════════════════

def make_figure1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))

    # ── Panel (a): dim_ratio vs k/N with bootstrap CIs ──
    exp2 = hardened['experiment_2_emergent_metric']
    k_over_n = []
    estimates = []
    ci_lo = []
    ci_hi = []
    p_vals = []

    for key in sorted(exp2.keys()):
        d = exp2[key]
        k_over_n.append(d['k_over_N'])
        bs = d['dim_ratio_bootstrap']
        estimates.append(bs['estimate'])
        ci_lo.append(bs['ci_low'])
        ci_hi.append(bs['ci_high'])
        p_vals.append(d['permutation_test']['p_value'])

    k_over_n = np.array(k_over_n)
    estimates = np.array(estimates)
    ci_lo = np.array(ci_lo)
    ci_hi = np.array(ci_hi)
    yerr = np.array([estimates - ci_lo, ci_hi - estimates])

    # Reference line
    ax1.axhline(y=1.0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1,
                label='No effect')

    # Data points with error bars
    ax1.errorbar(k_over_n, estimates, yerr=yerr, fmt='o', color=C_BLUE,
                 capsize=3, capthick=0.8, markersize=6, markeredgecolor='white',
                 markeredgewidth=0.5, zorder=3, label='Observed')

    # p-value annotations
    for x, y_hi, p in zip(k_over_n, ci_hi, p_vals):
        if p < 0.001:
            plabel = r'$p < 0.001$'
        else:
            plabel = r'$p = {:.3f}$'.format(p)
        ax1.annotate(plabel, xy=(x, y_hi + 0.03), fontsize=6.5,
                     ha='center', va='bottom', color=C_BLUE)

    ax1.set_xlabel(r'Observer fraction $k/N$')
    ax1.set_ylabel(r'Effective dimensionality ratio $d_{\mathrm{obs}}/d_{\mathrm{full}}$')
    ax1.set_xlim(0.3, 0.7)
    ax1.set_ylim(0.4, 1.35)
    ax1.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    panel_label(ax1, '(a)')

    # ── Panel (b): Correlation decay for k/N=0.375 ──
    exp5_h = hardened['experiment_5_correlation_decay']['N=8_k=3']
    # Use the raw real r values - pick a trial with strong negative r
    # to show as representative scatter
    # We need the actual distances and correlations for one trial
    # Since hardened_stats only has r values, use a representative approach:
    # Generate a synthetic scatter showing the decay pattern with reported r

    # Better: use the mean Pearson r and show the distribution
    raw_r = np.array(exp5_h['raw_real_r'])
    mean_r = exp5_h['pearson_r_bootstrap']['estimate']
    ci_r_lo = exp5_h['pearson_r_bootstrap']['ci_low']
    ci_r_hi = exp5_h['pearson_r_bootstrap']['ci_high']

    # Show histogram of Pearson r values across trials
    bins = np.linspace(-1, 1, 25)
    ax2.hist(raw_r, bins=bins, color=C_BLUE, alpha=0.7, edgecolor='white',
             linewidth=0.4, density=True, zorder=2)

    # Mark mean and CI
    ax2.axvline(mean_r, color=C_ORANGE, linewidth=1.5, linestyle='-',
                label=r'Mean $r = {:.2f}$'.format(mean_r), zorder=3)
    ax2.axvspan(ci_r_lo, ci_r_hi, alpha=0.2, color=C_ORANGE, zorder=1,
                label='95% CI')
    ax2.axvline(0, color=C_GRAY, linewidth=0.7, linestyle='--', zorder=1)

    ax2.set_xlabel(r'Pearson $r$ (MI distance vs $|C_{ij}|$)')
    ax2.set_ylabel('Density')
    ax2.set_xlim(-1.1, 1.1)
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='none', fontsize=7)
    ax2.set_title(r'$k/N = 0.375$, $N = 8$', fontsize=9, pad=4)
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig1_emergent_metric.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Controls
# ═════════════════════════════════════════════════════════════════════════════

def make_figure2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))

    # Gather comparable k/N=0.5 data across conditions
    # Our system (hardened stats)
    our_dr = hardened['experiment_2_emergent_metric']['N=8_k=4']
    our_dim_est = our_dr['dim_ratio_bootstrap']['estimate']
    our_dim_lo  = our_dr['dim_ratio_bootstrap']['ci_low']
    our_dim_hi  = our_dr['dim_ratio_bootstrap']['ci_high']

    our_r_data = hardened['experiment_5_correlation_decay']['N=8_k=4']
    our_r_est = our_r_data['pearson_r_bootstrap']['estimate']
    our_r_lo  = our_r_data['pearson_r_bootstrap']['ci_low']
    our_r_hi  = our_r_data['pearson_r_bootstrap']['ci_high']

    # 1D chain (control A) at k/N=0.50
    chain_sum = ctrl_a['summary']['0.50']
    chain_dr_mean = chain_sum['mean_dim_ratio']
    chain_dr_std  = chain_sum['std_r_mi_dist']  # use as proxy
    chain_r_mean  = chain_sum['mean_r_mi_dist']
    chain_r_std   = chain_sum['std_r_mi_dist']

    # Compute actual dim_ratio std from raw data
    chain_ratios = [d['dim_ratio'] for d in ctrl_a['all_data']
                    if abs(d['k_over_N'] - 0.5) < 0.01]
    chain_dr_std = np.std(chain_ratios) if chain_ratios else 0.1
    chain_rs = [d['r_pearson_mi_dist'] for d in ctrl_a['all_data']
                if abs(d['k_over_N'] - 0.5) < 0.01]
    chain_r_std = np.std(chain_rs) if chain_rs else 0.1

    # Haar random (control B) at k/N=0.50
    haar_sum = ctrl_b['summary']['0.50']
    haar_dr_mean = haar_sum['mean_dim_ratio']
    haar_r_mean  = haar_sum['mean_r_pearson']
    haar_r_std   = haar_sum['std_r_pearson']
    haar_ratios = [d['dim_ratio'] for d in ctrl_b['all_data']
                   if abs(d['k_over_N'] - 0.5) < 0.01]
    haar_dr_std = np.std(haar_ratios) if haar_ratios else 0.1

    # ── Panel (a): dim_ratio comparison ──
    conditions = ['All-to-all\n(PLC)', '1D chain\n(local $H$)', 'Haar random\n(no structure)']
    means = [our_dim_est, chain_dr_mean, haar_dr_mean]
    errs_lo = [our_dim_est - our_dim_lo, chain_dr_std, haar_dr_std]
    errs_hi = [our_dim_hi - our_dim_est, chain_dr_std, haar_dr_std]
    colors = [C_BLUE, C_GREEN, C_ORANGE]

    x_pos = np.arange(len(conditions))
    bars = ax1.bar(x_pos, means, width=0.55, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.8, zorder=2)
    ax1.errorbar(x_pos, means, yerr=[errs_lo, errs_hi], fmt='none',
                 ecolor=C_BLACK, capsize=4, capthick=0.8, zorder=3)

    ax1.axhline(1.0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(conditions, fontsize=8)
    ax1.set_ylabel(r'Dim. ratio $d_{\mathrm{obs}}/d_{\mathrm{full}}$')
    ax1.set_ylim(0, 1.4)
    ax1.set_title(r'$k/N = 0.5$, $N = 8$', fontsize=9, pad=4)
    panel_label(ax1, '(a)')

    # ── Panel (b): Pearson r comparison ──
    r_means = [our_r_est, chain_r_mean, haar_r_mean]
    r_errs_lo = [abs(our_r_est - our_r_lo), chain_r_std, haar_r_std]
    r_errs_hi = [abs(our_r_hi - our_r_est), chain_r_std, haar_r_std]

    bars2 = ax2.bar(x_pos, r_means, width=0.55, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=0.8, zorder=2)
    ax2.errorbar(x_pos, r_means, yerr=[r_errs_lo, r_errs_hi], fmt='none',
                 ecolor=C_BLACK, capsize=4, capthick=0.8, zorder=3)

    ax2.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conditions, fontsize=8)
    ax2.set_ylabel(r'Pearson $r$ (MI dist. vs $|C_{ij}|$)')
    ax2.set_ylim(-1.2, 0.6)
    ax2.set_title(r'Correlation decay at $k/N = 0.5$', fontsize=9, pad=4)
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig2_controls.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Scaling and Monotonicity
# ═════════════════════════════════════════════════════════════════════════════

def make_figure3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))

    # ── Panel (a): dim_ratio vs k/N from exp4_scaling, per N ──
    scaling = exp4['all_results']
    n_values = sorted(set(d['N'] for d in scaling))
    markers = ['o', 's', 'D']
    colors_n = [C_BLUE, C_ORANGE, C_GREEN]

    for i_n, N in enumerate(n_values):
        sub = [d for d in scaling if d['N'] == N]
        # Group by k_over_N and average
        k_vals = sorted(set(d['k_over_N'] for d in sub))
        mean_ratios = []
        std_ratios = []
        for kv in k_vals:
            ratios = [d['dim_ratio'] for d in sub if d['k_over_N'] == kv]
            mean_ratios.append(np.mean(ratios))
            std_ratios.append(np.std(ratios) / np.sqrt(len(ratios)))

        ax1.errorbar(k_vals, mean_ratios, yerr=std_ratios,
                     fmt=markers[i_n] + '-', color=colors_n[i_n],
                     capsize=2, capthick=0.6, markersize=4,
                     markeredgecolor='white', markeredgewidth=0.4,
                     label=r'$N = {}$'.format(N), zorder=3)

    ax1.axhline(1.0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)
    ax1.set_xlabel(r'Observer fraction $k/N$')
    ax1.set_ylabel(r'Dim. ratio $d_{\mathrm{obs}}/d_{\mathrm{full}}$')
    ax1.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax1.set_ylim(0.4, 1.3)
    panel_label(ax1, '(a)')

    # ── Panel (b): Pearson r vs k/N from hardened_stats ──
    exp5_h = hardened['experiment_5_correlation_decay']
    k_vals_r = []
    r_ests = []
    r_ci_lo = []
    r_ci_hi = []

    for key in sorted(exp5_h.keys()):
        d = exp5_h[key]
        k_vals_r.append(d['k_over_N'])
        bs = d['pearson_r_bootstrap']
        r_ests.append(bs['estimate'])
        r_ci_lo.append(bs['ci_low'])
        r_ci_hi.append(bs['ci_high'])

    k_vals_r = np.array(k_vals_r)
    r_ests = np.array(r_ests)
    r_ci_lo = np.array(r_ci_lo)
    r_ci_hi = np.array(r_ci_hi)

    # Shaded CI band
    ax2.fill_between(k_vals_r, r_ci_lo, r_ci_hi, alpha=0.25, color=C_BLUE,
                     zorder=1, label='95% CI')
    ax2.plot(k_vals_r, r_ests, 'o-', color=C_BLUE, markersize=5,
             markeredgecolor='white', markeredgewidth=0.5, zorder=3,
             label=r'Mean Pearson $r$')

    ax2.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.7, zorder=1)
    ax2.set_xlabel(r'Observer fraction $k/N$')
    ax2.set_ylabel(r'Pearson $r$ (MI dist. vs $|C_{ij}|$)')
    ax2.set_ylim(-0.8, 0.1)
    ax2.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax2.set_title(r'$N = 8$, hardened statistics', fontsize=9, pad=4)
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig3_scaling.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Additional Observables (supplemental)
# ═════════════════════════════════════════════════════════════════════════════

def make_figure4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))

    # Use N=8 analysis
    cmi_sum = obs['N8_analysis']['cmi_summary']
    spec_sum = obs['N8_analysis']['spectrum_summary']

    # ── Panel (a): CMI coefficient of variation vs k/N ──
    k_vals_cmi = sorted(cmi_sum.keys(), key=float)
    k_floats = [float(k) for k in k_vals_cmi]
    cv_means = [cmi_sum[k]['mean_cv'] for k in k_vals_cmi]
    cv_stds  = [cmi_sum[k]['std_cv'] for k in k_vals_cmi]

    ax1.errorbar(k_floats, cv_means, yerr=cv_stds, fmt='s-', color=C_BLUE,
                 capsize=3, capthick=0.8, markersize=5,
                 markeredgecolor='white', markeredgewidth=0.5, zorder=3)

    # Shade region of increasing structure
    ax1.fill_between(k_floats, [m - s for m, s in zip(cv_means, cv_stds)],
                     [m + s for m, s in zip(cv_means, cv_stds)],
                     alpha=0.15, color=C_BLUE, zorder=1)

    ax1.set_xlabel(r'Observer fraction $k/N$')
    ax1.set_ylabel(r'CMI coefficient of variation')
    ax1.set_title(r'Emergent structure ($N = 8$)', fontsize=9, pad=4)
    ax1.set_ylim(-0.3, 2.5)

    # Annotate trend
    ax1.annotate(r'More observation $\rightarrow$ more structure',
                 xy=(0.7, 1.1), fontsize=7, color=C_BLUE, ha='center',
                 style='italic')
    panel_label(ax1, '(a)')

    # ── Panel (b): Level spacing ratio vs k/N ──
    k_vals_sp = sorted(spec_sum.keys(), key=float)
    k_floats_sp = [float(k) for k in k_vals_sp]
    lsr_means = [spec_sum[k]['mean_level_spacing_ratio'] for k in k_vals_sp]

    # Reference lines for GOE and Poisson
    goe_val = 0.5307  # GOE level spacing ratio (Wigner surmise)
    poisson_val = 0.3863  # Poisson level spacing ratio

    ax2.axhline(goe_val, color=C_ORANGE, linestyle=':', linewidth=0.8,
                label=r'GOE ($\approx 0.531$)', zorder=1)
    ax2.axhline(poisson_val, color=C_GREEN, linestyle=':', linewidth=0.8,
                label=r'Poisson ($\approx 0.386$)', zorder=1)

    ax2.plot(k_floats_sp, lsr_means, 's-', color=C_BLUE, markersize=5,
             markeredgecolor='white', markeredgewidth=0.5, zorder=3,
             label='Observed')

    # PR normalized on secondary axis for context
    pr_means = [spec_sum[k]['mean_PR_normalized'] for k in k_vals_sp]

    ax2.set_xlabel(r'Observer fraction $k/N$')
    ax2.set_ylabel(r'Level spacing ratio $\langle r \rangle$')
    ax2.set_title(r'Spectral statistics ($N = 8$)', fontsize=9, pad=4)
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax2.set_ylim(0, 0.65)

    # Annotate sub-Poisson regime
    ax2.annotate('Sub-Poisson\n(localized)',
                 xy=(0.85, 0.15), fontsize=7, color=C_BLUE, ha='center',
                 style='italic')
    panel_label(ax2, '(b)')

    fig.tight_layout(w_pad=2.5)
    outpath = FIGDIR / 'fig4_observables.pdf'
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ═════════════════════════════════════════════════════════════════════════════
# Generate all figures
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating PLC publication figures...')
    print(f'Output directory: {FIGDIR}')
    print()

    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()

    print()
    print('All figures generated successfully.')
