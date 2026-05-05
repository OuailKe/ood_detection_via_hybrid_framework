#!/home/ouail.kerrak/.conda/envs/ood310/bin/python


import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_env = os.environ.get('CONDA_DEFAULT_ENV', '')
if _env != 'ood310':
    print(f'ERROR: wrong environment "{_env}". '
          f'Run: conda activate ood310 && python {sys.argv[0]}')
    sys.exit(1)

PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
OUTPUT_DIR   = PROJECT_ROOT / 'plots' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHI2_CSV     = PROJECT_ROOT / 'plots' / 'gatekeeperV2' / 'chi2_scores.csv'

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'figure.dpi':       150,
    'axes.grid':        True,
    'grid.linestyle':   '--',
    'grid.alpha':       0.4,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

COLORS = {
    'MSP':         '#e07b54',
    'Energy':      '#f5c542',
    'Mahalanobis': '#4c9be8',
    'Benford':     '#a0c878',
    'Hybrid':      '#7b5ea7',
}

DATASETS = ['LostAndFound', 'StreetHazards', 'SMIYC']

FPR95 = {
    'MSP':         [75.60, 40.25, 52.91],
    'Energy':      [43.85, 19.82, 11.80],
    'Mahalanobis': [36.58,  2.80,  0.58],
    'Benford':     [99.67, 68.40, 61.40],
    'Hybrid':      [36.24,  1.73,  0.58],
}

AUROC = {
    'MSP':         [59.48, 84.10, 77.18],
    'Energy':      [88.75, 94.57, 97.05],
    'Mahalanobis': [94.35, 99.04, 99.73],
    'Benford':     [45.03, 72.60, 73.90],
    'Hybrid':      [90.70, 94.98, 95.07],
}

# Per-stage breakdown
STAGE_DATA = {
    'LostAndFound':  {'Stage 1': 4,   'Stage 2': 763,  'False passes': 436},
    'StreetHazards': {'Stage 1': 474, 'Stage 2': 1000, 'False passes': 26},
    'SMIYC':         {'Stage 1': 132, 'Stage 2': 208,  'False passes': 2},
}
DATASET_TOTALS = {
    'LostAndFound': 1203,
    'StreetHazards': 1500,
    'SMIYC': 1500,
}



def plot_fpr95():
    methods = list(FPR95.keys())
    n_methods  = len(methods)
    n_datasets = len(DATASETS)

    x     = np.arange(n_datasets)
    width = 0.15
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        bars = ax.bar(x + offsets[i], FPR95[method], width,
                      label=method, color=COLORS[method],
                      edgecolor='white', linewidth=0.5)
        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            if h > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                        f'{h:.1f}', ha='center', va='bottom',
                        fontsize=7.5, color='#333333')

    ax.set_ylabel('FPR@95 (%)')
    ax.set_title('FPR@95 Comparison Across OOD Datasets and Detection Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlabel('OOD Dataset')

    plt.tight_layout()
    _save(fig, 'fpr95_comparison')
    print('  ✓ FPR@95 comparison saved')


def plot_auroc():
    methods    = list(AUROC.keys())
    n_methods  = len(methods)
    n_datasets = len(DATASETS)

    x       = np.arange(n_datasets)
    width   = 0.15
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        bars = ax.bar(x + offsets[i], AUROC[method], width,
                      label=method, color=COLORS[method],
                      edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 50:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f'{h:.1f}', ha='center', va='bottom',
                        fontsize=7.5, color='#333333')

    ax.set_ylabel('AUROC (%)')
    ax.set_title('AUROC Comparison Across OOD Datasets and Detection Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color='red', linestyle=':', linewidth=0.8,
               alpha=0.5, label='Random chance')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlabel('OOD Dataset')

    plt.tight_layout()
    _save(fig, 'auroc_comparison')
    print('  ✓ AUROC comparison saved')



def plot_stage_breakdown():
    datasets = list(STAGE_DATA.keys())
    stage1   = [STAGE_DATA[d]['Stage 1']      for d in datasets]
    stage2   = [STAGE_DATA[d]['Stage 2']      for d in datasets]
    fp       = [STAGE_DATA[d]['False passes'] for d in datasets]
    totals   = [DATASET_TOTALS[d]             for d in datasets]

    # Convert to percentages
    s1_pct = [100 * s / t for s, t in zip(stage1, totals)]
    s2_pct = [100 * s / t for s, t in zip(stage2, totals)]
    fp_pct = [100 * s / t for s, t in zip(fp,     totals)]

    x     = np.arange(len(datasets))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x, s1_pct, width, label='Rejected by Stage 1 (Benford)',
                color='#a0c878', edgecolor='white')
    b2 = ax.bar(x, s2_pct, width, bottom=s1_pct,
                label='Rejected by Stage 2 (Mahalanobis)',
                color='#4c9be8', edgecolor='white')
    b3 = ax.bar(x, fp_pct, width,
                bottom=[s1 + s2 for s1, s2 in zip(s1_pct, s2_pct)],
                label='False passes (missed)',
                color='#e07b54', edgecolor='white')

    # Annotate with raw counts
    for i, d in enumerate(datasets):
        ax.text(i, s1_pct[i] / 2,
                f'{stage1[i]}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        ax.text(i, s1_pct[i] + s2_pct[i] / 2,
                f'{stage2[i]}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        ax.text(i, s1_pct[i] + s2_pct[i] + fp_pct[i] / 2,
                f'{fp[i]}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    ax.set_ylabel('Percentage of OOD samples (%)')
    ax.set_title('Hybrid Gatekeeper — Per-Stage OOD Rejection Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlabel('OOD Dataset')

    plt.tight_layout()
    _save(fig, 'stage_breakdown')
    print('  ✓ Stage breakdown saved')


def plot_chi2_distributions():
    from scipy.stats import gaussian_kde

    if not CHI2_CSV.exists():
        print(f'  ✗ chi2_scores.csv not found at {CHI2_CSV} — skipping')
        return

    df = pd.read_csv(CHI2_CSV)

    datasets_plot = {
        'BDD100K (ID)':            df[df['ground_truth'] == 'ID']['chi2_score'].values,
        'LostAndFound (near-OOD)': df[df['dataset_name'] == 'lostandfound']['chi2_score'].values,
        'StreetHazards (far-OOD)': df[df['dataset_name'] == 'streethazards']['chi2_score'].values,
        'SMIYC (far-OOD)':         df[df['dataset_name'] == 'smiyc']['chi2_score'].values,
    }
    colors = {
        'BDD100K (ID)':            '#4c9be8',
        'LostAndFound (near-OOD)': '#e07b54',
        'StreetHazards (far-OOD)': '#a0c878',
        'SMIYC (far-OOD)':         '#7b5ea7',
    }
    linestyles = {
        'BDD100K (ID)':            '-',
        'LostAndFound (near-OOD)': '--',
        'StreetHazards (far-OOD)': '-.',
        'SMIYC (far-OOD)':         ':',
    }

    x = np.linspace(0, 300, 500)

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, scores in datasets_plot.items():
        # Clip extreme outliers for KDE fitting
        scores_clipped = scores[scores < 300]
        kde = gaussian_kde(scores_clipped, bw_method=0.15)
        y   = kde(x)
        ax.plot(x, y, label=label,
                color=colors[label],
                linestyle=linestyles[label],
                linewidth=2.0)
        ax.fill_between(x, y, alpha=0.08, color=colors[label])

    # Threshold line
    id_scores = datasets_plot['BDD100K (ID)']
    tau = np.percentile(id_scores, 95)
    ax.axvline(x=tau, color='red', linestyle='--', linewidth=1.5,
               label=f'$\\tau_1$ = {tau:.1f} (95th percentile ID)')

    # Shade rejected region
    ax.axvspan(tau, 300, alpha=0.04, color='red', label='Rejection region')

    ax.set_xlabel('Benford χ² score')
    ax.set_ylabel('Density')
    ax.set_title('Benford χ² Score Distributions — ID vs OOD Datasets')
    ax.set_xlim(0, 300)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.9, loc='upper right')

    plt.tight_layout()
    _save(fig, 'chi2_distributions')
    print('  ✓ χ² distributions saved')


# ── Save helper ───────────────────────────────────────────────────────────────
def _save(fig, name):
    fig.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{name}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Generating result plots...')
    plot_fpr95()
    plot_auroc()
    plot_stage_breakdown()
    plot_chi2_distributions()
    print(f'\n✓ All plots saved to {OUTPUT_DIR}')
    print('  ├── fpr95_comparison.pdf/.png')
    print('  ├── auroc_comparison.pdf/.png')
    print('  ├── stage_breakdown.pdf/.png')
    print('  └── chi2_distributions.pdf/.png')


if __name__ == '__main__':
    main()