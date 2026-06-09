#!/home/ouail.kerrak/.conda/envs/ood310/bin/python

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

_env = os.environ.get('CONDA_DEFAULT_ENV', '')
if _env != 'ood310':
    print(f'ERROR: wrong environment "{_env}". '
          f'Run: conda activate ood310 && python {sys.argv[0]}')
    sys.exit(1)

PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
RESULT_DIR    = PROJECT_ROOT / 'plots' / 'gatekeeperV3_dinov2'
OUTPUT_DIR    = RESULT_DIR / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FPR95_CSV = RESULT_DIR / 'gatekeeper_summary.csv'
SUMMARY_AUROC_CSV = RESULT_DIR / 'gatekeeper_auroc_summary.csv'
CHI2_CSV         = RESULT_DIR / 'chi2_scores.csv'
FALSE_PASS_DIR   = RESULT_DIR / 'false_passes'

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
    'Benford only':              '#a0c878',
    'Baseline MAHALANOBIS':      '#4c9be8',
    'Hybrid Benford+MAHALANOBIS': '#7b5ea7',
}

DATASETS = ['LostAndFound', 'StreetHazards', 'SMIYC']
CSV_KEYS = ['lostandfound', 'streethazards', 'smiyc']


def load_summary(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f'Summary CSV not found: {csv_path}')
    df = pd.read_csv(csv_path)
    df['Method'] = df['Method'].astype(str)
    return df.set_index('Method')


def plot_bar_metric(df, metric_name, ylabel, title, filename, clip_at=None,
                    hline=None, hline_label=None):
    methods = list(df.index)
    n_methods = len(methods)
    x = np.arange(len(DATASETS))
    width = 0.18
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2,
                          n_methods) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        values = [df.loc[method, key] for key in CSV_KEYS]
        bars = ax.bar(x + offsets[i], values, width,
                      label=method, color=COLORS.get(method, None),
                      edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if clip_at is None or h > clip_at:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                        f'{h:.2f}', ha='center', va='bottom',
                        fontsize=8, color='#333333')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_xlabel('OOD Dataset')
    if hline is not None:
        ax.axhline(y=hline, color='red', linestyle=':', linewidth=0.8,
                   alpha=0.7, label=hline_label)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    _save(fig, filename)
    print(f'  ✓ {filename} saved')


def load_stage_stats():
    dataset_stats = {}
    for dataset in CSV_KEYS:
        path = FALSE_PASS_DIR / f'{dataset}_mahalanobis_all_decisions.csv'
        if not path.exists():
            raise FileNotFoundError(f'Data file missing: {path}')
        df = pd.read_csv(path)
        counts = df['decision'].value_counts().to_dict()
        dataset_stats[dataset] = {
            'Stage 1': int(counts.get('stage1', 0)),
            'Stage 2': int(counts.get('stage2', 0)),
            'False passes': int(counts.get('pass', 0)),
            'Total': len(df),
        }
    return dataset_stats


def plot_stage_breakdown(dataset_stats):
    datasets = [d.title() if d != 'smiyc' else 'SMIYC' for d in CSV_KEYS]
    stage1 = [dataset_stats[d]['Stage 1'] for d in CSV_KEYS]
    stage2 = [dataset_stats[d]['Stage 2'] for d in CSV_KEYS]
    fp = [dataset_stats[d]['False passes'] for d in CSV_KEYS]
    totals = [dataset_stats[d]['Total'] for d in CSV_KEYS]

    s1_pct = [100 * s / t for s, t in zip(stage1, totals)]
    s2_pct = [100 * s / t for s, t in zip(stage2, totals)]
    fp_pct = [100 * s / t for s, t in zip(fp, totals)]

    x = np.arange(len(datasets))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, s1_pct, width, label='Rejected by Stage 1 (Benford)',
           color=COLORS['Benford only'], edgecolor='white')
    ax.bar(x, s2_pct, width, bottom=s1_pct,
           label='Rejected by Stage 2 (Mahalanobis)',
           color=COLORS['Baseline MAHALANOBIS'], edgecolor='white')
    ax.bar(x, fp_pct, width,
           bottom=[a + b for a, b in zip(s1_pct, s2_pct)],
           label='False passes (missed)',
           color='#e07b54', edgecolor='white')

    for i in range(len(datasets)):
        ax.text(i, s1_pct[i] / 2, f'{stage1[i]}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        ax.text(i, s1_pct[i] + s2_pct[i] / 2, f'{stage2[i]}',
                ha='center', va='center', fontsize=9, color='white',
                fontweight='bold')
        ax.text(i, s1_pct[i] + s2_pct[i] + fp_pct[i] / 2,
                f'{fp[i]}', ha='center', va='center', fontsize=9,
                color='white', fontweight='bold')

    ax.set_ylabel('Percentage of OOD samples (%)')
    ax.set_title('Hybrid Gatekeeper V3 (DINOv2) — Per-Stage OOD Rejection')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 110)
    ax.set_xlabel('OOD Dataset')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    _save(fig, 'dinov2_stage_breakdown')
    print('  ✓ dinov2_stage_breakdown saved')


def plot_chi2_distributions():
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
        if len(scores) == 0:
            continue
        scores_clipped = scores[scores < 300]
        if len(scores_clipped) < 10:
            continue
        kde = gaussian_kde(scores_clipped, bw_method=0.15)
        y = kde(x)
        ax.plot(x, y, label=label,
                color=colors[label], linestyle=linestyles[label], linewidth=2.0)
        ax.fill_between(x, y, alpha=0.08, color=colors[label])

    id_scores = datasets_plot['BDD100K (ID)']
    if len(id_scores) > 0:
        tau = np.percentile(id_scores, 95)
        ax.axvline(x=tau, color='red', linestyle='--', linewidth=1.5,
                   label=f'τ₁ = {tau:.1f} (95th percentile ID)')
        ax.axvspan(tau, 300, alpha=0.04, color='red', label='Rejection region')

    ax.set_xlabel('Benford χ² score')
    ax.set_ylabel('Density')
    ax.set_title('Benford χ² Distributions — ID vs OOD (DINOv2)')
    ax.set_xlim(0, 300)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.9, loc='upper right')
    plt.tight_layout()
    _save(fig, 'dinov2_chi2_distributions')
    print('  ✓ dinov2_chi2_distributions saved')


def _save(fig, name):
    fig.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{name}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def main():
    print('Generating DINOv2 gatekeeper plots...')
    fpr95_df = load_summary(SUMMARY_FPR95_CSV)
    auroc_df = load_summary(SUMMARY_AUROC_CSV)

    plot_bar_metric(fpr95_df, 'FPR@95', 'FPR@95 (%)',
                    'DINOv2 FPR@95 Comparison', 'dinov2_fpr95_comparison',
                    clip_at=5, hline=100, hline_label='100%')
    plot_bar_metric(auroc_df, 'AUROC', 'AUROC (%)',
                    'AUROC (%)', 'dinov2_auroc_comparison',
                    clip_at=50, hline=50, hline_label='Random chance')

    dataset_stats = load_stage_stats()
    plot_stage_breakdown(dataset_stats)
    plot_chi2_distributions()

    print(f'\n✓ All DINOv2 plots saved to {OUTPUT_DIR}')
    print('  ├── dinov2_fpr95_comparison.pdf/.png')
    print('  ├── dinov2_auroc_comparison.pdf/.png')
    print('  ├── dinov2_stage_breakdown.pdf/.png')
    print('  └── dinov2_chi2_distributions.pdf/.png')


if __name__ == '__main__':
    main()
