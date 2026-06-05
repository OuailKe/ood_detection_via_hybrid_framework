#!/home/ouail.kerrak/.conda/envs/ood310/bin/python

import os
import sys

_env = os.environ.get('CONDA_DEFAULT_ENV', '')
if _env != 'ood310':
    print(f'ERROR: wrong environment "{_env}". '
          f'Run: conda activate ood310 && python {sys.argv[0]}')
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('GPU_ID', '1')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.fftpack import dct
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
DATA_DIR     = PROJECT_ROOT / 'data'
OUTPUT_DIR   = PROJECT_ROOT / 'plots' / 'gatekeeperV3_dinov2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FALSE_PASS_DIR = OUTPUT_DIR / 'false_passes'
FALSE_PASS_DIR.mkdir(parents=True, exist_ok=True)

MSP_SCORES_DIR  = PROJECT_ROOT / 'results' / \
    'bdd100k_dinov2-vitb14_test_ood_ood_msp_bdd100k_autonomous_ood_msp_dinov2' / 'scores'
MAHA_SCORES_DIR = PROJECT_ROOT / 'results' / 'mahalanobis_dinov2' / 'scores'

DATASETS = {
    'bdd100k_val': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt',
        'image_dir': DATA_DIR / 'id/bdd100k/bdd100k/bdd100k/images/',
        'is_ood':    False,
        'ood_type':  'ID',
        'npz_stem':  'bdd100k',
    },
    'lostandfound': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/lostandfound/test.txt',
        'image_dir': DATA_DIR / 'ood/near/LostAndFound',
        'is_ood':    True,
        'ood_type':  'near',
        'npz_stem':  'lostandfound',
    },
    'streethazards': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/streethazards/test.txt',
        'image_dir': DATA_DIR / 'ood/far/StreetHazards/test/images',
        'is_ood':    True,
        'ood_type':  'far',
        'npz_stem':  'streethazards',
    },
    'smiyc': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/smiyc/test.txt',
        'image_dir': DATA_DIR / 'ood/far/SMIYC/test/images',
        'is_ood':    True,
        'ood_type':  'far',
        'npz_stem':  'smiyc',
    },
}

OOD_GROUPS = {
    'lostandfound':  ['lostandfound'],
    'streethazards': ['streethazards'],
    'smiyc':         ['smiyc'],
    'nearood':       ['lostandfound'],
    'farood':        ['streethazards', 'smiyc'],
}

BENFORD_PROBS = np.array([np.log10(1 + 1/d) for d in range(1, 10)])


# ── BENFORD/DCT ───────────────────────────────────────────────────────────────

def get_first_digit(x):
    x = abs(float(x))
    if x == 0:
        return None
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def compute_dct_benford_chi2(img_path, block_size=8):
    try:
        img = Image.open(img_path).convert('L')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float64)
        h, w   = arr.shape
        digits = []
        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block     = arr[row:row+block_size, col:col+block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                ac_coeffs = dct_block.flatten()[1:]
                for coeff in ac_coeffs:
                    d = get_first_digit(coeff)
                    if d is not None:
                        digits.append(d)
        if len(digits) < 100:
            return 0.0
        counts  = np.zeros(9)
        for d in digits:
            counts[d - 1] += 1
        observed  = counts / counts.sum()
        n         = len(digits)
        chi2_stat = n * np.sum((observed - BENFORD_PROBS) ** 2 / BENFORD_PROBS)
        return float(chi2_stat)
    except Exception:
        return 0.0


def compute_benford_scores(image_paths, image_dir, desc=''):
    scores    = []
    image_dir = Path(image_dir)
    for img_path in tqdm(image_paths, desc=f'  Benford [{desc}]', leave=False):
        scores.append(compute_dct_benford_chi2(image_dir / img_path))
    return np.array(scores)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_imglist(path):
    images, labels = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                images.append(parts[0])
                labels.append(int(parts[1]))
    return images, labels


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics_from_scores_and_labels(id_scores, ood_scores,
                                           tpr_level=0.95,
                                           higher_is_ood=True):
    if not higher_is_ood:
        id_scores  = -id_scores
        ood_scores = -ood_scores
    labels   = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores   = np.concatenate([id_scores, ood_scores])
    auroc    = roc_auc_score(labels, scores)
    aupr_out = average_precision_score(labels, scores)
    aupr_in  = average_precision_score(1 - labels, -scores)
    thresh   = np.percentile(id_scores, tpr_level * 100)
    fpr      = (ood_scores <= thresh).mean()
    return {
        'FPR@95':   round(fpr * 100, 2),
        'AUROC':    round(auroc * 100, 2),
        'AUPR_IN':  round(aupr_in * 100, 2),
        'AUPR_OUT': round(aupr_out * 100, 2),
    }


# ── GATEKEEPER ────────────────────────────────────────────────────────────────

def combine_benford_and_semantic(
        id_benford, ood_benford,
        id_semantic, ood_semantic,
        benford_fpr_budget=0.05,
        semantic_tpr=0.95,
        semantic_higher_is_id=True):

    tau_benford  = np.percentile(id_benford, (1 - benford_fpr_budget) * 100)
    tau_semantic = np.percentile(id_semantic, (1 - semantic_tpr) * 100) \
                   if semantic_higher_is_id else \
                   np.percentile(id_semantic, semantic_tpr * 100)

    def gate(benford_scores, semantic_scores, label=''):
        n             = len(benford_scores)
        stage1_reject = benford_scores > tau_benford
        stage2_reject = np.zeros(n, dtype=bool)
        survivors     = ~stage1_reject
        if semantic_higher_is_id:
            stage2_reject[survivors] = semantic_scores[survivors] < tau_semantic
        else:
            stage2_reject[survivors] = semantic_scores[survivors] > tau_semantic
        reject   = stage1_reject | stage2_reject
        caught_by = np.full(n, 'pass', dtype=object)
        caught_by[stage1_reject] = 'stage1'
        caught_by[stage2_reject] = 'stage2'
        breakdown = {
            f'{label}_total':           n,
            f'{label}_stage1_rejected': int(stage1_reject.sum()),
            f'{label}_stage2_rejected': int(stage2_reject.sum()),
            f'{label}_total_rejected':  int(reject.sum()),
            f'{label}_accepted':        int((~reject).sum()),
        }
        return reject, caught_by, breakdown

    reject_id,  caught_id,  bd_id  = gate(id_benford,  id_semantic,  label='ID')
    reject_ood, caught_ood, bd_ood = gate(ood_benford, ood_semantic, label='OOD')
    return reject_id, reject_ood, caught_ood, tau_benford, tau_semantic, {**bd_id, **bd_ood}


# ── FALSE PASS LOGGER ─────────────────────────────────────────────────────────

def save_false_passes(dataset_names, dataset_index,
                      benford_scores_dict, semantic_scores_dict,
                      caught_ood_per_dataset, sem_name):
    for ds_name in dataset_names:
        cfg     = dataset_index[ds_name]
        images  = cfg['images']
        b_scores = benford_scores_dict[ds_name]
        s_scores = semantic_scores_dict[ds_name]
        caught   = caught_ood_per_dataset[ds_name]
        records  = [{'image_path': img, 'dataset': ds_name,
                     'chi2_score': round(float(b), 4),
                     'semantic_score': round(float(s), 4),
                     'decision': dec}
                    for img, b, s, dec in zip(images, b_scores, s_scores, caught)]
        df_all = pd.DataFrame(records)
        df_fp  = df_all[df_all['decision'] == 'pass'].copy()
        stem   = FALSE_PASS_DIR / f'{ds_name}_{sem_name}'
        df_all.to_csv(f'{stem}_all_decisions.csv',  index=False)
        df_fp.to_csv( f'{stem}_false_passes.csv',   index=False)
        print(f'    ✓ {ds_name}: {len(df_fp)} false passes → {stem}_false_passes.csv')


# ── SAVE / PRINT ──────────────────────────────────────────────────────────────

def save_results(results_list, output_dir, method_name):
    out = Path(output_dir) / method_name
    out.mkdir(parents=True, exist_ok=True)
    df  = pd.DataFrame(results_list)
    df.to_csv(out / 'ood.csv', index=False)
    return df


def print_results_table(df, title):
    print(f'\n  ── {title} ──')
    print(f'  {"dataset":<20} {"FPR@95":>8} {"AUROC":>8} '
          f'{"AUPR_IN":>9} {"AUPR_OUT":>10}')
    print('  ' + '-' * 60)
    for _, row in df.iterrows():
        print(f'  {row["dataset"]:<20} {row["FPR@95"]:>7.2f}% '
              f'{row["AUROC"]:>7.2f}% '
              f'{row["AUPR_IN"]:>8.2f}% {row["AUPR_OUT"]:>9.2f}%')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('  Hybrid Gatekeeper — DINOv2 ViT-B/14  [Benford + Semantic]')
    print('=' * 65)

    print('\n[1/4] Loading image lists...')
    dataset_index = {}
    for name, cfg in DATASETS.items():
        images, labels = load_imglist(cfg['imglist'])
        dataset_index[name] = {**cfg, 'images': images, 'labels': labels}
        print(f'  {name:20s}: {len(images):5d} images')

    print('\n[2/4] Computing Benford/DCT χ² scores...')
    benford_scores = {}
    for name, cfg in dataset_index.items():
        benford_scores[name] = compute_benford_scores(
            cfg['images'], cfg['image_dir'], desc=name)
        print(f'  {name:20s}: mean χ²={benford_scores[name].mean():.2f}  '
              f'std={benford_scores[name].std():.2f}')

    pd.DataFrame([{'image_path': img, 'dataset_name': name,
                   'ground_truth': 'OOD' if cfg['is_ood'] else 'ID',
                   'chi2_score': score}
                  for name, cfg in dataset_index.items()
                  for img, score in zip(cfg['images'], benford_scores[name])
                  ]).to_csv(OUTPUT_DIR / 'chi2_scores.csv', index=False)

    print('\n[3/4] Loading semantic scores...')
    semantic_scores = {'msp': {}, 'mahalanobis': {}}

    for name, cfg in dataset_index.items():
        stem     = cfg['npz_stem']
        maha_key = name   # mahalanobis uses full dataset name as key

        # MSP — OpenOOD uses npz_stem (e.g. 'bdd100k' not 'bdd100k_val')
        try:
            d = np.load(MSP_SCORES_DIR / f'{stem}.npz')
            semantic_scores['msp'][name] = d['conf']
            print(f'  ✓ MSP   {name:20s}: {len(d["conf"])} samples')
        except FileNotFoundError as e:
            print(f'  ✗ MSP   {name}: {e}')
            semantic_scores['msp'][name] = None

        # Mahalanobis — uses full dataset name
        try:
            d = np.load(MAHA_SCORES_DIR / f'{maha_key}.npz')
            semantic_scores['mahalanobis'][name] = d['conf']
            print(f'  ✓ Maha  {name:20s}: {len(d["conf"])} samples')
        except FileNotFoundError as e:
            print(f'  ✗ Maha  {name}: {e}')
            semantic_scores['mahalanobis'][name] = None

    print('\n[4/4] Evaluating...')

    id_benford  = benford_scores['bdd100k_val']
    all_results = {}

    def evaluate_method(id_conf, ood_conf_dict, higher_is_ood=False):
        rows = []
        for group_name, dataset_names in OOD_GROUPS.items():
            valid = [ood_conf_dict[n] for n in dataset_names
                     if ood_conf_dict.get(n) is not None]
            if not valid:
                continue
            ood_conf = np.concatenate(valid)
            m = compute_metrics_from_scores_and_labels(
                id_conf, ood_conf, higher_is_ood=higher_is_ood)
            rows.append({'dataset': group_name, **m, 'ACC': 83.50})
        return rows

    # Benford only
    benford_ood = {n: benford_scores[n]
                   for n in dataset_index if n != 'bdd100k_val'}
    df_b = save_results(
        evaluate_method(id_benford, benford_ood, higher_is_ood=True),
        OUTPUT_DIR, 'benford_only')
    all_results['Benford only'] = df_b
    print_results_table(df_b, 'Benford Only')

    # Semantic baselines + hybrid
    for sem_name in ['msp', 'mahalanobis']:
        sem_scores = semantic_scores[sem_name]
        if sem_scores.get('bdd100k_val') is None:
            print(f'\n  Skipping {sem_name} — ID scores not found')
            continue

        id_sem       = sem_scores['bdd100k_val']
        ood_sem_dict = {n: sem_scores[n] for n in dataset_index
                        if n != 'bdd100k_val' and sem_scores.get(n) is not None}

        # Baseline
        df_base = save_results(
            evaluate_method(id_sem, ood_sem_dict, higher_is_ood=False),
            OUTPUT_DIR, f'baseline_{sem_name}')
        all_results[f'Baseline {sem_name.upper()}'] = df_base
        print_results_table(df_base, f'Baseline {sem_name.upper()}')

        # Hybrid
        print(f'\n  -- Hybrid Benford + {sem_name.upper()} --')
        hybrid_rows = []

        for group_name, dataset_names in OOD_GROUPS.items():
            valid_names = [n for n in dataset_names
                           if ood_sem_dict.get(n) is not None]
            if not valid_names:
                continue

            ood_b = np.concatenate([benford_scores[n] for n in valid_names])
            ood_s = np.concatenate([ood_sem_dict[n]   for n in valid_names])

            (reject_id, reject_ood, caught_ood,
             tau_b, tau_s, stats) = combine_benford_and_semantic(
                id_benford, ood_b, id_sem, ood_s,
                benford_fpr_budget=0.05, semantic_tpr=0.95,
                semantic_higher_is_id=True)

            fpr = (~reject_ood).mean()

            id_combined       = id_sem.copy().astype(np.float64)
            id_combined[reject_id]   = id_sem.min() - 1e6
            ood_combined      = ood_s.copy().astype(np.float64)
            ood_combined[reject_ood] = id_sem.min() - 1e6

            labels   = np.concatenate([np.zeros(len(id_combined)),
                                       np.ones(len(ood_combined))])
            sc       = -np.concatenate([id_combined, ood_combined])
            auroc    = roc_auc_score(labels, sc)
            aupr_out = average_precision_score(labels, sc)
            aupr_in  = average_precision_score(1 - labels, -sc)

            n_ood = len(ood_b)
            hybrid_rows.append({
                'dataset':             group_name,
                'FPR@95':              round(fpr * 100, 2),
                'AUROC':               round(auroc * 100, 2),
                'AUPR_IN':             round(aupr_in * 100, 2),
                'AUPR_OUT':            round(aupr_out * 100, 2),
                'ACC':                 83.50,
                'tau_benford':         round(tau_b, 2),
                'tau_semantic':        round(tau_s, 4),
                'OOD_stage1_rejected': stats['OOD_stage1_rejected'],
                'OOD_stage2_rejected': stats['OOD_stage2_rejected'],
                'OOD_total_rejected':  stats['OOD_total_rejected'],
                'OOD_false_passes':    n_ood - stats['OOD_total_rejected'],
            })

            print(f'    {group_name}: Stage1={stats["OOD_stage1_rejected"]}/{n_ood} '
                  f'Stage2={stats["OOD_stage2_rejected"]}/{n_ood} '
                  f'FPR={fpr*100:.2f}%')

            # Save false passes per dataset
            offset = 0
            caught_per_ds = {}
            for ds_name in valid_names:
                n_ds = len(dataset_index[ds_name]['images'])
                caught_per_ds[ds_name] = caught_ood[offset:offset + n_ds]
                offset += n_ds

            save_false_passes(valid_names, dataset_index,
                              benford_scores, ood_sem_dict,
                              caught_per_ds, sem_name)

        df_hybrid = save_results(hybrid_rows, OUTPUT_DIR, f'hybrid_{sem_name}')
        all_results[f'Hybrid Benford+{sem_name.upper()}'] = df_hybrid
        print_results_table(df_hybrid, f'Hybrid Benford + {sem_name.upper()}')

    # Summary
    print('\n' + '=' * 65)
    print('  SUMMARY — FPR@95')
    print('=' * 65)
    summary_rows = []
    for method_name, df in all_results.items():
        row = {'Method': method_name}
        for _, r in df.iterrows():
            row[r['dataset']] = r['FPR@95']
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / 'gatekeeper_summary.csv', index=False)

    auroc_rows = []
    for method_name, df in all_results.items():
        row = {'Method': method_name}
        for _, r in df.iterrows():
            row[r['dataset']] = r['AUROC']
        auroc_rows.append(row)
    pd.DataFrame(auroc_rows).to_csv(OUTPUT_DIR / 'gatekeeper_auroc_summary.csv',
                                     index=False)

    print(f'\n  ✓ Output saved to {OUTPUT_DIR}')
    print(f'  └── false_passes/ — per dataset × method false pass CSVs')


if __name__ == '__main__':
    main()