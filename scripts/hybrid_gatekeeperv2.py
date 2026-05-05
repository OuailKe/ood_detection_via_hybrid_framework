#!/home/ouail.kerrak/.conda/envs/ood310/bin/python


import os
import sys

# ── Environment guard ─────────────────────────────────────────────────────────
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
from scipy.stats import chi2_contingency
from scipy.fftpack import dct
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
DATA_DIR     = PROJECT_ROOT / 'data'
OUTPUT_DIR   = PROJECT_ROOT / 'plots' / 'gatekeeperV2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Where the existing semantic score .npz files 

MSP_SCORES_DIR   = PROJECT_ROOT / 'results' / \
    'bdd100k_vit-b-16_base_e30_lr0.001_bdd100k_13cls_ft_ood_ood_msp_bdd100k_13cls_ft' / 'scores'
MAHA_SCORES_DIR  = PROJECT_ROOT / 'plots' / 'mahalanobis' / 'scores'

DATASETS = {
    'bdd100k_val': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt',
        'image_dir': DATA_DIR / 'id/bdd100k/bdd100k/bdd100k/images/',
        'is_ood':    False,
        'ood_type':  'ID',
        'npz_stem':  'bdd100k',   #  used in OpenOOD .npz filename
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

BENFORD_PROBS = np.array([
    np.log10(1 + 1/d) for d in range(1, 10)
])  



# STAGE 1 — BENFORD/DCT  FILTER


def get_first_digit(x):
    """Return first significant digit (1-9) of absolute value."""
    x = abs(float(x))
    if x == 0:
        return None
    
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def compute_dct_benford_chi2(img_path, block_size=8):
    """
      1. Convert to grayscale
      2. Divide into non-overlapping 8×8 blocks
      3. Apply 2D DCT to each block
      4. Collect all AC coefficients (exclude DC = position [0,0])
      5. Extract first significant digit of each non-zero coefficient
      6. Compare digit frequency distribution to Benford's Law via χ²
    """
    try:
        img = Image.open(img_path).convert('L')  # grayscale
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float64)

        h, w     = arr.shape
        digits   = []

        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block    = arr[row:row+block_size, col:col+block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

                # AC coefficients only — skip DC at [0, 0]
                ac_coeffs = dct_block.flatten()[1:]

                for coeff in ac_coeffs:
                    d = get_first_digit(coeff)
                    if d is not None:
                        digits.append(d)

        if len(digits) < 100:
            # Too few coef
            return 0.0

        # Observed digit frequencies
        counts   = np.zeros(9)
        for d in digits:
            counts[d - 1] += 1

        observed  = counts / counts.sum()
        expected  = BENFORD_PROBS

        # χ² statistic: sum((O - E)² / E) scaled by N
        n         = len(digits)
        chi2_stat = n * np.sum((observed - expected) ** 2 / expected)

        return float(chi2_stat)

    except Exception as e:
        return 0.0


def compute_benford_scores(image_paths, image_dir, desc=''):
    
    scores = []
    image_dir = Path(image_dir)
    for img_path in tqdm(image_paths, desc=f'  Benford [{desc}]', leave=False):
        full_path = image_dir / img_path
        score     = compute_dct_benford_chi2(full_path)
        scores.append(score)
    return np.array(scores)





def load_imglist(path):
    images, labels = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                images.append(parts[0])
                labels.append(int(parts[1]))
    return images, labels


# LOAD EXISTING SEMANTIC SCORES FROM .NPZ FILES

def load_semantic_scores(scores_dir, stem):
    
    path = Path(scores_dir) / f'{stem}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Score file not found: {path}')
    data = np.load(path)
    return data['conf']   # shape [N]


# METRICS

def compute_metrics(id_conf, ood_conf, tpr_level=0.95):
    """
    Standard OOD metrics.
    Convention: higher conf = more ID-like.
    OOD samples have lower conf → label=1 when we negate.
    """
    # Negate so that higher score = more OOD = label 1
    id_score  = -id_conf
    ood_score = -ood_conf

    labels = np.concatenate([np.zeros(len(id_score)),
                              np.ones(len(ood_score))])
    scores = np.concatenate([id_score, ood_score])

    auroc    = roc_auc_score(labels, scores)
    aupr_out = average_precision_score(labels, scores)
    aupr_in  = average_precision_score(1 - labels, -scores)

    # FPR@95: threshold = 5th percentile of ID scores (after negation)
    thresh = np.percentile(id_conf, (1 - tpr_level) * 100)   
    fpr    = (ood_conf <= thresh).mean()                      

    return {
        'FPR@95':   round(fpr * 100, 2),
        'AUROC':    round(auroc * 100, 2),
        'AUPR_IN':  round(aupr_in * 100, 2),
        'AUPR_OUT': round(aupr_out * 100, 2),
    }


def compute_metrics_from_scores_and_labels(id_scores, ood_scores,
                                           tpr_level=0.95,
                                           higher_is_ood=True):
    """
    higher_is_ood=True  → use scores directly (Benford χ², Mahalanobis distance)
    higher_is_ood=False → negate first (MSP, Energy conf where higher=ID)
    """
    if not higher_is_ood:
        id_scores  = -id_scores
        ood_scores = -ood_scores

    labels = np.concatenate([np.zeros(len(id_scores)),
                              np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc    = roc_auc_score(labels, scores)
    aupr_out = average_precision_score(labels, scores)
    aupr_in  = average_precision_score(1 - labels, -scores)

    thresh = np.percentile(id_scores, tpr_level * 100)
    fpr    = (ood_scores <= thresh).mean()

    return {
        'FPR@95':   round(fpr * 100, 2),
        'AUROC':    round(auroc * 100, 2),
        'AUPR_IN':  round(aupr_in * 100, 2),
        'AUPR_OUT': round(aupr_out * 100, 2),
    }

# GATEKEEPER COMBINATION

def combine_benford_and_semantic(
        id_benford,   ood_benford,
        id_semantic,  ood_semantic,
        benford_fpr_budget=0.05,
        semantic_tpr=0.95,
        semantic_higher_is_id=True):
   
    # ── Calibrate thresholds on ID set ───────────────────────────────────────
    tau_benford  = np.percentile(id_benford,
                                 (1 - benford_fpr_budget) * 100)

    if semantic_higher_is_id:
        tau_semantic = np.percentile(id_semantic, (1 - semantic_tpr) * 100)
    else:
        tau_semantic = np.percentile(id_semantic, semantic_tpr * 100)

    def gate(benford_scores, semantic_scores, label=''):
        """
        True early-exit: Stage 2 is only applied to samples that pass Stage 1.
        Returns reject array and per-stage breakdown.
        """
        n = len(benford_scores)

        # Stage 1 — statistical filter
        stage1_reject = benford_scores > tau_benford      # [N] bool

        # Stage 2 — semantic filter (only on Stage 1 survivors)
        stage2_reject = np.zeros(n, dtype=bool)
        survivors     = ~stage1_reject                    # passed Stage 1

        if semantic_higher_is_id:
            stage2_reject[survivors] = (
                semantic_scores[survivors] < tau_semantic)
        else:
            stage2_reject[survivors] = (
                semantic_scores[survivors] > tau_semantic)

        reject = stage1_reject | stage2_reject

        breakdown = {
            f'{label}_total':          n,
            f'{label}_stage1_rejected': int(stage1_reject.sum()),
            f'{label}_stage2_rejected': int(stage2_reject.sum()),
            f'{label}_total_rejected':  int(reject.sum()),
            f'{label}_accepted':        int((~reject).sum()),
        }
        return reject, breakdown

    reject_id,  bd_id  = gate(id_benford,  id_semantic,  label='ID')
    reject_ood, bd_ood = gate(ood_benford, ood_semantic, label='OOD')

    stats = {**bd_id, **bd_ood}

    return reject_id, reject_ood, tau_benford, tau_semantic, stats


# SAVE RESULTS

def save_results(results_list, output_dir, method_name):
    out = Path(output_dir) / method_name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_list)
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



def main():
    print('=' * 65)
    print('  Hybrid Statistical-Semantic Gatekeeper')
    print('=' * 65)

    # Load image lists
    print('\n[1/4] Loading image lists...')
    dataset_index = {}
    for name, cfg in DATASETS.items():
        images, labels = load_imglist(cfg['imglist'])
        dataset_index[name] = {**cfg, 'images': images, 'labels': labels}
        print(f'  {name:20s}: {len(images):5d} images')

    #  Compute Benford χ²  
    print('\n[2/4] Computing Benford/DCT χ² scores...')
    benford_scores = {}
    for name, cfg in dataset_index.items():
        benford_scores[name] = compute_benford_scores(
            cfg['images'], cfg['image_dir'], desc=name)
        print(f'  {name:20s}: mean χ²={benford_scores[name].mean():.2f}  '
              f'std={benford_scores[name].std():.2f}')

    # Save per-sample χ² 
    chi2_records = []
    for name, cfg in dataset_index.items():
        for img, score in zip(cfg['images'], benford_scores[name]):
            chi2_records.append({
                'image_path':   img,
                'dataset_name': name,
                'ground_truth': 'OOD' if cfg['is_ood'] else 'ID',
                'chi2_score':   score,
            })
    chi2_df = pd.DataFrame(chi2_records)
    chi2_df.to_csv(OUTPUT_DIR / 'chi2_scores.csv', index=False)
    print(f'\n  ✓ χ² scores saved to {OUTPUT_DIR}/chi2_scores.csv')

    # Load existing semantic scores 
    print('\n[3/4] Loading semantic scores from .npz files...')

    # MSP and Energy come from OpenOOD results
    # Mahalanobis comes from our custom script
    semantic_scores = {'msp': {}, 'energy': {}, 'mahalanobis': {}}

    for name, cfg in dataset_index.items():
        stem = cfg['npz_stem']
        try:
            # MSP scores
            msp_data = np.load(MSP_SCORES_DIR / f'{stem}.npz')
            semantic_scores['msp'][name] = msp_data['conf']
            print(f'  ✓ MSP   {name:20s}: {len(msp_data["conf"])} samples')
        except FileNotFoundError as e:
            print(f'  ✗ MSP   {name}: {e}')
            semantic_scores['msp'][name] = None

        try:
            # Mahalanobis scores (our custom output)
            maha_stem = name  # our script uses full dataset name
            maha_data = np.load(MAHA_SCORES_DIR / f'{maha_stem}.npz')
            semantic_scores['mahalanobis'][name] = maha_data['conf']
            print(f'  ✓ Maha  {name:20s}: {len(maha_data["conf"])} samples')
        except FileNotFoundError as e:
            print(f'  ✗ Maha  {name}: {e}')
            semantic_scores['mahalanobis'][name] = None

    # Evaluate all methods and combinations 
    print('\n[4/4] Evaluating all methods...')

    id_benford = benford_scores['bdd100k_val']
    all_results = {}

    # Helper to evaluate over all OOD groups
    def evaluate_method(id_conf, ood_conf_dict, method_label, higher_is_ood=False):
        rows = []
        for group_name, dataset_names in OOD_GROUPS.items():
            valid = [ood_conf_dict[n] for n in dataset_names
                     if ood_conf_dict.get(n) is not None]
            if not valid:
                continue
            ood_conf = np.concatenate(valid)
            m = compute_metrics_from_scores_and_labels(
                id_conf, ood_conf, higher_is_ood=higher_is_ood)
            rows.append({'dataset': group_name, **m, 'ACC': 82.16})
        return rows

    # Benford only 
    benford_ood_dict = {n: benford_scores[n]
                        for n in dataset_index if n != 'bdd100k_val'}
    rows = evaluate_method(id_benford, benford_ood_dict,
                           'benford_only', higher_is_ood=True)
    df_benford = save_results(rows, OUTPUT_DIR, 'benford_only')
    all_results['Benford only'] = df_benford
    print_results_table(df_benford, 'Benford Only')

    #  Semantic baselines + Hybrid combinations 
    for sem_name in ['msp', 'mahalanobis']:
        sem_scores = semantic_scores[sem_name]
        if sem_scores.get('bdd100k_val') is None:
            print(f'\n  Skipping {sem_name} — scores not found')
            continue

        id_sem = sem_scores['bdd100k_val']
        ood_sem_dict = {n: sem_scores[n]
                        for n in dataset_index
                        if n != 'bdd100k_val' and sem_scores.get(n) is not None}

        # Baseline semantic only
        rows = evaluate_method(id_sem, ood_sem_dict,
                               sem_name, higher_is_ood=False)
        label = f'baseline_{sem_name}'
        df_base = save_results(rows, OUTPUT_DIR, label)
        all_results[f'Baseline {sem_name.upper()}'] = df_base
        print_results_table(df_base, f'Baseline {sem_name.upper()}')

        # Hybrid
        hybrid_rows = []
        for group_name, dataset_names in OOD_GROUPS.items():
            valid_names = [n for n in dataset_names
                           if ood_sem_dict.get(n) is not None]
            if not valid_names:
                continue

            ood_b = np.concatenate([benford_scores[n] for n in valid_names])
            ood_s = np.concatenate([ood_sem_dict[n]   for n in valid_names])

            reject_id, reject_ood, tau_b, tau_s, stats = \
                combine_benford_and_semantic(
                    id_benford, ood_b,
                    id_sem,     ood_s,
                    benford_fpr_budget=0.05,
                    semantic_tpr=0.95,
                    semantic_higher_is_id=True,
                )

            # FPR@95
            fpr = (~reject_ood).mean()

            # AUROC/AUPR
            id_combined  = id_sem.copy().astype(np.float64)
            id_combined[reject_id]  = id_sem.min() - 1e6

            ood_combined = ood_s.copy().astype(np.float64)
            ood_combined[reject_ood] = id_sem.min() - 1e6

            labels = np.concatenate([np.zeros(len(id_combined)),
                                      np.ones(len(ood_combined))])
            sc     = -np.concatenate([id_combined, ood_combined])

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
                'ACC':                 82.16,
                'tau_benford':         round(tau_b, 2),
                'tau_semantic':        round(tau_s, 4),
                'OOD_stage1_rejected': stats['OOD_stage1_rejected'],
                'OOD_stage2_rejected': stats['OOD_stage2_rejected'],
                'OOD_total_rejected':  stats['OOD_total_rejected'],
                'OOD_false_passes':    n_ood - stats['OOD_total_rejected'],
            })

            # Print per-stage breakdown
            print(f'    {group_name}: Stage1 caught '
                  f'{stats["OOD_stage1_rejected"]}/{n_ood} '
                  f'({100*stats["OOD_stage1_rejected"]/n_ood:.1f}%)  '
                  f'Stage2 caught {stats["OOD_stage2_rejected"]}/{n_ood} '
                  f'({100*stats["OOD_stage2_rejected"]/n_ood:.1f}%)  '
                  f'FPR={fpr*100:.2f}%')

        label = f'hybrid_{sem_name}'
        df_hybrid = save_results(hybrid_rows, OUTPUT_DIR, label)
        all_results[f'Hybrid Benford+{sem_name.upper()}'] = df_hybrid
        print_results_table(df_hybrid,
                            f'Hybrid Benford + {sem_name.upper()}')

    print('\n' + '=' * 65)
    print('  SUMMARY — FPR@95 comparison')
    print('=' * 65)

    # Pivot to show FPR@95 per dataset for each method
    summary_rows = []
    for method_name, df in all_results.items():
        row = {'Method': method_name}
        for _, r in df.iterrows():
            row[r['dataset']] = r['FPR@95']
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / 'gatekeeper_summary.csv', index=False)

    # AUROC summary
    auroc_rows = []
    for method_name, df in all_results.items():
        row = {'Method': method_name}
        for _, r in df.iterrows():
            row[r['dataset']] = r['AUROC']
        auroc_rows.append(row)
    auroc_df = pd.DataFrame(auroc_rows)
    auroc_df.to_csv(OUTPUT_DIR / 'gatekeeper_auroc_summary.csv', index=False)

    print(f'\n  ✓ Output saved to {OUTPUT_DIR}')
    print(f'  ├── benford_only/ood.csv')
    print(f'  ├── baseline_msp/ood.csv')
    print(f'  ├── baseline_mahalanobis/ood.csv')
    print(f'  ├── hybrid_msp/ood.csv')
    print(f'  ├── hybrid_mahalanobis/ood.csv')
    print(f'  ├── chi2_scores.csv')
    print(f'  ├── gatekeeper_summary.csv      ← FPR@95 comparison table')
    print(f'  └── gatekeeper_auroc_summary.csv ← AUROC comparison table')


if __name__ == '__main__':
    main()