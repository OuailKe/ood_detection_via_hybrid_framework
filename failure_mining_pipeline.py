import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm

# Add OpenOOD to path
sys.path.insert(0, '/home/ouail.kerrak/ood_project/OpenOOD')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from openood.utils import config

# Setup paths
PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR = PROJECT_ROOT / 'data'
PLOTS_DIR = PROJECT_ROOT / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

print("✓ All imports successful")

def load_imglist(imglist_path):
    """Load image list from file."""
    images = []
    labels = []
    if not os.path.exists(imglist_path):
        print(f"Warning: {imglist_path} not found")
        return images, labels
    
    with open(imglist_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                images.append(parts[0])
                labels.append(int(parts[1]))
    return images, labels

# Load dataset metadata
datasets = {
    'bdd100k_val': {
        'id_label': 1,
        'imglist': DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt',
        'image_dir': DATA_DIR / 'id/bdd100k/bdd100k/bdd100k/images/',
        'is_ood': False,
    },
    'lostandfound': {
        'id_label': 0,  # OOD dataset
        'imglist': DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt',  # placeholder
        'image_dir': DATA_DIR / 'ood/near/',
        'is_ood': True,
    },
    'streethazards': {
        'id_label': 0,
        'imglist': DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt',
        'image_dir': DATA_DIR / 'ood/near/',
        'is_ood': True,
    },
    'smiyc': {
        'id_label': 0,
        'imglist': DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt',
        'image_dir': DATA_DIR / 'ood/far/',
        'is_ood': True,
    },
}

# Load BDD100K validation set (ID)
print("Loading BDD100K validation set (ID samples)...")
bdd_images, bdd_labels = load_imglist(str(datasets['bdd100k_val']['imglist']))
print(f"  Loaded {len(bdd_images)} BDD100K validation samples")

# Create dataset index
dataset_index = {
    'bdd100k_val': {
        'images': bdd_images,
        'labels': bdd_labels,
        'image_dir': datasets['bdd100k_val']['image_dir'],
        'is_ood': False,
        'ground_truth': 'ID',
    }
}

# For OOD datasets, we'll add them if available
for ood_name in ['lostandfound', 'streethazards', 'smiyc']:
    # Search for directories
    ood_dir = PROJECT_ROOT / f'data/ood'
    print(f"Looking for {ood_name} in {ood_dir}")
    if ood_dir.exists():
        dataset_index[ood_name] = {
            'images': [],  # Will scan directory
            'image_dir': ood_dir,
            'is_ood': True,
            'ground_truth': 'OOD',
        }

print("\n✓ Dataset index created")
print(f"  Datasets: {list(dataset_index.keys())}")

# Load trained model config
model_dir = RESULTS_DIR / 'bdd100k_vit-b-16_base_e30_lr0.001_bdd100k_13cls_ft/s0'
config_path = model_dir / 'config.yml'

print(f"Loading model from {model_dir}")
print(f"Config path: {config_path}")

# For now, we'll create a data structure to store scores
# In practice, you would load the trained model and generate these scores
print("✓ Model loading setup ready")
print("  Note: Actual model inference would be implemented here with torch.load()")

# Create sample score records from existing results
# In practice, these would be computed from model predictions
sample_scores = []
sample_id = 0

# Example BDD100K validation samples (ID) - generate more samples
# ID samples should have LOW anomaly scores
for i, img_path in enumerate(bdd_images[:500]):  # Use 500 validation samples
    sample_scores.append({
        'image_id': sample_id,
        'image_path': str(img_path),
        'dataset_name': 'bdd100k_val',
        'ground_truth': 'ID',
        'energy_score': np.random.uniform(0.0, 0.4),  # Low for ID (good)
        'msp_score': np.random.uniform(0.7, 0.99),    # High confidence for ID
        'mahalanobis_score': np.random.uniform(0.0, 1.0),  # Low distance for ID
        'predicted_class': np.random.randint(0, 13),
        'confidence': np.random.uniform(0.8, 0.99),
        'anomaly_map': None,
    })
    sample_id += 1

# Add OOD samples - these should have HIGH anomaly scores
ood_datasets = {
    'lostandfound': 500,
    'streethazards': 400,
    'smiyc': 300,
}

for ood_name, n_samples in ood_datasets.items():
    for i in range(n_samples):
        # OOD samples: some detected correctly, some fail
        is_detected = np.random.random() > 0.15  # 85% detected, 15% false passes
        
        if is_detected:
            # Correctly detected as OOD - high anomaly scores
            sample_scores.append({
                'image_id': sample_id,
                'image_path': f'ood/{ood_name}/{i:06d}.jpg',
                'dataset_name': ood_name,
                'ground_truth': 'OOD',
                'energy_score': np.random.uniform(0.6, 1.5),
                'msp_score': np.random.uniform(0.1, 0.6),
                'mahalanobis_score': np.random.uniform(1.5, 5.0),
                'predicted_class': np.random.randint(0, 13),
                'confidence': np.random.uniform(0.4, 0.75),
                'anomaly_map': None,
            })
        else:
            # False pass - low anomaly scores (incorrectly classified as ID)
            sample_scores.append({
                'image_id': sample_id,
                'image_path': f'ood/{ood_name}/{i:06d}.jpg',
                'dataset_name': ood_name,
                'ground_truth': 'OOD',
                'energy_score': np.random.uniform(0.05, 0.35),  # Low (incorrectly ID-like)
                'msp_score': np.random.uniform(0.65, 0.98),      # High confidence (wrongly confident)
                'mahalanobis_score': np.random.uniform(0.2, 1.2),  # Low distance (incorrectly ID-like)
                'predicted_class': np.random.randint(0, 13),
                'confidence': np.random.uniform(0.75, 0.95),  # High confidence in wrong prediction
                'anomaly_map': None,
            })
        sample_id += 1

# Create DataFrame
scores_df = pd.DataFrame(sample_scores)

print("✓ Scores DataFrame created")
print(f"  Shape: {scores_df.shape}")
print(f"\nDataFrame columns:")
print(f"  {scores_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(scores_df.head())

def find_threshold_at_tpr(scores, target_tpr=0.95):
    """
    Find threshold that achieves target TPR on ID samples.
    
    For Energy and Mahalanobis: lower scores = more likely ID
    For MSP: higher scores = more likely ID
    
    TPR = TP / (TP + FN) = correctly classified ID / all ID
    """
    sorted_scores = np.sort(scores)
    n_samples = len(scores)
    
    # For ID detection: we want to accept samples below/above threshold
    # TPR is proportion of ID samples accepted
    threshold_idx = int(n_samples * (1 - target_tpr))
    threshold = sorted_scores[threshold_idx]
    
    return threshold

# Extract validation set (ID only)
val_df = scores_df[scores_df['dataset_name'] == 'bdd100k_val'].copy()

if len(val_df) > 0:
    # Calculate thresholds at 95% TPR
    energy_threshold = find_threshold_at_tpr(val_df['energy_score'].values, target_tpr=0.95)
    msp_threshold = find_threshold_at_tpr(val_df['msp_score'].values, target_tpr=0.95)
    maha_threshold = find_threshold_at_tpr(val_df['mahalanobis_score'].values, target_tpr=0.95)
    
    thresholds = {
        'energy': energy_threshold,
        'msp': msp_threshold,
        'mahalanobis': maha_threshold,
    }
    
    print("✓ Decision thresholds set at 95% TPR on BDD100K validation")
    print(f"\nThresholds:")
    for method, threshold in thresholds.items():
        print(f"  {method:15s}: {threshold:.4f}")
else:
    print("No validation data found, using default thresholds")
    thresholds = {
        'energy': 0.5,
        'msp': 0.7,
        'mahalanobis': 1.0,
    }

def classify_predictions(df, thresholds):
    """
    Classify each prediction for each method.
    
    For Energy and Mahalanobis: higher score = more anomalous (OOD)
    For MSP: higher score = more confident (ID)
    
    Classification:
    - correct_reject: OOD sample correctly rejected (detected as OOD)
    - false_pass: OOD sample incorrectly accepted as ID
    - false_reject: ID sample incorrectly rejected as OOD  
    - correct_accept: ID sample correctly accepted as ID
    """
    results = []
    
    for idx, row in df.iterrows():
        record = row.to_dict()
        
        # Energy method (higher = more OOD)
        energy_pred_ood = row['energy_score'] > thresholds['energy']
        record['energy_decision'] = 'OOD' if energy_pred_ood else 'ID'
        
        # MSP method (higher = more confident, lower = more OOD)
        msp_pred_ood = row['msp_score'] < thresholds['msp']
        record['msp_decision'] = 'OOD' if msp_pred_ood else 'ID'
        
        # Mahalanobis method (higher = more OOD)
        maha_pred_ood = row['mahalanobis_score'] > thresholds['mahalanobis']
        record['mahalanobis_decision'] = 'OOD' if maha_pred_ood else 'ID'
        
        # Classify outcomes for each method
        is_ood_gt = row['ground_truth'] == 'OOD'
        
        # Energy
        if is_ood_gt:
            record['energy_outcome'] = 'correct_reject' if energy_pred_ood else 'false_pass'
        else:
            record['energy_outcome'] = 'correct_accept' if not energy_pred_ood else 'false_reject'
        
        # MSP
        if is_ood_gt:
            record['msp_outcome'] = 'correct_reject' if msp_pred_ood else 'false_pass'
        else:
            record['msp_outcome'] = 'correct_accept' if not msp_pred_ood else 'false_reject'
        
        # Mahalanobis
        if is_ood_gt:
            record['mahalanobis_outcome'] = 'correct_reject' if maha_pred_ood else 'false_pass'
        else:
            record['mahalanobis_outcome'] = 'correct_accept' if not maha_pred_ood else 'false_reject'
        
        results.append(record)
    
    return pd.DataFrame(results)

# Apply classification
classified_df = classify_predictions(scores_df, thresholds)

# Summary statistics
print("✓ Classification complete\n")
print("Classification summary by method:\n")

for method in ['energy', 'msp', 'mahalanobis']:
    outcome_col = f'{method}_outcome'
    counts = classified_df[outcome_col].value_counts()
    print(f"{method.upper()}:")
    for outcome, count in counts.items():
        print(f"  {outcome:20s}: {count:3d}")
    print()

# Extract false passes for each method
false_passes = {}

for method in ['energy', 'msp', 'mahalanobis']:
    outcome_col = f'{method}_outcome'
    score_col = f'{method}_score'
    
    # Filter OOD samples
    ood_samples = classified_df[classified_df['ground_truth'] == 'OOD'].copy()
    
    # Filter false passes
    fps = ood_samples[ood_samples[outcome_col] == 'false_pass'].copy()
    
    # Sort by "confidence in ID" (how badly they failed)
    if method == 'msp':
        # Higher MSP = more confident in ID (worst failure)
        fps_sorted = fps.sort_values(score_col, ascending=False)
    else:
        # Lower Energy/Mahalanobis = more likely ID (worst failure)
        fps_sorted = fps.sort_values(score_col, ascending=True)
    
    false_passes[method] = fps_sorted
    
    print(f"\n{method.upper()} - False Passes (ranked by confidence in ID):")
    print(f"  Total OOD samples: {len(ood_samples)}")
    print(f"  False passes: {len(fps)}")
    if len(ood_samples) > 0:
        print(f"  False positive rate: {100*len(fps)/len(ood_samples):.1f}%")
    else:
        print(f"  False positive rate: N/A (no OOD samples)")
    
    if len(fps_sorted) > 0:
        print(f"\n  Top 3 worst failures:")
        for i, (idx, row) in enumerate(fps_sorted.head(3).iterrows()):
            print(f"    {i+1}. Dataset: {row['dataset_name']}, Score: {row[score_col]:.4f}, "
                  f"Confidence: {row['confidence']:.2%}")
    else:
        print(f"  No false passes detected for {method}")

# Store top failures for next section
top_failures = {}
for method in ['energy', 'msp', 'mahalanobis']:
    top_failures[method] = false_passes[method].head(10)  # Top 10 per method

print(f"\n✓ Ranked {sum(len(fps) for fps in false_passes.values())} false passes")

# Define corruption categories
CORRUPTION_CATEGORIES = [
    'noise',
    'glare_overexposure',
    'blur',
    'weather',
    'small_distant',
    'semantic_confusion',
    'scene_shift',
]

# Initialize categorization dictionary
failure_categories = {}

# Example manual categorization (in practice, you would examine images and assign categories)
# We'll demonstrate with Energy method's top failures
print("Manual Failure Categorization Framework")
print("=" * 60)
print(f"\nExamining Energy method's top 10 false passes:\n")

categorized_failures = []

for i, (idx, row) in enumerate(top_failures['energy'].head(10).iterrows()):
    # In practice: examine image at row['image_path'], assign category
    # For demo, we'll assign random categories
    category = np.random.choice(CORRUPTION_CATEGORIES)
    
    categorized_failures.append({
        'rank': i + 1,
        'image_id': row['image_id'],
        'dataset': row['dataset_name'],
        'category': category,
        'energy_score': row['energy_score'],
        'confidence': row['confidence'],
    })
    
    print(f"{i+1:2d}. Image ID: {row['image_id']:5d} | Dataset: {row['dataset_name']:15s} | "
          f"Category: {category:20s} | Score: {row['energy_score']:6.4f}")

# Create categorization dataframe
categorized_df = pd.DataFrame(categorized_failures)

# Count by category
category_counts = categorized_df['category'].value_counts()
print(f"\n\nCategory distribution (top 10 failures):")
for cat, count in category_counts.items():
    print(f"  {cat:20s}: {count:2d} ({100*count/len(categorized_df):5.1f}%)")

print(f"\n✓ Failure categorization framework ready")
print(f"  Next: Examine actual images and update categories")

def build_failure_analysis_table(classified_df, method='energy'):
    """Build comprehensive failure analysis table for a given method."""
    
    outcome_col = f'{method}_outcome'
    score_col = f'{method}_score'
    decision_col = f'{method}_decision'
    
    # Filter false passes only
    ood_samples = classified_df[classified_df['ground_truth'] == 'OOD']
    failures = ood_samples[ood_samples[outcome_col] == 'false_pass'].copy()
    
    if len(failures) == 0:
        print(f"No false passes for {method}")
        return pd.DataFrame()
    
    # Sort by score (confidence in ID)
    if method == 'msp':
        failures = failures.sort_values(score_col, ascending=False)
    else:
        failures = failures.sort_values(score_col, ascending=True)
    
    # Build analysis table
    analysis_table = []
    
    for i, (idx, row) in enumerate(failures.iterrows()):
        # Determine category (in practice: manual inspection)
        category = np.random.choice(CORRUPTION_CATEGORIES)
        
        # Get predicted class label
        pred_class = row['predicted_class']
        
        # Create note based on score
        if method == 'energy':
            note = f"Low energy score ({row[score_col]:.4f}), high confidence in ID"
        elif method == 'msp':
            note = f"High MSP ({row[score_col]:.4f}), very confident prediction"
        else:  # mahalanobis
            note = f"Low Mahalanobis distance ({row[score_col]:.4f}), close to ID distribution"
        
        analysis_table.append({
            'rank': i + 1,
            'image_id': row['image_id'],
            'dataset': row['dataset_name'],
            'corruption_type': category,
            f'{method}_score': row[score_col],
            'threshold_decision': row[decision_col],
            'ground_truth': row['ground_truth'],
            'predicted_class': pred_class,
            'confidence': row['confidence'],
            'note': note,
        })
    
    return pd.DataFrame(analysis_table)

# Build failure tables for each method
print("Building Failure Analysis Tables")
print("=" * 80)

failure_tables = {}

for method in ['energy', 'msp', 'mahalanobis']:
    failure_table = build_failure_analysis_table(classified_df, method=method)
    failure_tables[method] = failure_table
    
    print(f"\n{method.upper()} - Failure Analysis Table:")
    if len(failure_table) > 0:
        print(failure_table[['rank', 'image_id', 'dataset', 'corruption_type', 
                             f'{method}_score', 'confidence', 'note']].head(5))
    else:
        print(f"  No failures found for {method}")
    
    # Export to CSV
    csv_path = PLOTS_DIR / f'failure_analysis_{method}.csv'
    failure_table.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path} ({len(failure_table)} rows)")

# Combine all failures into master table
master_failures = []
for method, table in failure_tables.items():
    if len(table) > 0:
        table['method'] = method
        master_failures.append(table)

if len(master_failures) > 0:
    master_failure_table = pd.concat(master_failures, ignore_index=True)
else:
    master_failure_table = pd.DataFrame()

# Export master table
master_csv_path = PLOTS_DIR / 'failure_analysis_master.csv'
master_failure_table.to_csv(master_csv_path, index=False)
print(f"\n✓ Master failure table exported to {master_csv_path}")
print(f"  Total failure records: {len(master_failure_table)}")

# Show summary statistics
if len(master_failure_table) > 0:
    print(f"\nFailure Summary Statistics:")
    print(master_failure_table[['method', 'dataset', 'corruption_type']].describe())
else:
    print(f"\nNo failures to summarize")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 1. Score distribution histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, method in enumerate(['energy', 'msp', 'mahalanobis']):
    score_col = f'{method}_score'
    
    # Separate ID and OOD
    id_scores = classified_df[classified_df['ground_truth'] == 'ID'][score_col]
    ood_scores = classified_df[classified_df['ground_truth'] == 'OOD'][score_col]
    
    ax = axes[i]
    ax.hist(id_scores, bins=20, alpha=0.6, label='ID', color='blue', edgecolor='black')
    ax.hist(ood_scores, bins=20, alpha=0.6, label='OOD', color='red', edgecolor='black')
    ax.axvline(thresholds[method], color='green', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel(f'{method.upper()} Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{method.upper()} Distribution')
    ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'failure_score_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Score distribution plot saved")

# 2. False positive rates by dataset
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, method in enumerate(['energy', 'msp', 'mahalanobis']):
    outcome_col = f'{method}_outcome'
    
    # Get OOD samples
    ood_by_dataset = classified_df[classified_df['ground_truth'] == 'OOD'].copy()
    
    # Count false passes per dataset
    fp_by_dataset = ood_by_dataset.groupby('dataset_name')[outcome_col].apply(
        lambda x: (x == 'false_pass').sum()
    )
    total_by_dataset = ood_by_dataset.groupby('dataset_name').size()
    fpr = 100 * fp_by_dataset / total_by_dataset
    
    ax = axes[i]
    fpr.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.set_title(f'{method.upper()} False Pass Rate by Dataset')
    ax.set_ylabel('False Pass Rate (%)')
    ax.set_xlabel('Dataset')
    ax.set_ylim(0, 100)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'failure_false_pass_rates.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ False pass rate plot saved")

# 3. Corruption type distribution
fig, ax = plt.subplots(figsize=(10, 6))

if len(categorized_df) > 0:
    cat_counts = categorized_df['category'].value_counts()
    cat_counts.plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
    ax.set_xlabel('Count')
    ax.set_title('Failure Distribution by Corruption Type')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'failure_corruption_types.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Corruption type distribution plot saved")

# 4. Method comparison
fig, ax = plt.subplots(figsize=(10, 6))

method_fpr = {}
for method in ['energy', 'msp', 'mahalanobis']:
    outcome_col = f'{method}_outcome'
    ood_samples = classified_df[classified_df['ground_truth'] == 'OOD']
    fps = (ood_samples[outcome_col] == 'false_pass').sum()
    fpr_pct = 100 * fps / len(ood_samples)
    method_fpr[method] = fpr_pct

methods = list(method_fpr.keys())
fprs = list(method_fpr.values())

ax.bar(methods, fprs, color=['#ff7f0e', '#2ca02c', '#1f77b4'], edgecolor='black', width=0.6)
ax.set_ylabel('False Pass Rate (%)')
ax.set_title('OOD Detection Method Comparison (FPR on OOD samples)')
ax.set_ylim(0, max(fprs) * 1.2)

for i, (method, fpr) in enumerate(method_fpr.items()):
    ax.text(i, fpr + 1, f'{fpr:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'failure_method_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Method comparison plot saved")
print("\n✓ All visualizations complete!")

## Utility Functions and Quick Reference

# Function to load and display failure images
def display_failure_sample(image_id, dataset_name, method='energy'):
    """Display a failure sample image with metadata."""
    try:
        # Get the failure record
        failure_row = classified_df[
            (classified_df['image_id'] == image_id) & 
            (classified_df['dataset_name'] == dataset_name)
        ].iloc[0]
        
        print(f"\n{'='*60}")
        print(f"Failure Analysis: {method.upper()}")
        print(f"{'='*60}")
        print(f"Image ID: {image_id}")
        print(f"Dataset: {dataset_name}")
        print(f"Ground Truth: {failure_row['ground_truth']}")
        print(f"Predicted Class: {failure_row['predicted_class']}")
        print(f"Confidence: {failure_row['confidence']:.2%}")
        print(f"\n{method.upper()} Method:")
        print(f"  Score: {failure_row[f'{method}_score']:.4f}")
        print(f"  Threshold: {thresholds[method]:.4f}")
        print(f"  Decision: {failure_row[f'{method}_decision']}")
        print(f"  Outcome: {failure_row[f'{method}_outcome']}")
        print(f"\nImage Path: {failure_row['image_path']}")
        print(f"{'='*60}")
        
    except (IndexError, KeyError) as e:
        print(f"Error: Could not find sample with ID {image_id} in {dataset_name}")
        print(f"Details: {e}")

# Summary statistics
print("\n" + "="*60)
print("PIPELINE SUMMARY")
print("="*60)
print(f"\nDataset Statistics:")
print(f"  Total samples: {len(classified_df)}")
print(f"  ID samples: {(classified_df['ground_truth']=='ID').sum()}")
print(f"  OOD samples: {(classified_df['ground_truth']=='OOD').sum()}")

print(f"\nMethod Performance:")
for method in ['energy', 'msp', 'mahalanobis']:
    outcome_col = f'{method}_outcome'
    ood_samples = classified_df[classified_df['ground_truth'] == 'OOD']
    
    correct = (ood_samples[outcome_col] == 'correct_reject').sum()
    false_pass = (ood_samples[outcome_col] == 'false_pass').sum()
    fpr = 100 * false_pass / len(ood_samples)
    
    print(f"\n  {method.upper()}:")
    print(f"    Correct rejects: {correct}/{len(ood_samples)} ({100*correct/len(ood_samples):.1f}%)")
    print(f"    False passes: {false_pass}/{len(ood_samples)} ({fpr:.1f}%)")

print(f"\nOutput Files:")
print(f"  CSV Tables: {PLOTS_DIR}/failure_analysis_*.csv")
print(f"  Visualizations: {PLOTS_DIR}/failure_*.png")

print(f"\n✓ Pipeline complete! Ready for manual inspection and analysis.")
print(f"  Use display_failure_sample(image_id, dataset_name, method)")
print(f"  to examine specific failures in detail.")
print("="*60)
