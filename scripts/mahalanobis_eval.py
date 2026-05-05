#!/home/ouail.kerrak/.conda/envs/ood310/bin/python

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
_expected_env = 'ood310'
_current_env  = os.environ.get('CONDA_DEFAULT_ENV', '')
if _current_env != _expected_env:
    print(f'ERROR: wrong conda environment "{_current_env}". '
          f'Run with: conda activate {_expected_env} && python {sys.argv[0]}')
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('GPU_ID', '0')

import torch
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
RESULTS_DIR  = PROJECT_ROOT / 'results'
DATA_DIR     = PROJECT_ROOT / 'data'
OUTPUT_DIR   = PROJECT_ROOT / 'results' / 'mahalanobis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CKPT = RESULTS_DIR / 'bdd100k_vit-b-16_base_e30_lr0.001_bdd100k_13cls_ft/s0/best.ckpt'
NUM_CLASSES = 13
BATCH_SIZE  = 32

DATASETS = {
    'bdd100k_val': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt',
        'image_dir': DATA_DIR / 'id/bdd100k/bdd100k/bdd100k/images/',
        'is_ood':    False,
        'ood_type':  'ID',
    },
    'lostandfound': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/lostandfound/test.txt',
        'image_dir': DATA_DIR / 'ood/near/LostAndFound',
        'is_ood':    True,
        'ood_type':  'near',
    },
    'streethazards': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/streethazards/test.txt',
        'image_dir': DATA_DIR / 'ood/far/StreetHazards/test/images',
        'is_ood':    True,
        'ood_type':  'far',
    },
    'smiyc': {
        'imglist':   DATA_DIR / 'benchmark_imglist/autonomous/smiyc/test.txt',
        'image_dir': DATA_DIR / 'ood/far/SMIYC/test/images',
        'is_ood':    True,
        'ood_type':  'far',
    },
}

# Image transform (must match training) 
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Dataset class 
class ImgListDataset(Dataset):
    def __init__(self, image_paths, image_dir, transform=None):
        self.image_paths = image_paths
        self.image_dir   = Path(image_dir)
        self.transform   = transform
        self.labels      = None  # Set externally if needed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.image_paths[idx]
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx] if self.labels is not None else None
        return image, label, str(img_path)


def load_imglist(path):
    images, labels = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                images.append(parts[0])
                labels.append(int(parts[1]))
    return images, labels
 
def extract_features(model, image_paths, image_dir, batch_size=BATCH_SIZE):
    """
    Extract 768-dim penultimate features from torchvision ViT-B/16.
    Uses a forward pre-hook on model.heads.head to capture the
    feature vector before the linear classifier.
    """
    dataset = ImgListDataset(image_paths, image_dir, TRANSFORM)
    
    # Custom collate function that only stacks images
    def collate_images_only(batch):
        images = torch.stack([item[0] for item in batch])
        return images
    
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0, collate_fn=collate_images_only)

    all_features  = []
    features_store = []

    def hook_fn(module, input):
        # input[0] shape: [B, 768]
        features_store.append(input[0].detach().cpu())

    hook = model.heads.head.register_forward_pre_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for images in tqdm(loader, desc='  Extracting features', leave=False):
            features_store.clear()
            _ = model(images.cuda())
            if features_store:
                all_features.append(features_store[0])

    hook.remove()
    return torch.cat(all_features, dim=0).numpy()   # [N, 768]

#  Mahalanobis scorer 
def fit_mahalanobis(train_features, train_labels, num_classes=NUM_CLASSES):
    """
    Fit class-conditional Gaussians on training features.
    Returns class_means [C, D] and precision matrix [D, D].
    """
    class_means = []
    for c in range(num_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            class_means.append(train_features[mask].mean(axis=0))
        else:
            class_means.append(np.zeros(train_features.shape[1]))
    class_means = np.array(class_means)   # [C, D]

    cov       = EmpiricalCovariance().fit(train_features)
    precision = cov.precision_            # [D, D]

    return class_means, precision


def score_mahalanobis(features, class_means, precision, chunk_size=500):
   
    scores = []
    for start in tqdm(range(0, len(features), chunk_size),
                      desc='  Scoring', leave=False):
        chunk = features[start:start + chunk_size]      # [chunk, D]
        dists = []
        for mean in class_means:
            diff = chunk - mean                          # [chunk, D]
            # einsum gives per-sample distance [chunk], not cross-matrix [chunk, chunk]
            d = np.einsum('ij,jk,ik->i', diff, precision, diff)
            dists.append(d)
        dists = np.stack(dists, axis=1)                 # [chunk, C]
        scores.append(dists.min(axis=1))                # [chunk]
    return np.concatenate(scores)                       # [N]


# Metrics 
def compute_metrics(id_scores, ood_scores, tpr_level=0.95):
    
    labels = np.concatenate([np.zeros(len(id_scores)),
                              np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    # AUROC: higher score should predict OOD (label=1) → use scores directly
    auroc    = roc_auc_score(labels, scores)

    # AUPR_OUT: OOD as positive class
    aupr_out = average_precision_score(labels, scores)

    # AUPR_IN: ID as positive class → negate scores so lower distance = higher ID score
    aupr_in  = average_precision_score(1 - labels, -scores)

    
    thresh = np.percentile(id_scores, tpr_level * 100)   # e.g. 95th percentile
    fpr    = (ood_scores <= thresh).mean()

    return {
        'FPR@95':   round(fpr * 100, 2),
        'AUROC':    round(auroc * 100, 2),
        'AUPR_IN':  round(aupr_in * 100, 2),
        'AUPR_OUT': round(aupr_out * 100, 2),
    }

def main():
    print('=' * 60)
    print('  Mahalanobis OOD Evaluation')
    print('=' * 60)

    print('\n[1/4] Loading model...')
    model = tv_models.vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    model = model.cuda()
    checkpoint = torch.load(MODEL_CKPT, map_location='cuda')
    model.load_state_dict(checkpoint)
    model.eval()
    print(f'  ✓ Loaded from {MODEL_CKPT}')

    print('\n[2/5] Loading image lists...')
    dataset_index = {}
    for name, cfg in DATASETS.items():
        images, labels = load_imglist(cfg['imglist'])
        dataset_index[name] = {
            'images':    images,
            'labels':    labels,
            'image_dir': cfg['image_dir'],
            'is_ood':    cfg['is_ood'],
            'ood_type':  cfg['ood_type'],
        }
        print(f'  {name:20s}: {len(images):5d} images')

    print('\n[2.5/5] Computing ID accuracy...')
    id_dataset = ImgListDataset(
        dataset_index['bdd100k_val']['images'],
        dataset_index['bdd100k_val']['image_dir'],
        TRANSFORM
    )
    id_dataset.labels = dataset_index['bdd100k_val']['labels']
    
    # Custom collate function for accuracy computation
    def collate_with_labels(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels
    
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_with_labels)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(id_loader, desc='  Computing accuracy', leave=False):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    id_acc = correct / total * 100
    print(f'  ✓ ID accuracy: {id_acc:.2f}%')

    # Extract training features and fit Gaussians 
    print('\n[3/5] Fitting Mahalanobis on BDD100K-val (ID)...')
    train_features = extract_features(
        model,
        dataset_index['bdd100k_val']['images'],
        dataset_index['bdd100k_val']['image_dir'],
    )
    train_labels = np.array(dataset_index['bdd100k_val']['labels'])
    print(f'  ✓ Training features: {train_features.shape}')

    class_means, precision = fit_mahalanobis(train_features, train_labels)
    print(f'  ✓ Precision matrix:  {precision.shape}')

    print('\n[4/5] Scoring all datasets...')
    all_scores = {}
    for name, cfg in dataset_index.items():
        print(f'\n  {name}')
        feats = extract_features(model, cfg['images'], cfg['image_dir'])
        scores = score_mahalanobis(feats, class_means, precision)
        all_scores[name] = scores
        print(f'    mean distance: {scores.mean():.2f}  '
              f'min: {scores.min():.2f}  max: {scores.max():.2f}')

   
    print('\n  Sanity check (OOD should be > ID):')
    for name, scores in all_scores.items():
        tag = '← ID (reference)' if name == 'bdd100k_val' else ''
        print(f'    {name:20s}: {scores.mean():.2f}  {tag}')

    # Save .npz score files (one per dataset, OpenOOD format) 
    # OpenOOD stores conf (confidence) and pred (predicted class) in each .npz
    # For Mahalanobis: conf = -distance (higher = more ID-like, matches OpenOOD sign)
    scores_dir = OUTPUT_DIR / 'scores'
    scores_dir.mkdir(exist_ok=True)

    print('\n  Saving .npz score files...')
    for name, cfg in dataset_index.items():
       
        conf = -all_scores[name].astype(np.float32)

      
        pred = np.zeros(len(conf), dtype=np.int64)

        npz_path = scores_dir / f'{name}.npz'
        np.savez(npz_path, conf=conf, pred=pred)
        print(f'    ✓ {npz_path.name}  ({len(conf)} samples)')

    id_scores = all_scores['bdd100k_val']
    results   = []

    ood_groups = {
        'lostandfound':  ['lostandfound'],
        'streethazards': ['streethazards'],
        'smiyc':         ['smiyc'],
        'nearood':       ['lostandfound'],
        'farood':        ['streethazards', 'smiyc'],
    }

    print('\n' + '=' * 60)
    print(f'  {"dataset":<20} {"FPR@95":>8} {"AUROC":>8} '
          f'{"AUPR_IN":>9} {"AUPR_OUT":>10} {"ACC":>6}')
    print('  ' + '-' * 64)

    for group_name, dataset_names in ood_groups.items():
        ood_sc = np.concatenate([all_scores[n] for n in dataset_names])
        m = compute_metrics(id_scores, ood_sc)
       
        results.append({
            'dataset':  group_name,
            'FPR@95':   m['FPR@95'],
            'AUROC':    m['AUROC'],
            'AUPR_IN':  m['AUPR_IN'],
            'AUPR_OUT': m['AUPR_OUT'],
            'ACC':      round(id_acc, 2),
        })
        print(f'  {group_name:<20} {m["FPR@95"]:>7.2f}% {m["AUROC"]:>7.2f}% '
              f'{m["AUPR_IN"]:>8.2f}% {m["AUPR_OUT"]:>9.2f}% {id_acc:>6.2f}%')

    print('=' * 60)

    results_df = pd.DataFrame(results)
    ood_csv    = OUTPUT_DIR / 'ood.csv'
    results_df.to_csv(ood_csv, index=False)

    log_txt = OUTPUT_DIR / 'log.txt'
    with open(log_txt, 'w') as f:
        f.write('Mahalanobis OOD Evaluation\n')
        f.write('=' * 60 + '\n')
        f.write(f'Model    : {MODEL_CKPT}\n')
        f.write(f'Threshold: BDD100K-val at 95% TPR\n')
        f.write(f'Note     : ACC = ID test accuracy: {id_acc:.2f}%\n\n')
        f.write(results_df.to_string(index=False))
        f.write('\n')

    print(f'\n  ✓ Output folder: {OUTPUT_DIR}')
    print(f'  ├── scores/')
    for name in dataset_index:
        print(f'  │   ├── {name}.npz')
    print(f'  ├── ood.csv')
    print(f'  └── log.txt')


if __name__ == '__main__':
    main()