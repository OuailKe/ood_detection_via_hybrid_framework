#!/home/ouail.kerrak/.conda/envs/ood310/bin/python

import os
import sys

_expected_env = 'ood310'
_current_env  = os.environ.get('CONDA_DEFAULT_ENV', '')
if _current_env != _expected_env:
    print(f'ERROR: wrong conda environment "{_current_env}". '
          f'Run with: conda activate {_expected_env} && python {sys.argv[0]}')
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('GPU_ID', '1')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
DATA_DIR     = PROJECT_ROOT / 'data'
OUTPUT_DIR   = PROJECT_ROOT / 'results' / 'mahalanobis_dinov2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── DINOv2 checkpoint (written by train_dinov2_bdd100k.py) ───────────────────
MODEL_CKPT  = PROJECT_ROOT / 'results' / 'bdd100k_dinov2_vitb14_ft' / 'best.ckpt'
FEATURE_DIM = 768
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

# ── Transform — identical to training val transform ───────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """Mirror of the class in train_dinov2_bdd100k.py."""
    def __init__(self, num_classes=13):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            pretrained=False,   # weights come from our checkpoint
        )
        self.head = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImgListDataset(Dataset):
    def __init__(self, image_paths, image_dir, transform=None):
        self.image_paths = image_paths
        self.image_dir   = Path(image_dir)
        self.transform   = transform
        self.labels      = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img   = Image.open(self.image_dir / self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx] if self.labels is not None else -1
        return img, label, str(self.image_paths[idx])


def load_imglist(path):
    images, labels = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                images.append(parts[0])
                labels.append(int(parts[1]))
    return images, labels


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, image_paths, image_dir, batch_size=BATCH_SIZE):
    """
    Hook model.head (our linear classifier) with a forward pre-hook.
    input[0] at that point is the CLS-token vector [B, 768] from DINOv2.
    Identical pattern to the torchvision hook on model.heads.head.
    """
    dataset = ImgListDataset(image_paths, image_dir, TRANSFORM)

    def collate_images_only(batch):
        return torch.stack([item[0] for item in batch])

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_images_only)

    all_features   = []
    features_store = []

    def hook_fn(module, input):
        # input[0]: [B, 768] — CLS token just before the linear head
        features_store.append(input[0].detach().cpu())

    hook = model.head.register_forward_pre_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for images in tqdm(loader, desc='  Extracting features', leave=False):
            features_store.clear()
            _ = model(images.cuda())
            if features_store:
                all_features.append(features_store[0])

    hook.remove()
    return torch.cat(all_features, dim=0).numpy()   # [N, 768]


# ── Mahalanobis ───────────────────────────────────────────────────────────────

def fit_mahalanobis(train_features, train_labels, num_classes=NUM_CLASSES):
    class_means = []
    for c in range(num_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            class_means.append(train_features[mask].mean(axis=0))
        else:
            class_means.append(np.zeros(train_features.shape[1]))
    class_means = np.array(class_means)   # [C, D]
    cov         = EmpiricalCovariance().fit(train_features)
    precision   = cov.precision_           # [D, D]
    return class_means, precision


def score_mahalanobis(features, class_means, precision, chunk_size=500):
    scores = []
    for start in tqdm(range(0, len(features), chunk_size),
                      desc='  Scoring', leave=False):
        chunk = features[start:start + chunk_size]
        dists = []
        for mean in class_means:
            diff = chunk - mean
            d    = np.einsum('ij,jk,ik->i', diff, precision, diff)
            dists.append(d)
        dists = np.stack(dists, axis=1)    # [chunk, C]
        scores.append(dists.min(axis=1))   # [chunk]
    return np.concatenate(scores)          # [N]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(id_scores, ood_scores, tpr_level=0.95):
    labels   = np.concatenate([np.zeros(len(id_scores)),
                                np.ones(len(ood_scores))])
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  Mahalanobis OOD Evaluation — DINOv2 ViT-B/14')
    print('=' * 60)

    # Load model
    print('\n[1/5] Loading DINOv2 model...')
    model = DINOv2Classifier(num_classes=NUM_CLASSES).cuda()
    checkpoint = torch.load(MODEL_CKPT, map_location='cuda')
    model.load_state_dict(checkpoint)
    model.eval()
    print(f'  ✓ Loaded from {MODEL_CKPT}')

    # Load image lists
    print('\n[2/5] Loading image lists...')
    dataset_index = {}
    for name, cfg in DATASETS.items():
        images, labels = load_imglist(cfg['imglist'])
        dataset_index[name] = {**cfg, 'images': images, 'labels': labels}
        print(f'  {name:20s}: {len(images):5d} images')

    # ID accuracy
    print('\n[3/5] Computing ID accuracy...')
    id_ds = ImgListDataset(
        dataset_index['bdd100k_val']['images'],
        dataset_index['bdd100k_val']['image_dir'],
        TRANSFORM,
    )
    id_ds.labels = dataset_index['bdd100k_val']['labels']

    def collate_with_labels(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels

    id_loader = DataLoader(id_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, collate_fn=collate_with_labels)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in tqdm(id_loader, desc='  Accuracy', leave=False):
            preds    = model(imgs.cuda()).argmax(1)
            correct += (preds == lbls.cuda()).sum().item()
            total   += lbls.size(0)
    id_acc = correct / total * 100
    print(f'  ✓ ID accuracy: {id_acc:.2f}%')

    # Fit Mahalanobis on BDD100K val
    print('\n[4/5] Fitting Mahalanobis on BDD100K val (ID)...')
    train_features = extract_features(
        model,
        dataset_index['bdd100k_val']['images'],
        dataset_index['bdd100k_val']['image_dir'],
    )
    train_labels = np.array(dataset_index['bdd100k_val']['labels'])
    print(f'  ✓ Features shape : {train_features.shape}')
    class_means, precision = fit_mahalanobis(train_features, train_labels)
    print(f'  ✓ Precision matrix: {precision.shape}')

    # Score all datasets
    print('\n[5/5] Scoring all datasets...')
    all_scores = {}
    for name, cfg in dataset_index.items():
        print(f'\n  {name}')
        feats          = extract_features(model, cfg['images'], cfg['image_dir'])
        scores         = score_mahalanobis(feats, class_means, precision)
        all_scores[name] = scores
        print(f'    mean={scores.mean():.2f}  min={scores.min():.2f}  '
              f'max={scores.max():.2f}')

    print('\n  Sanity check (OOD mean distance should be > ID):')
    for name, scores in all_scores.items():
        tag = '← ID (reference)' if name == 'bdd100k_val' else ''
        print(f'    {name:20s}: {scores.mean():.2f}  {tag}')

    # Save .npz score files — same format as original mahalanobis_eval.py
    # conf = -distance  (higher = more ID-like, matches OpenOOD sign convention)
    scores_dir = OUTPUT_DIR / 'scores'
    scores_dir.mkdir(exist_ok=True)
    print('\n  Saving .npz score files...')
    for name in dataset_index:
        conf     = -all_scores[name].astype(np.float32)
        pred     = np.zeros(len(conf), dtype=np.int64)
        npz_path = scores_dir / f'{name}.npz'
        np.savez(npz_path, conf=conf, pred=pred)
        print(f'    ✓ {npz_path.name}  ({len(conf)} samples)')

    # Evaluate
    id_scores  = all_scores['bdd100k_val']
    ood_groups = {
        'lostandfound':  ['lostandfound'],
        'streethazards': ['streethazards'],
        'smiyc':         ['smiyc'],
        'nearood':       ['lostandfound'],
        'farood':        ['streethazards', 'smiyc'],
    }

    results = []
    print('\n' + '=' * 60)
    print(f'  {"dataset":<20} {"FPR@95":>8} {"AUROC":>8} '
          f'{"AUPR_IN":>9} {"AUPR_OUT":>10} {"ACC":>6}')
    print('  ' + '-' * 64)

    for group_name, dataset_names in ood_groups.items():
        ood_sc = np.concatenate([all_scores[n] for n in dataset_names])
        m      = compute_metrics(id_scores, ood_sc)
        results.append({'dataset': group_name, **m, 'ACC': round(id_acc, 2)})
        print(f'  {group_name:<20} {m["FPR@95"]:>7.2f}% {m["AUROC"]:>7.2f}% '
              f'{m["AUPR_IN"]:>8.2f}% {m["AUPR_OUT"]:>9.2f}% {id_acc:>6.2f}%')

    print('=' * 60)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'ood.csv', index=False)

    with open(OUTPUT_DIR / 'log.txt', 'w') as f:
        f.write('Mahalanobis OOD Evaluation — DINOv2 ViT-B/14\n')
        f.write('=' * 60 + '\n')
        f.write(f'Model    : {MODEL_CKPT}\n')
        f.write(f'Backbone : dinov2_vitb14 (fine-tuned on BDD100K 13-class)\n')
        f.write(f'ID acc   : {id_acc:.2f}%\n\n')
        f.write(results_df.to_string(index=False))
        f.write('\n')

    print(f'\n  ✓ Output saved to {OUTPUT_DIR}')
    print(f'  ├── scores/')
    for name in dataset_index:
        print(f'  │   ├── {name}.npz')
    print(f'  ├── ood.csv')
    print(f'  └── log.txt')


if __name__ == '__main__':
    main()