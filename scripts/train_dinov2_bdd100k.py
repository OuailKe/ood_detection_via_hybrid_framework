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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
DATA_DIR     = PROJECT_ROOT / 'data'

TRAIN_IMGLIST = DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/train_13cls.txt'
VAL_IMGLIST   = DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt'
IMAGE_DIR     = DATA_DIR / 'id/bdd100k/bdd100k/bdd100k/images/'

OUTPUT_DIR    = PROJECT_ROOT / 'results' / 'bdd100k_dinov2_vitb14_ft'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 13
EPOCHS      = 30
LR          = 1e-3
BATCH_SIZE  = 64
NUM_WORKERS = 4
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DINOv2 ViT-B/14 expects 224×224 input (patch size 14, so 224 is 16×16 patches)
IMG_SIZE = 224


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_imglist(path):
    images, labels = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                images.append(parts[0])
                labels.append(int(parts[1]))
    return images, labels


class BDD100KDataset(Dataset):
    def __init__(self, imglist_path, image_dir, transform=None):
        self.images, self.labels = load_imglist(imglist_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.images[idx]
        img      = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── Transforms ────────────────────────────────────────────────────────────────
# DINOv2 was trained with ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Model ─────────────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-B/14 backbone + linear classification head.
    The backbone is fully fine-tuned (not frozen).
    Feature dim for vitb14 = 768.
    """
    FEATURE_DIM = 768

    def __init__(self, num_classes=13):
        super().__init__()
        print('  Loading dinov2_vitb14 from torch.hub...')
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            pretrained=True,
        )
        # DINOv2 hub model exposes .forward() → CLS token [B, 768]
        self.head = nn.Linear(self.FEATURE_DIM, num_classes)

    def forward(self, x):
        features = self.backbone(x)   # [B, 768] CLS token
        return self.head(features)

    def get_features(self, x):
        """Return CLS-token features without the classification head."""
        return self.backbone(x)       # [B, 768]


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f'  Epoch {epoch:02d} [train]', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in tqdm(loader, desc='  Evaluating', leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits  = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += imgs.size(0)
    return correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('  DINOv2 ViT-B/14 Fine-tuning on BDD100K 13-class')
    print('=' * 65)
    print(f'  Device : {DEVICE}')
    print(f'  Epochs : {EPOCHS}')
    print(f'  LR     : {LR}')
    print(f'  Batch  : {BATCH_SIZE}')
    print(f'  Output : {OUTPUT_DIR}')

    # Datasets
    train_ds = BDD100KDataset(TRAIN_IMGLIST, IMAGE_DIR, train_transform)
    val_ds   = BDD100KDataset(VAL_IMGLIST,   IMAGE_DIR, val_transform)
    print(f'\n  Train: {len(train_ds)} samples')
    print(f'  Val  : {len(val_ds)} samples')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = DINOv2Classifier(num_classes=NUM_CLASSES).to(DEVICE)

    # Explicitly ensure all backbone parameters are unfrozen
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params    : {total_params:,}')
    print(f'  Trainable params: {trainable_params:,}')
    assert trainable_params == total_params, 'ERROR: some parameters are frozen!'

    criterion = nn.CrossEntropyLoss()

    # Differential LR: lower LR for backbone, higher for head
    # AdamW + weight_decay regularises the backbone without crushing features
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR * 0.01, 'weight_decay': 0.01},  # 1e-5
        {'params': model.head.parameters(),     'lr': LR,        'weight_decay': 0.01},  # 1e-3
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_acc      = 0.0
    history       = []
    patience      = 5
    no_improve    = 0

    print('\n  Starting training...\n')
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch)
        val_acc = evaluate(model, val_loader)
        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc   = val_acc
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best.ckpt')
        else:
            no_improve += 1

        # Always save latest
        torch.save(model.state_dict(), OUTPUT_DIR / 'last.ckpt')

        record = {
            'epoch':      epoch,
            'train_loss': round(train_loss, 6),
            'train_acc':  round(train_acc * 100, 4),
            'val_acc':    round(val_acc * 100, 4),
            'best_acc':   round(best_acc * 100, 4),
            'lr':         scheduler.get_last_lr()[0],
        }
        history.append(record)

        flag = ' ← best' if is_best else ''
        print(f'  Epoch {epoch:02d}/{EPOCHS}  '
              f'loss={train_loss:.4f}  '
              f'train_acc={train_acc*100:.2f}%  '
              f'val_acc={val_acc*100:.2f}%  '
              f'best={best_acc*100:.2f}%{flag}')

        # Save history after every epoch
        with open(OUTPUT_DIR / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if no_improve >= patience:
            print(f'\n  Early stopping at epoch {epoch} '
                  f'(no improvement for {patience} epochs)')
            break

    print(f'\n  ✓ Training complete. Best val accuracy: {best_acc*100:.2f}%')
    print(f'  ✓ Checkpoint saved to {OUTPUT_DIR}/best.ckpt')


if __name__ == '__main__':
    main()