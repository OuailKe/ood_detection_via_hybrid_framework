#!/home/ouail.kerrak/.conda/envs/ood310/bin/python


import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.fftpack import dct

_env = os.environ.get('CONDA_DEFAULT_ENV', '')
if _env != 'ood310':
    print(f'ERROR: wrong environment "{_env}". '
          f'Run: conda activate ood310 && python {sys.argv[0]}')
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('GPU_ID', '1')

import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from sklearn.covariance import EmpiricalCovariance

PROJECT_ROOT = Path('/home/ouail.kerrak/ood_project')
DATA_DIR     = PROJECT_ROOT / 'data'
RESULTS_DIR  = PROJECT_ROOT / 'results'
OUTPUT_DIR   = PROJECT_ROOT / 'plots' / 'timingV2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CKPT   = RESULTS_DIR / \
    'bdd100k_vit-b-16_base_e30_lr0.001_bdd100k_13cls_ft/s0/best.ckpt'
NUM_CLASSES  = 13
N_WARMUP     = 50    
N_MEASURE    = 200   

IMGLIST = DATA_DIR / 'benchmark_imglist/autonomous/bdd100k/val_13cls.txt'
IMG_DIR = DATA_DIR / 'id/bdd100k/bdd100k/bdd100k/images/'

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

BENFORD_PROBS = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

def get_first_digit(x):
    x = abs(float(x))
    if x == 0:
        return None
    while x < 1:  x *= 10
    while x >= 10: x /= 10
    return int(x)

def compute_dct_benford_chi2(img_array):

    arr       = img_array.astype(np.float64)
    block_size = 8
    h, w      = arr.shape
    digits    = []
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
    counts   = np.zeros(9)
    for d in digits: counts[d - 1] += 1
    observed  = counts / counts.sum()
    n         = len(digits)
    chi2_stat = n * np.sum((observed - BENFORD_PROBS)**2 / BENFORD_PROBS)
    return float(chi2_stat)



def load_model():
    model = tv_models.vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(
        model.heads.head.in_features, NUM_CLASSES)
    model = model.cuda()
    ckpt  = torch.load(MODEL_CKPT, map_location='cuda')
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_sample_images(n=N_WARMUP + N_MEASURE):
   
    images = []
    with open(IMGLIST) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                images.append(parts[0])
            if len(images) >= n:
                break
    return images


def preload_tensors(image_paths):
    tensors  = []
    grays    = []
    for p in tqdm(image_paths, desc='  Preloading images', leave=False):
        full = IMG_DIR / p
        img  = Image.open(full).convert('RGB')
        t    = TRANSFORM(img).unsqueeze(0).cuda()   # [1, 3, 224, 224]
        tensors.append(t)
        gray = np.array(img.convert('L').resize((224, 224)))
        grays.append(gray)
    return tensors, grays


def fit_mahalanobis(model, image_paths, tensors):
   
    print('  Fitting Mahalanobis on 500 ID samples...')
    features_store = []

    def hook_fn(module, input):
        features_store.append(input[0].detach().cpu())

    hook = model.heads.head.register_forward_pre_hook(hook_fn)

    all_feats  = []
    all_labels = []

    with open(IMGLIST) as f:
        lines = [l.strip().split() for l in f.readlines()[:500]]

    with torch.no_grad():
        for i, t in enumerate(tensors[:500]):
            features_store.clear()
            _ = model(t)
            if features_store:
                all_feats.append(features_store[0].numpy())
            lbl = int(lines[i][1]) if i < len(lines) else 0
            all_labels.append(lbl)

    hook.remove()

    feats  = np.concatenate(all_feats, axis=0)
    labels = np.array(all_labels)

    class_means = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            class_means.append(feats[mask].mean(axis=0))
        else:
            class_means.append(np.zeros(feats.shape[1]))
    class_means = np.array(class_means)

    cov       = EmpiricalCovariance().fit(feats)
    precision = cov.precision_

    return class_means, precision, hook


def time_msp_energy(model, tensors, n_warmup, n_measure):
    """Time MSP and Energy together — same forward pass."""
    # Warmup
    with torch.no_grad():
        for t in tensors[:n_warmup]:
            logits = model(t)
            _      = F.softmax(logits, dim=1).max(dim=1).values
            _      = -torch.logsumexp(logits, dim=1)
    torch.cuda.synchronize()

    msp_times    = []
    energy_times = []

    with torch.no_grad():
        for t in tensors[n_warmup:n_warmup + n_measure]:
            # MSP
            torch.cuda.synchronize()
            t0     = time.perf_counter()
            logits = model(t)
            _      = F.softmax(logits, dim=1).max(dim=1).values
            torch.cuda.synchronize()
            msp_times.append((time.perf_counter() - t0) * 1000)

            # Energy (same forward pass cost — just different postprocessing)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _  = -torch.logsumexp(logits, dim=1)
            torch.cuda.synchronize()
            energy_times.append((time.perf_counter() - t0) * 1000)

    # Energy total = forward pass (already in msp_times) + logsumexp overhead
    energy_total = [m + e for m, e in zip(msp_times, energy_times)]

    return msp_times, energy_total


def time_mahalanobis(model, tensors, class_means, precision,
                     n_warmup, n_measure):
    """Time Mahalanobis: forward pass + feature extraction + distance."""
    features_store = []

    def hook_fn(module, input):
        features_store.append(input[0].detach().cpu().numpy())

    hook = model.heads.head.register_forward_pre_hook(hook_fn)

    with torch.no_grad():
        for t in tensors[:n_warmup]:
            features_store.clear()
            _ = model(t)

    times = []
    with torch.no_grad():
        for t in tensors[n_warmup:n_warmup + n_measure]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            features_store.clear()
            _ = model(t)
            feat = features_store[0]                    # [1, 768]

            dists = []
            for mean in class_means:
                diff = feat[0] - mean
                d    = diff @ precision @ diff
                dists.append(d)
            _ = min(dists)

            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    hook.remove()
    return times


def time_benford(grays, n_warmup, n_measure):
    """Time Benford/DCT filter — CPU only, no GPU."""
    # Warmup
    for g in grays[:n_warmup]:
        _ = compute_dct_benford_chi2(g)

    times = []
    for g in grays[n_warmup:n_warmup + n_measure]:
        t0 = time.perf_counter()
        _  = compute_dct_benford_chi2(g)
        times.append((time.perf_counter() - t0) * 1000)

    return times


def time_hybrid(model, tensors, grays, class_means, precision,
                tau_benford, n_warmup, n_measure):
   
    features_store = []

    def hook_fn(module, input):
        features_store.append(input[0].detach().cpu().numpy())

    hook = model.heads.head.register_forward_pre_hook(hook_fn)

    # Warmup
    with torch.no_grad():
        for t, g in zip(tensors[:n_warmup], grays[:n_warmup]):
            chi2 = compute_dct_benford_chi2(g)
            if chi2 <= tau_benford:
                features_store.clear()
                _ = model(t)

    times          = []
    stage1_only    = 0   
    stage1_and_2   = 0   

    with torch.no_grad():
        for t, g in zip(tensors[n_warmup:n_warmup + n_measure],
                        grays[n_warmup:n_warmup + n_measure]):

            torch.cuda.synchronize()
            t0   = time.perf_counter()

            chi2 = compute_dct_benford_chi2(g)

            if chi2 > tau_benford:
                stage1_only += 1
            else:
                features_store.clear()
                _ = model(t)
                feat  = features_store[0][0]
                dists = []
                for mean in class_means:
                    diff = feat - mean
                    dists.append(diff @ precision @ diff)
                _ = min(dists)
                stage1_and_2 += 1

            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    hook.remove()
    print(f'    Early exits (Stage 1 only): {stage1_only}/{n_measure} '
          f'({100*stage1_only/n_measure:.1f}%)')
    print(f'    Full pipeline (Stage 1+2):  {stage1_and_2}/{n_measure} '
          f'({100*stage1_and_2/n_measure:.1f}%)')
    return times


def summarise(times, name):
    arr = np.array(times)
    return {
        'Method':        name,
        'Mean (ms)':     round(arr.mean(), 3),
        'Std (ms)':      round(arr.std(), 3),
        'Min (ms)':      round(arr.min(), 3),
        'Max (ms)':      round(arr.max(), 3),
        'Throughput (img/s)': round(1000 / arr.mean(), 1),
    }


def main():
    print('=' * 60)
    print('  Inference Time Benchmark')
    print('=' * 60)
    print(f'  Warmup iterations:  {N_WARMUP}')
    print(f'  Measured iterations: {N_MEASURE}')

    print('\n[1/4] Loading model...')
    model = load_model()
    print('  ✓ Model loaded')

    print('\n[2/4] Preloading images...')
    image_paths      = load_sample_images(N_WARMUP + N_MEASURE)
    tensors, grays   = preload_tensors(image_paths)
    print(f'  ✓ {len(tensors)} images preloaded')

    print('\n[3/4] Fitting Mahalanobis...')
    class_means, precision, _ = fit_mahalanobis(model, image_paths, tensors)
    tau_benford     = 73.3     
    tau_mahalanobis = 1204.01  
    print('  ✓ Mahalanobis fitted')

    print('\n[4/4] Timing methods...')
    results = []

    print('  → MSP + Energy...')
    msp_times, energy_times = time_msp_energy(
        model, tensors, N_WARMUP, N_MEASURE)
    results.append(summarise(msp_times,    'MSP'))
    results.append(summarise(energy_times, 'Energy'))

    print('  → Mahalanobis...')
    maha_times = time_mahalanobis(
        model, tensors, class_means, precision, N_WARMUP, N_MEASURE)
    results.append(summarise(maha_times, 'Mahalanobis'))

    print('  → Benford/DCT...')
    benford_times = time_benford(grays, N_WARMUP, N_MEASURE)
    results.append(summarise(benford_times, 'Benford/DCT'))

    print('  → Hybrid (early-exit)...')
    hybrid_times = time_hybrid(
        model, tensors, grays,
        class_means, precision,
        tau_benford,
        N_WARMUP, N_MEASURE)
    results.append(summarise(hybrid_times, 'Hybrid (early-exit Benford+Mahalanobis)'))

    df = pd.DataFrame(results)

    print('\n' + '=' * 60)
    print('  RESULTS')
    print('=' * 60)
    print(df.to_string(index=False))

    csv_path = OUTPUT_DIR / 'timing_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n  ✓ Saved to {csv_path}')

    # Pretty summary for thesis
    print('\n  Thesis table (Mean ± Std):')
    print(f'  {"Method":<35} {"ms/image":>10} {"img/s":>10}')
    print('  ' + '-' * 58)
    for _, row in df.iterrows():
        print(f'  {row["Method"]:<35} '
              f'{row["Mean (ms)"]:>7.2f} ± {row["Std (ms)"]:>5.2f}  '
              f'{row["Throughput (img/s)"]:>8.1f}')


if __name__ == '__main__':
    main()