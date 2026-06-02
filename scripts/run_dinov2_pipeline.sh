#!/bin/bash
#SBATCH --job-name=dinov2_ood
#SBATCH --output=/home/ouail.kerrak/ood_project/logs/dinov2_%j.log
#SBATCH --error=/home/ouail.kerrak/ood_project/logs/dinov2_%j.err
#SBATCH --partition=student
#SBATCH --nodelist=localhost
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=UNLIMITED

# ── Setup ─────────────────────────────────────────────────────────────────────
set -e   # abort on any error

LOG_DIR=/home/ouail.kerrak/ood_project/logs
mkdir -p $LOG_DIR

echo "========================================"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Start    : $(date)"
echo "========================================"

# ── Activate conda env ────────────────────────────────────────────────────────
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ood310

echo "  Python   : $(which python)"
echo "  Conda env: $CONDA_DEFAULT_ENV"

# ── Force GPU 1 (as per your existing setup) ─────────────────────────────────
export GPU_ID=0

# ── Step 1: Fine-tune DINOv2 ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  STEP 1/2 — Training DINOv2 ViT-B/14"
echo "  Start: $(date)"
echo "========================================"

python /home/ouail.kerrak/ood_project/scripts/train_dinov2_bdd100k.py

echo ""
echo "  Training complete: $(date)"

# ── Step 2: Mahalanobis evaluation ───────────────────────────────────────────
echo ""
echo "========================================"
echo "  STEP 2/2 — Mahalanobis Evaluation"
echo "  Start: $(date)"
echo "========================================"

python /home/ouail.kerrak/ood_project/scripts/mahalanobis_eval_dinov2.py

echo ""
echo "========================================"
echo "  ALL DONE: $(date)"
echo "========================================"