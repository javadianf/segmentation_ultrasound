#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian
#
# SLURM training job. Submit with: sbatch slurm/train_job.sh
# Set MLMT_WORK, MLMT_PROJECT and MLMT_ENV in the environment or edit the
# fallbacks below before submitting.

#SBATCH --job-name=mlmt_segformer
#SBATCH --output=logs/train_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g

set -euo pipefail

WORK="${MLMT_WORK:-$PWD}"
PROJECT="${MLMT_PROJECT:-$PWD}"
ENV_NAME="${MLMT_ENV:-MLMT}"
CONDA_SH="${MLMT_CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

export MLMT_DATA="${MLMT_DATA:-$WORK/data}"
export MLMT_CHECKPOINTS="${MLMT_CHECKPOINTS:-$WORK/checkpoints}"

# Compute nodes have no outbound network, so the encoder must already be cached.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} at $(date)"

source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$PROJECT"
mkdir -p logs "$MLMT_CHECKPOINTS"

nvidia-smi

python train.py \
    --data_dir "$MLMT_DATA" \
    --output_dir "$MLMT_CHECKPOINTS" \
    --model_name nvidia/mit-b2 \
    --img_size 512 \
    --batch_size 8 \
    --epochs 100 \
    --split_mode subject \
    --val_subject participant1

echo "finished at $(date)"
