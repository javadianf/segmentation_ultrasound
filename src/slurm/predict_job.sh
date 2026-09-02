#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian
#
# SLURM inference job. Submit with: sbatch slurm/predict_job.sh
# Override CHECKPOINT and PRED_DIR to run a different trained model, for example
# the random-split baseline:
#   sbatch --export=ALL,CHECKPOINT=$MLMT_WORK/checkpoints/best_model_random.pth,\
# PRED_DIR=$MLMT_WORK/predictions_random slurm/predict_job.sh

#SBATCH --job-name=mlmt_predict
#SBATCH --output=logs/predict_%j.log
#SBATCH --time=01:00:00
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
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CHECKPOINT="${CHECKPOINT:-$WORK/checkpoints/best_model.pth}"
PRED_DIR="${PRED_DIR:-$WORK/predictions}"

echo "job ${SLURM_JOB_ID:-local} at $(date)"
echo "checkpoint $CHECKPOINT"

source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$PROJECT"
mkdir -p logs "$PRED_DIR"

nvidia-smi

python predict.py \
    --test_dir "$MLMT_DATA/test/images" \
    --output_dir "$PRED_DIR" \
    --checkpoint "$CHECKPOINT" \
    --tta

( cd "$PRED_DIR" && zip -qr "$WORK/submission.zip" . )
echo "submission written to $WORK/submission.zip"

echo "finished at $(date)"
