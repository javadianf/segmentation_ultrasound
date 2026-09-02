# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian

import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from monai.metrics import (
    DiceMetric,
    ConfusionMatrixMetric,
    SurfaceDistanceMetric
)


def get_args():
    parser = argparse.ArgumentParser(description="Robust Binary Segmentation Evaluation")
    parser.add_argument("--pred_dir", type=str, required=True, help="Folder containing prediction masks (PNG)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Folder containing ground truth annotations (PNG)")
    return parser.parse_args()


def preprocess_image(image_path):
    """
    Load image, convert to single channel, scale to 0-1,
    and add batch/channel dimensions (B, C, H, W).
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale (L)
    img_np = np.array(img)

    # Convert 255 foreground to 1 (Binary)
    img_np = (img_np > 127).astype(np.float32)

    # Convert to Tensor (H, W) -> (1, H, W) -> (1, 1, H, W) to match MONAI format
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)

    return img_tensor


def main():
    args = get_args()

    # Initialize Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    conf_matrix = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["precision", "recall"],
        reduction="mean"
    )
    surface_distance_metric = SurfaceDistanceMetric(
        include_background=False,
        symmetric=True,
        distance_metric='euclidean'
    )

    # File lists
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.png")))
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.png")))

    total_files = len(pred_files)
    excluded_count = 0

    print(f"Processing {total_files} images...")
    print(f"{'Filename':<30} | {'Dice':<8} | {'Prec':<8} | {'Recall':<8} | {'ASSD (px)':<10}")
    print("-" * 80)

    results = []
    for pred_path, gt_path in zip(pred_files, gt_files):

        filename = os.path.basename(pred_path)
        y_pred = preprocess_image(pred_path)  # Prediction
        y = preprocess_image(gt_path)  # Ground Truth

        # --- Check for double empty ---
        is_empty_gt = (torch.sum(y) == 0)
        is_empty_pred = (torch.sum(y_pred) == 0)

        if is_empty_gt and is_empty_pred:
            excluded_count += 1
            dice_score = float('nan')
            ssd_score = float('nan')
            # Precision/Recall are technically undefined for TP=0, FP=0, FN=0
            # We set to NaN to exclude them from averages as well.
            precision_score = float('nan')
            recall_score = float('nan')
        else:
            # 1. Dice Score
            dice_metric.reset()
            dice_metric(y_pred=y_pred, y=y)
            dice_score = dice_metric.aggregate().item()

            # 2. Precision / Recall
            conf_matrix.reset()
            conf_matrix(y_pred=y_pred, y=y)
            conf_res = conf_matrix.aggregate()
            precision_score = conf_res[0].item()
            recall_score = conf_res[1].item()

            # 3. ASSD
            surface_distance_metric.reset()
            surface_distance_metric(y_pred=y_pred, y=y)
            if is_empty_gt or is_empty_pred:
                ssd_score = float('nan')
            else:
                ssd_score = surface_distance_metric.aggregate().item()


        print(f"{filename:<30} | {dice_score:.4f}   | {precision_score:.4f}   | {recall_score:.4f}   | {ssd_score:.4f}")

        results.append({
            "filename": filename,
            "dice": dice_score,
            "precision": precision_score,
            "recall": recall_score,
            "assd": ssd_score
        })

    if results:
        # Calculate Averages using np.nanmean (ignores NaNs automatically)
        avg_dice = np.nanmean([r['dice'] for r in results])
        avg_prec = np.nanmean([r['precision'] for r in results])
        avg_rec = np.nanmean([r['recall'] for r in results])
        avg_assd = np.nanmean([r['assd'] for r in results])

        effective_samples = total_files - excluded_count

        print("-" * 80)
        print(f"Total Images:      {total_files}")
        print(f"Excluded (Empty):  {excluded_count}")
        print(f"Effective Samples: {effective_samples}")
        print("-" * 80)
        print(f"Average Dice:      {avg_dice:.4f}")
        print(f"Average Precision: {avg_prec:.4f}")
        print(f"Average Recall:    {avg_rec:.4f}")
        print(f"Average ASSD:      {avg_assd:.4f}")


if __name__ == "__main__":
    main()
