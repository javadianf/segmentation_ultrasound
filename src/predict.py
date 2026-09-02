# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian


"""
Run a trained checkpoint over a directory of ultrasound frames and write binary
masks at the original frame resolution.

    python predict.py --test_dir ./data/test/images --output_dir ./predictions \
        --checkpoint ./checkpoints/best_model.pth --tta
"""

import os
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

DATA_ROOT = os.environ.get("MLMT_DATA", "./data")
CKPT_ROOT = os.environ.get("MLMT_CHECKPOINTS", "./checkpoints")
PRED_ROOT = os.environ.get("MLMT_PREDICTIONS", "./predictions")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_base_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model_name = ckpt_args.get("model_name", "nvidia/mit-b2")

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "background", 1: "bone"},
        label2id={"background": 0, "bone": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("loaded %s (epoch %s, val dice %s)"
          % (model_name, ckpt.get("epoch", "na"), ckpt.get("val_dice", "na")))
    return model, ckpt_args


@torch.no_grad()
def infer(model, img_tensor, orig_size, device, amp):
    img_tensor = img_tensor.to(device)
    with autocast(device.type, enabled=amp):
        logits = model(pixel_values=img_tensor).logits
    logits = F.interpolate(logits.float(), size=orig_size,
                           mode="bilinear", align_corners=False)
    return torch.softmax(logits, dim=1)[0, 1].cpu().numpy()


def tta_infer(model, img_np, transform, orig_size, device, amp):
    """
    Average probability maps over the four axis-aligned flips.

    Flips are the only transform used here because they are exactly invertible on
    the probability map, so no interpolation error is folded back into the
    average. The gain is concentrated on the boundary: single-view predictions
    tend to fray where the surface is faint, and averaging four views suppresses
    the frayed pixels without eroding the well-supported core of the surface.
    """
    flips = [
        (lambda x: x, lambda p: p),
        (np.fliplr, np.fliplr),
        (np.flipud, np.flipud),
        (lambda x: np.flipud(np.fliplr(x)), lambda p: np.fliplr(np.flipud(p))),
    ]

    probs = []
    for fwd, inv in flips:
        aug = np.ascontiguousarray(fwd(img_np))
        t = transform(image=aug)["image"].unsqueeze(0)
        probs.append(np.ascontiguousarray(inv(infer(model, t, orig_size, device, amp))))

    return np.mean(probs, axis=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str,
                   default=os.path.join(DATA_ROOT, "test", "images"))
    p.add_argument("--output_dir", type=str, default=PRED_ROOT)
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(CKPT_ROOT, "best_model.pth"))
    p.add_argument("--img_size", type=int, default=None,
                   help="override the image size stored in the checkpoint")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tta", action="store_true",
                   help="average over four flips before thresholding")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    model, ckpt_args = load_model(args.checkpoint, device)
    img_size = args.img_size or ckpt_args.get("img_size", 512)
    transform = get_base_transform(img_size)

    test_paths = sorted(
        glob.glob(os.path.join(args.test_dir, "*.png"))
        + glob.glob(os.path.join(args.test_dir, "*.jpg"))
        + glob.glob(os.path.join(args.test_dir, "*.jpeg"))
    )
    if not test_paths:
        raise FileNotFoundError("no images found in " + args.test_dir)

    print("predicting %d images, tta=%s, threshold=%.2f"
          % (len(test_paths), args.tta, args.threshold))

    for img_path in tqdm(test_paths):
        orig = Image.open(img_path).convert("RGB")
        orig_size = (orig.size[1], orig.size[0])
        img_np = np.array(orig)

        if args.tta:
            prob = tta_infer(model, img_np, transform, orig_size, device, amp)
        else:
            t = transform(image=img_np)["image"].unsqueeze(0)
            prob = infer(model, t, orig_size, device, amp)

        mask = ((prob >= args.threshold).astype(np.uint8)) * 255
        stem = os.path.splitext(os.path.basename(img_path))[0]
        Image.fromarray(mask).save(os.path.join(args.output_dir, stem + ".png"))

    print("wrote %d masks to %s" % (len(test_paths), args.output_dir))


if __name__ == "__main__":
    main()
