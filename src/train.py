# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian


"""
Train SegFormer for binary bone surface segmentation in knee ultrasound.

Two validation strategies are supported. The subject-aware mode holds out every
frame from one participant and is the setting reported in the paper. The random
mode is retained only to reproduce the leakage-inflated baseline discussed
there, and should not be used to report generalisation numbers.

    python train.py --data_dir ./data --split_mode subject --val_subject participant1
"""

import os
import re
import glob
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup
from monai.metrics import DiceMetric
from tqdm import tqdm

DATA_ROOT = os.environ.get("MLMT_DATA", "./data")
CKPT_ROOT = os.environ.get("MLMT_CHECKPOINTS", "./checkpoints")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Dataset

class KneeDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        assert len(img_paths) == len(mask_paths), "image/mask count mismatch"
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        return img, mask.unsqueeze(0)


def build_transforms(img_size, mode):
    """
    Augmentation pipeline for training or validation.

    The geometric block is deliberately mild. Bone surface annotations are thin,
    near-horizontal structures, so large rotations produce configurations that do
    not occur in real freehand scans and push the model away from the prior that
    the surface runs roughly across the image. Vertical flips are kept at low
    probability for the same reason: depth in an ultrasound frame is not
    symmetric, since the probe is always at the top.

    The intensity block matters more than the geometric one here. Speckle,
    gain and depth settings vary between participants and between sessions, and
    that variation is the main thing the model has to survive at test time on an
    unseen subject.
    """
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.10,
                rotate_limit=15, border_mode=0, p=0.5,
            ),
            A.OneOf([
                A.ElasticTransform(alpha=80, sigma=8, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(0.2, 0.2, p=1.0),
                A.CLAHE(clip_limit=2.0, p=1.0),
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.5),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, fill_value=0, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# Data discovery and splitting

def collect_pairs(data_dir):
    img_dir = os.path.join(data_dir, "train", "images")
    mask_dir = os.path.join(data_dir, "train", "masks")

    imgs = sorted(glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True))
    if not imgs:
        raise FileNotFoundError("no PNG images found under " + img_dir)

    masks = []
    for p in imgs:
        rel = os.path.relpath(p, img_dir)
        m = os.path.join(mask_dir, rel)
        if not os.path.exists(m):
            raise FileNotFoundError("no mask for image " + rel)
        masks.append(m)

    return imgs, masks


def subject_of(path, pattern):
    key = os.path.relpath(path).replace(os.sep, "/")
    m = re.search(pattern, key)
    return m.group(1) if m else None


def subject_aware_split(imgs, masks, val_subject, pattern):
    """
    Hold out every frame belonging to one participant.

    Consecutive frames from a single sweep are near duplicates of each other, so
    a random split places almost every validation frame within a few millimetres
    of a training frame. The resulting score measures interpolation within a
    subject rather than transfer to a new one, and in our runs it overstated Dice
    by roughly eleven points. Splitting on the participant is the only variant of
    this split that answers the question the application actually poses, which is
    how the model behaves on a patient it has never seen.
    """
    trn_i, trn_m, val_i, val_m = [], [], [], []
    unmatched = 0

    for ip, mp in zip(imgs, masks):
        s = subject_of(ip, pattern)
        if s is None:
            unmatched += 1
        if s == val_subject:
            val_i.append(ip)
            val_m.append(mp)
        else:
            trn_i.append(ip)
            trn_m.append(mp)

    if unmatched:
        raise ValueError(
            "subject pattern %r did not match %d of %d paths; pass --subject_regex"
            % (pattern, unmatched, len(imgs))
        )
    if not val_i:
        found = sorted({subject_of(p, pattern) for p in imgs})
        raise ValueError("no frames for %r; found %s" % (val_subject, found))

    return trn_i, trn_m, val_i, val_m


def random_split(imgs, masks, val_fraction, rng):
    idx = list(range(len(imgs)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx, trn_idx = idx[:n_val], idx[n_val:]
    return (
        [imgs[i] for i in trn_idx], [masks[i] for i in trn_idx],
        [imgs[i] for i in val_idx], [masks[i] for i in val_idx],
    )


# Loss

class DiceCELoss(nn.Module):
    """
    Weighted sum of cross entropy and soft Dice on 2-class logits.

    The bone surface occupies well under one percent of a frame. Dice alone is
    unstable at that ratio, because a handful of pixels moving across the
    threshold swings the gradient, and it gives no signal at all on frames where
    the surface is absent. Cross entropy is stable there but is dominated by
    background and will happily settle on predicting nothing. Weighting Dice
    above cross entropy at 0.6 to 0.4 keeps the overlap term in charge of the
    optimisation while cross entropy supplies a dense gradient early in training.
    """

    def __init__(self, smooth=1e-5, ce_weight=0.4):
        super().__init__()
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets.squeeze(1).long())

        probs = torch.softmax(logits, dim=1)[:, 1:2]
        inter = (probs * targets).sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        )
        return self.ce_weight * ce_loss + (1.0 - self.ce_weight) * dice_loss.mean()


# Model

def build_model(model_name):
    return SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "background", 1: "bone"},
        label2id={"background": 0, "bone": 1},
        ignore_mismatched_sizes=True,
    )


# Training

def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, amp):
    model.train()
    running = 0.0

    for imgs, masks in tqdm(loader, desc="train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=amp):
            logits = model(pixel_values=imgs).logits
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item()

    return running / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, amp):
    model.eval()
    running = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    for imgs, masks in tqdm(loader, desc="val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)

        with autocast(device.type, enabled=amp):
            logits = model(pixel_values=imgs).logits
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)

        running += loss.item()
        preds = (torch.softmax(logits.float(), 1)[:, 1:2] > 0.5).float()
        dice_metric(y_pred=preds, y=masks)

    return running / len(loader), dice_metric.aggregate().item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=DATA_ROOT,
                   help="root containing train/images and train/masks")
    p.add_argument("--output_dir", type=str, default=CKPT_ROOT)
    p.add_argument("--model_name", type=str, default="nvidia/mit-b2",
                   help="HuggingFace encoder id, mit-b0 through mit-b5")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--ce_weight", type=float, default=0.4,
                   help="weight of CE in DiceCE; Dice weight is 1 - ce_weight")
    p.add_argument("--split_mode", choices=["subject", "random"], default="subject")
    p.add_argument("--val_subject", type=str, default="participant1",
                   help="participant held out when --split_mode subject")
    p.add_argument("--subject_regex", type=str, default=r"(participant\d+)",
                   help="regex whose first group is the participant id in the path")
    p.add_argument("--val_split", type=float, default=0.15,
                   help="validation fraction when --split_mode random")
    p.add_argument("--patience", type=int, default=20,
                   help="early stopping patience in epochs")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    log.info("device=%s model=%s img_size=%d amp=%s",
             device, args.model_name, args.img_size, amp)

    imgs, masks = collect_pairs(args.data_dir)
    log.info("found %d image/mask pairs", len(imgs))

    if args.split_mode == "subject":
        trn_i, trn_m, val_i, val_m = subject_aware_split(
            imgs, masks, args.val_subject, args.subject_regex)
        log.info("subject-aware split, holding out %s", args.val_subject)
    else:
        trn_i, trn_m, val_i, val_m = random_split(imgs, masks, args.val_split, rng)
        log.warning("random split in use; validation Dice will be optimistic")

    log.info("train=%d val=%d", len(trn_i), len(val_i))

    train_ds = KneeDataset(trn_i, trn_m, build_transforms(args.img_size, "train"))
    val_ds = KneeDataset(val_i, val_m, build_transforms(args.img_size, "val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model_name).to(device)
    criterion = DiceCELoss(ce_weight=args.ce_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = GradScaler(device.type, enabled=amp)

    best_dice = 0.0
    no_improve = 0
    ckpt_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        trn_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                   scaler, scheduler, device, amp)
        val_loss, val_dice = validate(model, val_loader, criterion, device, amp)

        log.info("epoch %03d/%d | train %.4f | val %.4f | dice %.4f",
                 epoch, args.epochs, trn_loss, val_loss, val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "args": vars(args),
            }, ckpt_path)
            log.info("saved %s (dice %.4f)", ckpt_path, best_dice)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info("early stop after %d epochs without improvement", args.patience)
                break

    log.info("done, best val dice %.4f", best_dice)


if __name__ == "__main__":
    main()
