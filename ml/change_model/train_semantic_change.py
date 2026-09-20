"""Train a per-class semantic change detection model on the SECOND-derived dataset.

Architecture: shared Siamese CNN encoder (two parallel branches) ->
feature difference & fusion module -> two per-pixel class heads
(logits for date-1 and date-2 semantic labels).

Trained with fp16 mixed precision on RTX 2050, validated with per-class
IoU and derived binary change F1/IoU.

Data: ml/change_model/trained/data/{train,val}_{im1,im2,sem1,sem2}.npy
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.metrics.spatial_metrics import pixel_mask_f1_iou  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "trained", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "trained")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
IMG_SIZE = 256
NUM_CLASSES = 7
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = [
    (255, 255, 255), (0, 0, 255), (128, 128, 128),
    (0, 128, 0), (0, 255, 0), (128, 0, 0), (255, 0, 0),
]
CLASS_NAMES = [
    "unchanged", "water", "NVG_surface", "low_vegetation",
    "trees", "buildings", "playgrounds",
]

COLOR_TO_IDX = {(r, g, b): i for i, (r, g, b) in enumerate(COLORS)}


class SemanticDataset(Dataset):
    """Loads pre-computed .npy arrays for semantic change detection."""

    def __init__(self, split: str):
        self.split = split
        names = json.load(open(os.path.join(DATA_DIR, f"{split}_names.json")))
        self.names = names
        n = len(names)
        self.im1 = np.fromfile(os.path.join(DATA_DIR, f"{split}_im1.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        self.im2 = np.fromfile(os.path.join(DATA_DIR, f"{split}_im2.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        self.sem1 = np.fromfile(os.path.join(DATA_DIR, f"{split}_sem1.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE).astype(np.int64)
        self.sem2 = np.fromfile(os.path.join(DATA_DIR, f"{split}_sem2.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE).astype(np.int64)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        a = self.im1[idx] / 255.0
        b = self.im2[idx] / 255.0
        a = (a - self.mean) / self.std
        b = (b - self.mean) / self.std
        im1_t = torch.from_numpy(a).permute(2, 0, 1).contiguous()
        im2_t = torch.from_numpy(b).permute(2, 0, 1).contiguous()
        sem1_t = torch.from_numpy(self.sem1[idx]).long()
        sem2_t = torch.from_numpy(self.sem2[idx]).long()
        return im1_t, im2_t, sem1_t, sem2_t


class SiameseEncoder(nn.Module):
    """Shared-weight Siamese CNN encoder."""

    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.encoder = nn.Sequential(
            conv_block(3, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            nn.MaxPool2d(2),
            conv_block(128, 256),
            nn.MaxPool2d(2),
            conv_block(256, 512),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.encoder(x)


class FeatureDiffusionFusion(nn.Module):
    """Computes per-channel feature difference and fuses the two branch features."""

    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        x = torch.cat([diff, f1, f2, prod], dim=1)
        return self.fusion(x)


class SemanticChangeDecoder(nn.Module):
    """Two per-pixel class heads: one for date-1, one for date-2."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head1 = nn.Conv2d(32, num_classes, 1)
        self.head2 = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        f = self.decoder(x)
        logits1 = self.head1(f)
        logits2 = self.head2(f)
        return logits1, logits2


class SemanticChangeDetector(nn.Module):
    """Full model: Siamese encoder -> fusion -> dual class heads."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = SiameseEncoder()
        self.fusion = FeatureDiffusionFusion(512)
        self.decoder = SemanticChangeDecoder(512, num_classes)

    def forward(self, im1, im2):
        f1 = self.encoder(im1)
        f2 = self.encoder(im2)
        fused = self.fusion(f1, f2)
        logits1, logits2 = self.decoder(fused)
        return logits1, logits2


def compute_semantic_iou(pred_logits, gt_labels, num_classes=NUM_CLASSES):
    """Compute per-class IoU from logits and ground-truth labels."""
    pred = pred_logits.argmax(dim=1).cpu().numpy()
    gt = gt_labels.cpu().numpy()
    ious = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (gt == c))
        fp = np.sum((pred == c) & (gt != c))
        fn = np.sum((pred != c) & (gt == c))
        union = tp + fp + fn
        iou = tp / union if union > 0 else None
        ious.append(iou)
    return ious


def compute_miou(ious):
    """Mean IoU, ignoring None values."""
    valid = [i for i in ious if i is not None]
    return float(np.mean(valid)) if valid else None


def compute_binary_change_metrics(pred_sem1, pred_sem2, gt_sem1, gt_sem2):
    """Derive binary change metrics from predicted and ground-truth semantic labels."""
    pred_change = (pred_sem1 != pred_sem2).astype(bool)
    gt_change = (gt_sem1 != gt_sem2).astype(bool)
    result = pixel_mask_f1_iou(pred_change.astype(np.float32), gt_change.astype(np.float32))
    return result


def train():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = SemanticDataset("train")
    val_ds = SemanticDataset("val")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = SemanticChangeDetector().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    class_counts = np.array([89025858, 344918, 9499830, 7137163, 1732244, 3598759, 72428], dtype=np.float32)
    total = class_counts.sum()
    class_weights = total / (NUM_CLASSES * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler = GradScaler('cuda')

    best_miou = -1.0
    train_log = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (im1, im2, sem1, sem2) in enumerate(train_loader):
            im1, im2, sem1, sem2 = im1.to(DEVICE), im2.to(DEVICE), sem1.to(DEVICE), sem2.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                logits1, logits2 = model(im1, im2)
                loss1 = criterion(logits1, sem1)
                loss2 = criterion(logits2, sem2)
                loss = loss1 + loss2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        model.eval()
        val_ious_sem1 = []
        val_ious_sem2 = []
        val_binary_results = []
        val_loss_total = 0.0

        with torch.no_grad():
            for im1, im2, sem1, sem2 in val_loader:
                im1, im2, sem1, sem2 = im1.to(DEVICE), im2.to(DEVICE), sem1.to(DEVICE), sem2.to(DEVICE)
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits1, logits2 = model(im1, im2)
                    loss1 = criterion(logits1, sem1)
                    loss2 = criterion(logits2, sem2)
                    val_loss_total += (loss1.item() + loss2.item())

                ious1 = compute_semantic_iou(logits1, sem1)
                ious2 = compute_semantic_iou(logits2, sem2)
                val_ious_sem1.append(ious1)
                val_ious_sem2.append(ious2)

                pred_sem1 = logits1.argmax(dim=1).cpu().numpy()
                pred_sem2 = logits2.argmax(dim=1).cpu().numpy()
                for i in range(len(sem1)):
                    br = compute_binary_change_metrics(pred_sem1[i], pred_sem2[i], sem1[i].cpu().numpy(), sem2[i].cpu().numpy())
                    val_binary_results.append(br)

        avg_val_loss = val_loss_total / len(val_loader)
        miou_sem1 = compute_miou([iou[0] for iou in val_ious_sem1])
        miou_sem2 = compute_miou([iou[0] for iou in val_ious_sem2])
        mean_miou = (miou_sem1 + miou_sem2) / 2 if miou_sem1 is not None and miou_sem2 is not None else None

        all_binary_f1 = [r["f1"] for r in val_binary_results if r["f1"] is not None]
        all_binary_iou = [r["iou"] for r in val_binary_results if r["iou"] is not None]
        binary_f1 = float(np.mean(all_binary_f1)) if all_binary_f1 else None
        binary_iou = float(np.mean(all_binary_iou)) if all_binary_iou else None

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"mIoU(sem1): {miou_sem1:.4f}, mIoU(sem2): {miou_sem2:.4f}, "
              f"mean mIoU: {mean_miou:.4f}, "
              f"Binary F1: {binary_f1:.4f}, Binary IoU: {binary_iou:.4f}, "
              f"Time: {epoch_time:.1f}s")

        train_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "miou_sem1": miou_sem1,
            "miou_sem2": miou_sem2,
            "mean_miou": mean_miou,
            "binary_f1": binary_f1,
            "binary_iou": binary_iou,
            "time_seconds": epoch_time,
        })

        if mean_miou is not None and mean_miou > best_miou:
            best_miou = mean_miou
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_semantic_change_model.pt"))
            print(f"  -> Saved best model (mean mIoU={mean_miou:.4f}) to {OUTPUT_DIR}/best_semantic_change_model.pt")

    with open(os.path.join(OUTPUT_DIR, "semantic_training_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"\nTraining complete! Best mean mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    train()
