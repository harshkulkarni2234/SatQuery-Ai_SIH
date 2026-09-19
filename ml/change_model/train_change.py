"""Train a small Siamese CNN for binary change detection on the SECOND-derived dataset.

Architecture: shared Siamese encoder (two parallel CNN branches) ->
feature difference & fusion module -> decoder for binary change mask.
Trained with fp16 mixed precision on RTX 2050, validated with
evaluation.metrics.spatial_metrics.pixel_mask_f1_iou.

Data: ml/change_model/trained/data/{train,val}_{im1,im2,mask,sem1,sem2}.npy
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChangeDataset(Dataset):
    """Loads pre-computed .npy arrays for bi-temporal change detection."""

    def __init__(self, split: str):
        self.split = split
        names = json.load(open(os.path.join(DATA_DIR, f"{split}_names.json")))
        self.names = names

        n = len(names)
        self.im1 = np.fromfile(os.path.join(DATA_DIR, f"{split}_im1.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        self.im2 = np.fromfile(os.path.join(DATA_DIR, f"{split}_im2.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        self.mask = np.fromfile(os.path.join(DATA_DIR, f"{split}_mask.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE).astype(np.float32)
        self.sem1 = np.fromfile(os.path.join(DATA_DIR, f"{split}_sem1.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE).astype(np.float32)
        self.sem2 = np.fromfile(os.path.join(DATA_DIR, f"{split}_sem2.npy"), dtype=np.uint8).reshape(n, IMG_SIZE, IMG_SIZE).astype(np.float32)

        # Normalize im1/im2 to [0, 1] then standardize
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # im1/im2: HWC uint8 -> CHW float32 normalized
        a = self.im1[idx] / 255.0
        b = self.im2[idx] / 255.0
        a = (a - self.mean) / self.std
        b = (b - self.mean) / self.std
        im1_t = torch.from_numpy(a).permute(2, 0, 1).contiguous()
        im2_t = torch.from_numpy(b).permute(2, 0, 1).contiguous()

        mask_t = torch.from_numpy(self.mask[idx])
        sem1_t = torch.from_numpy(self.sem1[idx])
        sem2_t = torch.from_numpy(self.sem2[idx])

        return im1_t, im2_t, mask_t, sem1_t, sem2_t


class SiameseEncoder(nn.Module):
    """Shared-weight Siamese CNN encoder that processes both temporal images."""

    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # Shared encoder: 3 -> 64 -> 128 -> 256 -> 512
        self.encoder = nn.Sequential(
            conv_block(3, 64),          # 256x256
            nn.MaxPool2d(2),            # 128x128
            conv_block(64, 128),
            nn.MaxPool2d(2),            # 64x64
            conv_block(128, 256),
            nn.MaxPool2d(2),            # 32x32
            conv_block(256, 512),
            nn.MaxPool2d(2),            # 16x16
        )

    def forward(self, x):
        return self.encoder(x)


class FeatureDiffusionFusion(nn.Module):
    """Computes per-channel feature difference and fuses the two branch features.

    Takes the two encoded feature maps and produces a single fused tensor
    by concatenating |f1 - f2|, f1, f2, and elementwise product f1 * f2.
    """

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


class ChangeDecoder(nn.Module):
    """U-style decoder that upsamples the fused feature map to a binary mask."""

    def __init__(self, in_channels=512):
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
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.decoder(x)


class SiameseChangeDetector(nn.Module):
    """Full model: Siamese encoder -> feature diffusion/fusion -> decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = SiameseEncoder()
        self.fusion = FeatureDiffusionFusion(512)
        self.decoder = ChangeDecoder(512)

    def forward(self, im1, im2):
        f1 = self.encoder(im1)
        f2 = self.encoder(im2)
        fused = self.fusion(f1, f2)
        mask = self.decoder(fused)
        return mask


def compute_metrics(pred_masks, gt_masks):
    all_results = []
    for pred, gt in zip(pred_masks, gt_masks):
        pred_prob = torch.sigmoid(torch.from_numpy(pred)).numpy()
        pred_np = (pred_prob.squeeze() > 0.5).astype(np.float32)
        gt_np = gt.squeeze().astype(np.float32)
        result = pixel_mask_f1_iou(pred_np, gt_np)
        all_results.append(result)
    return all_results


def train():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets
    train_ds = ChangeDataset("train")
    val_ds = ChangeDataset("val")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model
    model = SiameseChangeDetector().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0.0
    train_log = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (im1, im2, masks, sem1, sem2) in enumerate(train_loader):
            im1, im2, masks = im1.to(DEVICE), im2.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                pred = model(im1, im2)
                loss = criterion(pred, masks.unsqueeze(1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Validation with F1/IoU
        model.eval()
        val_results = []
        val_loss = 0.0
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for im1, im2, masks, sem1, sem2 in val_loader:
                im1, im2, masks = im1.to(DEVICE), im2.to(DEVICE), masks.to(DEVICE)
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(im1, im2)
                    val_loss += criterion(pred, masks.unsqueeze(1)).item()

                all_preds.append(pred.cpu())
                all_gts.append(masks.cpu())

        avg_val_loss = val_loss / len(val_loader)

        # Compute pixel-level F1/IoU
        pred_stack = torch.cat(all_preds, dim=0).numpy()
        gt_stack = torch.cat(all_gts, dim=0).numpy()
        metrics = compute_metrics(pred_stack, gt_stack)

        f1s = [m["f1"] for m in metrics if m["f1"] is not None]
        ious = [m["iou"] for m in metrics if m["iou"] is not None]
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        mean_iou = float(np.mean(ious)) if ious else 0.0

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, F1: {mean_f1:.4f}, "
              f"IoU: {mean_iou:.4f}, Time: {epoch_time:.1f}s")

        train_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "val_f1": mean_f1,
            "val_iou": mean_iou,
            "time_seconds": epoch_time,
        })

        # Save best model by F1 score
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_path = os.path.join(OUTPUT_DIR, "best_change_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model (F1={mean_f1:.4f}, IoU={mean_iou:.4f}) to {best_path}")

    # Save training log
    with open(os.path.join(OUTPUT_DIR, "change_training_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nTraining complete! Best val F1: {best_f1:.4f}")
    print(f"Best model saved to {os.path.join(OUTPUT_DIR, 'best_change_model.pt')}")


if __name__ == "__main__":
    train()
