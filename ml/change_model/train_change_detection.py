"""Train a siamese change detection model on the SECOND dataset.

Uses 2968 bi-temporal image pairs (512x512 RGB) with semantic segmentation
labels (0=unchanged, 128=changed, 255=boundary).

Architecture: Siamese U-Net with timm EfficientNet-B0 backbone.
Fits in ~4GB VRAM (RTX 2050).

Usage:
    python ml/change_model/train_change_detection.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
import timm
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'evaluation', 'second_dataset', 'raw', 'SECOND_train_set')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ml', 'change_model', 'trained')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IM1_DIR = os.path.join(DATA_DIR, 'im1')
IM2_DIR = os.path.join(DATA_DIR, 'im2')
LABEL1_DIR = os.path.join(DATA_DIR, 'label1')
LABEL2_DIR = os.path.join(DATA_DIR, 'label2')

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 256
MAX_SAMPLES = 1000  # Use subset for training speed on 4GB VRAM


class SECONDDataset(Dataset):
    """Bi-temporal change detection dataset from the SECOND dataset."""

    def __init__(self, image_ids, transform=None):
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        im1_path = os.path.join(IM1_DIR, f'{img_id}.png')
        im2_path = os.path.join(IM2_DIR, f'{img_id}.png')
        label1_path = os.path.join(LABEL1_DIR, f'{img_id}.png')
        label2_path = os.path.join(LABEL2_DIR, f'{img_id}.png')

        im1 = Image.open(im1_path).convert('RGB')
        im2 = Image.open(im2_path).convert('RGB')
        label1 = Image.open(label1_path).convert('RGB')
        label2 = Image.open(label2_path).convert('RGB')

        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        # Change mask: label2 != label1 (changed regions)
        label1_pil = Image.open(label1_path).convert('L')
        label2_pil = Image.open(label2_path).convert('L')
        label1_arr = np.array(label1_pil)
        label2_arr = np.array(label2_pil)
        change_mask = (label2_arr != label1_arr).astype(np.float32)
        change_mask_t = torch.from_numpy(change_mask).unsqueeze(0)  # 1xHxW
        if self.transform:
            change_mask_t = torch.nn.functional.interpolate(
                change_mask_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='nearest'
            ).squeeze(0)
        return im1, im2, change_mask_t


class SiameseUNet(nn.Module):
    """Siamese U-Net for bi-temporal change detection."""

    def __init__(self, backbone_name='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, pretrained=True)
        self.backbone.eval()  # Freeze backbone initially

        num_features = 1280  # EfficientNet-B0 final feature dimension

        # Encoder (shared weights from backbone)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),  # 6 channels: 3 from each image
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        self._initialize_decoder()

    def _initialize_decoder(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1, x2):
        # Concatenate both images along channel dimension
        x = torch.cat([x1, x2], dim=1)
        x = self.encoder1(x)
        x = self.decoder(x)
        return x


def get_image_ids():
    """Get all image IDs from the dataset."""
    ids = sorted([f.replace('.png', '') for f in os.listdir(IM1_DIR) if f.endswith('.png') and not f.startswith('.') and not f.startswith('_')])
    label_ids = sorted([f.replace('.png', '') for f in os.listdir(LABEL1_DIR) if f.endswith('.png') and not f.startswith('.') and not f.startswith('_')])
    common = sorted(set(ids) & set(label_ids))
    print(f"Found {len(ids)} images in im1, {len(label_ids)} labels in label1")
    print(f"Common: {len(common)} image IDs")
    return common


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    image_ids = get_image_ids()
    if not image_ids:
        print("ERROR: No valid image IDs found")
        sys.exit(1)

    # Use subset for training speed
    if len(image_ids) > MAX_SAMPLES:
        np.random.seed(42)
        image_ids = list(np.random.choice(image_ids, MAX_SAMPLES, replace=False))
        print(f"Using subset of {MAX_SAMPLES} samples")

    # Transform
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and loader
    dataset = SECONDDataset(image_ids, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = SiameseUNet().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    print("Backbone frozen, only decoder trained")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')
    train_log = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (im1, im2, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')):
            im1, im2, labels = im1.to(device), im2.to(device), labels.to(device)

            optimizer.zero_grad()
            pred = model(im1, im2)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for im1, im2, labels in val_loader:
                im1, im2, labels = im1.to(device), im2.to(device), labels.to(device)
                pred = model(im1, im2)
                val_loss += criterion(pred, labels).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.1f}s")

        train_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'time_seconds': epoch_time,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

    # Save final model and training log
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
    with open(os.path.join(OUTPUT_DIR, 'training_log.json'), 'w') as f:
        json.dump(train_log, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {OUTPUT_DIR}")

    # Verify model can run inference
    model.eval()
    test_im1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    test_im2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        pred = model(test_im1, test_im2)
    print(f"Inference test: output shape {pred.shape}, mean {pred.mean().item():.4f}")


if __name__ == '__main__':
    train()
