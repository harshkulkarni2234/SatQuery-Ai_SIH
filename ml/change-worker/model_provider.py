"""Inference provider for the local change detection worker.

Loads the trained Siamese CNN (best_change_model.pt) once and runs
bi-temporal change detection on before/after image pairs.
"""

import os
import time

import numpy as np
import torch
from torch import nn
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "change_model", "trained")
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(MODEL_DIR, "best_change_model.pt")
MODEL_VERSION = "siamese-cnn-v1"
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SiameseEncoder(nn.Module):
    """Shared-weight Siamese CNN encoder."""

    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
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
    """Concatenates |f1-f2|, f1, f2, f1*f2 and fuses them."""

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
    """U-style decoder upsampling to a single-channel logit map."""

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


class ChangeModelProvider:
    """Loads best_change_model.pt and runs inference on image pairs."""

    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = None
        self.model_loaded = False

    def load(self):
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        self.model = SiameseChangeDetector().to(self.device)
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_loaded = True
        return self

    def _preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device).to(self.dtype)

    def detect(self, image_path_before, image_path_after):
        """Run change detection on a before/after image pair.

        Returns dict with mask (np.ndarray uint8 0/1), model_version, latency_ms.
        """
        if self.model is None or not self.model_loaded:
            self.load()

        t0 = time.time()
        im1 = self._preprocess(image_path_before)
        im2 = self._preprocess(image_path_after)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(im1, im2)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            else:
                logits = self.model(im1, im2)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        mask = (prob > 0.5).astype(np.uint8)
        latency_ms = int((time.time() - t0) * 1000)

        return {
            "mask": mask,
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms,
        }


_provider = None


def get_provider():
    global _provider
    if _provider is None:
        _provider = ChangeModelProvider()
        _provider.load()
    return _provider
