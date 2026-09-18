"""Prepare BigEarthNet VQA training data and train LoRA adapter.

This script builds a Q&A dataset from BigEarthNet imagery and labels,
then fine-tunes the SmolVLM-256M-Instruct base model with LoRA.

NOTE: BigEarthNet official labels were not available for download
from any tested source (bigearth.net 404, HuggingFace 401/unauthorized).
The testing/ folder patches have no label files.

This script will:
1. Use any available BigEarthNet metadata CSV if found
2. Fall back to creating a minimal training setup using the
   existing adapter as initialization
3. Train with whatever labeled data is available

Usage:
    python ml/adaptation/train_lora.py [--data-dir PATH] [--epochs N]
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, 'ml', 'vqa-worker'))
from model_provider import DecoderShim, _read_lora_config

MODEL_DIR = os.environ.get('MODEL_DIR', 'HuggingFaceTB/SmolVLM-256M-Instruct')
ADAPTER_DIR = os.path.join(BASE_DIR, 'ml', 'smolvlm', 'lora_stage3')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ml', 'adaptation', 'trained')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to find BigEarthNet metadata
BIGEARTHNET_METADATA = os.path.join(BASE_DIR, 'ml', 'adaptation', 'bigearthnet_metadata.csv')


class BigEarthNetVQADataset(Dataset):
    """BigEarthNet VQA dataset built from image-label pairs."""

    def __init__(self, image_paths, questions, processor, max_length=256):
        self.image_paths = image_paths
        self.questions = questions
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        question = self.questions[idx]

        image = Image.open(image_path).convert('RGB')
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'question': question,
        }


def load_bigearthnet_metadata():
    """Try to load BigEarthNet metadata CSV."""
    if not os.path.isfile(BIGEARTHNET_METADATA):
        print(f"BigEarthNet metadata not found at {BIGEARTHNET_METADATA}")
        return None

    print(f"Loading BigEarthNet metadata from {BIGEARTHNET_METADATA}")
    import csv
    rows = []
    with open(BIGEARTHNET_METADATA) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} BigEarthNet entries")
    return rows


def build_vqa_dataset_from_metadata(metadata_rows, testing_dir, processor):
    """Build Q&A pairs from BigEarthNet metadata."""
    image_paths = []
    questions = []

    presence_questions = [
        "Is there water present in the image?",
        "Is there vegetation in the image?",
        "Is there a forest in the image?",
        "Is there a city in the image?",
        "Is there a road in the image?",
        "Is there water present?",
        "Are there trees?",
        "Is there any human activity?",
    ]
    count_questions = [
        "How many buildings are visible?",
        "How many roads are visible?",
        "How many water bodies are present?",
    ]

    for row in metadata_rows:
        image_path = row.get('image_path', row.get('file_path', None))
        if not image_path:
            continue
        full_path = os.path.join(testing_dir, image_path)
        if not os.path.isfile(full_path):
            continue

        # Determine land cover class from metadata
        land_cover = row.get('land_cover', row.get('label', ''))
        if land_cover:
            questions.append(f"Is there {land_cover} in the image?")
            image_paths.append(full_path)

        # Add presence questions for diversity
        for q in presence_questions[:2]:
            questions.append(q)
            image_paths.append(full_path)

    # If we have enough data, use it; otherwise note the limitation
    if len(image_paths) > 0:
        print(f"Built VQA dataset with {len(image_paths)} image-question pairs")
        return BigEarthNetVQADataset(image_paths, questions, processor)
    else:
        print("Could not build VQA dataset from metadata - no valid image-label pairs found")
        return None


def load_adapter_weights(model, adapter_dir):
    """Load the existing Stage-3 LoRA adapter weights into the model."""
    import json

    adapter_config_path = os.path.join(adapter_dir, 'adapter_config.json')
    if os.path.isfile(adapter_config_path):
        with open(adapter_config_path) as f:
            cfg = json.load(f)
        lora_cfg = LoraConfig(
            r=int(cfg.get('r', 8)),
            lora_alpha=int(cfg.get('lora_alpha', 16)),
            target_modules=cfg.get('target_modules') or [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ],
            lora_dropout=float(cfg.get('lora_dropout', 0.1)),
            bias='none',
            task_type='CAUSAL_LM',
        )
    else:
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.1, bias='none', task_type='CAUSAL_LM',
        )

    shim = DecoderShim(model.model.text_model, model.lm_head)
    peft_lm = get_peft_model(shim, lora_cfg)

    adapter_weights = os.path.join(adapter_dir, 'adapter_model.safetensors')
    if os.path.isfile(adapter_weights):
        try:
            peft_lm.load_adapter(adapter_dir, adapter_name='default')
            print("Loaded existing Stage-3 adapter weights")
        except Exception as e:
            print(f"Could not load adapter weights: {e}")
            return model, None

    model.model.text_model = peft_lm
    return model, lora_cfg


def train_lora(model, train_dataset, adapter_dir, epochs=3, batch_size=2, lr=1e-4):
    """Fine-tune the LoRA adapter."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    model = model.to(device)
    model.train()

    # Only train LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' not in name and 'adapter' not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f'LoRA Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            loss = criterion(outputs.logits.view(-1, outputs.logits.shape[-1]), input_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Save the trained adapter
    adapter_path = os.path.join(OUTPUT_DIR, 'lora_stage4')
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    with open(os.path.join(adapter_path, 'training_log.json'), 'w') as f:
        json.dump({'epochs': epochs, 'loss': avg_loss}, f, indent=2)
    print(f"Trained adapter saved to {adapter_path}")
    return adapter_path


def collate_fn(batch):
    """Collate function for VQA dataset."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b['input_ids'] for b in batch], batch_first=True, padding_value=0
    )
    pixel_values = torch.nn.utils.rnn.pad_sequence(
        [b['pixel_values'] for b in batch], batch_first=True, padding_value=0
    )
    return {'input_ids': input_ids, 'pixel_values': pixel_values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None, help='Path to BigEarthNet data')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    testing_dir = os.path.join(BASE_DIR, 'testing')

    # Load processor
    print(f"Loading processor from {MODEL_DIR}...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    # Load base model
    print(f"Loading base model {MODEL_DIR}...")
    model = Idefics3ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.float32, device_map='auto'
    )

    # Try to load existing adapter
    model, lora_cfg = load_adapter_weights(model, ADAPTER_DIR)

    # Try to load BigEarthNet metadata
    metadata = load_bigearthnet_metadata()
    train_dataset = None

    if metadata and os.path.isdir(testing_dir):
        train_dataset = build_vqa_dataset_from_metadata(metadata, testing_dir, processor)

    if train_dataset is None:
        print("\nNo labeled BigEarthNet data available for training.")
        print("The existing Stage-3 adapter is already loaded as initialization.")
        print("To train a new LoRA adapter, BigEarthNet labels are needed.")
        print("See ml/adaptation/MODEL_CARD.md for details.")
        print("\nSaving existing adapter as the current best model...")
        adapter_path = os.path.join(OUTPUT_DIR, 'lora_stage4')
        os.makedirs(adapter_path, exist_ok=True)
        model.save_pretrained(adapter_path)
        with open(os.path.join(adapter_path, 'training_log.json'), 'w') as f:
            json.dump({'note': 'No labeled training data available; saved existing adapter',
                       'data_availability': 'unknown'}, f, indent=2)
        print(f"Adapter saved to {adapter_path}")
        return

    # Train
    train_lora(model, train_dataset, ADAPTER_DIR, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == '__main__':
    main()
