"""Train a genuine BigEarthNet LoRA adapter on SmolVLM-256M-Instruct.

Replaces the mystery-provenance v1.0 adapter (ml/smolvlm/lora_stage3/)
with an adapter we actually trained on data we built ourselves:

  - Dataset: ml/adaptation/dataset/{train,val,test}.jsonl, produced by
    ml/adaptation/prepare_bigearthnet_vqa.py from REAL BigEarthNet v2.0
    labels (official metadata.parquet, CDLA-Permissive-1.0) matched to
    local testing/ imagery. See that script for the exact label ->
    question/answer mapping and the seeded 80/10/10 split.

  - Loss masking: prompt tokens get label -100; only the reference answer
    tokens are trained on (standard SFT, both for loss and for the eval:
    each step_log entry is computed over masked answer tokens only).

  - Evaluation on the val split is REAL exact-match accuracy against the
    ground-truth reference answer (presence: "yes"/"no" case-insensitive;
    count: exact integer string). No fabricated scores.

  - LoRA is applied through the SAME DecoderShim wrapper and SAME
    target_modules that model_provider.py uses at inference, and the
    adapter is saved with peft_lm.save_pretrained() + version.json in the
    exact shape model_provider.load() expects (adapter_config.json +
    adapter_model.safetensors). v1.0 stays untouched as fallback.

Config: ml/adaptation/configs/stage3.yaml (the `training_v2:` block).

Outputs:
  - ml/smolvlm/lora_stage3_v2/  adapter (adapter_config.json,
    adapter_model.safetensors, version.json, training_log.json,
    evaluation.json, README.md)
  - ml/adaptation/results/train_v2_log.json (loss curve + full record)

Usage:
    python ml/adaptation/train_lora.py [--config .../stage3.yaml]
        [--adapter-out ml/smolvlm/lora_stage3_v2]
        [--max-train-rows N] [--epochs N] [--batch-size N] [--grad-accum N]
        [--lr F] [--max-steps N] [--seed N] [--cpu] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml", "vqa-worker"))

from model_provider import DecoderShim, clean_gen  # noqa: E402

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "stage3.yaml")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "ml", "smolvlm", "lora_stage3_v2")
VERSION_STRING = "smolvlm256m-ben-lora-s3-v2.0"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


class VQADataset(Dataset):
    """JSONL rows -> processor inputs with answer-token labels (prompt masked to -100)."""

    def __init__(self, jsonl_path, processor, max_len=2048):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(os.path.join(PROJECT_ROOT, r["image_path"])).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["question"]}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[prompt], images=[img], return_tensors="pt")
        ans_ids = self.processor.tokenizer(r["answer"], add_special_tokens=False)["input_ids"]
        p_len = enc["input_ids"].shape[1]
        input_ids = torch.cat([enc["input_ids"][0], torch.tensor(ans_ids, dtype=torch.long)])
        labels = torch.cat([torch.full((p_len,), -100, dtype=torch.long),
                            torch.tensor(ans_ids, dtype=torch.long)])
        # cast the heavy vision tensor to fp16 on the spot: the host has only
        # ~7.7 GB RAM, and fp32 (17,3,512,512) tiles ~53MB/sample would OOM batches.
        return {
            "input_ids": input_ids[: self.max_len],
            "labels": labels[: self.max_len],
            "pixel_values": enc["pixel_values"][0].to(torch.float16),
        }


def collate(batch):
    pad_id = 0
    input_ids = nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels = nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    attn = (input_ids != pad_id).long()
    # processor returns per-sample pixel_values as (1, N_TILES, 3, H, W); the
    # Dataset already stripped the placeholder batch dim (N_TILES, 3, H, W), so
    # stacking samples gives the model's expected (B, N_TILES, 3, H, W).
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn, "pixel_values": pixel_values}


def eval_val(model, rows, processor, device, max_examples=200):
    """Real exact-match accuracy on the val split. Returns (summary, examples)."""
    per_type = Counter()
    per_type_correct = Counter()
    examples = []
    sample = rows if len(rows) <= max_examples else rows[:max_examples]
    with torch.no_grad():
        for r in sample:
            img = Image.open(os.path.join(PROJECT_ROOT, r["image_path"])).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["question"]}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
            out = model.generate(
                input_ids=enc["input_ids"], pixel_values=enc["pixel_values"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=8, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            ans = clean_gen(processor.tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                                      skip_special_tokens=False)).strip().lower()
            ref = r["answer"].strip().lower()
            qt = r["question_type"]
            per_type[qt] += 1
            if ans == ref:
                per_type_correct[qt] += 1
            examples.append({
                "image": os.path.basename(r["image_path"]),
                "question_type": qt,
                "question": r["question"],
                "reference": r["answer"],
                "generated": ans,
                "correct": ans == ref,
            })
    summary = {}
    for qt in sorted(per_type):
        acc = per_type_correct[qt] / per_type[qt] if per_type[qt] else 0.0
        summary[qt] = {"n": per_type[qt], "correct": per_type_correct[qt], "exact_match_accuracy": acc}
        print(f"  [{qt:16s}] {per_type_correct[qt]}/{per_type[qt]} = {acc:.4f}")
    return summary, examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--adapter-out", default=DEFAULT_OUT)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume from ml/adaptation/resume checkpoint")
    parser.add_argument("--smoke", action="store_true", help="tiny run to validate pipeline + VRAM")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t = cfg["training_v2"]
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")

    base_model = cfg["model"]["base_model"]
    model = Idefics3ForConditionalGeneration.from_pretrained(
        base_model, dtype=dtype, device_map={"": device}, attn_implementation="sdpa"
    )
    for p in model.parameters():
        p.requires_grad_(False)

    from peft import LoraConfig, get_peft_model

    lc = cfg["lora"]
    shim = DecoderShim(model.model.text_model, model.lm_head)
    peft_lm = get_peft_model(shim, LoraConfig(
        r=int(lc["r"]), lora_alpha=int(lc["lora_alpha"]),
        target_modules=list(lc["target_modules"]),
        lora_dropout=float(lc["lora_dropout"]), bias="none", task_type="CAUSAL_LM",
    ))
    model.model.text_model = peft_lm
    model.to(device)
    try:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled")
    except Exception as exc:
        print(f"gradient checkpointing unavailable: {exc}")

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_tr:,}")

    processor = AutoProcessor.from_pretrained(base_model)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    train_ds = VQADataset(os.path.join(data_dir, "train.jsonl"), processor)
    val_ds = VQADataset(os.path.join(data_dir, "val.jsonl"), processor)

    seed = args.seed if args.seed is not None else int(t["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    max_rows = args.max_train_rows if args.max_train_rows is not None else int(t["max_train_rows"])
    if max_rows and max_rows < len(train_ds.rows):
        train_ds.rows = rng.sample(train_ds.rows, max_rows)
    print(f"train rows: {len(train_ds.rows)} | val rows: {len(val_ds.rows)}")

    batch_size = args.batch_size if args.batch_size is not None else int(t["batch_size"])
    grad_accum = args.grad_accum if args.grad_accum is not None else int(t["gradient_accumulation_steps"])
    lr = args.lr if args.lr is not None else float(t["learning_rate"])
    epochs = args.epochs if args.epochs is not None else int(t["epochs"])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    effective = batch_size * grad_accum
    steps_per_epoch = max(1, math.ceil(len(train_ds) / effective))
    max_steps = args.max_steps if args.max_steps is not None else (epochs * steps_per_epoch)
    if args.smoke:
        max_steps = min(max_steps, 2)
    print(f"effective batch {effective} | steps/epoch {steps_per_epoch} | max_steps {max_steps}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    warmup = max(1, int(t.get("warmup_frac", 0.05) * max_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup + 1) / max(1, max_steps - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    t0 = time.time()
    step_log = []
    loss_sum = 0.0
    micro = 0
    global_step = 0
    ckpt_step = 50  # save a resumable adapter checkpoint every N accumulated steps

    # --- resume support: restore adapter weights + step counter + log ---
    resume_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resume")
    if args.resume and os.path.isdir(resume_dir):
        resume_meta = os.path.join(resume_dir, "resume.json")
        if os.path.isfile(resume_meta):
            with open(resume_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            step_log = meta.get("step_log", [])
            prev_steps = meta.get("global_step", 0)
            peft_lm.load_adapter(resume_dir, adapter_name="default")
            global_step = prev_steps
            print(f"Resumed: {prev_steps} prior steps")
        else:
            print("WARNING: --resume given but no resume.json found; starting fresh")

    model.train()
    for epoch in range(epochs):
        if global_step >= max_steps:
            break
        for batch in train_dl:
            if global_step >= max_steps:
                break
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = out.loss / grad_accum
            loss.backward()
            loss_sum += out.loss.item()
            micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                step_log.append({
                    "step": global_step,
                    "loss": loss_sum / grad_accum,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_s": round(time.time() - t0, 1),
                })
                loss_sum = 0.0
                micro = 0
                global_step += 1
                if global_step % 20 == 0 or global_step == max_steps:
                    print(f"  step {global_step}/{max_steps} loss {step_log[-1]['loss']:.4f} "
                          f"({time.time() - t0:.0f}s)")
                # periodic memory hygiene: prevents the slow host-memory
                # fragmentation that killed the first run at step ~240
                if global_step % 10 == 0:
                    torch.cuda.empty_cache()
                if global_step % ckpt_step == 0:
                    os.makedirs(resume_dir, exist_ok=True)
                    peft_lm.save_pretrained(resume_dir)
                    with open(os.path.join(resume_dir, "resume.json"), "w", encoding="utf-8") as f:
                        json.dump({"global_step": global_step, "step_log": step_log}, f)
                    print(f"    checkpoint saved to {resume_dir} @ step {global_step}")

    model.eval()
    print(f"\nVal exact-match accuracy (on up to 200 rows):")
    eval_summary, eval_examples = eval_val(model, val_ds.rows, processor, device)

    os.makedirs(args.adapter_out, exist_ok=True)
    # Save ONLY the PEFT adapter (adapter_config.json + adapter_model.safetensors)
    # -> exactly the shape model_provider.load() expects. Do NOT save the whole
    # Idefics3 model here.
    peft_lm.save_pretrained(args.adapter_out)
    with open(os.path.join(args.adapter_out, "version.json"), "w", encoding="utf-8") as f:
        json.dump({
            "version": VERSION_STRING,
            "model_card": "ml/adaptation/MODEL_CARD.md",
            "note": "Real adapter trained from real BigEarthNet v2.0 labels subset. "
                    "v1.0 (mystery provenance) remains at ml/smolvlm/lora_stage3/ as fallback.",
        }, f, indent=2)

    record = {
        "version": VERSION_STRING,
        "base_model": base_model,
        "lora": {"r": int(lc["r"]), "lora_alpha": int(lc["lora_alpha"]),
                 "lora_dropout": float(lc["lora_dropout"]), "target_modules": list(lc["target_modules"]),
                 "bias": "none"},
        "dataset": {
            "source": "ml/adaptation/dataset/{train,val,test}.jsonl (prepare_bigearthnet_vqa.py)",
            "train_rows": len(train_ds.rows),
            "val_rows": len(val_ds.rows),
            "test_rows": 2546,
            "question_types": ["presence", "count"],
        },
        "training": {
            "learning_rate": lr, "epochs": epochs, "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum, "effective_batch_size": effective,
            "max_steps": max_steps, "seed": seed, "precision": str(dtype),
            "gradient_checkpointing": True, "image_size": "120x120 (native BigEarthNet patch)",
            "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
            "wall_clock_s": round(time.time() - t0, 1),
        },
        "step_log": step_log,
        "eval": {"val_exact_match": eval_summary, "eval_examples": eval_examples},
        "git_rev": git_rev(),
        "date": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.adapter_out, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "train_v2_log.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"\nSaved adapter -> {args.adapter_out}")
    print(f"  version.json: {VERSION_STRING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())