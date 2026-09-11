"""Inference provider for the local VQA worker.

Loads Base SmolVLM-256M-Instruct once and optionally attaches the
experimental BigEarthNet Stage 3 LoRA adapter on the same base model
(no duplicated model copy). LoRA is toggled per request via peft's
disable_adapter() context manager:
  - base mode       -> adapter disabled, pure base weights
  - specialist mode -> Stage 3 adapter active (experimental)
"""

import os
import re
import time

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from PIL import Image

BASE_MODEL_REPO = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smolvlm", "lora_stage3")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

MODEL_VERSION_BASE = "SmolVLM-256M-Instruct"
MODEL_VERSION_SPECIALIST = "SmolVLM-256M + Stage3LoRA (exp)"


class DecoderShim(nn.Module):
    """CausalLM facade over Idefics3's bare LlamaModel core so PEFT can wrap the decoder only."""

    def __init__(self, text_model, lm_head):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.gradient_checkpointing = False
        self.config = text_model.config

    def get_input_embeddings(self):
        return self.text_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, **kwargs):
        return self.text_model(**kwargs)


def clean_gen(text):
    """Strip generation scaffolding/special tokens from a decoded answer."""
    text = re.split(r"<end_of_utterance>|<\\s*end_of_utterance\\s*>", text)[0]
    text = re.sub(r"<[^>]*>", "", text)
    text = text.replace("Assistant", "").strip()
    return text


def _read_lora_config(adapter_dir):
    """Rebuild the LoRA tuning from the adapter's own config so shapes always match."""
    import json

    with open(os.path.join(adapter_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return LoraConfig(
        r=int(cfg.get("r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        target_modules=cfg.get("target_modules") or TARGET_MODULES,
        lora_dropout=float(cfg.get("lora_dropout", 0.1)),
        bias="none",
        task_type="CAUSAL_LM",
    )


class SmolVLMProvider:
    """Base SmolVLM-256M-Instruct with an optional experimental LoRA specialist."""

    def __init__(self, model_dir=None, adapter_dir=None):
        self.model_dir = model_dir or os.getenv("MODEL_DIR") or BASE_MODEL_REPO
        self.adapter_dir = adapter_dir or os.getenv("ADAPTER_DIR") or DEFAULT_ADAPTER_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self.specialist_available = False
        self.specialist_errors = []

    def load(self):
        t0 = time.time()
        kwargs = {"attn_implementation": "sdpa"}
        if self.device == "cuda":
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                self.model_dir, dtype=self.dtype, device_map={"": "cuda"}, **kwargs
            )
        else:
            from accelerate import dispatch_model

            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                self.model_dir, dtype=self.dtype, **kwargs
            )
            dispatch_model(self.model, device_map={"": "cpu"})
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(self.model_dir)

        cfg = _read_lora_config(self.adapter_dir) if os.path.isfile(
            os.path.join(self.adapter_dir, "adapter_config.json")
        ) else LoraConfig(
            r=8, lora_alpha=16, target_modules=TARGET_MODULES, lora_dropout=0.1,
            bias="none", task_type="CAUSAL_LM",
        )
        shim = DecoderShim(self.model.model.text_model, self.model.lm_head)
        peft_lm = get_peft_model(shim, cfg)
        # Proven Stage 3 sequence (equivalent to eval_vqa.py): get_peft_model
        # already creates and activates the "default" adapter, so load the saved
        # Stage 3 weights directly into that same adapter name. PEFT 0.20.0 has
        # no set_active_adapter; keep this exact loading shape.
        if os.path.isfile(os.path.join(self.adapter_dir, "adapter_model.safetensors")):
            try:
                peft_lm.load_adapter(self.adapter_dir, adapter_name="default")
                self.specialist_available = True
            except Exception as exc:  # worker must still operate in base mode
                self.specialist_errors.append(str(exc))
                self.specialist_available = False
        self.model.model.text_model = peft_lm
        self.model.eval()
        return self

    def _encode(self, query_text, image):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query_text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[prompt], images=[image], return_tensors="pt")
        return inp.to(self.model.device)

    def _generate(self, query_text, image, max_new_tokens):
        inputs = self._encode(query_text, image)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        return clean_gen(self.processor.tokenizer.decode(gen_ids, skip_special_tokens=False))

    def answer_question(self, image_path, query_text, use_specialist=False, max_new_tokens=256):
        """Run one VQA request. Returns structured result; never fabricates confidence."""
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")

        if self.model is None:
            self.load()

        specialist_used = False
        model_version = MODEL_VERSION_BASE
        if use_specialist and self.specialist_available:
            try:
                # after load() the active adapter is already "specialist", so a
                # plain generation runs the experimental LoRA on the base model.
                with torch.no_grad():
                    answer = self._generate(query_text, image, max_new_tokens)
                specialist_used = True
                model_version = MODEL_VERSION_SPECIALIST
            except Exception:
                # graceful fallback to the base model
                with torch.no_grad(), self.model.model.text_model.disable_adapter():
                    answer = self._generate(query_text, image, max_new_tokens)
                model_version = MODEL_VERSION_BASE
                specialist_used = False
        else:
            with torch.no_grad(), self.model.model.text_model.disable_adapter():
                answer = self._generate(query_text, image, max_new_tokens)

        return {
            "answer_text": answer,
            "model_version": model_version,
            "specialist": specialist_used,
            "confidence_score": None,
            "execution_time_ms": int((time.time() - t0) * 1000),
        }


_SPECIALIST_EXCLUDE = re.compile(
    r"\b(describe|explain|caption|overview|summar|what is visible|"
    r"what can you see|show me|find|locate|where is|compare|between|"
    r"change|detect|difference|optical|sar|multispectral|what changed)\b",
    re.IGNORECASE,
)
_SPECIALIST_PRESENCE = re.compile(
    r"\b(is there|are there|does the image contain|contain[s]?|"
    r"is any|present in the image|is .* present|exist[s]?)\b",
    re.IGNORECASE,
)
_SPECIALIST_COUNT = re.compile(r"\bhow many\b", re.IGNORECASE)


def should_use_specialist(query_text):
    """Deterministic, deliberately conservative gate for the experimental specialist.

    True only for obvious presence/existence or simple counting questions.
    Never for descriptions, captions, spatial reasoning, grounding,
    change detection, or optical/SAR questions.
    """
    query = (query_text or "").strip()
    if not query or len(query.split()) > 24:
        return False
    if _SPECIALIST_EXCLUDE.search(query):
        return False
    if _SPECIALIST_COUNT.search(query):
        return len(query.split()) <= 10
    return bool(_SPECIALIST_PRESENCE.search(query))