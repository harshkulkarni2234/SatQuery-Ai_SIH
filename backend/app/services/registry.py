"""Specialist registry (Phase A4).

A static list of SpecialistSpec entries plus selection logic. Each entry
pairs a serializable SpecialistSpec (for GET /specialists and for the
frontend) with a handler callable and an is_available() probe — kept
outside the pydantic model since callables aren't serializable.

Handlers wrap the existing B-owned service modules (vqa.py, grounding.py,
change_detection.py, cross_modal.py) via services/registry_adapters.py.
This module never imports those service modules directly, and never edits
them — only the adapters do, per AGENTS.md file ownership.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import requests

from app.contracts import SpecialistSpec
from app.services import registry_adapters as adapters

VQA_WORKER_URL = os.getenv("VQA_WORKER_URL", "http://127.0.0.1:8001").rstrip("/")
VQA_HEALTH_TIMEOUT_S = 1.5


def _vqa_worker_available() -> bool:
    try:
        resp = requests.get(f"{VQA_WORKER_URL}/health", timeout=VQA_HEALTH_TIMEOUT_S)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _always_available() -> bool:
    return True


def _semantic_change_available() -> bool:
    # No semantic change worker exists yet (Phase B8 not delivered).
    # Kept as an explicit hook rather than deleting the entry so the
    # registry/GET /specialists listing documents the planned model and
    # A7 only needs to flip this check once B8 ships a real worker.
    return False


@dataclass(frozen=True)
class RegistryEntry:
    spec: SpecialistSpec
    handler: Callable[..., dict]
    is_available: Callable[[], bool]


_REGISTRY: list[RegistryEntry] = [
    RegistryEntry(
        spec=SpecialistSpec(
            id="vqa.smolvlm_base",
            task="VQA",
            name="SmolVLM-256M-Instruct (base)",
            kind="learned_model",
            input_count=1,
            modalities=["OPTICAL", "SAR"],
            formats=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "jp2"],
            confidence_available=False,
            version="smolvlm-256m-instruct-base",
            priority=10,
            is_fallback=False,
            is_rs_adapted=False,
        ),
        handler=adapters.run_vqa,
        is_available=_vqa_worker_available,
    ),
    RegistryEntry(
        spec=SpecialistSpec(
            id="vqa.smolvlm_bigearthnet_lora_stage3",
            task="VQA",
            name="SmolVLM-256M-Instruct + BigEarthNet LoRA (Stage 3, narrow question types)",
            kind="learned_model",
            input_count=1,
            modalities=["OPTICAL", "SAR"],
            formats=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "jp2"],
            confidence_available=False,
            version="smolvlm256m-ben-lora-s3",
            priority=5,
            is_fallback=False,
            is_rs_adapted=True,
        ),
        # NOTE: base-vs-LoRA selection for a given question currently happens
        # INSIDE the VQA worker (vqa.should_use_specialist), not here — the
        # worker exposes one /vqa endpoint that internally decides. This
        # entry exists for GET /specialists introspection/documentation; it
        # is not independently dispatched by select() (see candidates_for).
        # The real model_version used is read back from the worker's own
        # response after the call, never assumed in advance.
        handler=adapters.run_vqa,
        is_available=_vqa_worker_available,
    ),
    RegistryEntry(
        spec=SpecialistSpec(
            id="grounding.deterministic_cv",
            task="GROUNDING",
            name="Deterministic visual grounding (HSV + morphology)",
            kind="deterministic_cv",
            input_count=1,
            modalities=["OPTICAL", "SAR"],
            formats=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "jp2"],
            confidence_available=True,
            version="cv-grounding-v1",
            priority=10,
            is_fallback=False,
            is_rs_adapted=False,
        ),
        handler=adapters.run_grounding,
        is_available=_always_available,
    ),
    RegistryEntry(
        spec=SpecialistSpec(
            id="change.semantic_model",
            task="CHANGE_DETECTION",
            name="Semantic change model (planned — Phase B8)",
            kind="learned_model",
            input_count=2,
            modalities=["OPTICAL"],
            formats=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
            confidence_available=False,
            version="unreleased",
            priority=5,
            is_fallback=False,
            is_rs_adapted=True,
        ),
        handler=adapters.run_change_detection,  # unreachable while unavailable
        is_available=_semantic_change_available,
    ),
    RegistryEntry(
        spec=SpecialistSpec(
            id="change.deterministic_cv",
            task="CHANGE_DETECTION",
            name="Deterministic pixel-difference change detection",
            kind="deterministic_cv",
            input_count=2,
            modalities=["OPTICAL", "SAR"],
            formats=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "jp2"],
            confidence_available=False,
            version="cv-change-v1",
            priority=10,
            is_fallback=True,
            is_rs_adapted=False,
        ),
        handler=adapters.run_change_detection,
        is_available=_always_available,
    ),
    RegistryEntry(
        spec=SpecialistSpec(
            id="cross_modal.feature_fusion",
            task="CROSS_MODAL",
            name="Optical + SAR statistical feature fusion",
            kind="rules",
            input_count=2,
            modalities=["OPTICAL", "SAR"],
            formats=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "jp2"],
            confidence_available=False,
            version="cross-modal-v1",
            priority=10,
            is_fallback=False,
            is_rs_adapted=False,
        ),
        handler=adapters.run_cross_modal,
        is_available=_always_available,
    ),
]


def list_specialists() -> list[SpecialistSpec]:
    return [entry.spec for entry in _REGISTRY]


def candidates_for(task: str, images: list[dict]) -> list[RegistryEntry]:
    """Entries matching *task* and compatible with the given images.

    images: list of {"modality": str, ...}. Only modality/input_count are
    checked here — deeper compatibility (overlap, CRS, coregistration) is
    services/compatibility.py's job and runs separately in routes/query.py.
    """
    n = len(images)
    modalities = {img.get("modality") for img in images}
    matches = []
    for entry in _REGISTRY:
        spec = entry.spec
        if spec.task != task:
            continue
        # The LoRA VQA entry is documentation-only (see its handler comment
        # above) — it is never independently selectable.
        if spec.id == "vqa.smolvlm_bigearthnet_lora_stage3":
            continue
        if spec.input_count != n:
            continue
        if not modalities.issubset(set(spec.modalities)):
            continue
        matches.append(entry)
    return sorted(matches, key=lambda e: e.spec.priority)


def select(task: str, images: list[dict]) -> tuple[RegistryEntry | None, str, list[dict]]:
    """Pick the best available specialist for *task*.

    Returns (entry_or_None, reason, rejected) where rejected is
    [{"id": spec_id, "reason": str}, ...] for every candidate not chosen.
    """
    candidates = candidates_for(task, images)
    if not candidates:
        return None, f"No registered specialist supports task={task} with {len(images)} image(s).", []

    rejected: list[dict] = []
    for entry in candidates:
        if entry.is_available():
            for other in candidates:
                if other is entry:
                    continue
                other_reason = (
                    "lower priority than the selected specialist"
                    if other.is_available()
                    else "currently unavailable"
                )
                rejected.append({"id": other.spec.id, "reason": other_reason})
            fallback_note = " (fallback)" if entry.spec.is_fallback else ""
            reason = f"Selected {entry.spec.id}{fallback_note}: highest-priority available specialist for {task}."
            return entry, reason, rejected

    # Nothing reported itself available — fall back to the highest-priority
    # candidate anyway so the handler's own honest "[unavailable]" / error
    # path can run and be surfaced, rather than silently answering nothing.
    best = candidates[0]
    for other in candidates[1:]:
        rejected.append({"id": other.spec.id, "reason": "currently unavailable"})
    reason = (
        f"No specialist for {task} reported itself available; attempting "
        f"{best.spec.id} anyway, which will report its own failure if it cannot run."
    )
    return best, reason, rejected
