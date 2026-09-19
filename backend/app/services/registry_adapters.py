"""Adapters: existing B-owned service return-dicts -> contracts.SpecialistResult.

These wrap vqa.py / grounding.py / change_detection.py / cross_modal.py
without editing them. Once those services return SpecialistResult natively
(Phase B7), the corresponding adapter here becomes a thin pass-through and
can eventually be removed (Phase A7).

Adapters do NOT know whether the registry picked them as a fallback for a
higher-priority (currently unavailable) specialist — that is a registry-
level fact, not something a single execution can see about itself. Callers
(routes/query.py) OR the registry's own is_fallback/used_fallback signal
into the SpecialistResult after calling an adapter.
"""

from __future__ import annotations

import os

import numpy as np
import requests
from PIL import Image

from app.contracts import SpecialistResult
from app.services import change_detection, cross_modal, grounding, vqa


CHANGE_WORKER_URL = os.getenv("CHANGE_WORKER_URL", "http://127.0.0.1:8002").rstrip("/")
CHANGE_WORKER_TIMEOUT_S = float(os.getenv("CHANGE_WORKER_TIMEOUT_S", "120"))
CHANGE_HEALTH_TIMEOUT_S = 1.5

_post = requests.post


def _check_image_sizes(path1: str, path2: str) -> bool:
    """Return True if both images have the same dimensions."""
    try:
        with Image.open(path1) as im1, Image.open(path2) as im2:
            return im1.size == im2.size
    except Exception:
        return False


def _call_change_worker(image_path_before: str, image_path_after: str) -> dict | None:
    """Call the change worker's /change endpoint. Returns the parsed
    JSON response, or None if the call fails."""
    try:
        with open(image_path_before, "rb") as f_before, open(image_path_after, "rb") as f_after:
            response = _post(
                f"{CHANGE_WORKER_URL}/change",
                data={},
                files={
                    "before_image": (os.path.basename(image_path_before), f_before),
                    "after_image": (os.path.basename(image_path_after), f_after),
                },
                timeout=CHANGE_WORKER_TIMEOUT_S,
            )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except ValueError:
        return None

    if not isinstance(data, dict) or "model_version" not in data or "latency_ms" not in data:
        return None

    return data


def _build_change_worker_result(data: dict) -> SpecialistResult:
    """Convert the change worker's /change response into a SpecialistResult."""
    mask_path = data.get("mask_path")
    model_version = data.get("model_version", "siamese-cnn-v1")
    latency_ms = data.get("latency_ms", 0)

    # Compute mask stats
    changed_pixels = 0
    total_pixels = 0
    change_percentage = 0.0
    if mask_path and os.path.isfile(mask_path):
        mask_img = np.array(Image.open(mask_path).convert("L"))
        changed_pixels = int(np.sum(mask_img > 0))
        total_pixels = int(mask_img.size)
        change_percentage = round(changed_pixels / total_pixels * 100, 2) if total_pixels else 0.0

    answer_text = (
        f"Learned Siamese CNN detected change covering approximately {change_percentage}% "
        f"of the frame ({changed_pixels}/{total_pixels} pixels). Model: {model_version}. "
        f"Inference latency: {latency_ms}ms."
    )

    stats = {
        "change_percentage": change_percentage,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "num_regions": 0,
        "changed_area_m2": None,
        "regions": [],
        "registration_applied": "learned_model",
        "alignment_method": "siamese_cnn",
    }

    return SpecialistResult(
        answer=answer_text,
        evidence={
            "boxes": None,
            "mask_path": mask_path,
            "overlay_path": None,
            "stats": stats,
        },
        confidence=None,
        confidence_source="unavailable",
        model_or_tool="change.semantic_model",
        model_version=model_version,
        used_fallback=False,
        fallback_reason=None,
        warnings=[],
    )


def run_vqa(image_path: str, query_text: str, grounding_target: str | None) -> SpecialistResult:
    result = vqa.answer_question(image_path, query_text)

    visual_evidence = None
    boxes = None
    if grounding_target:
        g = grounding.ground_object(image_path, grounding_target)
        g_boxes = g["bounding_boxes"]
        visual_evidence = {
            "available": bool(g_boxes),
            "target": grounding_target,
            "method": g["method"],
            "num_regions": len(g_boxes or []),
            "message": None if g_boxes else "Spatial localization unavailable for this analysis.",
        }
        if g_boxes:
            boxes = g_boxes
    else:
        cues = grounding.scene_cues(image_path)
        visual_evidence = {
            "available": False,
            "message": "Spatial localization unavailable for this analysis.",
            "detected_cues": cues["detected_cues"],
            "dominant_cue": cues["dominant_cue"],
        }

    is_error = bool(result.get("error"))
    used_lora = bool(result.get("specialist"))
    model_or_tool = "vqa.smolvlm_bigearthnet_lora_stage3" if used_lora else "vqa.smolvlm_base"

    return SpecialistResult(
        answer=result["answer_text"],
        evidence={
            "boxes": boxes,
            "visual_evidence": visual_evidence,
            "adapter_used": result.get("adapter_used", used_lora),
            "specialist_selection_reason": result.get("reason"),
        },
        confidence=result.get("confidence_score"),
        confidence_source="unavailable" if result.get("confidence_score") is None else "model-reported",
        model_or_tool=model_or_tool,
        model_version=result.get("model_version"),
        used_fallback=is_error,
        fallback_reason=result["answer_text"] if is_error else None,
        warnings=[] if not is_error else ["VQA worker unavailable; answer is an honest unavailability notice, not a model response."],
    )


def run_grounding(image_path: str, target: str) -> SpecialistResult:
    result = grounding.ground_object(image_path, target)
    confidence = result.get("confidence_score")

    return SpecialistResult(
        answer=result["answer_text"],
        evidence={
            "boxes": result.get("bounding_boxes"),
            "labels": [target] * len(result.get("bounding_boxes") or []),
            "regions": result.get("regions"),
        },
        confidence=confidence,
        confidence_source="unavailable" if confidence is None else "bbox fill ratio",
        model_or_tool="grounding.deterministic_cv",
        model_version="cv-grounding-v1",
        used_fallback=False,
        fallback_reason=None,
        warnings=[],
    )


def run_change_detection(
    image_path_before: str,
    image_path_after: str,
    metadata_before: dict | None = None,
    metadata_after: dict | None = None,
) -> SpecialistResult:
    """Try the learned Siamese CNN change worker first, then fall back
    to deterministic pixel-difference change detection if the worker is
    down or image sizes mismatch."""

    # Check image sizes match before attempting worker call.
    if not _check_image_sizes(image_path_before, image_path_after):
        result = change_detection.detect_change(
            image_path_before, image_path_after,
            metadata_before=metadata_before, metadata_after=metadata_after,
        )
        warnings = [] if not result.get("validation_failed") else [result.get("reason") or "Validation failed."]
        return _build_deterministic_result(result, warnings)

    # Try the learned change worker.
    worker_data = _call_change_worker(image_path_before, image_path_after)
    if worker_data is not None:
        return _build_change_worker_result(worker_data)

    # Worker down — graceful fallback to deterministic CV.
    result = change_detection.detect_change(
        image_path_before, image_path_after,
        metadata_before=metadata_before, metadata_after=metadata_after,
    )
    warnings = [] if not result.get("validation_failed") else [result.get("reason") or "Validation failed."]
    return _build_deterministic_result(result, warnings)


def _build_deterministic_result(result: dict, warnings: list) -> SpecialistResult:
    return SpecialistResult(
        answer=result["answer_text"],
        evidence={
            "boxes": result.get("bounding_boxes"),
            "mask_path": result.get("change_mask_path"),
            "overlay_path": result.get("overlay_path"),
            "stats": {
                "change_percentage": result.get("change_percentage"),
                "num_regions": result.get("num_regions"),
                "changed_pixels": result.get("changed_pixels"),
                "total_pixels": result.get("total_pixels"),
                "changed_area_m2": result.get("changed_area_m2"),
                "regions": result.get("regions"),
                "registration_applied": result.get("registration_applied"),
                "alignment_method": result.get("alignment_method"),
            },
        },
        confidence=None,
        confidence_source="unavailable",
        model_or_tool="change.deterministic_cv",
        model_version="cv-change-v1",
        used_fallback=bool(result.get("validation_failed")),
        fallback_reason=result.get("reason") if result.get("validation_failed") else None,
        warnings=warnings,
    )


def run_cross_modal(
    optical_path: str, sar_path: str, query_text: str = "", coregistration: str | None = None
) -> SpecialistResult:
    result = cross_modal.analyze_pair(optical_path, sar_path, query_text, coregistration=coregistration)

    return SpecialistResult(
        answer=result["answer_text"],
        evidence={
            # NOTE: "modality_evidence" is the {optical, sar, combined,
            # per_modality} dict from cross_modal.py itself (its own
            # per_modality key is the B9 fusion-attribution sub-dict) — kept
            # under a differently-named key here to avoid nesting confusion
            # with SpecialistResult.evidence's own shape.
            "modality_evidence": result.get("evidence"),
            "modality_contribution_note": result.get("modality_contribution_note"),
            "spatial_correspondence_note": result.get("spatial_correspondence_note"),
        },
        confidence=result.get("confidence_score"),
        confidence_source=result.get("confidence_source", "unavailable"),
        model_or_tool="cross_modal.feature_fusion",
        model_version=result.get("model_version"),
        used_fallback=False,
        fallback_reason=None,
        warnings=[],
    )
