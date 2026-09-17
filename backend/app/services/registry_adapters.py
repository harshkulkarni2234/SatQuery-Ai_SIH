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

from app.contracts import SpecialistResult
from app.services import change_detection, cross_modal, grounding, vqa


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
    result = change_detection.detect_change(
        image_path_before, image_path_after, metadata_before=metadata_before, metadata_after=metadata_after
    )
    warnings = [] if not result.get("validation_failed") else [result.get("reason") or "Validation failed."]

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
        used_fallback=False,
        fallback_reason=None,
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
