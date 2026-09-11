import os
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Image, Query, QueryResult
from app.schemas import QueryRequest, QueryResponse, ExecutionTrace
from app.services.router import classify_query, extract_grounding_target
from app.services.grounding import ground_object, scene_cues
from app.services.change_detection import detect_change
from app.services.vqa import answer_question
from app.services.cross_modal import analyze_pair

router = APIRouter(tags=["query"])


def _tracked_exception(message: str, status_code: int = 400) -> HTTPException:
    """Raise a client-safe error that never leaks internals."""
    return HTTPException(status_code=status_code, detail=message)


@router.post("/query", response_model=QueryResponse)
def post_query(body: QueryRequest, request: Request, db: Session = Depends(get_db)):
    images = db.query(Image).filter(Image.id.in_(body.image_ids)).all()
    found_ids = {str(img.id) for img in images}
    missing = [str(i) for i in body.image_ids if str(i) not in found_ids]
    if missing:
        raise _tracked_exception(
            f"Image(s) not found: {', '.join(missing)}", status_code=404
        )

    by_id = {img.id: img for img in images}
    ordered_images = [by_id[i] for i in body.image_ids]

    image_dicts = [
        {"id": str(img.id), "modality": img.modality, "capture_date": img.capture_date}
        for img in ordered_images
    ]
    classification = classify_query(body.query_text, image_dicts)

    if not classification["validation_passed"]:
        raise _tracked_exception(classification["reason"])

    task = classification["task_classified"]

    t0 = time.perf_counter()
    specialist_result = None
    model_version = None
    bounding_boxes = None
    confidence = None
    change_mask_url = None
    change_mask_path = None
    overlay_url = None
    overlay_path = None

    try:
        if task == "GROUNDING":
            img = ordered_images[0]
            if not os.path.isfile(img.file_path):
                raise _tracked_exception(
                    f"Image file for '{img.filename}' is missing on disk."
                )
            target = classification.get("grounding_target")
            specialist_result = ground_object(img.file_path, target)
            answer_text = specialist_result["answer_text"]
            confidence = specialist_result["confidence_score"]
            bounding_boxes = specialist_result["bounding_boxes"] or None
            top_meta = {
                "object_type": specialist_result["object_type"],
                "num_regions": len(bounding_boxes or []),
                "regions": specialist_result["regions"],
                "method": specialist_result["method"],
                "specialist": "grounding",
            }

        elif task == "CHANGE_DETECTION":
            img_before, img_after = ordered_images
            if not os.path.isfile(img_before.file_path) or not os.path.isfile(
                img_after.file_path
            ):
                raise _tracked_exception(
                    "One or more image files for change detection are missing on disk."
                )
            meta_before = {
                "crs": img_before.crs,
                "bbox_coords": img_before.bbox_coords,
                "capture_date": str(img_before.capture_date)
                if img_before.capture_date
                else None,
                "modality": img_before.modality,
            }
            meta_after = {
                "crs": img_after.crs,
                "bbox_coords": img_after.bbox_coords,
                "capture_date": str(img_after.capture_date)
                if img_after.capture_date
                else None,
                "modality": img_after.modality,
            }
            specialist_result = detect_change(
                img_before.file_path,
                img_after.file_path,
                metadata_before=meta_before,
                metadata_after=meta_after,
            )
            answer_text = specialist_result["answer_text"]
            bounding_boxes = specialist_result["bounding_boxes"] or None
            if specialist_result["change_mask_path"]:
                change_mask_path = specialist_result["change_mask_path"]
                change_mask_url = str(
                    request.url_for("masks", path=os.path.basename(change_mask_path))
                )
            if specialist_result["overlay_path"]:
                overlay_path = specialist_result["overlay_path"]
                overlay_url = str(
                    request.url_for("masks", path=os.path.basename(overlay_path))
                )
            top_meta = {
                "change_percentage": specialist_result["change_percentage"],
                "num_regions": specialist_result["num_regions"],
                "changed_pixels": specialist_result["changed_pixels"],
                "total_pixels": specialist_result["total_pixels"],
                "regions": specialist_result["regions"],
                "validation_failed": specialist_result["validation_failed"],
                "registration_applied": specialist_result["registration_applied"],
                "reason": specialist_result["reason"],
                "specialist": "change_detection",
            }

        elif task == "VQA":
            img = ordered_images[0]
            if not os.path.isfile(img.file_path):
                raise _tracked_exception(
                    f"Image file for '{img.filename}' is missing on disk."
                )
            specialist_result = answer_question(img.file_path, body.query_text)
            answer_text = specialist_result["answer_text"]
            confidence = specialist_result.get("confidence_score")
            model_version = specialist_result.get("model_version")

            # Visual evidence for the VQA answer — never fabricated.
            grounding_target = extract_grounding_target(body.query_text)
            visual_evidence = None
            if grounding_target:
                g = ground_object(img.file_path, grounding_target)
                g_boxes = g["bounding_boxes"]
                visual_evidence = {
                    "available": bool(g_boxes),
                    "target": grounding_target,
                    "method": g["method"],
                    "num_regions": len(g_boxes or []),
                    "message": None
                    if g_boxes
                    else "Spatial localization unavailable for this analysis.",
                }
                if g_boxes:
                    bounding_boxes = g_boxes
            else:
                cues = scene_cues(img.file_path)
                visual_evidence = {
                    "available": False,
                    "message": "Spatial localization unavailable for this analysis.",
                    "detected_cues": cues["detected_cues"],
                    "dominant_cue": cues["dominant_cue"],
                }

            top_meta = {
                "specialist": "vqa",
                "specialist_mode": "experimental"
                if specialist_result.get("specialist")
                else "base",
                "model_version": model_version,
                "visual_evidence": visual_evidence,
            }

        elif task == "CROSS_MODAL":
            if len(ordered_images) != 2 or len(set(
                img.modality for img in ordered_images
            )) != 2:
                raise _tracked_exception(
                    "CROSS_MODAL requires exactly 2 images with different "
                    "modalities (one OPTICAL, one SAR)."
                )
            optical_img = next(
                img for img in ordered_images if img.modality == "OPTICAL"
            )
            sar_img = next(
                img for img in ordered_images if img.modality == "SAR"
            )
            missing = [
                img for img in (optical_img, sar_img)
                if not os.path.isfile(img.file_path)
            ]
            if missing:
                raise _tracked_exception(
                    f"Image file for '{missing[0].filename}' is missing on disk."
                )
            specialist_result = analyze_pair(
                optical_img.file_path, sar_img.file_path, body.query_text
            )
            answer_text = specialist_result["answer_text"]
            confidence = specialist_result.get("confidence_score")
            model_version = specialist_result.get("model_version")
            top_meta = {
                "specialist": "cross_modal",
                "tool_version": model_version,
                "modality_contribution_note": specialist_result[
                    "modality_contribution_note"
                ],
                "spatial_correspondence_note": specialist_result.get(
                    "spatial_correspondence_note"
                ),
                "evidence": specialist_result["evidence"],
            }

        else:
            # Safety net — no other multi-image task exists today.
            answer_text = (
                f"[STUB] The {task} specialist is not implemented yet."
            )
            model_version = None
            top_meta = {"specialist": "stub"}
    except HTTPException:
        raise
    except Exception:
        raise _tracked_exception(
            "Specialist processing failed. Please check that the uploaded "
            "images are valid and re-try."
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if specialist_result is None:
        execution_status = "not_implemented"
    elif specialist_result.get("error"):
        execution_status = "unavailable"
    else:
        execution_status = "completed"

    trace = ExecutionTrace(
        selected_tool=task,
        model_version=model_version,
        modalities_detected=classification["modalities_detected"],
        confidence_score=confidence,
        execution_time_ms=elapsed_ms,
        reason=classification["reason"],
        execution_status=execution_status,
    )

    query_record = Query(
        query_text=body.query_text,
        image_ids=body.image_ids,
        task_classified=task,
        selected_tool=task,
        model_version=model_version,
        confidence_score=confidence,
        execution_time_ms=elapsed_ms,
    )
    db.add(query_record)
    db.flush()

    result_record = QueryResult(
        query_id=query_record.id,
        answer_text=answer_text,
        bounding_boxes=bounding_boxes,
        change_mask_path=change_mask_path,
        metadata_={
            **top_meta,
            "overlay_path": overlay_path,
            "reason": classification["reason"],
            "execution_status": execution_status,
            "execution_trace": trace.model_dump(),
        },
    )
    db.add(result_record)
    db.commit()
    db.refresh(query_record)

    return QueryResponse(
        query_id=query_record.id,
        task_classified=task,
        answer_text=answer_text,
        confidence_score=confidence,
        bounding_boxes=bounding_boxes,
        change_mask_url=change_mask_url,
        overlay_url=overlay_url,
        execution_trace=trace,
        metadata=top_meta,
    )


@router.get("/query/{query_id}", response_model=QueryResponse)
def get_query(query_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
    query_record = db.query(Query).filter(Query.id == query_id).first()
    if not query_record:
        raise HTTPException(status_code=404, detail="Query not found")

    result_record = (
        db.query(QueryResult).filter(QueryResult.query_id == query_id).first()
    )

    meta = (result_record.metadata_ or {}) if result_record else {}
    cached = meta.get("execution_trace") or {}

    trace = ExecutionTrace(
        selected_tool=cached.get("selected_tool") or query_record.selected_tool,
        model_version=cached.get("model_version") or query_record.model_version,
        modalities_detected=cached.get("modalities_detected") or [],
        confidence_score=(
            query_record.confidence_score
            if query_record.confidence_score is not None
            else cached.get("confidence_score")
        ),
        execution_time_ms=query_record.execution_time_ms,
        reason=cached.get("reason"),
        execution_status=cached.get("execution_status"),
    )

    change_mask_url = None
    overlay_url = None
    if result_record and result_record.change_mask_path:
        change_mask_url = str(
            request.url_for("masks", path=os.path.basename(result_record.change_mask_path))
        )
    overlay_path = (meta or {}).get("overlay_path")
    if overlay_path:
        overlay_url = str(
            request.url_for("masks", path=os.path.basename(overlay_path))
        )

    top_meta = {
        k: v for k, v in meta.items()
        if k not in ("execution_trace", "overlay_path")
    }

    return QueryResponse(
        query_id=query_record.id,
        task_classified=query_record.task_classified,
        answer_text=result_record.answer_text if result_record else "",
        confidence_score=query_record.confidence_score,
        bounding_boxes=result_record.bounding_boxes if result_record else None,
        change_mask_url=change_mask_url,
        overlay_url=overlay_url,
        execution_trace=trace,
        metadata=top_meta or None,
    )