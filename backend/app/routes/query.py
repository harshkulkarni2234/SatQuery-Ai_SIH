import os
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.contracts import specialist_result_to_legacy_response_fields
from app.database import get_db
from app.models import Image, Query, QueryResult
from app.schemas import QueryRequest, QueryResponse, ExecutionTrace
from app.services.router import extract_grounding_target
from app.services.planner import build_plan
from app.services.trace import TraceRecorder

router = APIRouter(tags=["query"])


def _tracked_exception(message: str, status_code: int = 400) -> HTTPException:
    """Raise a client-safe error that never leaks internals."""
    return HTTPException(status_code=status_code, detail=message)


def _tracked_exception_for_plan(plan, status_code: int) -> HTTPException:
    detail = {"message": plan.validation_reason, "suggestion": plan.suggestion}
    if plan.compatibility is not None:
        detail["compatibility"] = plan.compatibility.model_dump()
    return HTTPException(status_code=status_code, detail=detail)


def _persist_failed_query(
    db: Session, body: QueryRequest, recorder: TraceRecorder, task: str | None
) -> None:
    """Persist a Query row for a failed request so it stays auditable
    (Phase A6). No QueryResult is created — there is no answer."""
    query_record = Query(
        query_text=body.query_text,
        image_ids=body.image_ids,
        task_classified=task,
        trace_events=recorder.as_dicts(),
    )
    db.add(query_record)
    db.commit()


def _compat_dict(img: Image) -> dict:
    resolution = None
    if img.transform:
        resolution = (abs(img.transform[0]), abs(img.transform[4]))
    elif img.resolution_m is not None:
        resolution = (img.resolution_m, img.resolution_m)
    return {
        "modality": img.modality,
        "capture_date": str(img.capture_date) if img.capture_date else None,
        "is_georeferenced": bool(img.is_georeferenced),
        "crs": img.crs,
        "bounds_wgs84": tuple(img.bounds_wgs84) if img.bounds_wgs84 else None,
        "resolution": resolution,
        "transform": tuple(img.transform) if img.transform else None,
        "width": img.width,
        "height": img.height,
    }


@router.post("/query", response_model=QueryResponse)
def post_query(body: QueryRequest, request: Request, db: Session = Depends(get_db)):
    recorder = TraceRecorder()
    recorder.start("QUERY_RECEIVED")

    images = db.query(Image).filter(Image.id.in_(body.image_ids)).all()
    found_ids = {str(img.id) for img in images}
    missing = [str(i) for i in body.image_ids if str(i) not in found_ids]
    if missing:
        recorder.record("QUERY_RECEIVED", "FAILED", f"Image(s) not found: {', '.join(missing)}")
        recorder.error(f"Image(s) not found: {', '.join(missing)}")
        _persist_failed_query(db, body, recorder, task=None)
        raise _tracked_exception(
            f"Image(s) not found: {', '.join(missing)}", status_code=404
        )

    by_id = {img.id: img for img in images}
    ordered_images = [by_id[i] for i in body.image_ids]
    recorder.record("QUERY_RECEIVED", "COMPLETED", f"{len(ordered_images)} image(s) resolved")

    image_dicts = [{"id": str(img.id), **_compat_dict(img)} for img in ordered_images]

    plan_result = build_plan(body.query_text, image_dicts, recorder=recorder)
    plan = plan_result.plan

    if not plan.validation_passed:
        recorder.error(plan.validation_reason, {"task": plan.task})
        _persist_failed_query(db, body, recorder, task=plan.task)
        status_code = 422 if plan.compatibility is not None else 400
        raise _tracked_exception_for_plan(plan, status_code)

    task = plan.task
    selected_entry = plan_result.selected_entry
    select_reason = plan.selection_reason
    rejected_specialists = plan.rejected_specialists

    t0 = time.perf_counter()
    specialist_result = None
    model_version = None
    bounding_boxes = None
    confidence = None
    change_mask_url = None
    change_mask_path = None
    overlay_url = None
    overlay_path = None

    recorder.start("EXECUTION")
    try:
        if task == "GROUNDING":
            img = ordered_images[0]
            if not os.path.isfile(img.file_path):
                raise _tracked_exception(
                    f"Image file for '{img.filename}' is missing on disk."
                )
            target = plan.target
            specialist_result = selected_entry.handler(img.file_path, target)
            legacy = specialist_result_to_legacy_response_fields(specialist_result)
            answer_text = legacy["answer_text"]
            confidence = legacy["confidence_score"]
            bounding_boxes = legacy["bounding_boxes"] or None
            top_meta = {
                "object_type": target,
                "num_regions": len(bounding_boxes or []),
                "regions": specialist_result.evidence.get("regions"),
                "method": "deterministic visual grounding",
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
            specialist_result = selected_entry.handler(
                img_before.file_path,
                img_after.file_path,
                metadata_before=meta_before,
                metadata_after=meta_after,
            )
            legacy = specialist_result_to_legacy_response_fields(specialist_result)
            answer_text = legacy["answer_text"]
            bounding_boxes = legacy["bounding_boxes"] or None
            change_mask_path = legacy["change_mask_url"]  # path, not URL yet
            overlay_path = legacy["overlay_url"]  # path, not URL yet
            if change_mask_path:
                change_mask_url = str(
                    request.url_for("masks", path=os.path.basename(change_mask_path))
                )
            if overlay_path:
                overlay_url = str(
                    request.url_for("masks", path=os.path.basename(overlay_path))
                )
            stats = specialist_result.evidence.get("stats") or {}
            top_meta = {
                "change_percentage": stats.get("change_percentage"),
                "num_regions": stats.get("num_regions"),
                "changed_pixels": stats.get("changed_pixels"),
                "total_pixels": stats.get("total_pixels"),
                "regions": stats.get("regions"),
                "validation_failed": bool(specialist_result.warnings),
                "registration_applied": stats.get("registration_applied"),
                "reason": specialist_result.warnings[0] if specialist_result.warnings else None,
                "specialist": "change_detection",
            }

        elif task == "VQA":
            img = ordered_images[0]
            if not os.path.isfile(img.file_path):
                raise _tracked_exception(
                    f"Image file for '{img.filename}' is missing on disk."
                )
            grounding_target = extract_grounding_target(body.query_text)
            specialist_result = selected_entry.handler(
                img.file_path, body.query_text, grounding_target
            )
            legacy = specialist_result_to_legacy_response_fields(specialist_result)
            answer_text = legacy["answer_text"]
            confidence = legacy["confidence_score"]
            model_version = legacy["model_version"]
            bounding_boxes = specialist_result.evidence.get("boxes") or None

            top_meta = {
                "specialist": "vqa",
                "specialist_mode": "experimental"
                if specialist_result.model_or_tool == "vqa.smolvlm_bigearthnet_lora_stage3"
                else "base",
                "model_version": model_version,
                "visual_evidence": specialist_result.evidence.get("visual_evidence"),
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
            specialist_result = selected_entry.handler(
                optical_img.file_path, sar_img.file_path, body.query_text
            )
            legacy = specialist_result_to_legacy_response_fields(specialist_result)
            answer_text = legacy["answer_text"]
            confidence = legacy["confidence_score"]
            model_version = legacy["model_version"]
            top_meta = {
                "specialist": "cross_modal",
                "tool_version": model_version,
                "modality_contribution_note": specialist_result.evidence.get(
                    "modality_contribution_note"
                ),
                "spatial_correspondence_note": specialist_result.evidence.get(
                    "spatial_correspondence_note"
                ),
                "evidence": specialist_result.evidence.get("per_modality"),
            }

        else:
            # Safety net — no other multi-image task exists today.
            answer_text = (
                f"[STUB] The {task} specialist is not implemented yet."
            )
            model_version = None
            top_meta = {"specialist": "stub"}
    except HTTPException as exc:
        recorder.record("EXECUTION", "FAILED", str(exc.detail))
        recorder.error(str(exc.detail), {"task": task})
        _persist_failed_query(db, body, recorder, task=task)
        raise
    except Exception:
        recorder.record("EXECUTION", "FAILED", "Unexpected error during specialist execution")
        recorder.error("Specialist processing failed unexpectedly.", {"task": task})
        _persist_failed_query(db, body, recorder, task=task)
        raise _tracked_exception(
            "Specialist processing failed. Please check that the uploaded "
            "images are valid and re-try."
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if specialist_result is None:
        execution_status = "not_implemented"
    elif specialist_result.used_fallback and specialist_result.fallback_reason:
        execution_status = "unavailable"
    else:
        execution_status = "completed"
    recorder.record(
        "EXECUTION",
        "COMPLETED" if execution_status == "completed" else "PASSED",
        f"{selected_entry.spec.id if selected_entry else task} finished ({execution_status})",
    )
    recorder.record(
        "EVIDENCE",
        "COMPLETED",
        f"{len(bounding_boxes or [])} box(es), mask={'yes' if change_mask_path else 'no'}",
    )

    used_fallback = bool(specialist_result and specialist_result.used_fallback) or bool(
        selected_entry and selected_entry.spec.is_fallback
    )
    confidence_source = specialist_result.confidence_source if specialist_result else "unavailable"
    warnings = specialist_result.warnings if specialist_result else []
    if selected_entry is not None:
        top_meta["specialist_id"] = selected_entry.spec.id
        top_meta["selection_reason"] = select_reason
        top_meta["rejected_specialists"] = rejected_specialists
        top_meta["used_fallback"] = used_fallback
        top_meta["confidence_source"] = confidence_source
        top_meta["warnings"] = warnings
    top_meta["plan"] = plan.model_dump()

    recorder.record("RESULT", "COMPLETED", "Answer produced" if answer_text else "No answer produced")

    modalities_detected = [img.modality for img in ordered_images if img.modality in ("OPTICAL", "SAR")]
    trace = ExecutionTrace(
        selected_tool=task,
        model_version=model_version,
        modalities_detected=modalities_detected,
        confidence_score=confidence,
        execution_time_ms=elapsed_ms,
        reason=plan.validation_reason,
        execution_status=execution_status,
    )
    trace_events = recorder.as_dicts()
    compatibility = plan.compatibility

    query_record = Query(
        query_text=body.query_text,
        image_ids=body.image_ids,
        task_classified=task,
        selected_tool=task,
        model_version=model_version,
        confidence_score=confidence,
        execution_time_ms=elapsed_ms,
        trace_events=trace_events,
        compatibility=compatibility.model_dump() if compatibility else None,
        confidence_source=confidence_source,
        warnings=warnings,
        used_fallback=used_fallback,
        specialist_id=selected_entry.spec.id if selected_entry else None,
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
            "reason": plan.validation_reason,
            "execution_status": execution_status,
            "execution_trace": trace.model_dump(),
            "trace_events": trace_events,
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
        trace_events=trace_events,
        compatibility=compatibility,
        confidence_source=confidence_source,
        warnings=warnings,
        used_fallback=used_fallback,
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
        if k not in ("execution_trace", "overlay_path", "trace_events")
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
        trace_events=query_record.trace_events,
        compatibility=query_record.compatibility,
        confidence_source=query_record.confidence_source,
        warnings=query_record.warnings or [],
        used_fallback=bool(query_record.used_fallback),
    )