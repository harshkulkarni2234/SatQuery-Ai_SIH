import uuid
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Image, Query, QueryResult
from app.schemas import QueryRequest, QueryResponse, ExecutionTrace
from app.services.router import classify_query

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def post_query(body: QueryRequest, db: Session = Depends(get_db)):
    images = db.query(Image).filter(Image.id.in_(body.image_ids)).all()
    found_ids = {str(img.id) for img in images}
    missing = [str(i) for i in body.image_ids if str(i) not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"Image(s) not found: {', '.join(missing)}")

    image_dicts = [{"id": str(img.id), "modality": img.modality} for img in images]
    classification = classify_query(body.query_text, image_dicts)

    t0 = time.perf_counter()

    answer_text = (
        f"[STUB] Specialist services not yet implemented. "
        f"Task classified as: {classification['task_classified'] or 'UNKNOWN'}. "
        f"No AI model was executed."
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    trace = ExecutionTrace(
        selected_tool=classification["task_classified"],
        model_version=None,
        modalities_detected=classification["modalities_detected"],
        confidence_score=None,
        execution_time_ms=elapsed_ms,
        reason=classification["reason"],
    )

    query_record = Query(
        query_text=body.query_text,
        image_ids=body.image_ids,
        task_classified=classification["task_classified"],
        selected_tool=classification["task_classified"],
        model_version=None,
        confidence_score=None,
        execution_time_ms=elapsed_ms,
    )
    db.add(query_record)
    db.flush()

    result_record = QueryResult(
        query_id=query_record.id,
        answer_text=answer_text,
        bounding_boxes=None,
        change_mask_path=None,
        metadata_={
            "validation_passed": classification["validation_passed"],
            "stub": True,
        },
    )
    db.add(result_record)
    db.commit()
    db.refresh(query_record)

    return QueryResponse(
        query_id=query_record.id,
        task_classified=classification["task_classified"],
        answer_text=answer_text,
        confidence_score=None,
        bounding_boxes=None,
        change_mask_url=None,
        execution_trace=trace,
    )


@router.get("/query/{query_id}", response_model=QueryResponse)
def get_query(query_id: uuid.UUID, db: Session = Depends(get_db)):
    query_record = db.query(Query).filter(Query.id == query_id).first()
    if not query_record:
        raise HTTPException(status_code=404, detail="Query not found")

    result_record = (
        db.query(QueryResult).filter(QueryResult.query_id == query_id).first()
    )

    return QueryResponse(
        query_id=query_record.id,
        task_classified=query_record.task_classified,
        answer_text=result_record.answer_text if result_record else "",
        confidence_score=query_record.confidence_score,
        bounding_boxes=result_record.bounding_boxes if result_record else None,
        change_mask_url=result_record.change_mask_path if result_record else None,
        execution_trace=ExecutionTrace(
            selected_tool=query_record.selected_tool,
            model_version=query_record.model_version,
            modalities_detected=[],
            confidence_score=query_record.confidence_score,
            execution_time_ms=query_record.execution_time_ms,
            reason=result_record.metadata_.get("reason") if result_record and result_record.metadata_ else None,
        ),
    )
