from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.contracts import RasterMetadata


# ── Image upload ──────────────────────────────────────────────────────

class ImageUploadResponse(BaseModel):
    image_id: uuid.UUID
    filename: str
    modality: str
    crs: Optional[str] = None
    resolution_m: Optional[float] = None
    metadata: Optional[RasterMetadata] = None


class ImageDetailResponse(BaseModel):
    image_id: uuid.UUID
    filename: str
    modality: str
    capture_date: Optional[date] = None
    crs: Optional[str] = None
    resolution_m: Optional[float] = None
    metadata: Optional[RasterMetadata] = None


# ── Query ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    image_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=2)


class ExecutionTrace(BaseModel):
    selected_tool: Optional[str] = None
    model_version: Optional[str] = None
    modalities_detected: list[str] = []
    confidence_score: Optional[float] = None
    execution_time_ms: Optional[int] = None
    reason: Optional[str] = None
    execution_status: Optional[str] = None


class QueryResponse(BaseModel):
    query_id: uuid.UUID
    task_classified: Optional[str] = None
    answer_text: str
    confidence_score: Optional[float] = None
    bounding_boxes: Optional[list[list[float]]] = None
    change_mask_url: Optional[str] = None
    overlay_url: Optional[str] = None
    execution_trace: ExecutionTrace
    metadata: Optional[dict] = None
