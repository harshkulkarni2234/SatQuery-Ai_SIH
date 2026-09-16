"""Shared data contracts between the backend, frontend, and ML workers.

Pure pydantic models only — no DB/ORM imports here. These are the types that
cross the frontend/backend/model boundary; see docs/CONTRACTS.md for the
narrative version with JSON examples. Don't change a field here silently:
the frontend (Person/Phase C) and the specialists (Person/Phase B) both
depend on these shapes staying stable and additive.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ── Raster metadata (A1/A2) ──────────────────────────────────────────────

AcquisitionDateSource = Literal["user", "file_metadata", "unknown"]


class RasterMetadata(BaseModel):
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    band_count: Optional[int] = None
    dtype: Optional[str] = None
    crs: Optional[str] = None
    bounds: Optional[tuple[float, float, float, float]] = None  # minx, miny, maxx, maxy, in native CRS
    bounds_wgs84: Optional[tuple[float, float, float, float]] = None
    resolution: Optional[tuple[float, float]] = None  # (x, y) in CRS units
    transform: Optional[tuple[float, float, float, float, float, float]] = None
    nodata: Optional[float] = None
    acquisition_date: Optional[date] = None
    acquisition_date_source: AcquisitionDateSource = "unknown"
    is_georeferenced: bool = False
    file_size_bytes: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)


# ── Compatibility checks (A3) ────────────────────────────────────────────

CheckStatus = Literal["PASS", "FAIL", "WARN", "SKIPPED"]
CoregistrationStatus = Literal["verified", "assumed", "unverified", "failed"]


class CompatibilityCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str


class CompatibilityReport(BaseModel):
    ok: bool
    checks: list[CompatibilityCheck] = Field(default_factory=list)
    overlap_ratio: Optional[float] = None
    coregistration: Optional[CoregistrationStatus] = None
    alignment_possible: Optional[bool] = None


# ── Specialist results (A0 contract; A4 registry integration) ───────────

SpecialistTask = Literal["VQA", "GROUNDING", "CHANGE_DETECTION", "CROSS_MODAL"]


class SpecialistResult(BaseModel):
    answer: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    confidence_source: str = "unavailable"
    model_or_tool: str
    model_version: Optional[str] = None
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class SpecialistSpec(BaseModel):
    id: str
    task: SpecialistTask
    name: str
    kind: Literal["learned_model", "deterministic_cv", "rules"]
    input_count: int
    modalities: list[str]
    formats: list[str]
    required_metadata: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    confidence_available: bool = False
    version: str
    priority: int
    is_fallback: bool = False
    is_rs_adapted: bool = False


# ── Execution plan (A5) ───────────────────────────────────────────────────

QuestionType = Literal["presence", "count", "description", "location", "change", "comparison"]


class ExecutionPlan(BaseModel):
    task: Optional[SpecialistTask] = None
    target: Optional[str] = None
    question_type: Optional[QuestionType] = None
    requested_outputs: list[str] = Field(default_factory=list)
    validation_passed: bool
    validation_reason: str
    suggestion: Optional[str] = None
    compatibility: Optional[CompatibilityReport] = None
    selected_specialist_id: Optional[str] = None
    selection_reason: Optional[str] = None
    rejected_specialists: list[dict[str, str]] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── Execution trace (A5/A6) ──────────────────────────────────────────────

TraceStep = Literal[
    "QUERY_RECEIVED",
    "INPUT_VALIDATION",
    "COMPATIBILITY",
    "TASK_SELECTED",
    "SPECIALIST_SELECTED",
    "EXECUTION",
    "EVIDENCE",
    "RESULT",
    "ERROR",
]
TraceStatus = Literal["STARTED", "PASSED", "FAILED", "COMPLETED", "SKIPPED"]


class TraceEvent(BaseModel):
    step: TraceStep
    status: TraceStatus
    detail: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    duration_ms: Optional[int] = None


# ── Legacy compatibility shim ─────────────────────────────────────────────

def specialist_result_to_legacy_response_fields(result: SpecialistResult) -> dict[str, Any]:
    """Map a SpecialistResult onto the pre-existing QueryResponse field names.

    Existing code (schemas.QueryResponse, routes/query.py) predates the
    registry/SpecialistResult contract. Until every specialist is migrated to
    return SpecialistResult natively and QueryResponse is updated to consume
    it directly, this adapter keeps the old response fields populated so
    nothing breaks.
    """
    return {
        "answer_text": result.answer,
        "confidence_score": result.confidence,
        "bounding_boxes": result.evidence.get("boxes"),
        "change_mask_url": result.evidence.get("mask_path"),
        "overlay_url": result.evidence.get("overlay_path"),
        "model_version": result.model_version,
    }
