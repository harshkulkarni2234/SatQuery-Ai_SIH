from datetime import date, datetime, timezone

from app.contracts import (
    CompatibilityCheck,
    CompatibilityReport,
    RasterMetadata,
    SpecialistResult,
    SpecialistSpec,
    TraceEvent,
    specialist_result_to_legacy_response_fields,
)


def test_raster_metadata_georeferenced():
    m = RasterMetadata(
        format="GTiff",
        width=512,
        height=512,
        band_count=4,
        dtype="uint16",
        crs="EPSG:32643",
        bounds=(500000.0, 3100000.0, 505120.0, 3105120.0),
        bounds_wgs84=(76.90, 28.02, 76.95, 28.07),
        resolution=(10.0, 10.0),
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 3105120.0),
        nodata=0.0,
        acquisition_date=date(2023, 6, 14),
        acquisition_date_source="file_metadata",
        is_georeferenced=True,
        file_size_bytes=2097152,
    )
    assert m.is_georeferenced is True
    assert m.warnings == []


def test_raster_metadata_non_georeferenced_defaults():
    m = RasterMetadata(format="PNG", width=4, height=4)
    assert m.crs is None
    assert m.bounds is None
    assert m.acquisition_date_source == "unknown"
    assert m.is_georeferenced is False


def test_compatibility_report_fail():
    r = CompatibilityReport(
        ok=False,
        checks=[
            CompatibilityCheck(name="modality_match", status="PASS", detail="Both OPTICAL"),
            CompatibilityCheck(name="overlap", status="FAIL", detail="Overlap 0.12 below 0.5"),
        ],
        overlap_ratio=0.12,
        coregistration=None,
        alignment_possible=False,
    )
    assert r.ok is False
    assert r.checks[1].status == "FAIL"


def test_specialist_result_no_confidence():
    r = SpecialistResult(
        answer="Water is visible in the lower-left quadrant.",
        evidence={"boxes": [[120, 340, 210, 410]], "labels": ["water"]},
        model_or_tool="grounding.deterministic_cv",
        model_version="cv-grounding-v1",
    )
    assert r.confidence is None
    assert r.confidence_source == "unavailable"
    assert r.used_fallback is False


def test_specialist_spec():
    s = SpecialistSpec(
        id="vqa.smolvlm_base",
        task="VQA",
        name="SmolVLM-256M-Instruct",
        kind="learned_model",
        input_count=1,
        modalities=["OPTICAL", "SAR"],
        formats=["tif", "png", "jpg"],
        confidence_available=False,
        version="smolvlm-256m-instruct-base",
        priority=5,
    )
    assert s.is_fallback is False
    assert s.is_rs_adapted is False


def test_trace_event():
    e = TraceEvent(
        step="SPECIALIST_SELECTED",
        status="COMPLETED",
        detail="Selected grounding.deterministic_cv",
        data={"spec_id": "grounding.deterministic_cv"},
        timestamp=datetime.now(timezone.utc),
        duration_ms=3,
    )
    assert e.duration_ms >= 0


def test_specialist_result_to_legacy_response_fields():
    r = SpecialistResult(
        answer="A region in the upper-right quadrant changed.",
        evidence={"boxes": [[1, 2, 3, 4]], "mask_path": "m.png", "overlay_path": "o.png"},
        confidence=0.8,
        confidence_source="box fill ratio",
        model_or_tool="vqa.smolvlm_base",
        model_version="smolvlm-256m-instruct-base",
    )
    legacy = specialist_result_to_legacy_response_fields(r)
    assert legacy["answer_text"] == r.answer
    assert legacy["confidence_score"] == 0.8
    assert legacy["bounding_boxes"] == [[1, 2, 3, 4]]
    assert legacy["change_mask_url"] == "m.png"
    assert legacy["overlay_url"] == "o.png"
    assert legacy["model_version"] == "smolvlm-256m-instruct-base"
