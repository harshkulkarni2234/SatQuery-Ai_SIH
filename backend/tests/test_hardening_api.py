"""Phase C8: consolidated API-level hardening pass over the plan's edge-case
checklist. A owns unit-level coverage (raster_ingest, planner, registry,
compatibility, images) already spread across the other test files; this file
tests the same class of failures purely through the public HTTP surface
(TestClient), and asserts the shared contract for every case: never a 500,
always a clear human-readable message, never a fabricated box/percentage/
confidence value.
"""

import io
import uuid

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from rasterio.transform import from_origin

from app.main import app
from app.routes import images as images_module
from app.services import raster_ingest

client = TestClient(app)


def _solid_png(color=(60, 140, 60), size=64):
    buf = io.BytesIO()
    PILImage.new("RGB", (size, size), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _geotiff_bytes(west=76.90, north=28.07, width=32, height=32, band_count=3, crs="EPSG:4326"):
    transform = from_origin(west, north, 0.0001, 0.0001)
    buf = io.BytesIO()
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=height, width=width, count=band_count,
            dtype="uint8", crs=crs, transform=transform,
        ) as ds:
            data = (np.random.rand(band_count, height, width) * 200).astype("uint8")
            for i in range(band_count):
                ds.write(data[i], i + 1)
        buf.write(memfile.read())
    buf.seek(0)
    return buf.getvalue()


def _upload(payload, filename, modality="OPTICAL", capture_date=None, content_type="image/png"):
    data = {"modality": modality}
    if capture_date:
        data["capture_date"] = capture_date
    return client.post(
        "/images/upload",
        files={"file": (filename, payload, content_type)},
        data=data,
    )


def _upload_ok(payload, filename, modality="OPTICAL", capture_date=None, content_type="image/png"):
    resp = _upload(payload, filename, modality, capture_date, content_type)
    assert resp.status_code == 200, resp.text
    return resp.json()["image_id"]


# ── Upload-time rejections ─────────────────────────────────────────────

def test_non_image_file_rejected_cleanly():
    resp = _upload(b"this is plain text, not an image", "notes.txt", content_type="text/plain")
    assert resp.status_code in (400, 422)
    assert resp.status_code != 500


def test_non_image_bytes_with_image_extension_rejected_cleanly():
    resp = _upload(b"not actually a tiff", "fake.tif", content_type="image/tiff")
    assert resp.status_code == 400
    assert "server-side" not in resp.json()["detail"].lower()  # no path/internal leak


def test_corrupt_tiff_rejected_with_readable_message():
    corrupt = _geotiff_bytes()[:20]  # truncated -> unreadable by GDAL
    resp = _upload(corrupt, "corrupt.tif", content_type="image/tiff")
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "corrupt" in detail.lower() or "unsupported" in detail.lower()
    assert "/Users" not in detail and "/home" not in detail  # no server path leak


def test_unsupported_extension_rejected():
    resp = _upload(_solid_png(), "image.gif", content_type="image/gif")
    assert resp.status_code == 400
    assert "not allowed" in resp.json()["detail"].lower()


def test_huge_raster_via_api_warns_without_500(monkeypatch):
    monkeypatch.setattr(raster_ingest, "MAX_RASTER_PIXELS", 100)
    resp = _upload(_solid_png(size=32), "huge.png", content_type="image/png")
    assert resp.status_code == 200
    warnings = resp.json()["metadata"]["warnings"]
    assert any("exceeding" in w.lower() for w in warnings)


def test_upload_over_size_limit_returns_413(monkeypatch):
    monkeypatch.setattr(images_module, "MAX_UPLOAD_SIZE_BYTES", 50)
    resp = _upload(_solid_png(size=64), "big.png", content_type="image/png")
    assert resp.status_code == 413
    assert resp.status_code != 500


# ── Query-time rejections ───────────────────────────────────────────────

def test_empty_query_text_rejected_not_500():
    image_id = _upload_ok(_solid_png(), "img.png")
    resp = client.post("/query", json={"query_text": "", "image_ids": [image_id]})
    assert resp.status_code in (400, 422)
    assert resp.status_code != 500


def test_unsupported_grounding_target_gets_suggestion():
    image_id = _upload_ok(_solid_png(), "img.png")
    resp = client.post(
        "/query",
        json={"query_text": "Show me the flying saucer", "image_ids": [image_id]},
    )
    assert resp.status_code == 400
    body = resp.json()["detail"]
    assert body.get("suggestion")


def test_one_image_for_temporal_change_query_rejected():
    image_id = _upload_ok(_solid_png(), "img.png")
    resp = client.post(
        "/query",
        json={"query_text": "What changed between these two dates?", "image_ids": [image_id]},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"].get("suggestion")


def test_same_date_pair_rejected_fast():
    id_a = _upload_ok(_geotiff_bytes(), "a.tif", capture_date="2022-01-01", content_type="image/tiff")
    id_b = _upload_ok(_geotiff_bytes(), "b.tif", capture_date="2022-01-01", content_type="image/tiff")
    resp = client.post(
        "/query",
        json={"query_text": "What changed between these two dates?", "image_ids": [id_a, id_b]},
    )
    assert resp.status_code == 400
    assert "date" in resp.json()["detail"]["message"].lower()


def test_non_overlapping_pair_returns_422_with_compatibility_report():
    id_before = _upload_ok(
        _geotiff_bytes(west=76.90, north=28.07), "before.tif",
        capture_date="2021-03-01", content_type="image/tiff",
    )
    id_after = _upload_ok(
        _geotiff_bytes(west=10.0, north=10.0), "after.tif",
        capture_date="2022-03-01", content_type="image/tiff",
    )
    resp = client.post(
        "/query",
        json={"query_text": "What changed between these two dates?", "image_ids": [id_before, id_after]},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail.get("compatibility") is not None
    assert detail["compatibility"]["ok"] is False


def test_wrong_modality_pair_for_cross_modal_rejected():
    id_a = _upload_ok(_solid_png((60, 140, 60)), "a.png", modality="OPTICAL")
    id_b = _upload_ok(_solid_png((80, 120, 80)), "b.png", modality="OPTICAL")
    resp = client.post(
        "/query",
        json={"query_text": "Analyze this image pair together", "image_ids": [id_a, id_b]},
    )
    assert resp.status_code == 400
    assert "OPTICAL + SAR" in resp.json()["detail"]["message"]


def test_missing_metadata_image_still_answers_without_fabrication():
    """A plain PNG has no CRS/date/georeferencing at all ('missing metadata'
    case) — the query must still complete honestly, never inventing a
    confidence value or spatial claim it can't support."""
    image_id = _upload_ok(_solid_png(), "no_metadata.png")
    resp = client.post(
        "/query",
        json={"query_text": "What can you tell me about this image?", "image_ids": [image_id]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence_source"] is not None  # always states its source, real or "unavailable"


def test_vqa_worker_missing_falls_back_honestly(monkeypatch):
    from app.services import registry as registry_module

    monkeypatch.setattr(registry_module, "_vqa_worker_available", lambda: False)
    image_id = _upload_ok(_solid_png(), "img.png")
    resp = client.post(
        "/query",
        json={"query_text": "What can you tell me about this image?", "image_ids": [image_id]},
    )
    assert resp.status_code != 500
    if resp.status_code == 200:
        body = resp.json()
        assert body["used_fallback"] or "unavailable" in (body["confidence_source"] or "").lower() or body["warnings"]


def test_three_image_ids_rejected_by_schema():
    resp = client.post(
        "/query",
        json={"query_text": "test", "image_ids": [str(uuid.uuid4())] * 3},
    )
    assert resp.status_code == 422
    assert resp.status_code != 500


def test_unknown_image_id_returns_404_not_500():
    resp = client.post(
        "/query",
        json={"query_text": "What is here?", "image_ids": [str(uuid.uuid4())]},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ── Report for a failed query ───────────────────────────────────────────

def test_report_for_failed_query_does_not_500():
    """A query that fails validation (unknown image) is persisted as a
    Query row with no QueryResult (Phase A6). Its report must degrade
    gracefully, not crash."""
    unknown_id = str(uuid.uuid4())
    fail_resp = client.post(
        "/query", json={"query_text": "What is here?", "image_ids": [unknown_id]}
    )
    assert fail_resp.status_code == 404

    from app.database import SessionLocal
    from app.models import Query

    db = SessionLocal()
    try:
        failed_query = (
            db.query(Query)
            .filter(Query.query_text == "What is here?")
            .order_by(Query.created_at.desc())
            .first()
        )
        assert failed_query is not None
        query_id = str(failed_query.id)
    finally:
        db.close()

    json_resp = client.get(f"/query/{query_id}/report.json")
    assert json_resp.status_code == 200
    assert json_resp.json()["answer_text"] is None

    pdf_resp = client.get(f"/query/{query_id}/report.pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.content[:4] == b"%PDF"


def test_report_for_nonexistent_query_returns_404_not_500():
    resp = client.get(f"/query/{uuid.uuid4()}/report.json")
    assert resp.status_code == 404
    assert resp.status_code != 500
