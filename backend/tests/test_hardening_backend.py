"""Phase A8: unit-level backend hardening tests not already covered by
test_raster_ingest.py / test_images_metadata.py / test_planner.py / test_query_compatibility_api.py.

Covers: upload size limit, the global exception handler (no internals
leaked to the client on an unexpected error), and /health reporting DB
status honestly even when the DB check itself fails.
"""

import io
import os

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.main import app
from app.routes import images as images_module
from app.routes import query as query_module

client = TestClient(app)


def _png_bytes(size=(8, 8)):
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_over_size_limit_rejected_and_file_not_left_on_disk(monkeypatch):
    monkeypatch.setattr(images_module, "MAX_UPLOAD_SIZE_BYTES", 100)
    payload = _png_bytes((64, 64))  # a real PNG comfortably over 100 bytes
    assert len(payload) > 100

    before = set(os.listdir(images_module.DATA_DIR)) if os.path.isdir(images_module.DATA_DIR) else set()
    resp = client.post(
        "/images/upload",
        files={"file": ("big.png", payload, "image/png")},
        data={"modality": "OPTICAL"},
    )
    assert resp.status_code == 413
    assert "exceeds maximum upload size" in resp.json()["detail"]

    after = set(os.listdir(images_module.DATA_DIR)) if os.path.isdir(images_module.DATA_DIR) else set()
    assert after == before  # partial file was cleaned up, nothing orphaned


def test_upload_under_size_limit_still_succeeds(monkeypatch):
    monkeypatch.setattr(images_module, "MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
    resp = client.post(
        "/images/upload",
        files={"file": ("small.png", _png_bytes((8, 8)), "image/png")},
        data={"modality": "OPTICAL"},
    )
    assert resp.status_code == 200


def test_unhandled_exception_returns_clean_500_without_leaking_internals(monkeypatch):
    """An error raised outside the route's own try/except (e.g. a planner
    crash) must not reach the client as a raw traceback or exception
    message — the global handler in main.py should catch it."""

    def _boom(*args, **kwargs):
        raise RuntimeError("super secret internal db connection string leaked")

    upload_resp = client.post(
        "/images/upload",
        files={"file": ("probe.png", _png_bytes((8, 8)), "image/png")},
        data={"modality": "OPTICAL"},
    )
    assert upload_resp.status_code == 200
    image_id = upload_resp.json()["image_id"]

    monkeypatch.setattr(query_module, "build_plan", _boom)

    # The default TestClient re-raises server exceptions (useful for other
    # tests' stack traces); here we want the real HTTP response the global
    # handler produces, exactly as a real client over the network would see it.
    no_raise_client = TestClient(app, raise_server_exceptions=False)
    resp = no_raise_client.post(
        "/query",
        json={"query_text": "what is here?", "image_ids": [image_id]},
    )
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error"}
    assert "secret" not in resp.text
    assert "RuntimeError" not in resp.text


def test_health_reports_db_down_without_crashing(monkeypatch):
    from app import main as main_module

    def _raise(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(main_module, "SessionLocal", _raise)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["database"] == "down"
