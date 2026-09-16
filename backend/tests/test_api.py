import uuid
import io
import pytest
from PIL import Image as PILImage


# ── 1. Health endpoint ────────────────────────────────────────────────

def test_health():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── 2. Image upload accepts valid image ──────────────────────────────

def test_upload_valid_image():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), color=(100, 200, 50)).save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        "/images/upload",
        files={"file": ("satellite.png", buf, "image/png")},
        data={"modality": "OPTICAL"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "image_id" in body
    assert body["filename"] == "satellite.png"
    assert body["modality"] == "OPTICAL"


# ── 3. Invalid modality is rejected ──────────────────────────────────

def test_upload_invalid_modality():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    buf = io.BytesIO()
    PILImage.new("RGB", (4, 4)).save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        "/images/upload",
        files={"file": ("img.png", buf, "image/png")},
        data={"modality": "INVALID"},
    )
    assert resp.status_code == 400


# ── 4. Query rejects nonexistent image IDs ───────────────────────────

def test_query_rejects_nonexistent_ids():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    fake_id = str(uuid.uuid4())
    resp = client.post(
        "/query",
        json={"query_text": "What is here?", "image_ids": [fake_id]},
    )
    assert resp.status_code == 404


# ── 5. Query accepts valid image IDs ─────────────────────────────────

def test_query_valid_image_ids(uploaded_image_id, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.services import vqa as vqa_module
    client = TestClient(app)

    def fake_answer(image_path, query_text):
        return {
            "answer_text": "A terrain sample.",
            "model_version": "SmolVLM-256M-Instruct",
            "specialist": False,
            "confidence_score": None,
            "execution_time_ms": 42,
            "error": False,
        }

    monkeypatch.setattr(vqa_module, "answer_question", fake_answer)

    resp = client.post(
        "/query",
        json={"query_text": "What is visible here?", "image_ids": [uploaded_image_id]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "query_id" in body
    assert body["task_classified"] == "VQA"
    assert "execution_trace" in body
    assert body["execution_trace"]["reason"] is not None
    assert "[STUB]" not in body["answer_text"]
    assert body["execution_trace"]["execution_status"] == "completed"
    assert body["execution_trace"]["model_version"] == "SmolVLM-256M-Instruct"


# ── 6. Query retrieval works ─────────────────────────────────────────

def test_query_retrieval(uploaded_image_id):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    post_resp = client.post(
        "/query",
        json={"query_text": "Show me the water", "image_ids": [uploaded_image_id]},
    )
    query_id = post_resp.json()["query_id"]

    get_resp = client.get(f"/query/{query_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["query_id"] == query_id


# ── 7. Nonexistent query ID returns 404 ──────────────────────────────

def test_get_nonexistent_query():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.get(f"/query/{uuid.uuid4()}")
    assert resp.status_code == 404
