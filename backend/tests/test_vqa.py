"""Hermetic VQA tests — mocked worker HTTP, no GPU or model download required."""

import requests

from app.services import vqa as vqa_module
from tests.conftest import client, upload_png_bytes


# ── Helpers ───────────────────────────────────────────────────────────


def _base_success(*args, **kwargs):
    return _make_response({
        "answer_text": "There is a cloud-free terrain.",
        "model_version": "SmolVLM-256M-Instruct",
        "specialist": False,
        "adapter_used": False,
        "reason": "base mode (not requested for this question)",
        "confidence_score": None,
        "execution_time_ms": 33,
    })


def _specialist_success(*args, **kwargs):
    return _make_response({
        "answer_text": "Yes.",
        "model_version": "smolvlm256m-ben-lora-s3-v1.0",
        "specialist": True,
        "adapter_used": True,
        "reason": "specialist requested and available; ran successfully",
        "confidence_score": None,
        "execution_time_ms": 48,
    })


def _make_response(payload, status_code=200):
    class FakeResp:
        def __init__(self, payload, status_code):
            self.status_code = status_code
            self._payload = payload
        def json(self):
            return self._payload
    return FakeResp(payload, status_code)


def _error_post(*args, **kwargs):
    raise requests.ConnectionError("refused")


def _malformed_post(*args, **kwargs):
    return _make_response("not-a-dict")


def _server_error_post(*args, **kwargs):
    return _make_response({}, status_code=500)


def _png_bytes():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(0, 100, 200)).save(buf, format="PNG")
    return buf.getvalue()


# ── should_use_specialist unit tests ──────────────────────────────────

class TestSpecialistGate:

    def test_presence_true(self):
        assert vqa_module.should_use_specialist("Is there a road?")
        assert vqa_module.should_use_specialist("Are there buildings?")
        assert vqa_module.should_use_specialist("Does the image contain water?")

    def test_count_true(self):
        assert vqa_module.should_use_specialist("How many buildings are visible?")
        assert vqa_module.should_use_specialist("How many roads?")

    def test_open_ended_false(self):
        assert not vqa_module.should_use_specialist("What is visible here?")
        assert not vqa_module.should_use_specialist("Describe this image.")
        assert not vqa_module.should_use_specialist("Show me the water")
        assert not vqa_module.should_use_specialist("What changed?")
        assert not vqa_module.should_use_specialist("Compare the two images.")
        assert not vqa_module.should_use_specialist("Is the terrain optical?")

    def test_empty_false(self):
        assert not vqa_module.should_use_specialist("")
        assert not vqa_module.should_use_specialist(None)

    def test_long_query_false(self):
        assert not vqa_module.should_use_specialist(
            "Is there any building in the upper left region of this satellite "
            "image which was captured by Sentinel-2 entirely during a clear "
            "morning with absolutely no cloud cover whatsoever?"
        )


# ── vqa.py answer_question unit tests ────────────────────────────────

class TestVQAServicer:

    def test_base_request(self, tmp_path, monkeypatch):
        fake_img = tmp_path / "a.png"
        fake_img.write_bytes(_png_bytes())
        monkeypatch.setattr(vqa_module, "_post", _base_success)
        result = vqa_module.answer_question(str(fake_img), "What is visible here?")
        assert result["answer_text"]
        assert result["model_version"] == "SmolVLM-256M-Instruct"
        assert result["specialist"] is False
        assert result["adapter_used"] is False
        assert result["reason"] == "base mode (not requested for this question)"
        assert result["confidence_score"] is None
        assert result["error"] is False
        assert isinstance(result["execution_time_ms"], int)

    def test_specialist_selected(self, tmp_path, monkeypatch):
        fake_img = tmp_path / "b.png"
        fake_img.write_bytes(_png_bytes())
        monkeypatch.setattr(vqa_module, "_post", _specialist_success)
        result = vqa_module.answer_question(str(fake_img), "Is there a road?")
        assert result["specialist"] is True
        assert result["adapter_used"] is True
        assert result["model_version"] == "smolvlm256m-ben-lora-s3-v1.0"
        assert "specialist requested and available" in result["reason"]
        assert result["confidence_score"] is None

    def test_worker_offline(self, tmp_path, monkeypatch):
        fake_img = tmp_path / "c.png"
        fake_img.write_bytes(_png_bytes())
        monkeypatch.setattr(vqa_module, "_post", _error_post)
        result = vqa_module.answer_question(str(fake_img), "What is here?")
        assert result["error"] is True
        assert "[VQA unavailable]" in result["answer_text"]
        assert result["model_version"] is None
        assert result["adapter_used"] is False
        assert "worker unavailable" in result["reason"]
        assert result["confidence_score"] is None

    def test_malformed_worker_response(self, tmp_path, monkeypatch):
        fake_img = tmp_path / "d.png"
        fake_img.write_bytes(_png_bytes())
        monkeypatch.setattr(vqa_module, "_post", _malformed_post)
        result = vqa_module.answer_question(str(fake_img), "What is here?")
        assert result["error"] is True
        assert result["model_version"] is None

    def test_non_200_worker(self, tmp_path, monkeypatch):
        fake_img = tmp_path / "e.png"
        fake_img.write_bytes(_png_bytes())
        monkeypatch.setattr(vqa_module, "_post", _server_error_post)
        result = vqa_module.answer_question(str(fake_img), "What is here?")
        assert result["error"] is True

    def test_confidence_null_on_success(self, tmp_path, monkeypatch):
        fake_img = tmp_path / "f.png"
        fake_img.write_bytes(_png_bytes())
        monkeypatch.setattr(vqa_module, "_post", _base_success)
        result = vqa_module.answer_question(str(fake_img), "What is here?")
        assert result["confidence_score"] is None

    def test_model_version_returned(self, tmp_path, monkeypatch):
        fake_img = tmp_path / "g.png"
        fake_img.write_bytes(_png_bytes())
        monkeypatch.setattr(vqa_module, "_post", _specialist_success)
        result = vqa_module.answer_question(str(fake_img), "Is there a road?")
        assert isinstance(result["model_version"], str)
        assert len(result["model_version"]) > 0


# ── Route-level VQA integration (mocked worker, real DB) ─────────────

class TestVQARoute:

    def test_specialist_selected_request(self, monkeypatch):
        image_id = upload_png_bytes(_png_bytes(), modality="OPTICAL", name="spec.png")

        def fake_answer(image_path, query_text):
            return {
                "answer_text": "Yes.",
                "model_version": "SmolVLM-256M + Stage3LoRA (exp)",
                "specialist": True,
                "confidence_score": None,
                "execution_time_ms": 60,
                "error": False,
            }

        monkeypatch.setattr(vqa_module, "answer_question", fake_answer)

        resp = client.post(
            "/query",
            json={"query_text": "Is there a road in this image?", "image_ids": [image_id]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_classified"] == "VQA"
        assert body["execution_trace"]["execution_status"] == "completed"
        assert body["execution_trace"]["model_version"] == "SmolVLM-256M + Stage3LoRA (exp)"
        assert body["confidence_score"] is None
        assert body["metadata"]["specialist_mode"] == "experimental"

    def test_worker_offline_via_service(self, monkeypatch):
        image_id = upload_png_bytes(_png_bytes(), modality="OPTICAL", name="off.png")
        monkeypatch.setattr(vqa_module, "_post", _error_post)

        resp = client.post(
            "/query",
            json={"query_text": "What is visible here?", "image_ids": [image_id]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_classified"] == "VQA"
        assert "[VQA unavailable]" in body["answer_text"]
        assert body["execution_trace"]["execution_status"] == "unavailable"
        assert body["execution_trace"]["model_version"] is None
        assert body["confidence_score"] is None

    def test_confidence_null_in_trace(self, monkeypatch):
        image_id = upload_png_bytes(_png_bytes(), modality="OPTICAL", name="trace.png")

        def fake_answer(image_path, query_text):
            return {
                "answer_text": "Open terrain.",
                "model_version": "SmolVLM-256M-Instruct",
                "specialist": False,
                "confidence_score": None,
                "execution_time_ms": 31,
                "error": False,
            }

        monkeypatch.setattr(vqa_module, "answer_question", fake_answer)
        resp = client.post(
            "/query",
            json={"query_text": "What is visible?", "image_ids": [image_id]},
        )
        body = resp.json()
        assert body["confidence_score"] is None
        assert body["execution_trace"]["confidence_score"] is None