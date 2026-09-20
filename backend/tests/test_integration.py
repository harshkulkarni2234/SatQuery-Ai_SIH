"""Integration tests: specialists wired into POST /query."""

import io

from PIL import Image as PILImage

from tests.conftest import client, upload_png_bytes


# ── Helpers ───────────────────────────────────────────────────────────

def _water_image_png(size=200, region=(40, 40, 120, 120)):
    """Solid tan background with a blue 'water' region. Returns PNG bytes."""
    img = PILImage.new("RGB", (size, size), color=(120, 120, 60))
    px = img.load()
    x1, y1, x2, y2 = region
    for y in range(y1, y2):
        for x in range(x1, x2):
            px[x, y] = (0, 80, 200)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _solid_png(color, size=100):
    buf = io.BytesIO()
    PILImage.new("RGB", (size, size), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _solid_png_with_white_region(size=100, region=(20, 20, 70, 70)):
    """Dark background with a large white rectangular region."""
    img = PILImage.new("RGB", (size, size), color=(40, 40, 40))
    px = img.load()
    x1, y1, x2, y2 = region
    for y in range(y1, y2):
        for x in range(x1, x2):
            px[x, y] = (255, 255, 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── GROUNDING ─────────────────────────────────────────────────────────

class TestGroundingIntegration:

    def test_grounding_query_runs_real_service(self):
        image_id = upload_png_bytes(_water_image_png(), modality="OPTICAL", name="water.png")

        resp = client.post(
            "/query",
            json={"query_text": "Show me the water", "image_ids": [image_id]},
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["task_classified"] == "GROUNDING"
        assert body["bounding_boxes"], "Expected at least one bounding box"
        assert len(body["bounding_boxes"][0]) == 4
        assert body["confidence_score"] is not None and body["confidence_score"] > 0
        assert "water" in body["answer_text"].lower()
        assert "Detected 1 water" in body["answer_text"]
        assert body["metadata"]["object_type"] == "water"
        assert body["metadata"]["num_regions"] >= 1

        trace = body["execution_trace"]
        assert trace["selected_tool"] == "GROUNDING"
        assert trace["execution_status"] == "completed"
        assert trace["model_version"] is None
        assert trace["confidence_score"] is not None
        assert trace["execution_time_ms"] is not None and trace["execution_time_ms"] >= 0
        assert "keyword" in trace["reason"]

    def test_grounding_result_is_persisted(self):
        image_id = upload_png_bytes(_water_image_png(), modality="OPTICAL")

        post_resp = client.post(
            "/query",
            json={"query_text": "Show me the water", "image_ids": [image_id]},
        )
        query_id = post_resp.json()["query_id"]

        get_resp = client.get(f"/query/{query_id}")
        body = get_resp.json()
        assert body["query_id"] == query_id
        assert body["task_classified"] == "GROUNDING"
        assert body["bounding_boxes"]
        assert body["execution_trace"]["execution_status"] == "completed"

    def test_grounding_no_target_found_is_truthful(self):
        # A green image with no water content
        image_id = upload_png_bytes(_solid_png((50, 150, 50)), modality="OPTICAL")

        resp = client.post(
            "/query",
            json={"query_text": "Show me the water", "image_ids": [image_id]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_classified"] == "GROUNDING"
        assert not body["bounding_boxes"]
        assert "No water" in body["answer_text"]

    def test_grounding_with_two_images_rejected(self):
        id1 = upload_png_bytes(_water_image_png())
        id2 = upload_png_bytes(_solid_png((120, 120, 60)))

        resp = client.post(
            "/query",
            json={"query_text": "Show me the water", "image_ids": [id1, id2]},
        )
        assert resp.status_code == 400
        assert "exactly 1 image" in resp.json()["detail"]["message"]


# ── CHANGE DETECTION ──────────────────────────────────────────────────

class TestChangeDetectionIntegration:

    def test_change_detection_executes_real_service(self):
        before_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="OPTICAL", name="before.png")
        after_id = upload_png_bytes(
            _solid_png_with_white_region(), modality="OPTICAL", name="after.png"
        )

        resp = client.post(
            "/query",
            json={
                "query_text": "What changed between these two images?",
                "image_ids": [before_id, after_id],
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["task_classified"] == "CHANGE_DETECTION"
        assert body["bounding_boxes"], "Expected at least one change region"
        assert body["change_mask_url"] is not None
        assert body["metadata"]["change_percentage"] > 0
        assert body["metadata"]["num_regions"] >= 1
        assert body["metadata"]["specialist"] == "change_detection"
        # The worker is stubbed unavailable (conftest), so the deterministic
        # specialist is the one that executed — and the record must say so.
        assert body["metadata"]["specialist_id"] == "change.deterministic_cv"
        assert "were detected" in body["answer_text"]

        trace = body["execution_trace"]
        assert trace["selected_tool"] == "CHANGE_DETECTION"
        assert trace["execution_status"] == "completed"
        assert trace["model_version"] is None
        assert trace["execution_time_ms"] is not None and trace["execution_time_ms"] >= 0

    def test_change_mask_url_is_retrievable(self):
        before_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="OPTICAL")
        after_id = upload_png_bytes(_solid_png_with_white_region(), modality="OPTICAL")

        resp = client.post(
            "/query",
            json={
                "query_text": "What changed?",
                "image_ids": [before_id, after_id],
            },
        )
        body = resp.json()
        mask_url = body["change_mask_url"]
        assert mask_url is not None

        mask_resp = client.get(mask_url)
        assert mask_resp.status_code == 200
        assert len(mask_resp.content) > 0  # actual binary mask body

    def test_change_detection_result_is_persisted(self):
        before_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="OPTICAL")
        after_id = upload_png_bytes(_solid_png_with_white_region(), modality="OPTICAL")

        post_resp = client.post(
            "/query",
            json={
                "query_text": "What changed between these two images?",
                "image_ids": [before_id, after_id],
            },
        )
        query_id = post_resp.json()["query_id"]

        get_resp = client.get(f"/query/{query_id}")
        body = get_resp.json()
        assert body["query_id"] == query_id
        assert body["task_classified"] == "CHANGE_DETECTION"
        assert body["bounding_boxes"]
        assert body["change_mask_url"] is not None
        assert body["metadata"]["change_percentage"] > 0
        assert body["execution_trace"]["execution_status"] == "completed"

    def test_change_detection_with_one_image_rejected(self):
        image_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="OPTICAL")

        resp = client.post(
            "/query",
            json={"query_text": "What changed?", "image_ids": [image_id]},
        )
        assert resp.status_code == 400
        assert "exactly 2 images" in resp.json()["detail"]["message"]

    def test_optical_sar_pair_routed_to_cross_modal_even_with_change_wording(self):
        # An OPTICAL + SAR pair is routed to CROSS_MODAL before change
        # validation, so change-related wording is no longer rejected.
        optical_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="OPTICAL")
        sar_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="SAR")

        resp = client.post(
            "/query",
            json={
                "query_text": "What changed between these two images?",
                "image_ids": [optical_id, sar_id],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["task_classified"] == "CROSS_MODAL"


# ── STUBS REMAIN HONEST ───────────────────────────────────────────────

class TestStubs:

    def test_vqa_executes_via_worker_path(self, monkeypatch):
        from app.services import vqa as vqa_module

        image_id = upload_png_bytes(_solid_png((120, 120, 60)), modality="OPTICAL")

        def fake_answer(image_path, query_text):
            return {
                "answer_text": "The base SmolVLM model describes a uniform terrain patch.",
                "model_version": "SmolVLM-256M-Instruct",
                "specialist": False,
                "confidence_score": None,
                "execution_time_ms": 55,
                "error": False,
            }

        monkeypatch.setattr(vqa_module, "answer_question", fake_answer)

        resp = client.post(
            "/query",
            json={"query_text": "What is visible here?", "image_ids": [image_id]},
        )
        body = resp.json()
        assert body["task_classified"] == "VQA"
        assert "[STUB]" not in body["answer_text"]
        assert body["bounding_boxes"] is None
        assert body["execution_trace"]["execution_status"] == "completed"
        assert body["execution_trace"]["model_version"] == "SmolVLM-256M-Instruct"
        assert body["confidence_score"] is None
        assert body["metadata"]["specialist"] == "vqa"
        assert body["metadata"]["specialist_mode"] == "base"

    def test_vqa_target_query_never_fabricates_boxes(self, monkeypatch):
        """Object-specific VQA questions get real deterministic grounding boxes
        when the object is detectable — and none otherwise."""
        from app.services import vqa as vqa_module

        # Green optical image: vegetation IS detectable
        green_id = upload_png_bytes(_solid_png((60, 140, 60)), modality="OPTICAL")

        def fake_answer(image_path, query_text):
            return {
                "answer_text": "Yes, vegetation is present.",
                "model_version": "SmolVLM-256M-Instruct",
                "specialist": False,
                "confidence_score": None,
                "execution_time_ms": 10,
                "error": False,
            }

        monkeypatch.setattr(vqa_module, "answer_question", fake_answer)

        resp = client.post(
            "/query",
            json={"query_text": "Is there vegetation in the image?", "image_ids": [green_id]},
        )
        body = resp.json()
        assert body["task_classified"] == "VQA"
        assert body["bounding_boxes"], "Vegetation is present, grounding must find it"
        assert body["metadata"]["visual_evidence"]["available"] is True
        assert body["metadata"]["visual_evidence"]["target"] == "vegetation"

        # Blue-only image: no building should be found -> honest None, not a box
        blue_id = upload_png_bytes(_solid_png((160, 60, 60)), modality="OPTICAL")
        resp = client.post(
            "/query",
            json={
                "query_text": "Is there a building in the image?",
                "image_ids": [blue_id],
            },
        )
        body = resp.json()
        assert body["task_classified"] == "VQA"
        assert body["bounding_boxes"] is None
        assert body["metadata"]["visual_evidence"]["available"] is False
        assert "Spatial localization unavailable" in body["metadata"]["visual_evidence"]["message"]

    def test_vqa_descriptive_query_reports_caption_cues_only(self, monkeypatch):
        """Open-ended descriptions must not claim localized boxes."""
        from app.services import vqa as vqa_module

        image_id = upload_png_bytes(_solid_png((60, 140, 60)), modality="OPTICAL")

        def fake_answer(image_path, query_text):
            return {
                "answer_text": "A uniform vegetation-like terrain patch.",
                "model_version": "SmolVLM-256M-Instruct",
                "specialist": False,
                "confidence_score": None,
                "execution_time_ms": 10,
                "error": False,
            }

        monkeypatch.setattr(vqa_module, "answer_question", fake_answer)

        resp = client.post(
            "/query",
            json={"query_text": "What is visible in this area?", "image_ids": [image_id]},
        )
        body = resp.json()
        assert body["bounding_boxes"] is None
        ve = body["metadata"]["visual_evidence"]
        assert ve["available"] is False
        assert "dominant_cue" in ve or "detected_cues" in ve

    def test_cross_modal_reports_per_sensor_regions(self):
        optical_id = upload_png_bytes(_solid_png((60, 140, 60)), modality="OPTICAL")
        sar_id = upload_png_bytes(_solid_png((30, 30, 30)), modality="SAR")

        resp = client.post(
            "/query",
            json={"query_text": "Analyze this image pair together", "image_ids": [optical_id, sar_id]},
        )
        assert resp.status_code == 200
        body = resp.json()
        opt_regions = body["metadata"]["evidence"]["optical"]["regions"]
        sar_regions = body["metadata"]["evidence"]["sar"]["regions"]
        assert opt_regions, "Optical sensor must report land-cover regions"
        assert sar_regions, "SAR sensor must report backscatter regions"
        assert all(r["box"] and len(r["box"]) == 4 for r in opt_regions)
        labels = {r["label"] for r in opt_regions + sar_regions}
        assert labels, "Per-region labels must be provided"
        assert "spatial_correspondence_note" in body["metadata"]
        assert "could not be verified" in body["metadata"]["spatial_correspondence_note"].lower()

    def test_cross_modal_executes_deterministic_service(self):
        optical_id = upload_png_bytes(_solid_png((60, 140, 60)), modality="OPTICAL")
        sar_id = upload_png_bytes(_solid_png((30, 30, 30)), modality="SAR")

        resp = client.post(
            "/query",
            json={"query_text": "Analyze this image pair together", "image_ids": [optical_id, sar_id]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_classified"] == "CROSS_MODAL"
        assert "[STUB]" not in body["answer_text"]
        assert body["confidence_score"] is None
        assert body["bounding_boxes"] is None
        assert body["execution_trace"]["execution_status"] == "completed"
        assert body["execution_trace"]["model_version"] == "cross-modal-deterministic-v2"
        assert body["metadata"]["specialist"] == "cross_modal"
        assert "modality_contribution_note" in body["metadata"]
        assert "optical" in body["metadata"]["modality_contribution_note"].lower()
        assert body["metadata"]["evidence"]["optical"]["vegetation_green_dominance"] >= 0.9
        assert body["metadata"]["evidence"]["sar"]["dark_low_backscatter_fraction"] >= 0.9

    def test_cross_modal_same_modality_pair_rejected(self):
        optical_id = upload_png_bytes(_solid_png((60, 140, 60)), modality="OPTICAL")
        optical2_id = upload_png_bytes(_solid_png((80, 120, 80)), modality="OPTICAL")

        resp = client.post(
            "/query",
            json={"query_text": "Analyze this image pair together", "image_ids": [optical_id, optical2_id]},
        )
        assert resp.status_code == 400
        assert "OPTICAL + SAR" in resp.json()["detail"]["message"]


# ── Generic validation ────────────────────────────────────────────────

class TestValidation:

    def test_nonexistent_image_id_rejected(self):
        import uuid

        resp = client.post(
            "/query",
            json={"query_text": "What is here?", "image_ids": [str(uuid.uuid4())]},
        )
        assert resp.status_code == 404