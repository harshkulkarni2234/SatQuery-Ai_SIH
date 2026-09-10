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
        assert "exactly 1 image" in resp.json()["detail"]


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
        assert "changed" in body["answer_text"]

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
        assert "exactly 2 images" in resp.json()["detail"]

    def test_incompatible_modalities_rejected(self):
        optical_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="OPTICAL")
        sar_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="SAR")

        resp = client.post(
            "/query",
            json={
                "query_text": "What changed between these two images?",
                "image_ids": [optical_id, sar_id],
            },
        )
        assert resp.status_code == 400
        assert "different modalities" in resp.json()["detail"]


# ── STUBS REMAIN HONEST ───────────────────────────────────────────────

class TestStubs:

    def test_vqa_remains_honest_stub(self):
        image_id = upload_png_bytes(_solid_png((120, 120, 60)), modality="OPTICAL")

        resp = client.post(
            "/query",
            json={"query_text": "What is visible here?", "image_ids": [image_id]},
        )
        body = resp.json()
        assert body["task_classified"] == "VQA"
        assert "[STUB]" in body["answer_text"]
        assert body["bounding_boxes"] is None
        assert body["execution_trace"]["execution_status"] == "not_implemented"
        assert body["execution_trace"]["model_version"] is None

    def test_cross_modal_remains_honest_stub(self):
        optical_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="OPTICAL")
        sar_id = upload_png_bytes(_solid_png((40, 40, 40)), modality="SAR")

        resp = client.post(
            "/query",
            json={"query_text": "Analyze this image pair together", "image_ids": [optical_id, sar_id]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_classified"] == "CROSS_MODAL"
        assert "[STUB]" in body["answer_text"]
        assert body["execution_trace"]["execution_status"] == "not_implemented"


# ── Generic validation ────────────────────────────────────────────────

class TestValidation:

    def test_nonexistent_image_id_rejected(self):
        import uuid

        resp = client.post(
            "/query",
            json={"query_text": "What is here?", "image_ids": [str(uuid.uuid4())]},
        )
        assert resp.status_code == 404