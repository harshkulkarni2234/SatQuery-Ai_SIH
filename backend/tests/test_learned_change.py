"""Learned change specialist (change.siamese_binary_cnn): guards, honest
fallbacks, artifact handling, and the /query wiring.

Hermetic: the change worker's HTTP call is stubbed (change_specialists._post),
so no torch, weights, or running worker is needed. What is under test is the
backend's handling of the worker contract — NOT the model's quality.
"""

import base64
import io

import numpy as np
import pytest
import requests
from PIL import Image as PILImage

from app.services import change_detection, change_specialists as cs, registry
from tests.conftest import client, upload_png_bytes

LEARNED = "change.siamese_binary_cnn"
DETERMINISTIC = "change.deterministic_cv"


# ── Helpers ───────────────────────────────────────────────────────────

def _png_bytes(color=(40, 40, 40), size=(100, 100), white_box=None):
    img = PILImage.new("RGB", size, color=color)
    if white_box:
        x1, y1, x2, y2 = white_box
        px = img.load()
        for y in range(y1, y2):
            for x in range(x1, x2):
                px[x, y] = (255, 255, 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def _mask_b64(changed_fraction=0.25, size=64, values=(0, 255)):
    arr = np.full((size, size), values[0], dtype=np.uint8)
    n = int(size * size * changed_fraction)
    arr.reshape(-1)[:n] = values[1]
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, bad_json=False):
        self._payload = payload
        self.status_code = status_code
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


def _worker_ok(changed_fraction=0.25, **overrides):
    payload = {
        "model_version": "siamese-cnn-v1",
        "latency_ms": 42,
        "mask_png_base64": _mask_b64(changed_fraction),
        "mask_width": 64,
        "mask_height": 64,
    }
    payload.update(overrides)
    return lambda *a, **k: _FakeResponse(payload)


@pytest.fixture
def pair(tmp_path):
    before = _write(tmp_path, "before.png", _png_bytes())
    after = _write(tmp_path, "after.png", _png_bytes(white_box=(20, 20, 70, 70)))
    return before, after


# ── pixel_size_m / skip_reason ───────────────────────────────────────

class TestSkipReason:
    def test_pixel_size_unknown_without_metadata(self):
        assert cs.pixel_size_m(None) is None
        assert cs.pixel_size_m({"resolution": None}) is None

    def test_pixel_size_projected_crs_is_metres(self):
        assert cs.pixel_size_m({"resolution": (10.0, 10.0), "crs": "EPSG:32643", "transform": (10, 0, 0, 0, -10, 0)}) == 10.0

    def test_pixel_size_geographic_crs_is_converted(self):
        size = cs.pixel_size_m({"resolution": (0.0001, 0.0001), "crs": "EPSG:4326", "transform": (1e-4, 0, 0, 0, -1e-4, 0)})
        assert 10 < size < 12

    def test_pair_inside_domain_is_allowed(self, pair):
        assert cs.skip_reason(*pair, None, None) is None
        fine = {"resolution": (0.5, 0.5), "crs": "EPSG:32643", "transform": (0.5, 0, 0, 0, -0.5, 0)}
        assert cs.skip_reason(*pair, fine, fine) is None

    def test_size_mismatch_is_skipped(self, tmp_path):
        a = _write(tmp_path, "a.png", _png_bytes(size=(100, 100)))
        b = _write(tmp_path, "b.png", _png_bytes(size=(120, 100)))
        assert "differ in size" in cs.skip_reason(a, b, None, None)

    def test_unreadable_input_is_skipped(self, tmp_path, pair):
        junk = _write(tmp_path, "junk.png", b"not an image")
        assert "could not be read" in cs.skip_reason(junk, pair[1], None, None)

    def test_provably_different_areas_are_skipped(self, pair):
        reason = cs.skip_reason(*pair, {"tile_id": "T43QGF"}, {"tile_id": "T44RKV"})
        assert "do not cover the same area" in reason

    def test_coarse_imagery_is_outside_training_domain(self, pair):
        coarse = {"resolution": (10.0, 10.0), "crs": "EPSG:32643", "transform": (10, 0, 0, 0, -10, 0)}
        reason = cs.skip_reason(*pair, coarse, coarse)
        assert "coarser than" in reason and "SECOND" in reason


# ── worker client / mask decoding ────────────────────────────────────

class TestWorkerClient:
    def test_valid_response_is_returned(self, monkeypatch, pair):
        monkeypatch.setattr(cs, "_post", _worker_ok())
        data = cs.call_worker(*pair)
        assert data["model_version"] == "siamese-cnn-v1"

    @pytest.mark.parametrize(
        "post",
        [
            lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")),
            lambda *a, **k: _FakeResponse({}, status_code=500),
            lambda *a, **k: _FakeResponse(bad_json=True),
            lambda *a, **k: _FakeResponse(["not", "a", "dict"]),
            lambda *a, **k: _FakeResponse({"model_version": "v", "latency_ms": 1}),  # no mask
            lambda *a, **k: _FakeResponse({"latency_ms": 1, "mask_png_base64": "AAAA"}),  # no version
        ],
    )
    def test_any_bad_response_is_none(self, monkeypatch, pair, post):
        monkeypatch.setattr(cs, "_post", post)
        assert cs.call_worker(*pair) is None

    def test_decode_mask_accepts_binary_png(self):
        mask = cs.decode_mask(_mask_b64(0.25, size=8))
        assert mask.shape == (8, 8) and mask.sum() == 16

    def test_decode_mask_rejects_garbage_and_non_binary(self):
        assert cs.decode_mask("!!!not base64!!!") is None
        assert cs.decode_mask(base64.b64encode(b"not a png").decode()) is None
        assert cs.decode_mask(_mask_b64(0.25, values=(0, 128))) is None  # grey values


# ── run_learned_change ───────────────────────────────────────────────

class TestRunLearnedChange:
    def test_success_is_reported_as_learned_and_binary(self, monkeypatch, pair):
        monkeypatch.setattr(cs, "_post", _worker_ok(0.25))
        r = cs.run_learned_change(*pair)
        assert r.model_or_tool == LEARNED
        assert r.used_fallback is False and r.fallback_reason is None
        assert r.model_version == "siamese-cnn-v1"
        assert r.confidence is None and r.confidence_source == "unavailable"
        stats = r.evidence["stats"]
        assert stats["change_percentage"] == 25.0
        assert stats["changed_pixels"] == 1024 and stats["total_pixels"] == 4096
        # No fabricated deterministic-only fields.
        assert stats["num_regions"] is None and stats["changed_area_m2"] is None
        assert r.evidence["boxes"] is None
        # Honest scope in the answer text.
        assert "not what they changed to" in r.answer
        assert "SECOND" in r.answer
        # The eval scorer takes the FIRST number as the percentage.
        import re
        assert re.search(r"-?\d+(?:\.\d+)?", r.answer).group() == "25.0"
        # Unknown resolution is surfaced as a warning, not hidden.
        assert any("resolution is unknown" in w for w in r.warnings)

    def test_mask_and_overlay_are_saved_by_the_backend_and_unique(self, monkeypatch, pair):
        monkeypatch.setattr(cs, "_post", _worker_ok(0.25))
        r1 = cs.run_learned_change(*pair)
        r2 = cs.run_learned_change(*pair)
        m1, m2 = r1.evidence["mask_path"], r2.evidence["mask_path"]
        assert m1 != m2  # no filename collision between requests
        for path in (m1, m2, r1.evidence["overlay_path"]):
            assert path.startswith(change_detection.MASK_DIR)
            assert PILImage.open(path).size == (100, 100)  # upscaled to the after image

    def test_worker_down_falls_back_and_says_so(self, monkeypatch, pair):
        monkeypatch.setattr(cs, "_post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
        r = cs.run_learned_change(*pair)
        assert r.model_or_tool == DETERMINISTIC
        assert r.used_fallback is True
        assert r.evidence["skipped_planned_reason"]
        assert any("deterministic pixel-difference method ran instead" in w for w in r.warnings)
        assert "Learned" not in r.answer and "Siamese" not in r.answer

    def test_invalid_mask_never_becomes_a_zero_percent_result(self, monkeypatch, pair):
        monkeypatch.setattr(cs, "_post", _worker_ok(mask_png_base64="AAAA"))
        r = cs.run_learned_change(*pair)
        assert r.model_or_tool == DETERMINISTIC and r.used_fallback is True
        assert "invalid mask" in r.evidence["skipped_planned_reason"]

    def test_near_full_frame_change_is_treated_as_non_corresponding(self, monkeypatch, pair):
        monkeypatch.setattr(cs, "_post", _worker_ok(0.95))
        r = cs.run_learned_change(*pair)
        assert r.model_or_tool == DETERMINISTIC and r.used_fallback is True
        assert "do not show the same area" in r.evidence["skipped_planned_reason"]

    def test_coarse_imagery_never_calls_the_worker(self, monkeypatch, pair):
        def boom(*a, **k):
            raise AssertionError("worker must not be called outside the training domain")

        monkeypatch.setattr(cs, "_post", boom)
        coarse = {"resolution": (10.0, 10.0), "crs": "EPSG:32643", "transform": (10, 0, 0, 0, -10, 0)}
        r = cs.run_learned_change(*pair, coarse, coarse)
        assert r.model_or_tool == DETERMINISTIC and r.used_fallback is True
        assert "coarser" in r.evidence["skipped_planned_reason"]

    def test_provable_non_correspondence_is_refused_not_answered(self, monkeypatch, pair):
        def boom(*a, **k):
            raise AssertionError("worker must not be called for provably different areas")

        monkeypatch.setattr(cs, "_post", boom)
        r = cs.run_learned_change(*pair, {"tile_id": "T43QGF"}, {"tile_id": "T44RKV"})
        # Same refusal the deterministic path always gave — not a change % at all.
        assert r.model_or_tool == DETERMINISTIC
        assert r.evidence["validation_failed"] is True
        assert r.answer == change_detection.INCOMPATIBLE_MSG
        assert r.evidence["stats"]["change_percentage"] is None

    def test_deterministic_direct_run_is_not_marked_fallback(self, pair):
        r = cs.run_deterministic_change(*pair)
        assert r.model_or_tool == DETERMINISTIC
        assert r.used_fallback is False and r.fallback_reason is None
        assert "skipped_planned_reason" not in r.evidence


# ── POST /query wiring ───────────────────────────────────────────────

def _upload_pair(before_size=(100, 100), after_size=(100, 100)):
    before = upload_png_bytes(_png_bytes(size=before_size), name="b.png")
    after = upload_png_bytes(_png_bytes(size=after_size, white_box=(20, 20, 70, 70)), name="a.png")
    return before, after


def _ask(before, after):
    resp = client.post(
        "/query",
        json={"query_text": "What changed between these two images?", "image_ids": [before, after]},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestQueryWiring:
    def test_learned_path_end_to_end(self, monkeypatch):
        monkeypatch.setattr(registry, "_change_worker_available", lambda: True)
        monkeypatch.setattr(cs, "_post", _worker_ok(0.25))
        body = _ask(*_upload_pair())
        meta = body["metadata"]

        assert meta["specialist_id"] == LEARNED
        assert meta["validation_failed"] is False
        assert meta["change_percentage"] == 25.0
        assert body["used_fallback"] is False
        assert "planned_specialist_id" not in meta
        assert "not what they changed to" in body["answer_text"]
        assert body["execution_trace"]["execution_status"] == "completed"
        # The served mask URL really resolves.
        assert client.get(body["change_mask_url"]).status_code == 200
        assert client.get(body["overlay_url"]).status_code == 200

    def test_learned_result_survives_persistence_round_trip(self, monkeypatch):
        monkeypatch.setattr(registry, "_change_worker_available", lambda: True)
        monkeypatch.setattr(cs, "_post", _worker_ok(0.25))
        body = _ask(*_upload_pair())
        again = client.get(f"/query/{body['query_id']}").json()
        assert again["change_mask_url"] == body["change_mask_url"]
        assert again["metadata"]["change_percentage"] == 25.0

    def test_runtime_fallback_is_reported_honestly(self, monkeypatch):
        """Worker passes the health probe but the call then fails: the trace
        must record what ACTUALLY ran (deterministic), what was planned, why."""
        monkeypatch.setattr(registry, "_change_worker_available", lambda: True)
        monkeypatch.setattr(cs, "_post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
        body = _ask(*_upload_pair())
        meta = body["metadata"]

        assert meta["specialist_id"] == DETERMINISTIC
        assert meta["planned_specialist_id"] == LEARNED
        assert "Planned change.siamese_binary_cnn but ran change.deterministic_cv" in meta["execution_note"]
        assert body["used_fallback"] is True
        assert any("learned change model was not used" in w for w in body["warnings"])
        # A benign fallback note must NOT flip the UI into "validation failed".
        assert meta["validation_failed"] is False
        assert meta["change_percentage"] > 0
        assert body["execution_trace"]["execution_status"] == "completed"
        assert client.get(body["change_mask_url"]).status_code == 200

    def test_size_mismatch_routes_to_deterministic_and_says_so(self, monkeypatch):
        monkeypatch.setattr(registry, "_change_worker_available", lambda: True)
        monkeypatch.setattr(cs, "_post", _worker_ok(0.25))
        body = _ask(*_upload_pair(after_size=(120, 100)))
        meta = body["metadata"]
        assert meta["specialist_id"] == DETERMINISTIC
        assert meta["planned_specialist_id"] == LEARNED
        assert "differ in size" in meta["execution_note"]

    def test_worker_down_at_plan_time_is_a_plain_deterministic_fallback(self):
        # conftest default: worker unavailable
        body = _ask(*_upload_pair())
        meta = body["metadata"]
        assert meta["specialist_id"] == DETERMINISTIC
        assert "planned_specialist_id" not in meta
        assert body["used_fallback"] is True  # registry-level fallback
        assert any(r["id"] == LEARNED for r in meta["rejected_specialists"])


def test_execution_trace_names_the_specialist_that_actually_ran(monkeypatch):
    """The EXECUTION trace line must not name the planned specialist when a
    different one ran — that would contradict the fallback reason beside it."""
    monkeypatch.setattr(registry, "_change_worker_available", lambda: True)
    monkeypatch.setattr(cs, "_post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    body = _ask(*_upload_pair())
    exec_events = [e for e in body["execution_trace_events"] if e["step"] == "EXECUTION"] \
        if "execution_trace_events" in body else \
        [e for e in client.get(f"/query/{body['query_id']}").json()["trace_events"] if e["step"] == "EXECUTION"]
    assert exec_events, "no EXECUTION trace event recorded"
    detail = exec_events[0]["detail"]
    assert DETERMINISTIC in detail and LEARNED not in detail, detail
