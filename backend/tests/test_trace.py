import uuid

from app.database import SessionLocal
from app.models import Query
from app.services.trace import TraceRecorder
from tests.conftest import client


def test_trace_recorder_produces_ordered_events_with_nonnegative_durations():
    recorder = TraceRecorder()
    recorder.start("QUERY_RECEIVED")
    recorder.record("QUERY_RECEIVED", "COMPLETED", "ok")
    recorder.start("TASK_SELECTED")
    recorder.record("TASK_SELECTED", "COMPLETED", "VQA")
    recorder.record("RESULT", "COMPLETED", "done")

    steps = [e.step for e in recorder.events]
    assert steps == ["QUERY_RECEIVED", "TASK_SELECTED", "RESULT"]
    for e in recorder.events:
        assert e.duration_ms is not None and e.duration_ms >= 0
        assert e.timestamp is not None


def test_trace_recorder_error_appends_error_step():
    recorder = TraceRecorder()
    recorder.record("QUERY_RECEIVED", "COMPLETED", "ok")
    recorder.error("something failed")

    assert recorder.events[-1].step == "ERROR"
    assert recorder.events[-1].status == "FAILED"


def test_successful_query_response_has_real_trace_events(uploaded_image_id, monkeypatch):
    from app.services import vqa as vqa_module

    def fake_answer(image_path, query_text):
        return {
            "answer_text": "A terrain sample.",
            "model_version": "SmolVLM-256M-Instruct",
            "specialist": False,
            "confidence_score": None,
            "execution_time_ms": 5,
            "error": False,
        }

    monkeypatch.setattr(vqa_module, "answer_question", fake_answer)

    resp = client.post(
        "/query",
        json={"query_text": "What is visible here?", "image_ids": [uploaded_image_id]},
    )
    assert resp.status_code == 200
    body = resp.json()
    events = body["trace_events"]
    assert events, "trace_events must be present and non-empty"
    steps = [e["step"] for e in events]
    assert steps[0] == "QUERY_RECEIVED"
    assert steps[-1] == "RESULT"
    assert "TASK_SELECTED" in steps
    assert "SPECIALIST_SELECTED" in steps
    assert "EXECUTION" in steps
    for e in events:
        assert e["duration_ms"] is not None and e["duration_ms"] >= 0


def test_get_query_returns_persisted_trace_events(uploaded_image_id, monkeypatch):
    from app.services import vqa as vqa_module

    monkeypatch.setattr(
        vqa_module,
        "answer_question",
        lambda image_path, query_text: {
            "answer_text": "ok", "model_version": "m", "specialist": False,
            "confidence_score": None, "execution_time_ms": 1, "error": False,
        },
    )
    post_resp = client.post(
        "/query", json={"query_text": "Describe this.", "image_ids": [uploaded_image_id]}
    )
    query_id = post_resp.json()["query_id"]

    get_resp = client.get(f"/query/{query_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["trace_events"]


def test_empty_query_failure_persists_a_query_row_with_error_trace(uploaded_image_id):
    db = SessionLocal()
    try:
        count_before = db.query(Query).count()
    finally:
        db.close()

    resp = client.post("/query", json={"query_text": "   ", "image_ids": [uploaded_image_id]})
    assert resp.status_code == 400

    db = SessionLocal()
    try:
        count_after = db.query(Query).count()
        latest = db.query(Query).order_by(Query.created_at.desc()).first()
    finally:
        db.close()

    assert count_after == count_before + 1
    assert latest.trace_events
    steps = [e["step"] for e in latest.trace_events]
    assert "ERROR" in steps


def test_unknown_image_id_failure_persists_query_row():
    db = SessionLocal()
    try:
        count_before = db.query(Query).count()
    finally:
        db.close()

    fake_id = str(uuid.uuid4())
    resp = client.post("/query", json={"query_text": "What is here?", "image_ids": [fake_id]})
    assert resp.status_code == 404

    db = SessionLocal()
    try:
        count_after = db.query(Query).count()
        latest = db.query(Query).order_by(Query.created_at.desc()).first()
    finally:
        db.close()
    assert count_after == count_before + 1
    steps = [e["step"] for e in latest.trace_events]
    assert "ERROR" in steps
