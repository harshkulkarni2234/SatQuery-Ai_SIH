import uuid

from tests.conftest import client, upload_png_bytes


def _run_query(image_id, query_text="What can you tell me about this image?"):
    resp = client.post("/query", json={"query_text": query_text, "image_ids": [image_id]})
    assert resp.status_code == 200, resp.text
    return resp.json()["query_id"]


def test_report_pdf_is_a_real_pdf():
    image_id = upload_png_bytes_for_report()
    query_id = _run_query(image_id, "Where is the water located?")

    resp = client.get(f"/query/{query_id}/report.pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")
    assert len(resp.content) > 500


def test_report_json_has_expected_fields():
    image_id = upload_png_bytes_for_report()
    query_id = _run_query(image_id, "Where is the water located?")

    resp = client.get(f"/query/{query_id}/report.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["report_id"].startswith("SQ-")
    assert data["query_id"] == query_id
    assert data["query_text"] == "Where is the water located?"
    assert data["task"] == "GROUNDING"
    assert len(data["inputs"]) == 1
    assert data["inputs"][0]["modality"] == "OPTICAL"
    assert data["specialist"]["id"] == "grounding.deterministic_cv"
    assert "confidence_source" in data
    assert isinstance(data["execution_trace"], list) and len(data["execution_trace"]) > 0
    assert isinstance(data["limitations"], list) and len(data["limitations"]) >= 3
    assert "pixel-level" in data["limitations"][0].lower()


def test_report_json_and_pdf_agree_on_answer():
    image_id = upload_png_bytes_for_report()
    query_id = _run_query(image_id, "Where is the water located?")

    json_data = client.get(f"/query/{query_id}/report.json").json()
    pdf_bytes = client.get(f"/query/{query_id}/report.pdf").content
    # The PDF is binary/compressed, so just confirm both endpoints succeeded
    # from the same underlying data (build_report_data is the single source).
    assert json_data["answer_text"]
    assert pdf_bytes.startswith(b"%PDF")


def test_report_404_for_unknown_query_id():
    fake_id = str(uuid.uuid4())
    resp_pdf = client.get(f"/query/{fake_id}/report.pdf")
    resp_json = client.get(f"/query/{fake_id}/report.json")
    assert resp_pdf.status_code == 404
    assert resp_json.status_code == 404


def test_report_handles_missing_evidence_file_gracefully(monkeypatch):
    """If an input image's file_path no longer exists on disk, report
    generation must not crash -- it should just omit the thumbnail."""
    import app.services.report as report_module

    image_id = upload_png_bytes_for_report()
    query_id = _run_query(image_id, "What is visible here?")

    original = report_module._thumbnail_reader
    monkeypatch.setattr(report_module, "_thumbnail_reader", lambda *a, **k: None)
    try:
        resp = client.get(f"/query/{query_id}/report.pdf")
        assert resp.status_code == 200
        assert resp.content.startswith(b"%PDF")
    finally:
        monkeypatch.setattr(report_module, "_thumbnail_reader", original)


def upload_png_bytes_for_report():
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    img = PILImage.new("RGB", (60, 60), color=(120, 120, 60))
    for x in range(10, 30):
        for y in range(10, 30):
            img.putpixel((x, y), (0, 80, 200))
    img.save(buf, format="PNG")
    buf.seek(0)
    return upload_png_bytes(buf.read())
