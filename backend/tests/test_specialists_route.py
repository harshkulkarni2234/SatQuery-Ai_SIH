from tests.conftest import client


def test_get_specialists_returns_registry_list():
    resp = client.get("/specialists")
    assert resp.status_code == 200
    body = resp.json()
    ids = {s["id"] for s in body}
    assert "vqa.smolvlm_base" in ids
    assert "change.deterministic_cv" in ids
    grounding_spec = next(s for s in body if s["id"] == "grounding.deterministic_cv")
    assert grounding_spec["kind"] == "deterministic_cv"
    assert grounding_spec["task"] == "GROUNDING"
