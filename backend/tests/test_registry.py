from app.services import registry


def test_list_specialists_returns_all_specs():
    specs = registry.list_specialists()
    ids = {s.id for s in specs}
    assert "vqa.smolvlm_base" in ids
    assert "vqa.smolvlm_bigearthnet_lora_stage3" in ids
    assert "grounding.deterministic_cv" in ids
    assert "change.semantic_model" in ids
    assert "change.deterministic_cv" in ids
    assert "cross_modal.feature_fusion" in ids


def test_candidates_for_vqa_excludes_lora_documentation_entry():
    candidates = registry.candidates_for("VQA", [{"modality": "OPTICAL"}])
    ids = [c.spec.id for c in candidates]
    assert ids == ["vqa.smolvlm_base"]


def test_candidates_for_change_detection_two_images():
    candidates = registry.candidates_for(
        "CHANGE_DETECTION", [{"modality": "OPTICAL"}, {"modality": "OPTICAL"}]
    )
    ids = {c.spec.id for c in candidates}
    assert ids == {"change.semantic_model", "change.deterministic_cv"}


def test_select_grounding_returns_deterministic_cv():
    entry, reason, rejected = registry.select("GROUNDING", [{"modality": "OPTICAL"}])
    assert entry.spec.id == "grounding.deterministic_cv"
    assert rejected == []
    assert "grounding.deterministic_cv" in reason


def test_select_change_detection_falls_back_to_deterministic():
    """change.semantic_model is unavailable (Phase B8 not delivered) so
    selection must fall back to the deterministic specialist with a reason."""
    entry, reason, rejected = registry.select(
        "CHANGE_DETECTION", [{"modality": "OPTICAL"}, {"modality": "OPTICAL"}]
    )
    assert entry.spec.id == "change.deterministic_cv"
    assert entry.spec.is_fallback is True
    rejected_ids = {r["id"] for r in rejected}
    assert "change.semantic_model" in rejected_ids
    unavailable_reasons = [r["reason"] for r in rejected if r["id"] == "change.semantic_model"]
    assert unavailable_reasons and "unavailable" in unavailable_reasons[0]


def test_select_vqa_when_worker_down_still_returns_the_only_specialist(monkeypatch):
    """There is no VQA fallback specialist — selection still returns
    vqa.smolvlm_base so its own honest '[VQA unavailable]' path can run,
    rather than fabricating a different result."""
    monkeypatch.setattr(registry, "_vqa_worker_available", lambda: False)
    entry, reason, rejected = registry.select("VQA", [{"modality": "OPTICAL"}])
    assert entry.spec.id == "vqa.smolvlm_base"
    assert "no specialist" in reason.lower() or "not reported itself available" in reason.lower()


def test_select_cross_modal_returns_feature_fusion():
    entry, reason, rejected = registry.select(
        "CROSS_MODAL", [{"modality": "OPTICAL"}, {"modality": "SAR"}]
    )
    assert entry.spec.id == "cross_modal.feature_fusion"


def test_select_unknown_task_returns_none():
    entry, reason, rejected = registry.select("UNKNOWN_TASK", [{"modality": "OPTICAL"}])
    assert entry is None
    assert "no registered specialist" in reason.lower()
