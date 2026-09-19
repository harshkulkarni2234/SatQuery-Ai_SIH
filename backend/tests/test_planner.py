from app.services import planner

OPTICAL_A = {"modality": "OPTICAL", "capture_date": "2021-01-01", "is_georeferenced": False,
             "crs": None, "bounds_wgs84": None, "resolution": None, "transform": None,
             "width": 100, "height": 100}
OPTICAL_B = {"modality": "OPTICAL", "capture_date": "2023-01-01", "is_georeferenced": False,
             "crs": None, "bounds_wgs84": None, "resolution": None, "transform": None,
             "width": 100, "height": 100}
SAR_A = {"modality": "SAR", "capture_date": None, "is_georeferenced": False,
         "crs": None, "bounds_wgs84": None, "resolution": None, "transform": None,
         "width": 100, "height": 100}


def test_empty_query_rejected_with_suggestion():
    result = planner.build_plan("   ", [OPTICAL_A])
    assert result.plan.validation_passed is False
    assert "empty" in result.plan.validation_reason.lower()
    assert result.plan.suggestion
    assert result.selected_entry is None


def test_grounding_plan_has_target_and_parameters():
    result = planner.build_plan("Where is the water located?", [OPTICAL_A])
    plan = result.plan
    assert plan.validation_passed is True
    assert plan.task == "GROUNDING"
    assert plan.target == "water"
    assert plan.question_type == "location"
    assert plan.parameters == {"target": "water"}
    assert plan.selected_specialist_id == "grounding.deterministic_cv"
    assert result.selected_entry is not None


def test_grounding_unsupported_target_gets_suggestion_listing_targets():
    result = planner.build_plan("Where is the spaceship located?", [OPTICAL_A])
    plan = result.plan
    assert plan.validation_passed is False
    assert "water" in plan.suggestion  # one of the supported targets is listed


def test_grounding_wrong_image_count_gets_suggestion():
    result = planner.build_plan("Where is the water located?", [OPTICAL_A, OPTICAL_B])
    plan = result.plan
    assert plan.validation_passed is False
    assert "1 image" in plan.suggestion


def test_change_detection_plan_selects_deterministic_fallback(monkeypatch):
    """When the change worker is unavailable, the planner should
    select the deterministic fallback."""
    from app.services import registry
    monkeypatch.setattr(registry, "_change_worker_available", lambda: False)
    result = planner.build_plan(
        "What changed between these two images?", [OPTICAL_A, OPTICAL_B]
    )
    plan = result.plan
    assert plan.validation_passed is True
    assert plan.task == "CHANGE_DETECTION"
    assert plan.question_type == "change"
    assert plan.compatibility is not None
    assert plan.compatibility.ok is True
    assert plan.selected_specialist_id == "change.deterministic_cv"
    assert any(r["id"] == "change.semantic_model" for r in plan.rejected_specialists)


def test_change_detection_same_date_rejected_by_fast_classifier():
    """The fast keyword classifier's own same-date short-circuit
    (services/router.py) rejects this before the planner's own
    compatibility.check_temporal_pair() ever runs, so no compatibility
    report is attached here -- see test_change_detection_non_overlapping_
    rejected_with_compatibility_report for a case that does attach one."""
    same_date = dict(OPTICAL_B, capture_date="2021-01-01")
    result = planner.build_plan("What changed?", [OPTICAL_A, same_date])
    plan = result.plan
    assert plan.validation_passed is False
    assert plan.compatibility is None
    assert result.selected_entry is None


def test_change_detection_non_overlapping_rejected_with_compatibility_report():
    georeferenced_a = dict(
        OPTICAL_A, is_georeferenced=True, crs="EPSG:4326",
        bounds_wgs84=(0.0, 0.0, 1.0, 1.0), resolution=(10.0, 10.0),
    )
    georeferenced_b = dict(
        OPTICAL_B, is_georeferenced=True, crs="EPSG:4326",
        bounds_wgs84=(50.0, 50.0, 51.0, 51.0), resolution=(10.0, 10.0),
    )
    result = planner.build_plan("What changed?", [georeferenced_a, georeferenced_b])
    plan = result.plan
    assert plan.validation_passed is False
    assert plan.compatibility is not None
    assert plan.compatibility.ok is False
    assert result.selected_entry is None


def test_change_detection_one_image_gets_suggestion():
    result = planner.build_plan("What changed?", [OPTICAL_A])
    plan = result.plan
    assert plan.validation_passed is False
    assert "different date" in plan.suggestion


def test_cross_modal_plan():
    result = planner.build_plan("Compare the optical and SAR images.", [OPTICAL_A, SAR_A])
    plan = result.plan
    assert plan.validation_passed is True
    assert plan.task == "CROSS_MODAL"
    assert plan.selected_specialist_id == "cross_modal.feature_fusion"


def test_vqa_plan_presence_question_type():
    result = planner.build_plan("Is there a road in this image?", [OPTICAL_A])
    plan = result.plan
    assert plan.task == "VQA"
    assert plan.question_type == "presence"
    assert plan.selected_specialist_id == "vqa.smolvlm_base"


def test_vqa_plan_count_question_type():
    result = planner.build_plan("How many buildings are visible?", [OPTICAL_A])
    assert result.plan.question_type == "count"


def test_vqa_plan_description_question_type():
    result = planner.build_plan("What can you tell me about this image?", [OPTICAL_A])
    assert result.plan.question_type == "description"


def test_ambiguous_two_image_query_rejected_with_suggestion():
    result = planner.build_plan("Tell me about these images", [OPTICAL_A, OPTICAL_B])
    plan = result.plan
    assert plan.validation_passed is False
    assert "cross-modal" in plan.suggestion.lower() or "change" in plan.suggestion.lower()


def test_plan_requested_outputs_match_task():
    result = planner.build_plan("Where is the water located?", [OPTICAL_A])
    assert "boxes" in result.plan.requested_outputs
