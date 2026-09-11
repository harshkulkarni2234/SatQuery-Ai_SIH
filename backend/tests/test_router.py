"""Focused unit tests for query routing priorities in app.services.router."""

from datetime import date

from app.services.router import classify_query


def _img(modality, capture_date=None, label=""):
    return {"id": label or modality, "modality": modality, "capture_date": capture_date}


def _opt(capture_date=None):
    return _img("OPTICAL", capture_date, "opt")


def _sar(capture_date=None):
    return _img("SAR", capture_date, "sar")


# ── CROSS_MODAL: OPTICAL + SAR pair wins over keyword logic ──────────

class TestCrossModalPriority:

    def test_optical_sar_pair_routes_cross_modal_without_change_keyword(self):
        result = classify_query(
            "Analyze this image pair together", [_opt(), _sar()]
        )
        assert result["task_classified"] == "CROSS_MODAL"
        assert result["validation_passed"] is True

    def test_optical_sar_pair_routes_cross_modal_even_with_change_wording(self):
        # Regression for the live-demo routing bug: "compare"/"changed"
        # wording on a genuine OPTICAL + SAR pair must NOT be rejected.
        for query in (
            "What changed between these two images?",
            "Compare the optical and SAR imagery",
            "What is different between the radar and optical image?",
        ):
            result = classify_query(query, [_opt(), _sar()])
            assert result["task_classified"] == "CROSS_MODAL", query
            assert result["validation_passed"] is True, query
            assert "change" not in result["task_classified"].lower()

    def test_cross_modal_pair_does_not_require_change_keyword(self):
        result = classify_query("Describe what both sensors show", [_opt(), _sar()])
        assert result["task_classified"] == "CROSS_MODAL"
        assert result["validation_passed"] is True


# ── CHANGE_DETECTION: same-modality pair + change intent ─────────────

class TestChangeDetectionRouting:

    def test_same_modality_pair_with_change_query_routes_change_detection(self):
        result = classify_query(
            "What changed between these two images?",
            [_opt(date(2017, 8, 6)), _opt(date(2017, 8, 8))],
        )
        assert result["task_classified"] == "CHANGE_DETECTION"
        assert result["validation_passed"] is True

    def test_same_modality_pair_routes_change_detection_without_dates(self):
        # Missing temporal metadata is not treated as a violation.
        result = classify_query(
            "What changed?", [_opt(), _opt()]
        )
        assert result["task_classified"] == "CHANGE_DETECTION"
        assert result["validation_passed"] is True

    def test_change_detection_same_capture_date_rejected(self):
        result = classify_query(
            "What changed between these two images?",
            [_opt(date(2017, 8, 8)), _opt(date(2017, 8, 8))],
        )
        assert result["task_classified"] == "CHANGE_DETECTION"
        assert result["validation_passed"] is False
        assert "same capture date" in result["reason"]

    def test_single_image_with_change_query_still_rejected(self):
        result = classify_query("What changed?", [_opt()])
        assert result["task_classified"] == "CHANGE_DETECTION"
        assert result["validation_passed"] is False
        assert "exactly 2 images" in result["reason"]


# ── Rejections ───────────────────────────────────────────────────────

class TestTwoImageViewRejection:

    def test_same_modality_pair_without_change_query_rejected(self):
        result = classify_query("Analyze this image pair together", [_opt(), _opt()])
        assert result["task_classified"] is None
        assert result["validation_passed"] is False
        assert "change-related question" in result["reason"]
        assert "OPTICAL + SAR" in result["reason"]


# ── Single-image routing preserved ───────────────────────────────────

class TestSingleImageRouting:

    def test_grounding_single_image_routed(self):
        result = classify_query("Show me the water", [_opt()])
        assert result["task_classified"] == "GROUNDING"
        assert result["validation_passed"] is True
        assert result["grounding_target"] == "water"

    def test_descriptive_single_image_routed_to_vqa(self):
        result = classify_query("What is visible in this image?", [_opt()])
        assert result["task_classified"] == "VQA"
        assert result["validation_passed"] is True


# ── GROUNDING: natural phrasing + target synonyms ───────────────────

class TestGroundingPhrasing:

    def test_where_is_water_routes_grounding(self):
        result = classify_query("Where is the water?", [_opt()])
        assert result["task_classified"] == "GROUNDING"
        assert result["validation_passed"] is True
        assert result["grounding_target"] == "water"

    def test_where_contraction_located_phrasing(self):
        result = classify_query("Where's the water located?", [_opt()])
        assert result["task_classified"] == "GROUNDING"
        assert result["grounding_target"] == "water"

    def test_find_me_lake_maps_to_water(self):
        result = classify_query("Find me the lake", [_opt()])
        assert result["task_classified"] == "GROUNDING"
        assert result["grounding_target"] == "water"

    def test_locate_the_lake_maps_to_water(self):
        result = classify_query("Locate the lake", [_opt()])
        assert result["grounding_target"] == "water"

    def test_highlight_road_maps_to_roads(self):
        result = classify_query("Highlight the road", [_opt()])
        assert result["grounding_target"] == "roads"

    def test_farm_maps_to_farmland(self):
        result = classify_query("Show me the farms", [_opt()])
        assert result["grounding_target"] == "farmland"

    def test_show_me_vegetation(self):
        result = classify_query("Show me the vegetation", [_opt()])
        assert result["grounding_target"] == "vegetation"

    def test_which_area_contains_water(self):
        result = classify_query("Which area of the image contains water?", [_opt()])
        assert result["task_classified"] == "GROUNDING"
        assert result["grounding_target"] == "water"


# ── CHANGE_DETECTION: comparison/change phrasing wins over grounding ─

class TestChangePhrasing:

    def test_show_me_the_change_wins_over_grounding_wording(self):
        # "show me" is grounding phrasing, but with 2 same-modality images and
        # a change-word, CHANGE_DETECTION must win.
        result = classify_query(
            "Show me the change between these images", [_opt(), _opt()]
        )
        assert result["task_classified"] == "CHANGE_DETECTION"
        assert result["validation_passed"] is True

    def test_what_is_different(self):
        result = classify_query("What is different?", [_opt(), _opt()])
        assert result["task_classified"] == "CHANGE_DETECTION"

    def test_compare_these_images(self):
        result = classify_query("Compare these images", [_opt(), _opt()])
        assert result["task_classified"] == "CHANGE_DETECTION"

    def test_before_after_comparison(self):
        result = classify_query("Before and after comparison", [_opt(), _opt()])
        assert result["task_classified"] == "CHANGE_DETECTION"

    def test_what_changed_over_time(self):
        result = classify_query("What changed over time?", [_opt(), _opt()])
        assert result["task_classified"] == "CHANGE_DETECTION"

    def test_show_me_changed_areas(self):
        result = classify_query("Show me the changed areas", [_opt(), _opt()])
        assert result["task_classified"] == "CHANGE_DETECTION"


# ── CROSS_MODAL: different-modality pair wins regardless of wording ──

class TestCrossModalPhrasing:

    def test_compare_optical_and_sar(self):
        result = classify_query("Compare the optical and SAR images", [_opt(), _sar()])
        assert result["task_classified"] == "CROSS_MODAL"
        assert result["validation_passed"] is True

    def test_both_modalities_together(self):
        result = classify_query("Let's look at both modalities together", [_opt(), _sar()])
        assert result["task_classified"] == "CROSS_MODAL"

    def test_check_both_images_together(self):
        result = classify_query("Check both images together", [_opt(), _sar()])
        assert result["task_classified"] == "CROSS_MODAL"

    def test_analyze_two_images_jointly(self):
        result = classify_query("Analyze the two images jointly", [_opt(), _sar()])
        assert result["task_classified"] == "CROSS_MODAL"


# ── VQA fallback stays safe ─────────────────────────────────────────

class TestVQAFallback:

    def test_describe_this_image(self):
        result = classify_query("Describe this image", [_opt()])
        assert result["task_classified"] == "VQA"
        assert result["validation_passed"] is True

    def test_what_can_you_tell_me(self):
        result = classify_query("What can you tell me about this image?", [_opt()])
        assert result["task_classified"] == "VQA"
        assert result["validation_passed"] is True

    def test_is_there_water_question_is_vqa_not_grounding(self):
        # "is there water" is a yes/no description, not a locate request.
        result = classify_query("Is there water in this image?", [_opt()])
        assert result["task_classified"] == "VQA"
        assert result["validation_passed"] is True


# ── Natural-language variation coverage (required for the NL demo set) ──

class TestNaturalGroundingCoverage:
    """A single optical image + any locate wording must route to GROUNDING
    with the correct canonical target."""

    CASES = [
        ("Where is the water located?", "water"),
        ("Find the lake", "water"),
        ("Show me where the river is", "water"),
        ("Where exactly is the pond?", "water"),
        ("Show me the location of the forest", "vegetation"),
        ("Find me the location of the lake", "water"),
        ("Show me the trees", "vegetation"),
        ("Highlight the buildings", "built-up"),
        ("Pinpoint the urban areas", "built-up"),
        ("Point me to the wetland", "water"),
        ("Locate the grassland", "vegetation"),
        ("Where can I find the settlement?", "built-up"),
        ("Which region of the image contains cropland?", "vegetation"),
        ("Mark the infrastructure", "built-up"),
        ("Outline the water body", "water"),
        ("Show me the location of the vegetation", "vegetation"),
        ("Identify the location of the road", "roads"),
        ("Identify locations of water bodies", "water"),
        ("Where is the vegetation located?", "vegetation"),
        ("Find me the houses", "built-up"),
        ("Show me where the highway is", "roads"),
        ("Locate the farmland", "farmland"),
        ("Where are the fields?", "farmland"),
    ]

    def test_all_natural_grounding_queries(self):
        for query, expected_target in self.CASES:
            result = classify_query(query, [_opt()])
            assert result["task_classified"] == "GROUNDING", query
            assert result["validation_passed"] is True, query
            assert result["grounding_target"] == expected_target, query


class TestNaturalChangeCoverage:
    """Two same-modality images + comparison wording must route to
    CHANGE_DETECTION."""

    CASES = [
        "What changed between these two images?",
        "Show me the changes between the two dates",
        "What is different between the July and August images?",
        "Detect changes in this area over time",
        "How has this area changed?",
        "How has this area been changed?",
        "Which areas have changed?",
        "What areas have changed?",
        "Describe the changes",
        "Before and after comparison",
        "Show me the differences between these images",
        "Compare these two images",
        "Compare these two images and highlight changes",
        "Has this area changed?",
        "Has the land changed?",
        "What changed between the images?",
        "Any changes detected?",
        "Show me the change from the first image to the second",
        "What has changed since the last survey?",
        "The two images look different",
    ]

    def test_all_natural_change_queries(self):
        for query in self.CASES:
            result = classify_query(query, [_opt(), _opt()])
            assert result["task_classified"] == "CHANGE_DETECTION", query
            assert result["validation_passed"] is True, query


class TestNaturalCrossModalCoverage:
    """An OPTICAL + SAR pair must route to CROSS_MODAL regardless of wording."""

    CASES = [
        "Analyze the optical and SAR images together",
        "Compare both sensors",
        "Fuse the optical and SAR information",
        "What do the optical and SAR images show together?",
        "Combine the optical and SAR data",
        "Identify the same features across both modalities",
        "Show me what the optical and SAR images reveal together",
        "Is there a difference between optical and SAR?",
        "Merge the optical and SAR channels",
        "Joint analysis of the optical and SAR imagery",
        "Correlate the optical and SAR data",
        "Review the optical and SAR pair",
        "Synthesize the optical and SAR views",
    ]

    def test_all_natural_cross_modal_queries(self):
        for query in self.CASES:
            result = classify_query(query, [_opt(), _sar()])
            assert result["task_classified"] == "CROSS_MODAL", query
            assert result["validation_passed"] is True, query


class TestNaturalVQACoverage:
    """Counting / descriptive questions must stay VQA even when they mention
    a grounding target."""

    CASES = [
        "Describe this image",
        "What can you see in this scene?",
        "Is there water in this image?",
        "What is the overall land cover?",
        "How many buildings are there?",
        "What percentage of the area is water?",
        "Give me a summary of the scene",
        "What does the image show?",
        "Are there any roads?",
        "What kind of terrain is this?",
        "How built up is this area?",
        "What is the state of the crops?",
        "How many water bodies are in this image?",
    ]

    def test_all_natural_vqa_queries(self):
        for query in self.CASES:
            result = classify_query(query, [_opt()])
            assert result["task_classified"] == "VQA", query
            assert result["validation_passed"] is True, query