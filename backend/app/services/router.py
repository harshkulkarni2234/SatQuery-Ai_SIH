import re

VALID_MODALITIES = {"OPTICAL", "SAR"}

# Canonical grounding targets understood by the grounding specialist.
GROUNDING_TARGETS = {"water", "vegetation", "built-up", "roads", "farmland"}

# Natural-language synonyms mapped onto canonical targets. The router keeps
# the synonym -> canonical mapping so downstream specialists receive the
# canonical target while users phrase the object however they like.
GROUNDING_TARGET_SYNONYMS = {
    "water": [
        "water", "waters", "water body", "waterbody", "water bodies", "waterbodies",
        "lake", "lakes", "river", "rivers", "stream", "streams", "pond", "ponds",
        "ocean", "sea", "canal", "canals", "wetland", "wetlands",
    ],
    "vegetation": [
        "vegetation", "forest", "forests", "tree", "trees", "grass", "grassland",
        "crop", "crops", "cropland", "green area", "green areas", "greenery",
        "canopy",
    ],
    "built-up": [
        "built-up", "built up", "building", "buildings", "house", "houses",
        "urban", "urban area", "urban areas", "settlement", "settlements",
        "structure", "structures", "infrastructure",
    ],
    "roads": [
        "road", "roads", "highway", "highways", "street", "streets", "avenue",
        "avenues", "lane", "lanes", "carriageway", "carriageways", "motorway",
        "motorways", "arterial",
    ],
    "farmland": [
        "farmland", "farms", "farm", "agriculture", "agricultural",
        "crop area", "field", "fields", "pasture", "cultivated",
    ],
}

_TARGET_SYNONYM_LIST = [
    (synonym, canonical)
    for canonical, synonyms in GROUNDING_TARGET_SYNONYMS.items()
    for synonym in synonyms
]
_GROUNDING_TARGET_RE = re.compile(
    r"\b(" + "|".join(re.escape(synonym) for synonym, _ in _TARGET_SYNONYM_LIST) + r")\b",
    re.IGNORECASE,
)
_TARGET_CANONICAL = {
    synonym.lower(): canonical for synonym, canonical in _TARGET_SYNONYM_LIST
}

# Locate / show / highlight / "where" phrasing. Kept intentionally narrow: a
# grounding-specific verb or question word must be present so generic
# descriptive questions are not hijacked.
GROUNDING_KEYWORDS = re.compile(
    r"\b(show me|show the|show me where|show where|where is|where are|where's|"
    r"where can i find|where is the location of|where exactly (is|are)|"
    r"show me the location|find me the location|find the location|"
    r"identify the location|identify locations of|identify where|identify which|"
    r"locate|location of|point out|point me to|find me|find the|"
    r"which (part|area|region) of|highlight|pinpoint|outline|mark)\b",
    re.IGNORECASE,
)

# Change / comparison phrasing. Phrase-based entries capture natural wording
# ("show me the change", "what is different") beyond single keywords.
CHANGE_KEYWORDS = re.compile(
    r"\b(change|changes|changed|changing|changed areas|changed regions|"
    r"difference|differences|different|differ|compare|compared|comparing|"
    r"comparison|before|after|over time|loss|gain|lost|gained|increased|"
    r"decreased|new construction|newly built|removed|appeared|disappeared)\b"
    r"|(what changed|what has changed|what is different|what's different|"
    r"what are the differences|what is the difference|show me the change|"
    r"show the change|show me the changes|show the changes|show changes|"
    r"detect changes|identify changes|before and after|changes between|"
    r"different between|how has this area changed|how has this area been changed|"
    r"has this area changed|has the land changed|which areas have changed|"
    r"what areas have changed|describe the changes|any changes|what changed between)",
    re.IGNORECASE,
)

# NOTE: This is deterministic keyword classification, NOT an AI agent.
# It only decides which specialist tool applies and validates the input
# combination; it makes no claims about image content or model behaviour.


def _temporal_metadata_compatible(images: list[dict]) -> tuple[bool, str | None]:
    """Gate same-modality CHANGE_DETECTION on temporal metadata.

    Missing dates are not treated as a violation (the deterministic diff still
    runs), but two identical capture dates positively indicate there is no
    temporal basis for change detection, so such a pair is rejected.
    """
    dates = [img.get("capture_date") for img in images]
    if all(dates) and dates[0] == dates[1]:
        return False, (
            "Both images share the same capture date; change detection "
            "requires images captured at different dates."
        )
    return True, None


def classify_query(query_text: str, images: list[dict]) -> dict:
    """Classify a query into a task and validate the image combination.

    Intent precedence:
        1. CROSS_MODAL       – 2 compatible images of different modalities.
        2. CHANGE_DETECTION  – 2 (same-modality) images + change/comparison intent.
        3. GROUNDING         – single image + locate/show/highlight/where intent.
        4. CHANGE_DETECTION  – change wording without the required image pair.
        5. generic rejection  – 2 images with no detectable intent.
        6. VQA               – single image fallback.

    Returns
    -------
    dict with keys:
        task_classified     – str | None
        modalities_detected – list[str]
        validation_passed   – bool
        reason              – plain-English justification
        grounding_target    – str | None (only meaningful for GROUNDING)
    """
    n = len(images)
    modalities = [img.get("modality", "").upper() for img in images]
    modalities_detected = [m for m in modalities if m in VALID_MODALITIES]

    change_match = CHANGE_KEYWORDS.search(query_text)
    grounding_match = GROUNDING_KEYWORDS.search(query_text)
    target_match = _GROUNDING_TARGET_RE.search(query_text.lower())
    grounding_target = (
        _TARGET_CANONICAL.get(target_match.group(1)) if target_match else None
    )

    # Priority 1: exactly 2 images with different modalities (OPTICAL + SAR)
    # -> CROSS_MODAL. Detected before any keyword-based logic so that a valid
    # optical + SAR pair is never rejected for missing change-related wording.
    if n == 2 and len(set(modalities)) == 2 and "OPTICAL" in modalities and "SAR" in modalities:
        return {
            "task_classified": "CROSS_MODAL",
            "modalities_detected": modalities_detected,
            "validation_passed": True,
            "reason": (
                "Two images provided with different modalities (OPTICAL + SAR) "
                "— routed to cross-modal analysis."
            ),
        }

    # Priority 2: exactly 2 images (an OPTICAL + SAR pair was already routed
    # to CROSS_MODAL above, so both images share a modality) with an explicit
    # change / comparison intent -> CHANGE_DETECTION. Also wins over GROUNDING
    # wording ("show me the change") because two images cue comparison here.
    if change_match and n == 2:
        compatible, incompat_reason = _temporal_metadata_compatible(images)
        if not compatible:
            return {
                "task_classified": "CHANGE_DETECTION",
                "modalities_detected": modalities_detected,
                "validation_passed": False,
                "reason": incompat_reason,
            }
        return {
            "task_classified": "CHANGE_DETECTION",
            "modalities_detected": modalities_detected,
            "validation_passed": True,
            "reason": (
                f"Query contains change-related keyword '{change_match.group()}' "
                "and 2 compatible same-modality images were provided."
            ),
        }

    # Grounding intent: must be a single image with a detectable target
    if grounding_match:
        if n != 1:
            return {
                "task_classified": "GROUNDING",
                "modalities_detected": modalities_detected,
                "validation_passed": False,
                "reason": (
                    f"Grounding requires exactly 1 image but {n} were provided."
                ),
                "grounding_target": grounding_target,
            }
        if grounding_target is None:
            return {
                "task_classified": "GROUNDING",
                "modalities_detected": modalities_detected,
                "validation_passed": False,
                "reason": (
                    "Could not infer a grounding target from the query. "
                    f"Supported targets: {', '.join(sorted(GROUNDING_TARGETS))}."
                ),
                "grounding_target": grounding_target,
            }
        return {
            "task_classified": "GROUNDING",
            "modalities_detected": modalities_detected,
            "validation_passed": True,
            "reason": (
                f"Query contains localisation keyword '{grounding_match.group()}' "
                f"for target '{grounding_target}' with a single image."
            ),
            "grounding_target": grounding_target,
        }

    # Priority 4: change wording without the required pair (e.g. one image)
    if change_match:
        return {
            "task_classified": "CHANGE_DETECTION",
            "modalities_detected": modalities_detected,
            "validation_passed": False,
            "reason": (
                f"Change detection requires exactly 2 images but {n} were provided."
            ),
        }

    # Two images with no change intent and not an OPTICAL + SAR pair
    if n == 2:
        return {
            "task_classified": None,
            "modalities_detected": modalities_detected,
            "validation_passed": False,
            "reason": (
                "Two images supplied without a change-related question. "
                "Ask a change question or provide an OPTICAL + SAR pair."
            ),
        }

    # Single image, descriptive question
    return {
        "task_classified": "VQA",
        "modalities_detected": modalities_detected,
        "validation_passed": True,
        "reason": "Single image with a descriptive question, routed to VQA (SmolVLM worker).",
        "grounding_target": None,
    }


def extract_grounding_target(query_text: str) -> str | None:
    """Return the canonical grounding target mentioned in *query_text*, if any.

    Reuses the same synonym table as :func:`classify_query` so VQA evidence
    and GROUNDING never disagree about what an object phrase means. This is
    deliberately target-only: it does NOT decide routing.
    """
    if not query_text:
        return None
    match = _GROUNDING_TARGET_RE.search(query_text.lower())
    if not match:
        return None
    return _TARGET_CANONICAL.get(match.group(1))