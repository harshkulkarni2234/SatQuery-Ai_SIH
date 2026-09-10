import re

CHANGE_KEYWORDS = re.compile(
    r"\b(changed?|change|compare|between|increased?|decreased?|difference|before|after|loss|gain|new|removed|appeared|disappeared)\b",
    re.IGNORECASE,
)
GROUNDING_KEYWORDS = re.compile(
    r"\b(show me|where is|locate|highlight|point out|find|identify the|mark)\b",
    re.IGNORECASE,
)

VALID_MODALITIES = {"OPTICAL", "SAR"}
GROUNDING_TARGETS = {"water", "vegetation", "built-up"}
_GROUNDING_TARGET_RE = re.compile(
    r"\b(" + "|".join(GROUNDING_TARGETS) + r")\b", re.IGNORECASE
)

# NOTE: This is deterministic keyword classification, NOT an AI agent.
# It only decides which specialist tool applies and validates the input
# combination; it makes no claims about image content or model behaviour.


def classify_query(query_text: str, images: list[dict]) -> dict:
    """Classify a query into a task and validate the image combination.

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
    target_match = _GROUNDING_TARGET_RE.search(query_text)
    grounding_target = target_match.group(1).lower() if target_match else None

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

    # Change intent: must be exactly 2 same-modality images
    if change_match:
        if n != 2:
            return {
                "task_classified": "CHANGE_DETECTION",
                "modalities_detected": modalities_detected,
                "validation_passed": False,
                "reason": (
                    f"Change detection requires exactly 2 images but {n} were provided."
                ),
            }
        if len(set(modalities)) != 1:
            return {
                "task_classified": "CHANGE_DETECTION",
                "modalities_detected": modalities_detected,
                "validation_passed": False,
                "reason": (
                    "Change-related query detected but the 2 images have "
                    f"different modalities ({', '.join(set(modalities))}). "
                    "Change detection requires same-modality images."
                ),
            }
        return {
            "task_classified": "CHANGE_DETECTION",
            "modalities_detected": modalities_detected,
            "validation_passed": True,
            "reason": (
                f"Query contains change-related keyword '{change_match.group()}' "
                "and 2 same-modality images were provided."
            ),
        }

    # Two images with no explicit change/grounding intent
    if n == 2:
        if len(set(modalities)) == 2 and "OPTICAL" in modalities and "SAR" in modalities:
            return {
                "task_classified": "CROSS_MODAL",
                "modalities_detected": modalities_detected,
                "validation_passed": True,
                "reason": "Two images with different modalities (OPTICAL + SAR) provided.",
            }
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
        "reason": "Single image with a descriptive question — routed to VQA (not yet implemented).",
        "grounding_target": None,
    }