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


def classify_query(query_text: str, images: list[dict]) -> dict:
    n = len(images)
    modalities = [img.get("modality", "").upper() for img in images]
    modalities_detected = [m for m in modalities if m in VALID_MODALITIES]

    change_match = CHANGE_KEYWORDS.search(query_text)
    grounding_match = GROUNDING_KEYWORDS.search(query_text)

    if change_match and n == 2 and len(set(modalities)) == 1:
        return {
            "task_classified": "CHANGE_DETECTION",
            "modalities_detected": modalities_detected,
            "validation_passed": True,
            "reason": (
                f"Query contains change-related keyword '{change_match.group()}' "
                f"and 2 images of the same modality were provided."
            ),
        }

    if grounding_match and n == 1:
        return {
            "task_classified": "GROUNDING",
            "modalities_detected": modalities_detected,
            "validation_passed": True,
            "reason": (
                f"Query contains localisation keyword '{grounding_match.group()}' "
                f"with a single image."
            ),
        }

    if n == 2 and len(set(modalities)) == 2 and "OPTICAL" in modalities and "SAR" in modalities:
        return {
            "task_classified": "CROSS_MODAL",
            "modalities_detected": modalities_detected,
            "validation_passed": True,
            "reason": "Two images with different modalities (OPTICAL + SAR) provided.",
        }

    if n == 1:
        return {
            "task_classified": "VQA",
            "modalities_detected": modalities_detected,
            "validation_passed": True,
            "reason": "Single image with a descriptive question — routed to VQA.",
        }

    if change_match and n == 2 and len(set(modalities)) > 1:
        return {
            "task_classified": "CHANGE_DETECTION",
            "modalities_detected": modalities_detected,
            "validation_passed": False,
            "reason": (
                "Change-related query detected but the 2 images have different "
                f"modalities ({', '.join(set(modalities))}). "
                "Change detection requires same-modality images."
            ),
        }

    if change_match and n == 1:
        return {
            "task_classified": "CHANGE_DETECTION",
            "modalities_detected": modalities_detected,
            "validation_passed": False,
            "reason": "Change detection requires 2 images but only 1 was provided.",
        }

    if n > 2:
        return {
            "task_classified": None,
            "modalities_detected": modalities_detected,
            "validation_passed": False,
            "reason": f"Too many images provided ({n}). Maximum supported is 2.",
        }

    return {
        "task_classified": "VQA",
        "modalities_detected": modalities_detected,
        "validation_passed": True,
        "reason": "Defaulting to VQA for single-image descriptive query.",
    }
