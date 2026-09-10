"""
Deterministic image grounding service.

Uses HSV colour-space thresholding + contour detection to locate
land-cover categories (water, vegetation, built-up) in a satellite image.
"""

import cv2
import numpy as np

# ── Tunable thresholds ────────────────────────────────────────────────

# Water: blue/cyan hue range in HSV
WATER_HUE_LOW = 85
WATER_HUE_HIGH = 135
WATER_SAT_MIN = 40
WATER_VAL_MIN = 40

# Vegetation: green hue range in HSV
VEG_HUE_LOW = 25
VEG_HUE_HIGH = 85
VEG_SAT_MIN = 30
VEG_VAL_MIN = 30

# Built-up: low saturation + moderate value (grey / tan urban surfaces)
BUILTUP_HUE_LOW = 0
BUILTUP_HUE_HIGH = 180
BUILTUP_SAT_MAX = 80
BUILTUP_VAL_LOW = 60
BUILTUP_VAL_HIGH = 220
BUILTUP_EDGE_RATIO = 0.02  # minimum Canny edge pixel ratio for built-up

_MIN_AREA_RATIO = 0.002  # discard contours smaller than 0.2 % of image

SUPPORTED_TARGETS = {"water", "vegetation", "built-up"}


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _bboxes_from_mask(mask: np.ndarray, total_pixels: int) -> list[list[int]]:
    """Find contours on a binary mask, filter by area, return bboxes."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = total_pixels * _MIN_AREA_RATIO
    bboxes: list[list[int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append([int(x), int(y), int(x + w), int(y + h)])
    bboxes.sort(key=lambda b: (b[1], b[0]))
    return bboxes


def _confidence_from_mask(mask: np.ndarray, bboxes: list[list[int]]) -> float | None:
    """Simple confidence: fraction of bbox area filled by the mask."""
    if not bboxes:
        return None
    total_bbox_area = 0
    filled_area = 0
    for box in bboxes:
        x1, y1, x2, y2 = box
        roi = mask[y1:y2, x1:x2]
        total_bbox_area += roi.size
        filled_area += int(np.count_nonzero(roi))
    return round(filled_area / total_bbox_area, 3) if total_bbox_area > 0 else None


def ground_object(image_path: str, object_type: str) -> dict:
    """Detect regions of *object_type* in *image_path*.

    Returns
    -------
    dict with keys:
        bounding_boxes   – list of [x_min, y_min, x_max, y_max]
        confidence_score – float or None
        object_type      – echoed back
        answer_text      – human-readable summary
    """
    target = object_type.lower().strip()
    if target not in SUPPORTED_TARGETS:
        return {
            "bounding_boxes": [],
            "confidence_score": None,
            "object_type": object_type,
            "answer_text": (
                f"Unsupported target '{object_type}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_TARGETS))}"
            ),
        }

    bgr = _load_bgr(image_path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_size, w_size = bgr.shape[:2]
    total_pixels = h_size * w_size

    if target == "water":
        mask = cv2.inRange(h, WATER_HUE_LOW, WATER_HUE_HIGH)
        mask_sat = cv2.inRange(s, WATER_SAT_MIN, 255)
        mask_val = cv2.inRange(v, WATER_VAL_MIN, 255)
        mask = cv2.bitwise_and(mask, cv2.bitwise_and(mask_sat, mask_val))

    elif target == "vegetation":
        mask = cv2.inRange(h, VEG_HUE_LOW, VEG_HUE_HIGH)
        mask_sat = cv2.inRange(s, VEG_SAT_MIN, 255)
        mask_val = cv2.inRange(v, VEG_VAL_MIN, 255)
        mask = cv2.bitwise_and(mask, cv2.bitwise_and(mask_sat, mask_val))

    elif target == "built-up":
        # Low saturation = grey/concrete surfaces
        mask_low_sat = cv2.inRange(s, 0, BUILTUP_SAT_MAX)
        mask_val_range = cv2.inRange(v, BUILTUP_VAL_LOW, BUILTUP_VAL_HIGH)
        color_mask = cv2.bitwise_and(mask_low_sat, mask_val_range)
        # Canny edge density boosts areas with structure
        edges = cv2.Canny(v, 50, 150)
        edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        edge_mask = cv2.inRange(
            edge_dilated, int(255 * BUILTUP_EDGE_RATIO), 255
        )
        # Combine: grey surfaces that also have nearby edges
        combined = cv2.bitwise_and(color_mask, edge_mask)
        # Also keep pure grey areas above edge ratio threshold for large blocks
        mask = combined

    bboxes = _bboxes_from_mask(mask, total_pixels)
    confidence = _confidence_from_mask(mask, bboxes)

    if not bboxes:
        answer_text = f"No {target} regions detected in this image."
    else:
        region_word = "region" if len(bboxes) == 1 else "regions"
        answer_text = (
            f"Detected {len(bboxes)} {target} {region_word}."
        )

    return {
        "bounding_boxes": bboxes,
        "confidence_score": confidence,
        "object_type": object_type,
        "answer_text": answer_text,
    }
