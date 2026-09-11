"""
Deterministic image grounding service.

Uses HSV colour-space thresholding, edge density and linear-morphology cues
to locate land-cover categories in a satellite image:

  water, vegetation, built-up, roads, farmland

All detections come from deterministic computer-vision operations (no learned
model) and are therefore reproducible. Confidences reflect a real, measured
quantity (bbox fill ratio) or are omitted entirely — nothing is fabricated.
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

# Roads: grey surfaces with linear structure (low saturation, mid value)
ROADS_VAL_LOW = 90
ROADS_MIN_ELONGATION = 2.5  # long relative to wide => road-like

_MIN_AREA_RATIO = 0.002  # discard components smaller than 0.2 % of image

# Display cap: keep overlays readable, never show 18+ overlapping boxes.
_MAX_DISPLAY_REGIONS = 10

SUPPORTED_TARGETS = {"water", "vegetation", "built-up", "roads", "farmland"}

# Human-readable labels used by the frontend and caption evidence
TARGET_LABELS = {
    "water": "Water",
    "vegetation": "Vegetation",
    "built-up": "Built-up area",
    "roads": "Road",
    "farmland": "Farmland",
}

# Minimum cover fraction before a land-cover cue is reported for captions
CUE_MIN_FRACTION = {
    "water": 0.01,
    "vegetation": 0.03,
    "built-up": 0.03,
    "roads": 0.01,
    "farmland": 0.02,
}


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def build_target_mask(bgr: np.ndarray, target: str) -> np.ndarray:
    """Build a binary (0/255) candidate mask for *target* from a BGR image.

    Roads get a linear-morphology pass so only strip-like grey regions
    survive; farmland is the vegetation mask after a large closing (merging
    individual fields into blocks).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

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

    elif target == "farmland":
        base = cv2.inRange(h, VEG_HUE_LOW, VEG_HUE_HIGH)
        base_sat = cv2.inRange(s, VEG_SAT_MIN, 255)
        base_val = cv2.inRange(v, VEG_VAL_MIN, 255)
        base = cv2.bitwise_and(base, cv2.bitwise_and(base_sat, base_val))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel)

    elif target == "built-up":
        # Low saturation = grey/concrete surfaces
        mask_low_sat = cv2.inRange(s, 0, BUILTUP_SAT_MAX)
        mask_val_range = cv2.inRange(v, BUILTUP_VAL_LOW, BUILTUP_VAL_HIGH)
        color_mask = cv2.bitwise_and(mask_low_sat, mask_val_range)
        # Canny edge density boosts areas with structure
        edges = cv2.Canny(v, 50, 150)
        edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        edge_mask = cv2.inRange(edge_dilated, int(255 * BUILTUP_EDGE_RATIO), 255)
        mask = cv2.bitwise_and(color_mask, edge_mask)

    elif target == "roads":
        # Roads are low-saturation, mid-value (grey/tan) surfaces. Keep only
        # components that are long relative to their width via linear openings
        # plus an elongation check at region level; this separates roads from
        # compact built-up blocks.
        low_sat = cv2.inRange(s, 0, BUILTUP_SAT_MAX)
        val_range = cv2.inRange(v, ROADS_VAL_LOW, BUILTUP_VAL_HIGH)
        color = cv2.bitwise_and(low_sat, val_range)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        h_line = cv2.morphologyEx(color, cv2.MORPH_OPEN, h_kernel)
        v_line = cv2.morphologyEx(color, cv2.MORPH_OPEN, v_kernel)
        mask = cv2.bitwise_or(h_line, v_line)

    else:
        raise ValueError(f"Unsupported target: {target}")

    return mask


def _extract_regions(
    mask: np.ndarray, total_pixels: int, min_elongation: float = 1.0
) -> tuple[list[list[int]], list[dict]]:
    """Return ([bboxes], [region dicts]) from connected components.

    Regions carry box, area_px, area_pct and centroid, all in pixel coords.
    """
    min_area = total_pixels * _MIN_AREA_RATIO
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    bboxes: list[list[int]] = []
    regions: list[dict] = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if min_elongation > 1.0:
            elong = max(w, h) / max(1, min(w, h))
            if elong < min_elongation:
                continue

        # Fine-grained centroid from the actual foreground pixels
        ys, xs = np.where(mask[y:y + h, x:x + w] > 0)
        if len(xs) == 0:
            continue
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        bboxes.append(bbox)
        regions.append(
            {
                "box": bbox,
                "area_px": area,
                "area_pct": round(area / total_pixels * 100, 3),
                "centroid": [
                    int(round(float(xs.mean()) + x)),
                    int(round(float(ys.mean()) + y)),
                ],
                "label": None,  # filled in by caller
            }
        )

    order = sorted(range(len(bboxes)), key=lambda i: bboxes[i][1] * 1e6 + bboxes[i][0])
    bboxes = [bboxes[i] for i in order]
    regions = [regions[i] for i in order]
    return bboxes, regions


def _confidence_from_mask(mask: np.ndarray, bboxes: list[list[int]]) -> float | None:
    """Confidence: fraction of bbox area filled by the mask (a real quantity)."""
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
        confidence_score – float or None (only when meaningfully measurable)
        object_type      – echoed back
        regions          – per-region metrics (box, area_px, area_pct, centroid)
        method           – description of the deterministic pipeline
        answer_text      – human-readable summary
    """
    target = object_type.lower().strip()
    if target not in SUPPORTED_TARGETS:
        return {
            "bounding_boxes": None,
            "confidence_score": None,
            "object_type": object_type,
            "regions": None,
            "method": "deterministic visual grounding",
            "answer_text": (
                f"Unsupported target '{object_type}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_TARGETS))}"
            ),
        }

    bgr = _load_bgr(image_path)
    h_size, w_size = bgr.shape[:2]
    total_pixels = h_size * w_size

    mask = build_target_mask(bgr, target)
    min_elongation = ROADS_MIN_ELONGATION if target == "roads" else 1.0
    bboxes, regions = _extract_regions(mask, total_pixels, min_elongation=min_elongation)

    for region in regions:
        region["label"] = TARGET_LABELS[target]

    confidence = _confidence_from_mask(mask, bboxes)

    # Keep overlays readable: sort by area and cap at a display-friendly count.
    ranked = sorted(range(len(regions)), key=lambda i: regions[i]["area_px"], reverse=True)
    ranked = ranked[:_MAX_DISPLAY_REGIONS]
    bboxes = [bboxes[i] for i in ranked]
    regions = [regions[i] for i in ranked]

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
        "regions": regions,
        "method": "deterministic visual grounding",
        "answer_text": answer_text,
    }


def scene_cues(image_path: str) -> dict:
    """Deterministic land-cover fractions for caption-style evidence.

    Every fraction is measured from the actual image via the same masks the
    bounding-box grounding uses, so the cues can never contradict the boxes.
    Only cues above ``CUE_MIN_FRACTION`` are reported.
    """
    bgr = _load_bgr(image_path)
    h_size, w_size = bgr.shape[:2]
    total_pixels = h_size * w_size
    if total_pixels == 0:
        return {"fractions": {}, "detected_cues": [], "dominant_cue": None}

    fractions: dict[str, float] = {}
    for target in sorted(SUPPORTED_TARGETS):
        mask = build_target_mask(bgr, target)
        min_elongation = ROADS_MIN_ELONGATION if target == "roads" else 1.0
        _, regions = _extract_regions(mask, total_pixels, min_elongation=min_elongation)
        fractions[target] = round(
            sum(r["area_px"] for r in regions) / total_pixels, 4
        )

    detected = [
        {
            "cue": TARGET_LABELS[target],
            "target": target,
            "fraction": fractions[target],
        }
        for target in sorted(SUPPORTED_TARGETS)
        if fractions[target] >= CUE_MIN_FRACTION[target]
    ]

    dominant = None
    if detected:
        top = max(detected, key=lambda d: d["fraction"])
        dominant = f"{top['cue']}-dominant ({top['fraction']:.1%} of the frame)"

    return {
        "fractions": fractions,
        "detected_cues": detected,
        "dominant_cue": dominant,
    }