"""
Deterministic bi-temporal change detection service (lightweight, CPU-only).

Pipeline:
  1. Input validation — readability, aspect-ratio compatibility, and any
     spatial metadata (tile id / CRS+bounds) that would prove the two images
     do NOT cover the same geographic area.
  2. Optional translation registration — phase correlation to remove small
     spatial offsets before differencing (no learned model).
  3. Robust difference — median-smoothed grayscale difference plus the
     max-channel colour difference, magnitude-gated threshold.
  4. Morphological cleaning — opening (speckle removal) then closing
     (connecting fragmented regions) so the mask is readable.
  5. Connected components — per-region metrics, merging of near/overlapping
     boxes, and a TOP-N cap so the report shows a few meaningful regions.
  6. Outputs — binary mask, a colour change-overlay on the AFTER image,
     pixel coordinates scaled back to the *original* AFTER resolution so
     frontend overlays always line up, and statistics measured from the
     *cleaned* mask.

The service NEVER interprets semantic change (e.g. "building constructed").
It reports pixel-level visual differences only.
"""

import os
import uuid

import cv2
import numpy as np

MASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "change_masks")

# ── Tunable parameters ────────────────────────────────────────────────

_MEDIAN_KERNEL = 5
_THRESHOLD = 26            # minimum smoothed difference to count a pixel as changed
_MORPH_KERNEL = 5          # cleaning kernel (opening then closing)
_MORPH_CLOSE_ITERS = 2
_MIN_REGION_AREA_RATIO = 0.0004  # components smaller than this are noise
_MAX_REGIONS = 8           # sensible cap on displayed change regions
_MERGE_IOU = 0.30          # boxes with IoU above this are merged
_MERGE_PX_RATIO = 0.015    # boxes whose centres are this close (in frame px) are merged
_ASPECT_RATIO_TOL = 0.20   # max relative aspect-ratio drift allowed before a pair is "unverifiable"
_NON_CORRESPONDENCE_MAX_FRAME_FRACTION = 0.85  # aligned diff covering almost the whole frame ⇒ not the same area
_OVERLAY_COLOR = (0, 0, 235)  # BGR highlight for the change overlay

# Registration (phase correlation) guards
_REGISTER_MIN_SHIFT = 1.0
_REGISTER_MIN_RESPONSE = 0.05
_REGISTER_MAX_SHIFT_FRACTION = 0.5

INCOMPATIBLE_MSG = (
    "These images cannot be reliably compared for temporal change detection "
    "because their spatial correspondence could not be verified."
)


def _ensure_mask_dir():
    os.makedirs(MASK_DIR, exist_ok=True)


def _load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _resize_to_match(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _bounds_overlap(b1, b2) -> bool:
    """True if two [xmin, ymin, xmax, ymax] bounds share any area."""
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
        return True  # missing bounds => cannot prove a mismatch
    x1a, y1a, x2a, y2a = map(float, b1[:4])
    x1b, y1b, x2b, y2b = map(float, b2[:4])
    ix = min(x2a, x2b) - max(x1a, x1b)
    iy = min(y2a, y2b) - max(y1a, y1b)
    if ix <= 0 or iy <= 0:
        return False
    a = (x2a - x1a) * (y2a - y1a)
    inter = ix * iy
    return inter / a >= 0.02  # at least ~2% geographic overlap


def _spatial_conflict(meta_before, meta_after) -> str | None:
    """Return a conflict message if metadata proves the areas are different.

    Uses tile identifiers when present, otherwise CRS + bounds. Only a
    positive, provable mismatch triggers a refusal — missing metadata never
    does.
    """
    before = meta_before or {}
    after = meta_after or {}

    tile_b = (before.get("tile_id") or "").strip().lower()
    tile_a = (after.get("tile_id") or "").strip().lower()
    if tile_b and tile_a and tile_b != tile_a:
        return INCOMPATIBLE_MSG

    crs_b = (before.get("crs") or "").strip().upper()
    crs_a = (after.get("crs") or "").strip().upper()
    bounds_b = before.get("bbox_coords")
    bounds_a = after.get("bbox_coords")
    if crs_b and crs_a and crs_b == crs_a and bounds_b and bounds_a:
        if not _bounds_overlap(bounds_b, bounds_a):
            return INCOMPATIBLE_MSG
    return None


def register_after(after: np.ndarray, before: np.ndarray) -> tuple[np.ndarray, bool, tuple[float, float]]:
    """Translation-align *after* toward *before* using phase correlation.

    Returns (aligned, applied, shift). Alignment is only applied when the
    phase correlation is confident and the shift is non-trivial but small.
    """
    h, w = after.shape
    if after.shape != before.shape:
        raise ValueError("Registration requires equal-sized inputs")

    if float(after.std()) < 1.0 or float(before.std()) < 1.0:
        return after, False, (0.0, 0.0)

    try:
        shift, response = cv2.phaseCorrelate(
            np.float32(before), np.float32(after)
        )
    except cv2.error:
        return after, False, (0.0, 0.0)

    dx, dy = float(shift[0]), float(shift[1])
    if response < _REGISTER_MIN_RESPONSE:
        return after, False, (0.0, 0.0)
    if max(abs(dx), abs(dy)) < _REGISTER_MIN_SHIFT:
        return after, False, (0.0, 0.0)
    if abs(dx) > _REGISTER_MAX_SHIFT_FRACTION * w or abs(dy) > _REGISTER_MAX_SHIFT_FRACTION * h:
        return after, False, (0.0, 0.0)

    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(after, transform, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return aligned, True, (dx, dy)


def _robust_diff_mask(gray_before, gray_after, bgr_before, bgr_after) -> np.ndarray:
    """Smoothed, magnitude-gated binary mask of pixels that visually changed."""
    gray_diff = cv2.absdiff(gray_before, gray_after)
    gray_diff = cv2.medianBlur(gray_diff, _MEDIAN_KERNEL)

    color_diff = cv2.absdiff(bgr_before, bgr_after)
    if color_diff.ndim == 3 and color_diff.shape[2] >= 3:
        color_diff = color_diff.max(axis=2).astype(np.uint8)
        color_diff = cv2.medianBlur(color_diff, _MEDIAN_KERNEL)
    else:
        color_diff = gray_diff

    combined = cv2.max(gray_diff, color_diff)
    _, thresh = cv2.threshold(combined, _THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_MORPH_KERNEL, _MORPH_KERNEL))
    # Opening removes isolated specks; keep the shape crisp.
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing bridges fragmented regions so the result reads as fewer, larger
    # blobs (but does not fuse genuinely separated structures).
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=_MORPH_CLOSE_ITERS)
    return cleaned


def _iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_regions(boxes: list[list[int]], frame_px: int) -> list[list[int]]:
    """Merge overlapping / adjacent boxes, then keep the largest TOP-N."""
    if not boxes:
        return []
    merge_dist = max(2, int(_MERGE_PX_RATIO * max(frame_px, 64)))

    def _centre(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    ordered = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    merged: list[list[int]] = []
    for box in ordered:
        placed = False
        for m in merged:
            if _iou(box, m) >= _MERGE_IOU:
                cm = _centre(m)
                cb = _centre(box)
                if abs(cm[0] - cb[0]) <= merge_dist and abs(cm[1] - cb[1]) <= merge_dist:
                    m[0] = min(m[0], box[0])
                    m[1] = min(m[1], box[1])
                    m[2] = max(m[2], box[2])
                    m[3] = max(m[3], box[3])
                    placed = True
                    break
        if not placed:
            merged.append(list(box))

    merged.sort(key=lambda b: -abs((b[2] - b[0]) * (b[3] - b[1])))
    return merged[:_MAX_REGIONS]


def _region_metrics(box, clean_mask):
    x1, y1, x2, y2 = box
    roi = clean_mask[y1:y2, x1:x2]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return None
    area_px = int(len(xs))
    cx = int(round(float(xs.mean()) + x1))
    cy = int(round(float(ys.mean()) + y1))
    return {
        "box": [int(x1), int(y1), int(x2), int(y2)],
        "area_px": area_px,
        "area_pct": round(area_px / clean_mask.size * 100, 3),
        "centroid": [int(cx), int(cy)],
    }


def _render_overlay(after_bgr: np.ndarray, clean_mask: np.ndarray, path: str) -> None:
    """Save AFTER image with semi-transparent highlight over changed pixels."""
    overlay = after_bgr.copy()
    alpha = 0.45
    colored = np.zeros_like(after_bgr)
    colored[clean_mask > 0] = _OVERLAY_COLOR
    blend = cv2.addWeighted(after_bgr, 1 - alpha, colored, alpha, 0)
    mask_bool = clean_mask > 0
    overlay[mask_bool] = blend[mask_bool]
    cv2.imwrite(path, overlay)


def detect_change(
    image_path_before: str,
    image_path_after: str,
    metadata_before: dict | None = None,
    metadata_after: dict | None = None,
) -> dict:
    """Compare two images and return a change analysis dict.

    Returns
    -------
    dict with keys:
        validation_failed    – True when the pair cannot be compared reliably
        reason               – plain-English explanation (None on success)
        error                – True when validation/processing failed
        registration_applied – whether a translation alignment was applied
        change_mask_path     – path to the saved binary mask (or None)
        overlay_path         – path to the saved change-overlay PNG (or None)
        change_percentage    – float 0-100 measured on the CLEANED mask (or None)
        answer_text          – human-readable, honest summary
        bounding_boxes       – [x1, y1, x2, y2] scaled to the ORIGINAL after image
        regions              – per-region metrics (box, area_px, area_pct, centroid)
        num_regions          – int
        changed_pixels       – int (cleaned mask)
        total_pixels         – int
    """
    # ── STEP 1 — INPUT VALIDATION ─────────────────────────────────────
    before_gray = _load_grayscale(image_path_before)
    after_gray = _load_grayscale(image_path_after)
    before_bgr = _load_bgr(image_path_before)
    after_bgr = _load_bgr(image_path_after)

    h_b, w_b = before_gray.shape
    h_a, w_a = after_gray.shape
    aspect_b = w_b / h_b if h_b else 0.0
    aspect_a = w_a / h_a if h_a else 0.0
    aspect_drift = (
        abs(aspect_b - aspect_a) / max(aspect_b, aspect_a) if max(aspect_b, aspect_a) else 0.0
    )

    conflict = _spatial_conflict(metadata_before, metadata_after)
    if conflict or aspect_drift > _ASPECT_RATIO_TOL:
        reason = conflict or INCOMPATIBLE_MSG
        return {
            "validation_failed": True,
            "reason": reason,
            "error": True,
            "registration_applied": False,
            "change_mask_path": None,
            "overlay_path": None,
            "change_percentage": None,
            "answer_text": reason,
            "bounding_boxes": None,
            "regions": None,
            "num_regions": 0,
            "changed_pixels": None,
            "total_pixels": None,
        }

    # ── STEP 2 — ALIGNMENT / REGISTRATION ─────────────────────────────
    after_gray_resized = _resize_to_match(after_gray, before_gray.shape[:2])
    after_bgr_resized = _resize_to_match(after_bgr, before_bgr.shape[:2])
    aligned_gray, registration_applied, shift = register_after(after_gray_resized, before_gray)
    if registration_applied:
        transform = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        aligned_bgr = cv2.warpAffine(
            after_bgr_resized, transform, (w_b, h_b), borderMode=cv2.BORDER_REPLICATE
        )
    else:
        aligned_bgr = after_bgr_resized

    h, w = aligned_gray.shape
    total_pixels = h * w

    # ── STEP 3+4 — ROBUST DIFFERENCE + MORPHOLOGICAL CLEANING ─────────
    cleaned = _robust_diff_mask(before_gray, aligned_gray, before_bgr, aligned_bgr)

    # ── STEP 5 — CONNECTED COMPONENTS → MERGED TOP-N REGIONS ──────────
    boxes = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_area = total_pixels * _MIN_REGION_AREA_RATIO
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, x + bw, y + bh])

    final_boxes = _merge_regions(boxes, min(w, h))

    regions = []
    for box in final_boxes:
        metrics = _region_metrics(box, cleaned)
        if metrics:
            regions.append(metrics)

    changed_pixels = int(np.count_nonzero(cleaned))
    change_percentage = (changed_pixels / total_pixels * 100) if total_pixels else 0.0

    # ── GUARD — near-full-frame difference ⇒ not the same area ──────────
    # A genuine change in the same footprint would never flip almost the
    # entire frame. When the *aligned* difference still covers nearly the
    # whole image, the pair does not actually correspond (e.g. neighbouring
    # scenes with a spurious periodic phase-correlation peak). Refusing
    # honestly beats reporting a misleading "~99% changed".
    if (
        cleaned.any()
        and change_percentage > _NON_CORRESPONDENCE_MAX_FRAME_FRACTION * 100
    ):
        return {
            "validation_failed": True,
            "reason": INCOMPATIBLE_MSG,
            "error": True,
            "registration_applied": registration_applied,
            "change_mask_path": None,
            "overlay_path": None,
            "change_percentage": None,
            "answer_text": INCOMPATIBLE_MSG,
            "bounding_boxes": None,
            "regions": None,
            "num_regions": 0,
            "changed_pixels": None,
            "total_pixels": None,
        }

    # ── STEP 6 — WRITE OUTPUTS ─────────────────────────────────────────
    mask_path = None
    overlay_path = None
    if cleaned.any():
        _ensure_mask_dir()
        stamp = uuid.uuid4().hex
        mask_path = os.path.join(MASK_DIR, f"mask_{stamp}.png")
        cv2.imwrite(mask_path, cleaned)
        overlay_path = os.path.join(MASK_DIR, f"overlay_{stamp}.png")
        _render_overlay(aligned_bgr, cleaned, overlay_path)

    # Scale boxes back to the ORIGINAL after-image resolution so frontend
    # overlays drawn over the native AFTER image never drift.
    scale_x = w_a / w if w else 1.0
    scale_y = h_a / h if h else 1.0
    scaled_boxes = []
    scaled_regions = []
    for r in regions:
        box = r["box"]
        x1 = int(round(box[0] * scale_x))
        y1 = int(round(box[1] * scale_y))
        x2 = int(round(box[2] * scale_x))
        y2 = int(round(box[3] * scale_y))
        if x2 <= x1 or y2 <= y1:
            continue
        scaled_boxes.append([x1, y1, x2, y2])
        scaled_regions.append(
            {
                **r,
                "box": [x1, y1, x2, y2],
                "centroid": [
                    int(round(r["centroid"][0] * scale_x)),
                    int(round(r["centroid"][1] * scale_y)),
                ],
            }
        )

    # Sort so the largest scaled region is listed first.
    def _area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    scaled_boxes.sort(key=_area, reverse=True)
    scaled_regions.sort(key=lambda r: _area(r["box"]), reverse=True)
    if not scaled_boxes:
        scaled_regions = []

    # ── STEPS 7+8 — HONEST SUMMARY + STATISTICS FROM THE CLEANED MASK ──
    if not scaled_boxes:
        answer_text = (
            "Pixel-level visual differences were not detected above the "
            "sensitivity threshold between the two images."
        )
    else:
        area_word = "region" if len(scaled_boxes) == 1 else "regions"
        answer_text = (
            f"Pixel-level visual differences were detected in {len(scaled_boxes)} "
            f"highlighted {area_word}, covering approximately "
            f"{change_percentage:.1f}% of the frame. Differences are measured at "
            "the pixel level; semantic change classification was not performed."
        )

    registration_note = (
        "Images were translation-aligned before differencing. "
        if registration_applied
        else ""
    )
    if scaled_boxes:
        answer_text = registration_note + answer_text

    result = {
        "validation_failed": False,
        "reason": None,
        "error": False,
        "registration_applied": registration_applied,
        "change_mask_path": mask_path,
        "overlay_path": overlay_path,
        "change_percentage": round(change_percentage, 2),
        "answer_text": answer_text,
        "bounding_boxes": scaled_boxes,
        "regions": scaled_regions,
        "num_regions": len(scaled_boxes),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
    }
    return result