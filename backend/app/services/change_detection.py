"""
Deterministic bi-temporal change detection service.

Pipeline: grayscale → absolute difference → Gaussian blur → binary threshold
→ morphological cleanup → connected-component filtering → bounding boxes + stats.
"""

import os
import uuid

import cv2
import numpy as np

MASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "change_masks")
_MIN_REGION_AREA_RATIO = 0.001  # regions smaller than 0.1 % of frame are noise
_GAUSSIAN_KERNEL = (5, 5)
_GAUSSIAN_SIGMA = 0
_THRESHOLD = 30  # absolute grayscale difference above which a pixel counts as changed
_MORPH_KERNEL_SIZE = 3
_MORPH_ITERATIONS = 2


def _ensure_mask_dir():
    os.makedirs(MASK_DIR, exist_ok=True)


def _load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _resize_to_match(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def detect_change(image_path_before: str, image_path_after: str) -> dict:
    """Compare two images and return a change analysis dict.

    Returns
    -------
    dict with keys:
        change_mask_path  – path to the saved binary mask image (or None)
        change_percentage – float 0-100
        answer_text       – human-readable summary
        bounding_boxes    – list of [x_min, y_min, x_max, y_max] for each region
        num_regions       – int
        changed_pixels    – int
        total_pixels      – int
    """
    gray_before = _load_grayscale(image_path_before)
    gray_after = _load_grayscale(image_path_after)

    # Resize 'after' to match 'before' if needed
    gray_after = _resize_to_match(gray_after, gray_before.shape[:2])

    h, w = gray_before.shape
    total_pixels = h * w

    # Absolute pixel-wise difference
    diff = cv2.absdiff(gray_before, gray_after)

    # Gaussian blur to suppress sensor noise
    blurred = cv2.GaussianBlur(diff, _GAUSSIAN_KERNEL, _GAUSSIAN_SIGMA)

    # Binary threshold
    _, thresh = cv2.threshold(blurred, _THRESHOLD, 255, cv2.THRESH_BINARY)

    # Morphological close to merge nearby pixels, then open to remove specks
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE)
    )
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=_MORPH_ITERATIONS)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected-component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    min_area = total_pixels * _MIN_REGION_AREA_RATIO
    bounding_boxes: list[list[int]] = []
    changed_pixels = 0

    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bounding_boxes.append([x, y, x + bw, y + bh])
        changed_pixels += int(np.count_nonzero(labels == i))

    change_percentage = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    # Sort boxes top-to-bottom, left-to-right
    bounding_boxes.sort(key=lambda b: (b[1], b[0]))

    # Save mask
    mask_path = None
    if bounding_boxes:
        _ensure_mask_dir()
        mask_name = f"mask_{uuid.uuid4().hex}.png"
        mask_path = os.path.join(MASK_DIR, mask_name)
        cv2.imwrite(mask_path, cleaned)

    # Human-readable summary
    if not bounding_boxes:
        answer_text = "No significant change detected between the two images."
    else:
        region_word = "region" if len(bounding_boxes) == 1 else "regions"
        answer_text = (
            f"Detected {len(bounding_boxes)} changed {region_word} "
            f"covering approximately {change_percentage:.1f}% of the frame "
            f"({changed_pixels:,} of {total_pixels:,} pixels)."
        )

    return {
        "change_mask_path": mask_path,
        "change_percentage": round(change_percentage, 2),
        "answer_text": answer_text,
        "bounding_boxes": bounding_boxes,
        "num_regions": len(bounding_boxes),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
    }
