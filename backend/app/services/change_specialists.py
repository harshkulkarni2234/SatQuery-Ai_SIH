"""Change-detection specialists: the learned Siamese CNN client and the
deterministic pixel-difference specialist.

The learned model is a *binary* change/no-change segmenter trained on
0.5-3 m aerial RGB pairs (SECOND dataset, see ml/change_model/MODEL_CARD.md).
It says where pixels changed, never what they changed to. This module keeps
it honest in three ways:

  1. It only runs when the same "can these two images be compared at all?"
     rules the deterministic method applies say it may, plus a training-domain
     check (resolution) the deterministic method doesn't need.
  2. Anything short of a fully valid model answer (worker down, invalid mask,
     size mismatch, coarse imagery, non-corresponding pair) falls back to the
     deterministic method and SAYS SO in the result — it never reports a
     made-up "0%" or a result that was not actually produced by the model.
  3. The mask is returned by the worker as bytes and stored by the backend in
     its own served mask directory, so what the UI links to always exists.
"""

from __future__ import annotations

import base64
import io
import os
import uuid

import numpy as np
import requests
from PIL import Image

from app.contracts import SpecialistResult
from app.services import change_detection

LEARNED_SPECIALIST_ID = "change.siamese_binary_cnn"
DETERMINISTIC_SPECIALIST_ID = "change.deterministic_cv"

CHANGE_WORKER_URL = os.getenv("CHANGE_WORKER_URL", "http://127.0.0.1:8002").rstrip("/")
CHANGE_WORKER_TIMEOUT_S = float(os.getenv("CHANGE_WORKER_TIMEOUT_S", "120"))

# The model was trained on 0.5-3 m aerial imagery; anything coarser is outside
# what it has ever seen, so it is not used (the deterministic method runs).
MAX_TRAINED_PIXEL_SIZE_M = float(os.getenv("CHANGE_MODEL_MAX_PIXEL_SIZE_M", "3.0"))
TRAINING_DOMAIN = "0.5-3 m aerial RGB image pairs (SECOND dataset)"

_GEOGRAPHIC_CRS = {"EPSG:4326", "EPSG:4269", "EPSG:4258"}
_METERS_PER_DEGREE = 111_320.0

# Module-level alias so tests can mock the HTTP call without touching the
# shared requests module (same convention as services/vqa.py).
_post = requests.post


def pixel_size_m(meta: dict | None) -> float | None:
    """Best-effort pixel size in metres from stored metadata, or None when it
    is unknown. Geographic (degree-based) CRSs are converted approximately —
    good enough for a coarse "is this far outside the training domain" gate."""
    meta = meta or {}
    res = meta.get("resolution")
    if not res:
        return None
    try:
        size = float(min(res))
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    crs = (meta.get("crs") or "").strip().upper()
    # Resolution is in CRS units only when it came from a real geotransform
    # (a user-supplied resolution_m is already metres).
    if meta.get("transform") and crs in _GEOGRAPHIC_CRS:
        size *= _METERS_PER_DEGREE
    return size


def skip_reason(
    path_before: str,
    path_after: str,
    metadata_before: dict | None,
    metadata_after: dict | None,
) -> str | None:
    """Why the learned model must NOT be used for this pair, or None if it may."""
    try:
        with Image.open(path_before) as im1, Image.open(path_after) as im2:
            size_before, size_after = im1.size, im2.size
    except Exception:  # noqa: BLE001 - any unreadable input means "don't guess"
        return "an input image could not be read for the pre-check"
    if size_before != size_after:
        return (
            f"the before/after images differ in size ({size_before[0]}x{size_before[1]} vs "
            f"{size_after[0]}x{size_after[1]}); this model only accepts pre-aligned, same-size pairs"
        )

    # Same provable-mismatch rule the deterministic method uses, so the two
    # specialists can never disagree about whether a pair may be compared.
    if change_detection._spatial_conflict(metadata_before, metadata_after):
        return "the image metadata shows the two images do not cover the same area"

    for meta in (metadata_before, metadata_after):
        size = pixel_size_m(meta)
        if size is not None and size > MAX_TRAINED_PIXEL_SIZE_M:
            return (
                f"the imagery is about {size:.0f} m per pixel, coarser than the "
                f"{TRAINING_DOMAIN} this model was trained on"
            )
    return None


def call_worker(path_before: str, path_after: str) -> dict | None:
    """POST the pair to the change worker. Returns its parsed JSON, or None on
    any failure (never raises, never returns a partial/unvalidated dict)."""
    try:
        with open(path_before, "rb") as f_before, open(path_after, "rb") as f_after:
            response = _post(
                f"{CHANGE_WORKER_URL}/change",
                files={
                    "before_image": (os.path.basename(path_before), f_before),
                    "after_image": (os.path.basename(path_after), f_after),
                },
                timeout=CHANGE_WORKER_TIMEOUT_S,
            )
    except (requests.RequestException, OSError):
        return None
    if response.status_code != 200:
        return None
    try:
        data = response.json()
    except ValueError:
        return None
    if (
        not isinstance(data, dict)
        or not data.get("model_version")
        or data.get("latency_ms") is None
        or not isinstance(data.get("mask_png_base64"), str)
    ):
        return None
    return data


def decode_mask(mask_png_base64: str) -> np.ndarray | None:
    """Decode the worker's PNG mask into a boolean array. Returns None unless
    it is a non-empty 2-D image containing only 0/255 — a corrupt or
    non-binary payload is treated as "no answer", not silently coerced."""
    try:
        raw = base64.b64decode(mask_png_base64, validate=True)
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
    except Exception:  # noqa: BLE001
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not set(np.unique(arr).tolist()) <= {0, 255}:
        return None
    return arr > 0


def save_artifacts(mask: np.ndarray, after_path: str) -> tuple[str, str | None]:
    """Write the change mask (and, when the after image is readable, an overlay)
    into the backend's own served mask directory under unique names. The mask
    is upscaled with nearest-neighbour to the after image's size for display;
    the model itself only ever sees a 256x256 resize."""
    os.makedirs(change_detection.MASK_DIR, exist_ok=True)
    stamp = uuid.uuid4().hex[:12]

    with Image.open(after_path) as after:
        after_rgb = after.convert("RGB")
    size = after_rgb.size

    mask_u8 = mask.astype(np.uint8) * 255
    mask_full = Image.fromarray(mask_u8).resize(size, Image.NEAREST)
    mask_path = os.path.join(change_detection.MASK_DIR, f"learned_mask_{stamp}.png")
    mask_full.save(mask_path)

    overlay_path: str | None = None
    try:
        arr = np.array(after_rgb)
        changed = np.array(mask_full) > 0
        red = np.array([255, 0, 0], dtype=np.float32)
        arr[changed] = (0.5 * arr[changed].astype(np.float32) + 0.5 * red).astype(np.uint8)
        overlay_path = os.path.join(change_detection.MASK_DIR, f"learned_overlay_{stamp}.png")
        Image.fromarray(arr).save(overlay_path)
    except Exception:  # noqa: BLE001 - the mask alone is still valid evidence
        overlay_path = None
    return mask_path, overlay_path


def build_deterministic_result(result: dict, skipped_reason: str | None = None) -> SpecialistResult:
    """Wrap a change_detection.detect_change() dict as a SpecialistResult.

    skipped_reason is set only when the learned model was planned but not used;
    the deterministic result then carries used_fallback=True and says why."""
    validation_failed = bool(result.get("validation_failed"))
    warnings: list[str] = []
    if skipped_reason:
        warnings.append(
            f"The learned change model was not used because {skipped_reason}; "
            "the deterministic pixel-difference method ran instead."
        )
    if validation_failed:
        warnings.append(result.get("reason") or "Validation failed.")

    evidence: dict = {
        "boxes": result.get("bounding_boxes"),
        "mask_path": result.get("change_mask_path"),
        "overlay_path": result.get("overlay_path"),
        "validation_failed": validation_failed,
        "validation_reason": (result.get("reason") or "Validation failed.") if validation_failed else None,
        "stats": {
            "change_percentage": result.get("change_percentage"),
            "num_regions": result.get("num_regions"),
            "changed_pixels": result.get("changed_pixels"),
            "total_pixels": result.get("total_pixels"),
            "changed_area_m2": result.get("changed_area_m2"),
            "regions": result.get("regions"),
            "registration_applied": result.get("registration_applied"),
            "alignment_method": result.get("alignment_method"),
        },
    }
    if skipped_reason:
        evidence["skipped_planned_reason"] = skipped_reason

    return SpecialistResult(
        answer=result["answer_text"],
        evidence=evidence,
        confidence=None,
        confidence_source="unavailable",
        model_or_tool=DETERMINISTIC_SPECIALIST_ID,
        model_version="cv-change-v1",
        used_fallback=bool(skipped_reason),
        fallback_reason=None,
        warnings=warnings,
    )


def run_deterministic_change(
    image_path_before: str,
    image_path_after: str,
    metadata_before: dict | None = None,
    metadata_after: dict | None = None,
) -> SpecialistResult:
    result = change_detection.detect_change(
        image_path_before, image_path_after,
        metadata_before=metadata_before, metadata_after=metadata_after,
    )
    return build_deterministic_result(result)


def _fallback(before, after, meta_before, meta_after, reason: str) -> SpecialistResult:
    result = change_detection.detect_change(
        before, after, metadata_before=meta_before, metadata_after=meta_after
    )
    return build_deterministic_result(result, skipped_reason=reason)


def run_learned_change(
    image_path_before: str,
    image_path_after: str,
    metadata_before: dict | None = None,
    metadata_after: dict | None = None,
) -> SpecialistResult:
    """Run the learned Siamese CNN if — and only if — it can give a fully valid
    answer for this pair; otherwise run the deterministic method and record why."""

    def fallback(reason: str) -> SpecialistResult:
        return _fallback(image_path_before, image_path_after, metadata_before, metadata_after, reason)

    reason = skip_reason(image_path_before, image_path_after, metadata_before, metadata_after)
    if reason:
        return fallback(reason)

    data = call_worker(image_path_before, image_path_after)
    if data is None:
        return fallback("the learned change worker did not return a usable response")

    mask = decode_mask(data["mask_png_base64"])
    if mask is None:
        return fallback("the learned change worker returned an invalid mask")

    total_pixels = int(mask.size)
    changed_pixels = int(mask.sum())
    change_percentage = round(changed_pixels / total_pixels * 100, 2)

    # Same guard the deterministic method applies: a real change in one
    # footprint never flips nearly the whole frame — that means the two images
    # do not actually show the same place.
    max_pct = change_detection._NON_CORRESPONDENCE_MAX_FRAME_FRACTION * 100
    if change_percentage > max_pct:
        return fallback(
            f"the model marked {change_percentage}% of the frame as changed, which indicates "
            "the two images do not show the same area"
        )

    try:
        mask_path, overlay_path = save_artifacts(mask, image_path_after)
    except OSError:
        return fallback("the change mask could not be saved")

    model_version = data["model_version"]
    latency_ms = data["latency_ms"]
    # NOTE: the change percentage is deliberately the FIRST number in the
    # answer — the CDVQA scorer (evaluation/metrics/vqa_metrics.py) extracts
    # the first number it finds, so a version string or resolution ahead of it
    # would be scored as the "percentage".
    answer = (
        f"A learned Siamese CNN marks approximately {change_percentage}% of the frame as changed "
        f"({changed_pixels} of {total_pixels} pixels at the model's {mask.shape[1]}x{mask.shape[0]} "
        f"working resolution; model {model_version}, inference {latency_ms} ms). "
        "This is a binary change/no-change result: it shows where pixels changed, not what they "
        "changed to, and the model does not classify land cover. "
        f"It was trained on {TRAINING_DOMAIN}; results on other imagery are unvalidated."
    )

    warnings: list[str] = []
    if pixel_size_m(metadata_before) is None and pixel_size_m(metadata_after) is None:
        warnings.append(
            "Image resolution is unknown, so this model's training domain "
            f"({TRAINING_DOMAIN}) could not be checked against your input."
        )

    return SpecialistResult(
        answer=answer,
        evidence={
            "boxes": None,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "validation_failed": False,
            "stats": {
                "change_percentage": change_percentage,
                "num_regions": None,
                "changed_pixels": changed_pixels,
                "total_pixels": total_pixels,
                "changed_area_m2": None,
                "regions": None,
                "registration_applied": False,
                "alignment_method": "none",
            },
        },
        confidence=None,
        confidence_source="unavailable",
        model_or_tool=LEARNED_SPECIALIST_ID,
        model_version=model_version,
        used_fallback=False,
        fallback_reason=None,
        warnings=warnings,
    )
