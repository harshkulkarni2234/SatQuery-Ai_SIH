"""Geospatial compatibility checks for multi-image tasks (Phase A3).

Pure functions, no DB access — callers pass plain dicts built from whatever
image metadata they have (ORM rows, RasterMetadata, or test fixtures). Every
FAIL/WARN/SKIPPED check carries a plain-English `detail` so the API caller
always gets a reason, never a bare rejection.
"""

from __future__ import annotations

import os

from app.contracts import CompatibilityCheck, CompatibilityReport

MIN_OVERLAP = float(os.environ.get("MIN_OVERLAP", 0.5))
RESOLUTION_WARN_RATIO = 1.5


def _bbox_overlap_ratio(bounds_a, bounds_b) -> float:
    """Intersection-over-union of two axis-aligned WGS84 bounding boxes.

    This is a bounding-box approximation (not true polygon/footprint
    intersection), which is what bounds_wgs84 already is.
    """
    minx1, miny1, maxx1, maxy1 = bounds_a
    minx2, miny2, maxx2, maxy2 = bounds_b
    ix0, iy0 = max(minx1, minx2), max(miny1, miny2)
    ix1, iy1 = min(maxx1, maxx2), min(maxy1, maxy2)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, maxx1 - minx1) * max(0.0, maxy1 - miny1)
    area_b = max(0.0, maxx2 - minx2) * max(0.0, maxy2 - miny2)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _geo_checks(img_a: dict, img_b: dict) -> tuple[list[CompatibilityCheck], float | None, bool]:
    """Shared CRS/overlap/resolution checks for any georeferenced pair.

    Returns (checks, overlap_ratio, all_geo_checks_passed).
    """
    checks: list[CompatibilityCheck] = []
    overlap_ratio = None
    passed = True

    geo_a, geo_b = img_a.get("is_georeferenced"), img_b.get("is_georeferenced")
    if not (geo_a and geo_b):
        checks.append(
            CompatibilityCheck(
                name="georeferencing",
                status="SKIPPED",
                detail="One or both images are not georeferenced; same-area "
                "coverage is not verifiable — pixel alignment only.",
            )
        )
        return checks, None, passed

    crs_a, crs_b = img_a.get("crs"), img_b.get("crs")
    if crs_a and crs_b:
        if crs_a == crs_b:
            checks.append(
                CompatibilityCheck(name="crs_match", status="PASS", detail=f"Both images use {crs_a}")
            )
        else:
            checks.append(
                CompatibilityCheck(
                    name="crs_match",
                    status="WARN",
                    detail=f"Images use different CRS ({crs_a} vs {crs_b}); reprojection assumed possible",
                )
            )
    else:
        checks.append(
            CompatibilityCheck(name="crs_match", status="WARN", detail="CRS missing on one or both images")
        )

    bounds_a, bounds_b = img_a.get("bounds_wgs84"), img_b.get("bounds_wgs84")
    if bounds_a and bounds_b:
        overlap_ratio = _bbox_overlap_ratio(bounds_a, bounds_b)
        if overlap_ratio < MIN_OVERLAP:
            checks.append(
                CompatibilityCheck(
                    name="overlap",
                    status="FAIL",
                    detail=f"Footprint overlap ratio {overlap_ratio:.2f} is below minimum {MIN_OVERLAP}",
                )
            )
            passed = False
        else:
            checks.append(
                CompatibilityCheck(
                    name="overlap", status="PASS", detail=f"Footprint overlap ratio {overlap_ratio:.2f}"
                )
            )
    else:
        checks.append(
            CompatibilityCheck(name="overlap", status="SKIPPED", detail="Bounds not available for one or both images")
        )

    res_a, res_b = img_a.get("resolution"), img_b.get("resolution")
    if res_a and res_b and res_a[0] > 0 and res_b[0] > 0:
        ratio = max(res_a[0], res_b[0]) / min(res_a[0], res_b[0])
        if ratio > RESOLUTION_WARN_RATIO:
            checks.append(
                CompatibilityCheck(
                    name="resolution_ratio",
                    status="WARN",
                    detail=f"Resolutions differ by {ratio:.2f}x ({res_a[0]}m vs {res_b[0]}m)",
                )
            )
        else:
            checks.append(
                CompatibilityCheck(
                    name="resolution_ratio", status="PASS", detail=f"Resolutions within tolerance ({ratio:.2f}x)"
                )
            )
    else:
        checks.append(
            CompatibilityCheck(name="resolution_ratio", status="SKIPPED", detail="Resolution not available for one or both images")
        )

    return checks, overlap_ratio, passed


def check_temporal_pair(img_a: dict, img_b: dict) -> CompatibilityReport:
    """Compatibility for a same-modality before/after (CHANGE_DETECTION) pair."""
    checks: list[CompatibilityCheck] = []
    ok = True

    modality_a, modality_b = img_a.get("modality"), img_b.get("modality")
    if modality_a == modality_b:
        checks.append(CompatibilityCheck(name="modality_match", status="PASS", detail=f"Both images are {modality_a}"))
    else:
        checks.append(
            CompatibilityCheck(
                name="modality_match",
                status="FAIL",
                detail=f"Images have different modalities ({modality_a} vs {modality_b}); "
                "use cross-modal analysis instead of change detection",
            )
        )
        ok = False

    date_a, date_b = img_a.get("capture_date"), img_b.get("capture_date")
    if date_a and date_b:
        if date_a == date_b:
            checks.append(
                CompatibilityCheck(
                    name="date_distinct",
                    status="FAIL",
                    detail="Both images share the same capture date; change detection requires different dates",
                )
            )
            ok = False
        else:
            checks.append(
                CompatibilityCheck(name="date_distinct", status="PASS", detail=f"{date_a} vs {date_b}")
            )
    else:
        checks.append(
            CompatibilityCheck(
                name="date_distinct",
                status="WARN",
                detail="One or both capture dates are unknown; temporal order is unverified",
            )
        )

    geo_checks, overlap_ratio, geo_ok = _geo_checks(img_a, img_b)
    checks.extend(geo_checks)
    ok = ok and geo_ok

    return CompatibilityReport(
        ok=ok,
        checks=checks,
        overlap_ratio=overlap_ratio,
        coregistration=None,
        alignment_possible=ok,
    )


def check_optical_sar_pair(img_a: dict, img_b: dict, coregistered_hint: bool = False) -> CompatibilityReport:
    """Compatibility for a CROSS_MODAL (1 OPTICAL + 1 SAR) pair."""
    checks: list[CompatibilityCheck] = []
    ok = True

    modalities = {img_a.get("modality"), img_b.get("modality")}
    if modalities == {"OPTICAL", "SAR"}:
        checks.append(CompatibilityCheck(name="modality_pair", status="PASS", detail="One OPTICAL and one SAR image"))
    else:
        checks.append(
            CompatibilityCheck(
                name="modality_pair",
                status="FAIL",
                detail=f"Cross-modal analysis requires one OPTICAL and one SAR image, got {sorted(modalities)}",
            )
        )
        ok = False

    geo_checks, overlap_ratio, geo_ok = _geo_checks(img_a, img_b)
    checks.extend(geo_checks)
    ok = ok and geo_ok

    geo_a, geo_b = img_a.get("is_georeferenced"), img_b.get("is_georeferenced")
    if geo_a and geo_b:
        transform_a, transform_b = img_a.get("transform"), img_b.get("transform")
        crs_a, crs_b = img_a.get("crs"), img_b.get("crs")
        if crs_a and crs_b and crs_a == crs_b and transform_a and transform_b and _transforms_match(transform_a, transform_b):
            coregistration = "verified"
            checks.append(
                CompatibilityCheck(
                    name="coregistration", status="PASS", detail="Same CRS and matching pixel grid — verified"
                )
            )
        else:
            coregistration = "unverified"
            checks.append(
                CompatibilityCheck(
                    name="coregistration",
                    status="WARN",
                    detail="Pixel grids do not provably match; co-registration is unverified",
                )
            )
    elif coregistered_hint:
        coregistration = "assumed"
        checks.append(
            CompatibilityCheck(
                name="coregistration",
                status="WARN",
                detail="Images are not georeferenced; co-registration is assumed from user-provided dataset flag, not verified",
            )
        )
    else:
        coregistration = "unverified"
        checks.append(
            CompatibilityCheck(
                name="coregistration",
                status="WARN",
                detail="Images are not georeferenced; co-registration cannot be verified",
            )
        )

    return CompatibilityReport(
        ok=ok,
        checks=checks,
        overlap_ratio=overlap_ratio,
        coregistration=coregistration,
        alignment_possible=ok,
    )


def _transforms_match(transform_a, transform_b, tolerance: float = 1e-6) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(transform_a, transform_b))


def check_single_image(img: dict) -> CompatibilityReport:
    """Compatibility for a single-image task (VQA/GROUNDING)."""
    checks: list[CompatibilityCheck] = []
    ok = True

    if img.get("width") and img.get("height"):
        checks.append(CompatibilityCheck(name="readable", status="PASS", detail="Image dimensions are known"))
    else:
        checks.append(
            CompatibilityCheck(name="readable", status="WARN", detail="Image dimensions could not be determined")
        )

    modality = img.get("modality")
    if modality in ("OPTICAL", "SAR"):
        checks.append(CompatibilityCheck(name="modality_supported", status="PASS", detail=f"Modality {modality} is supported"))
    else:
        checks.append(
            CompatibilityCheck(
                name="modality_supported", status="FAIL", detail=f"Unsupported modality: {modality}"
            )
        )
        ok = False

    return CompatibilityReport(ok=ok, checks=checks, overlap_ratio=None, coregistration=None, alignment_possible=ok)
