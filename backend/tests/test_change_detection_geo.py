"""Phase B7: geographic-grid alignment + real area-in-m2 for change detection."""

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.change_detection import detect_change

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DEMO_BEFORE = os.path.join(REPO_ROOT, "data", "demo", "scenario_B_temporal", "before.tif")
DEMO_AFTER = os.path.join(REPO_ROOT, "data", "demo", "scenario_B_temporal", "after.tif")


def _write_geotiff(path, west, north, res=0.0001, size=80, region=None, seed=0):
    rng = np.random.default_rng(seed)
    transform = from_origin(west, north, res, res)
    data = (rng.random((3, size, size)) * 60 + 80).astype("uint8")
    if region:
        x, y, w, h = region
        data[:, y : y + h, x : x + w] = 240
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype="uint8", crs="EPSG:4326", transform=transform,
    ) as ds:
        ds.write(data)


def _meta(is_georeferenced, resolution=None, crs=None):
    return {
        "is_georeferenced": is_georeferenced,
        "resolution": resolution,
        "crs": crs,
        "bbox_coords": None,
        "capture_date": None,
        "modality": "OPTICAL",
    }


def test_geographic_alignment_used_when_both_georeferenced(tmp_path):
    before_path = str(tmp_path / "before.tif")
    after_path = str(tmp_path / "after.tif")
    _write_geotiff(before_path, 76.90, 28.07, region=(20, 20, 25, 25), seed=1)
    _write_geotiff(after_path, 76.90, 28.07, region=(20, 20, 25, 25), seed=1)

    result = detect_change(
        before_path, after_path,
        metadata_before=_meta(True, resolution=(10.0, 10.0)),
        metadata_after=_meta(True, resolution=(10.0, 10.0)),
    )
    assert result["alignment_method"] == "geographic_reprojection"
    assert result["registration_applied"] is False  # phase correlation didn't run


def test_pixel_based_fallback_when_not_georeferenced(tmp_path):
    before_path = str(tmp_path / "before.png")
    after_path = str(tmp_path / "after.png")
    import cv2
    cv2.imwrite(before_path, np.full((80, 80, 3), 100, dtype=np.uint8))
    cv2.imwrite(after_path, np.full((80, 80, 3), 100, dtype=np.uint8))

    result = detect_change(before_path, after_path)
    assert result["alignment_method"] == "none"


def test_real_area_in_m2_computed_when_resolution_known(tmp_path):
    before_path = str(tmp_path / "before.tif")
    after_path = str(tmp_path / "after.tif")
    _write_geotiff(before_path, 76.90, 28.07, region=None, seed=2)
    _write_geotiff(after_path, 76.90, 28.07, region=(10, 10, 30, 30), seed=2)

    result = detect_change(
        before_path, after_path,
        metadata_before=_meta(True, resolution=(10.0, 10.0)),
        metadata_after=_meta(True, resolution=(10.0, 10.0)),
    )
    if result["changed_pixels"]:
        assert result["changed_area_m2"] is not None
        expected = result["changed_pixels"] * 10.0 * 10.0
        assert result["changed_area_m2"] == pytest.approx(expected, rel=0.01)
        assert "m²" in result["answer_text"]


def test_area_in_m2_is_none_when_resolution_unknown(tmp_path):
    before_path = str(tmp_path / "before.tif")
    after_path = str(tmp_path / "after.tif")
    _write_geotiff(before_path, 76.90, 28.07, region=None, seed=3)
    _write_geotiff(after_path, 76.90, 28.07, region=(10, 10, 30, 30), seed=3)

    result = detect_change(
        before_path, after_path,
        metadata_before=_meta(True, resolution=None),
        metadata_after=_meta(True, resolution=None),
    )
    assert result["changed_area_m2"] is None
    assert "m²" not in result["answer_text"]


def test_falls_back_gracefully_when_geo_alignment_raises(tmp_path, monkeypatch):
    """If rasterio.warp.reproject fails for any reason, the pipeline should
    still produce a result via the phase-correlation fallback, not crash."""
    import app.services.change_detection as cd_module

    before_path = str(tmp_path / "before.tif")
    after_path = str(tmp_path / "after.tif")
    _write_geotiff(before_path, 76.90, 28.07, seed=4)
    _write_geotiff(after_path, 76.90, 28.07, seed=4)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated reproject failure")

    monkeypatch.setattr(cd_module, "_geographic_align", _boom)

    result = detect_change(
        before_path, after_path,
        metadata_before=_meta(True, resolution=(10.0, 10.0)),
        metadata_after=_meta(True, resolution=(10.0, 10.0)),
    )
    assert result["alignment_method"] in ("phase_correlation", "none")


@pytest.mark.skipif(
    not (os.path.isfile(DEMO_BEFORE) and os.path.isfile(DEMO_AFTER)),
    reason="real demo temporal pair not present",
)
def test_real_lake_mead_demo_pair_produces_meaningful_output():
    result = detect_change(
        DEMO_BEFORE, DEMO_AFTER,
        metadata_before=_meta(True, resolution=(10.0, 10.0), crs="EPSG:32611"),
        metadata_after=_meta(True, resolution=(10.0, 10.0), crs="EPSG:32611"),
    )
    assert result["validation_failed"] is False
    assert result["alignment_method"] == "geographic_reprojection"
    assert result["total_pixels"] > 0
    # Real, measured — not asserting a specific magnitude, only that a real
    # figure was produced and is internally consistent.
    if result["changed_pixels"]:
        assert result["changed_area_m2"] == pytest.approx(
            result["changed_pixels"] * 100.0, rel=0.01
        )
    assert "semantic change classification was not performed" in result["answer_text"]
