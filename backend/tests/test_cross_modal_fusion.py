"""Phase B9: SAR speckle filtering, VV/VH, NDVI/NDWI, per-modality fusion
attribution, and coregistration-aware pixel-level agreement confidence."""

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.cross_modal import analyze_pair, TOOL_VERSION

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DEMO_OPTICAL = os.path.join(REPO_ROOT, "data", "demo", "scenario_C_optical_sar", "optical.png")
DEMO_SAR = os.path.join(REPO_ROOT, "data", "demo", "scenario_C_optical_sar", "sar.png")


def _write_geotiff(path, west, north, bands, values, dtype="uint8", res=0.0001, size=60):
    transform = from_origin(west, north, res, res)
    data = np.zeros((bands, size, size), dtype=dtype)
    for b in range(bands):
        data[b, :, :] = values[b] if b < len(values) else values[-1]
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=bands,
        dtype=dtype, crs="EPSG:4326", transform=transform,
    ) as ds:
        ds.write(data)


def test_confidence_unavailable_when_not_coregistered(tmp_path):
    optical = str(tmp_path / "optical.png")
    sar = str(tmp_path / "sar.png")
    import cv2
    cv2.imwrite(optical, np.full((60, 60, 3), (60, 140, 60), dtype=np.uint8))
    cv2.imwrite(sar, np.full((60, 60), 20, dtype=np.uint8))

    result = analyze_pair(optical, sar, coregistration=None)
    assert result["confidence_score"] is None
    assert result["confidence_source"] == "unavailable"
    assert "could not be verified" in result["spatial_correspondence_note"]


def test_confidence_unavailable_when_unverified_explicitly(tmp_path):
    optical = str(tmp_path / "optical.png")
    sar = str(tmp_path / "sar.png")
    import cv2
    cv2.imwrite(optical, np.full((60, 60, 3), (60, 140, 60), dtype=np.uint8))
    cv2.imwrite(sar, np.full((60, 60), 20, dtype=np.uint8))

    result = analyze_pair(optical, sar, coregistration="unverified")
    assert result["confidence_score"] is None


def test_real_pixel_agreement_when_verified_and_georeferenced(tmp_path):
    optical_path = str(tmp_path / "optical.tif")
    sar_path = str(tmp_path / "sar.tif")
    # Optical: blue-dominant (water-like) -> B=200,G=80,R=80
    _write_geotiff(optical_path, 76.90, 28.07, bands=3, values=[200, 80, 80])
    # SAR: very low backscatter (dark, water-like)
    _write_geotiff(sar_path, 76.90, 28.07, bands=1, values=[10])

    result = analyze_pair(optical_path, sar_path, coregistration="verified")
    assert result["confidence_score"] is not None
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert result["confidence_source"] == "optical/SAR pixel mask agreement"
    assert len(result["confidence_source"]) <= 50  # must fit the DB column
    assert "verified" in result["spatial_correspondence_note"]


def test_assumed_coregistration_still_labeled_honestly(tmp_path):
    optical_path = str(tmp_path / "optical.tif")
    sar_path = str(tmp_path / "sar.tif")
    _write_geotiff(optical_path, 76.90, 28.07, bands=3, values=[200, 80, 80])
    _write_geotiff(sar_path, 76.90, 28.07, bands=1, values=[10])

    result = analyze_pair(optical_path, sar_path, coregistration="assumed")
    assert "assumed" in result["spatial_correspondence_note"]
    assert "not independently verified" in result["spatial_correspondence_note"]


def test_vv_vh_polarization_stats_from_2band_sar(tmp_path):
    sar_path = str(tmp_path / "sar_2band.tif")
    _write_geotiff(sar_path, 76.90, 28.07, bands=2, values=[100, 50])
    optical_path = str(tmp_path / "optical.png")
    import cv2
    cv2.imwrite(optical_path, np.full((60, 60, 3), (100, 100, 100), dtype=np.uint8))

    result = analyze_pair(optical_path, sar_path)
    pol = result["evidence"]["sar"]["polarization"]
    assert pol["vv_mean"] == pytest.approx(100.0, abs=1)
    assert pol["vh_mean"] == pytest.approx(50.0, abs=1)
    assert pol["vh_vv_ratio"] == pytest.approx(0.5, abs=0.05)


def test_single_band_sar_has_no_polarization_stats(tmp_path):
    sar_path = str(tmp_path / "sar_1band.png")
    import cv2
    cv2.imwrite(sar_path, np.full((60, 60), 30, dtype=np.uint8))
    optical_path = str(tmp_path / "optical.png")
    cv2.imwrite(optical_path, np.full((60, 60, 3), (100, 100, 100), dtype=np.uint8))

    result = analyze_pair(optical_path, sar_path)
    pol = result["evidence"]["sar"]["polarization"]
    assert pol["vv_mean"] is None
    assert "single-band" in pol["note"]


def test_ndvi_ndwi_from_4band_optical(tmp_path):
    optical_path = str(tmp_path / "optical_4band.tif")
    # bands: blue, green, red, nir -- high NIR relative to red -> high NDVI
    _write_geotiff(optical_path, 76.90, 28.07, bands=4, values=[50, 60, 40, 200])
    sar_path = str(tmp_path / "sar.png")
    import cv2
    cv2.imwrite(sar_path, np.full((60, 60), 100, dtype=np.uint8))

    result = analyze_pair(optical_path, sar_path)
    spectral = result["evidence"]["optical"]["spectral_indices"]
    assert spectral["ndvi_mean"] is not None
    assert spectral["ndvi_mean"] > 0.5  # NIR >> Red


def test_ndvi_ndwi_none_for_3band_optical(tmp_path):
    optical_path = str(tmp_path / "optical.png")
    import cv2
    cv2.imwrite(optical_path, np.full((60, 60, 3), (100, 100, 100), dtype=np.uint8))
    sar_path = str(tmp_path / "sar.png")
    cv2.imwrite(sar_path, np.full((60, 60), 100, dtype=np.uint8))

    result = analyze_pair(optical_path, sar_path)
    spectral = result["evidence"]["optical"]["spectral_indices"]
    assert spectral["ndvi_mean"] is None
    assert "requires 4+ bands" in spectral["note"]


def test_per_modality_fusion_attribution_water_both(tmp_path):
    optical_path = str(tmp_path / "optical.png")
    sar_path = str(tmp_path / "sar.png")
    import cv2
    # Clear blue-dominant optical (water-like) + very low SAR backscatter
    cv2.imwrite(optical_path, np.full((60, 60, 3), (200, 80, 80), dtype=np.uint8))
    cv2.imwrite(sar_path, np.full((60, 60), 10, dtype=np.uint8))

    result = analyze_pair(optical_path, sar_path)
    assert result["evidence"]["per_modality"]["water"] == "both"


def test_cloud_covered_optical_routes_water_to_sar_only(tmp_path):
    optical_path = str(tmp_path / "cloudy_optical.png")
    sar_path = str(tmp_path / "sar.png")
    import cv2
    # Bright, low-saturation (cloud-like) optical -- no blue water signature
    cv2.imwrite(optical_path, np.full((60, 60, 3), (230, 230, 225), dtype=np.uint8))
    # SAR: very low backscatter (water-like)
    cv2.imwrite(sar_path, np.full((60, 60), 10, dtype=np.uint8))

    result = analyze_pair(optical_path, sar_path)
    optical_ev = result["evidence"]["optical"]
    assert optical_ev["cloud_covered_fraction"] > 0.5
    attribution = result["evidence"]["per_modality"]
    assert attribution["water"] == "sar_only"


def test_model_version_bumped_for_b9():
    assert TOOL_VERSION == "cross-modal-deterministic-v2"


@pytest.mark.skipif(
    not (os.path.isfile(DEMO_OPTICAL) and os.path.isfile(DEMO_SAR)),
    reason="real B1 demo optical+SAR pair not present",
)
def test_real_demo_pair_produces_joint_evidence_with_attribution():
    """Acceptance: 'identify built-up and water-covered regions' style query
    returns joint evidence with modality attribution, even though this real
    demo pair ships as plain PNG (is_georeferenced=False) so coregistration
    is honestly unverified and confidence stays unavailable."""
    result = analyze_pair(
        DEMO_OPTICAL, DEMO_SAR,
        query_text="Identify built-up and water-covered regions.",
        coregistration=None,
    )
    assert result["evidence"]["per_modality"]
    assert set(result["evidence"]["per_modality"]) == {"water", "built_up", "vegetation"}
    assert result["confidence_score"] is None
    assert result["confidence_source"] == "unavailable"
    assert "could not be verified" in result["spatial_correspondence_note"]
