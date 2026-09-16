"""Phase B1 acceptance: demo scenario pairs are genuine and readable.

Verifies data/demo/ files load correctly and, for the georeferenced pair,
that CRS/bounds/overlap are real and consistent — not that any specific
land-cover claim is true (that's for the actual specialists to measure).
"""

import json
import os

from PIL import Image as PILImage

from app.services.raster_ingest import extract_metadata

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DEMO_DIR = os.path.join(REPO_ROOT, "data", "demo")


def test_manifest_is_valid_json_with_three_scenarios():
    with open(os.path.join(REPO_ROOT, "data", "manifest.json")) as f:
        manifest = json.load(f)
    scenario_ids = {s["id"] for s in manifest["demo_scenarios"]}
    assert scenario_ids == {"scenario_A_single", "scenario_B_temporal", "scenario_C_optical_sar"}


def test_scenario_a_single_image_readable():
    path = os.path.join(DEMO_DIR, "scenario_A_single", "single_image.jpg")
    meta = extract_metadata(path)
    assert meta.width and meta.height
    assert meta.is_georeferenced is False


def test_scenario_b_temporal_pair_same_real_georeferenced_area():
    before = extract_metadata(os.path.join(DEMO_DIR, "scenario_B_temporal", "before.tif"))
    after = extract_metadata(os.path.join(DEMO_DIR, "scenario_B_temporal", "after.tif"))

    assert before.is_georeferenced and after.is_georeferenced
    assert before.crs == after.crs == "EPSG:32611"
    assert before.bounds_wgs84 == after.bounds_wgs84  # same crop window, genuine overlap = 100%
    assert before.resolution == after.resolution == (10.0, 10.0)
    # These are TCI visualization products with no acquisition-date tag —
    # the real dates are documented in data/manifest.json instead.
    assert before.acquisition_date_source == "unknown"


def test_scenario_c_optical_sar_pair_same_patch_grid_dimensions():
    optical = PILImage.open(os.path.join(DEMO_DIR, "scenario_C_optical_sar", "optical.png"))
    sar = PILImage.open(os.path.join(DEMO_DIR, "scenario_C_optical_sar", "sar.png"))
    assert optical.size == sar.size == (120, 120)
    assert optical.mode == "RGB"
    assert sar.mode == "L"
