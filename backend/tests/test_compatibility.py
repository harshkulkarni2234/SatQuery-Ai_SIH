from app.services.compatibility import (
    check_optical_sar_pair,
    check_single_image,
    check_temporal_pair,
)

GEOREF_A = dict(
    modality="OPTICAL",
    capture_date="2021-03-01",
    is_georeferenced=True,
    crs="EPSG:4326",
    bounds_wgs84=(76.90, 28.00, 76.95, 28.05),
    resolution=(10.0, 10.0),
    transform=(0.0001, 0.0, 76.90, 0.0, -0.0001, 28.07),
)


def _make(base, **overrides):
    d = dict(base)
    d.update(overrides)
    return d


def test_temporal_same_date_fails():
    a = _make(GEOREF_A)
    b = _make(GEOREF_A, capture_date="2021-03-01")
    report = check_temporal_pair(a, b)
    assert report.ok is False
    date_check = next(c for c in report.checks if c.name == "date_distinct")
    assert date_check.status == "FAIL"


def test_temporal_different_dates_pass():
    a = _make(GEOREF_A)
    b = _make(GEOREF_A, capture_date="2023-06-14")
    report = check_temporal_pair(a, b)
    date_check = next(c for c in report.checks if c.name == "date_distinct")
    assert date_check.status == "PASS"


def test_temporal_missing_dates_warns_not_fails():
    a = _make(GEOREF_A, capture_date=None)
    b = _make(GEOREF_A, capture_date=None)
    report = check_temporal_pair(a, b)
    date_check = next(c for c in report.checks if c.name == "date_distinct")
    assert date_check.status == "WARN"
    # A missing date alone shouldn't fail the whole report.
    assert report.ok is True


def test_temporal_non_overlapping_fails():
    a = _make(GEOREF_A, capture_date="2021-03-01")
    b = _make(
        GEOREF_A,
        capture_date="2023-06-14",
        bounds_wgs84=(10.0, 10.0, 10.1, 10.1),
    )
    report = check_temporal_pair(a, b)
    assert report.ok is False
    assert report.overlap_ratio == 0.0
    overlap_check = next(c for c in report.checks if c.name == "overlap")
    assert overlap_check.status == "FAIL"


def test_temporal_different_crs_warns():
    a = _make(GEOREF_A, capture_date="2021-03-01")
    b = _make(GEOREF_A, capture_date="2023-06-14", crs="EPSG:32643")
    report = check_temporal_pair(a, b)
    crs_check = next(c for c in report.checks if c.name == "crs_match")
    assert crs_check.status == "WARN"


def test_temporal_wrong_modality_fails():
    a = _make(GEOREF_A, capture_date="2021-03-01")
    b = _make(GEOREF_A, capture_date="2023-06-14", modality="SAR")
    report = check_temporal_pair(a, b)
    assert report.ok is False
    modality_check = next(c for c in report.checks if c.name == "modality_match")
    assert modality_check.status == "FAIL"


def test_temporal_non_georeferenced_pair_skips_geo_checks():
    a = _make(GEOREF_A, capture_date="2021-03-01", is_georeferenced=False, crs=None, bounds_wgs84=None)
    b = _make(GEOREF_A, capture_date="2023-06-14", is_georeferenced=False, crs=None, bounds_wgs84=None)
    report = check_temporal_pair(a, b)
    geo_check = next(c for c in report.checks if c.name == "georeferencing")
    assert geo_check.status == "SKIPPED"
    assert report.overlap_ratio is None
    # Missing georeferencing shouldn't fail the report by itself.
    assert report.ok is True


def test_optical_sar_wrong_modality_combo_fails():
    a = _make(GEOREF_A, modality="OPTICAL")
    b = _make(GEOREF_A, modality="OPTICAL")
    report = check_optical_sar_pair(a, b)
    assert report.ok is False
    modality_check = next(c for c in report.checks if c.name == "modality_pair")
    assert modality_check.status == "FAIL"


def test_optical_sar_verified_coregistration():
    a = _make(GEOREF_A, modality="OPTICAL")
    b = _make(GEOREF_A, modality="SAR")
    report = check_optical_sar_pair(a, b)
    assert report.coregistration == "verified"
    coreg_check = next(c for c in report.checks if c.name == "coregistration")
    assert coreg_check.status == "PASS"


def test_optical_sar_unverified_when_transforms_differ():
    a = _make(GEOREF_A, modality="OPTICAL")
    b = _make(GEOREF_A, modality="SAR", transform=(0.0002, 0.0, 76.90, 0.0, -0.0002, 28.07))
    report = check_optical_sar_pair(a, b)
    assert report.coregistration == "unverified"


def test_optical_sar_non_georeferenced_unverified():
    a = _make(GEOREF_A, modality="OPTICAL", is_georeferenced=False, crs=None, bounds_wgs84=None, transform=None)
    b = _make(GEOREF_A, modality="SAR", is_georeferenced=False, crs=None, bounds_wgs84=None, transform=None)
    report = check_optical_sar_pair(a, b)
    assert report.coregistration == "unverified"


def test_optical_sar_non_georeferenced_assumed_with_hint():
    a = _make(GEOREF_A, modality="OPTICAL", is_georeferenced=False, crs=None, bounds_wgs84=None, transform=None)
    b = _make(GEOREF_A, modality="SAR", is_georeferenced=False, crs=None, bounds_wgs84=None, transform=None)
    report = check_optical_sar_pair(a, b, coregistered_hint=True)
    assert report.coregistration == "assumed"


def test_single_image_supported_modality():
    report = check_single_image({"modality": "OPTICAL", "width": 100, "height": 100})
    assert report.ok is True


def test_single_image_unsupported_modality_fails():
    report = check_single_image({"modality": "RGB", "width": 100, "height": 100})
    assert report.ok is False
