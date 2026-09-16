import os
from datetime import date

import numpy as np
import pytest
import rasterio
from PIL import Image as PILImage
from rasterio.transform import from_origin

from app.services import raster_ingest
from app.services.raster_ingest import (
    RasterIngestError,
    extract_metadata,
    load_rgb_preview,
)


def _write_geotiff(path, width=16, height=16, band_count=3, dtype="uint8", crs="EPSG:4326", tags=None):
    transform = from_origin(76.90, 28.07, 0.0001, 0.0001)
    data = (np.random.rand(band_count, height, width) * 200).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=band_count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as ds:
        for i in range(band_count):
            ds.write(data[i], i + 1)
        if tags:
            ds.update_tags(**tags)
    return path


def test_geotiff_georeferenced_metadata(tmp_path):
    path = str(tmp_path / "test.tif")
    _write_geotiff(path)
    meta = extract_metadata(path)

    assert meta.format == "GTiff"
    assert meta.width == 16 and meta.height == 16
    assert meta.band_count == 3
    assert meta.dtype == "uint8"
    assert meta.crs == "EPSG:4326"
    assert meta.is_georeferenced is True
    assert meta.bounds is not None
    assert meta.bounds_wgs84 is not None
    assert meta.resolution == pytest.approx((0.0001, 0.0001))
    assert meta.acquisition_date_source == "unknown"
    assert meta.acquisition_date is None
    assert any("acquisition date" in w.lower() for w in meta.warnings)


def test_geotiff_file_metadata_date(tmp_path):
    path = str(tmp_path / "dated.tif")
    _write_geotiff(path, tags={"TIFFTAG_DATETIME": "2023:06:14 10:30:00"})
    meta = extract_metadata(path)

    assert meta.acquisition_date == date(2023, 6, 14)
    assert meta.acquisition_date_source == "file_metadata"


def test_user_date_overrides_file_metadata(tmp_path):
    path = str(tmp_path / "dated2.tif")
    _write_geotiff(path, tags={"TIFFTAG_DATETIME": "2023:06:14 10:30:00"})
    meta = extract_metadata(path, user_capture_date=date(2024, 1, 1))

    assert meta.acquisition_date == date(2024, 1, 1)
    assert meta.acquisition_date_source == "user"


def test_geotiff_no_crs_is_not_georeferenced(tmp_path):
    path = str(tmp_path / "nocrs.tif")
    transform = from_origin(0, 0, 1, 1)
    data = (np.random.rand(1, 8, 8) * 200).astype("uint8")
    with rasterio.open(
        path, "w", driver="GTiff", height=8, width=8, count=1, dtype="uint8", transform=transform
    ) as ds:
        ds.write(data[0], 1)
    meta = extract_metadata(path)

    assert meta.crs is None
    assert meta.is_georeferenced is False
    assert any("no crs" in w.lower() for w in meta.warnings)


def test_png_metadata(tmp_path):
    path = str(tmp_path / "test.png")
    PILImage.new("RGB", (10, 20), color=(0, 100, 200)).save(path)
    meta = extract_metadata(path)

    assert meta.format == "PNG"
    assert meta.width == 10 and meta.height == 20
    assert meta.crs is None
    assert meta.bounds is None
    assert meta.is_georeferenced is False
    assert meta.acquisition_date_source == "unknown"


def test_corrupt_tif_raises(tmp_path):
    path = tmp_path / "corrupt.tif"
    path.write_bytes(os.urandom(256))
    with pytest.raises(RasterIngestError) as exc_info:
        extract_metadata(str(path))
    assert str(tmp_path) not in str(exc_info.value)


def test_non_image_renamed_to_tif_raises(tmp_path):
    path = tmp_path / "fake.tif"
    path.write_text("this is definitely not an image, just text pretending to be one")
    with pytest.raises(RasterIngestError) as exc_info:
        extract_metadata(str(path))
    assert str(tmp_path) not in str(exc_info.value)


def test_missing_file_raises(tmp_path):
    with pytest.raises(RasterIngestError):
        extract_metadata(str(tmp_path / "nope.tif"))


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "file.gif"
    path.write_bytes(b"GIF89a")
    with pytest.raises(RasterIngestError):
        extract_metadata(str(path))


def test_huge_raster_warns_but_does_not_read_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr(raster_ingest, "MAX_RASTER_PIXELS", 100)
    path = str(tmp_path / "big.tif")
    _write_geotiff(path, width=16, height=16)
    meta = extract_metadata(path)

    assert any("exceeding max_raster_pixels" in w.lower() for w in meta.warnings)


def test_load_rgb_preview_geotiff_rgb(tmp_path):
    path = str(tmp_path / "preview.tif")
    _write_geotiff(path, width=32, height=32, band_count=3)
    arr = load_rgb_preview(path)

    assert arr.dtype == np.uint8
    assert arr.shape == (32, 32, 3)


def test_load_rgb_preview_multispectral_picks_432(tmp_path):
    path = str(tmp_path / "multispectral.tif")
    _write_geotiff(path, width=16, height=16, band_count=6, dtype="uint16")
    arr = load_rgb_preview(path)

    assert arr.dtype == np.uint8
    assert arr.shape == (16, 16, 3)


def test_load_rgb_preview_png_grayscale(tmp_path):
    path = str(tmp_path / "gray.png")
    PILImage.new("L", (10, 10), color=128).save(path)
    arr = load_rgb_preview(path)

    assert arr.dtype == np.uint8
    assert arr.shape[0] == 10 and arr.shape[1] == 10


def test_load_rgb_preview_downsamples_large_image(tmp_path):
    path = str(tmp_path / "large.tif")
    _write_geotiff(path, width=4000, height=2000, band_count=3)
    arr = load_rgb_preview(path, max_side=500)

    assert max(arr.shape[0], arr.shape[1]) <= 500
