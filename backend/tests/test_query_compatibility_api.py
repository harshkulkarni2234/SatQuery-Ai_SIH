import io

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.main import app

client = TestClient(app)


def _geotiff_bytes(west, north, width=32, height=32, band_count=3, crs="EPSG:4326", capture_date=None):
    transform = from_origin(west, north, 0.0001, 0.0001)
    buf = io.BytesIO()
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=height, width=width, count=band_count,
            dtype="uint8", crs=crs, transform=transform,
        ) as ds:
            data = (np.random.rand(band_count, height, width) * 200).astype("uint8")
            for i in range(band_count):
                ds.write(data[i], i + 1)
        buf.write(memfile.read())
    buf.seek(0)
    return buf


def _upload(name, buf, modality, capture_date=None):
    data = {"modality": modality}
    if capture_date:
        data["capture_date"] = capture_date
    resp = client.post(
        "/images/upload",
        files={"file": (name, buf, "image/tiff")},
        data=data,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["image_id"]


def test_change_detection_non_overlapping_pair_returns_422_with_report():
    id_before = _upload(
        "before.tif", _geotiff_bytes(76.90, 28.07), "OPTICAL", capture_date="2021-03-01"
    )
    id_after = _upload(
        "after.tif", _geotiff_bytes(10.0, 10.0), "OPTICAL", capture_date="2023-06-14"
    )

    resp = client.post(
        "/query",
        json={"query_text": "What changed between these two images?", "image_ids": [id_before, id_after]},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "compatibility" in detail
    assert detail["compatibility"]["ok"] is False
    overlap_check = next(c for c in detail["compatibility"]["checks"] if c["name"] == "overlap")
    assert overlap_check["status"] == "FAIL"


def test_change_detection_overlapping_pair_proceeds():
    id_before = _upload(
        "before2.tif", _geotiff_bytes(76.90, 28.07), "OPTICAL", capture_date="2021-03-01"
    )
    id_after = _upload(
        "after2.tif", _geotiff_bytes(76.90, 28.07), "OPTICAL", capture_date="2023-06-14"
    )

    resp = client.post(
        "/query",
        json={"query_text": "What changed between these two images?", "image_ids": [id_before, id_after]},
    )
    assert resp.status_code == 200


def test_cross_modal_optical_sar_overlapping_pair_proceeds():
    id_optical = _upload("optical.tif", _geotiff_bytes(76.90, 28.07), "OPTICAL")
    id_sar = _upload("sar.tif", _geotiff_bytes(76.90, 28.07), "SAR")

    resp = client.post(
        "/query",
        json={"query_text": "Compare the optical and SAR images.", "image_ids": [id_optical, id_sar]},
    )
    assert resp.status_code == 200
