import io
import os
import uuid

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.database import SessionLocal
from app.main import app
from app.models import Image

client = TestClient(app)


def _geotiff_bytes(width=32, height=32, band_count=3, crs="EPSG:4326"):
    transform = from_origin(76.90, 28.07, 0.0001, 0.0001)
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


def test_upload_geotiff_returns_full_metadata():
    buf = _geotiff_bytes()
    resp = client.post(
        "/images/upload",
        files={"file": ("aoi.tif", buf, "image/tiff")},
        data={"modality": "OPTICAL"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["metadata"] is not None
    meta = body["metadata"]
    assert meta["crs"] == "EPSG:4326"
    assert meta["bounds"] is not None
    assert meta["resolution"] is not None
    assert meta["band_count"] == 3
    assert meta["is_georeferenced"] is True

    db = SessionLocal()
    try:
        row = db.get(Image, uuid.UUID(body["image_id"]))
        assert row.crs == "EPSG:4326"
        assert row.band_count == 3
        assert row.is_georeferenced is True
        assert row.bounds is not None
    finally:
        db.close()


def test_upload_corrupt_tif_returns_400_and_no_row():
    db = SessionLocal()
    try:
        count_before = db.query(Image).count()
    finally:
        db.close()

    resp = client.post(
        "/images/upload",
        files={"file": ("corrupt.tif", io.BytesIO(os.urandom(128)), "image/tiff")},
        data={"modality": "OPTICAL"},
    )
    assert resp.status_code == 400
    assert "detail" in resp.json()

    db = SessionLocal()
    try:
        count_after = db.query(Image).count()
    finally:
        db.close()
    assert count_after == count_before


def test_get_image_detail():
    buf = _geotiff_bytes()
    upload_resp = client.post(
        "/images/upload",
        files={"file": ("detail.tif", buf, "image/tiff")},
        data={"modality": "SAR"},
    )
    image_id = upload_resp.json()["image_id"]

    get_resp = client.get(f"/images/{image_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["image_id"] == image_id
    assert body["modality"] == "SAR"
    assert body["metadata"]["crs"] == "EPSG:4326"


def test_get_unknown_image_returns_404():
    resp = client.get(f"/images/{uuid.uuid4()}")
    assert resp.status_code == 404
