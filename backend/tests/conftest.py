import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models import Image, Query, QueryResult
from app.services.change_detection import MASK_DIR

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db_and_files():
    """Remove all test-created records and generated files after the suite."""
    yield
    db = SessionLocal()
    try:
        db.query(QueryResult).delete()
        db.query(Query).delete()
        for img in db.query(Image).all():
            if img.file_path and os.path.isfile(img.file_path):
                try:
                    os.remove(img.file_path)
                except OSError:
                    pass
        db.query(Image).delete()
        db.commit()
    finally:
        db.close()
    if os.path.isdir(MASK_DIR):
        for f in os.listdir(MASK_DIR):
            try:
                os.remove(os.path.join(MASK_DIR, f))
            except OSError:
                pass


@pytest.fixture(scope="module")
def uploaded_image_id():
    """Upload a tiny valid PNG and return its image_id."""
    import io
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (4, 4), color=(0, 100, 200)).save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        "/images/upload",
        files={"file": ("test.png", buf, "image/png")},
        data={"modality": "OPTICAL"},
    )
    assert resp.status_code == 200
    return resp.json()["image_id"]


@pytest.fixture(scope="module")
def uploaded_image_ids():
    """Upload two small images and return their IDs."""
    import io
    from PIL import Image as PILImage

    ids = []
    for i, mod in enumerate(["OPTICAL", "SAR"]):
        buf = io.BytesIO()
        PILImage.new("RGB", (4, 4), color=(i * 50, 100, 200)).save(buf, format="PNG")
        buf.seek(0)
        resp = client.post(
            "/images/upload",
            files={"file": (f"test_{mod.lower()}.png", buf, "image/png")},
            data={"modality": mod},
        )
        assert resp.status_code == 200
        ids.append(resp.json()["image_id"])
    return ids


def upload_png_bytes(png_bytes: bytes, modality: str = "OPTICAL", name: str = "img.png"):
    """Upload raw PNG bytes and return the image_id."""
    resp = client.post(
        "/images/upload",
        files={"file": (name, png_bytes, "image/png")},
        data={"modality": modality},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["image_id"]