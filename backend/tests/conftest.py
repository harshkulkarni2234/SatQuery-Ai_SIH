import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


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
