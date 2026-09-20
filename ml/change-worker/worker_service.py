"""Local change detection worker HTTP service.

Serves the trained Siamese CNN change detection model over HTTP
so the FastAPI backend stays torch-free. Requests are served
serially (single inference slot).

Endpoints:
  GET  /health   -> service/model status
  POST /change   -> multipart: before_image, after_image
                    returns model_version, latency_ms, and the binary change
                    mask as a base64 PNG (0 = unchanged, 255 = changed) at the
                    model's 256x256 working resolution. The worker keeps no
                    files: the backend stores and serves the mask itself.
"""

import base64
import io
import os
import tempfile
import threading
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image as PILImage
from pydantic import BaseModel

from model_provider import get_provider as _get_model_provider, MODEL_VERSION

_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _safe_suffix(filename: str | None) -> str:
    suffix = os.path.splitext(filename or "")[1].lower()
    return suffix if suffix in _ALLOWED_SUFFIXES else ".png"


class HealthStatus(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_version: str


class ChangeResponse(BaseModel):
    model_version: str
    latency_ms: int
    mask_png_base64: str
    mask_width: int
    mask_height: int


app = FastAPI(title="SatQuery Change Detection Worker")
_change_lock = threading.Lock()
_provider = None


def get_provider():
    global _provider
    if _provider is None:
        _provider = _get_model_provider()
    return _provider


@app.on_event("startup")
def _startup():
    try:
        get_provider()
    except Exception as exc:
        raise RuntimeError(f"Failed to load change model: {exc}") from exc


@app.get("/health", response_model=HealthStatus)
def health():
    p = _provider
    if p is None or not p.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthStatus(
        status="ok",
        device=p.device,
        model_loaded=True,
        model_version=MODEL_VERSION,
    )


@app.post("/change")
def change(before_image: UploadFile = File(...), after_image: UploadFile = File(...)):
    if before_image.filename is None or after_image.filename is None:
        raise HTTPException(status_code=400, detail="Both before_image and after_image are required")

    before_path = None
    after_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=_safe_suffix(before_image.filename), delete=False) as tmp:
            tmp.write(before_image.file.read())
            before_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=_safe_suffix(after_image.filename), delete=False) as tmp:
            tmp.write(after_image.file.read())
            after_path = tmp.name

        with _change_lock:
            result = get_provider().detect(before_path, after_path)

        mask = result["mask"]
        buf = io.BytesIO()
        PILImage.fromarray(mask * 255).save(buf, format="PNG")

        return ChangeResponse(
            model_version=result["model_version"],
            latency_ms=result["latency_ms"],
            mask_png_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
            mask_width=int(mask.shape[1]),
            mask_height=int(mask.shape[0]),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Change detection failed: {exc}") from exc
    finally:
        for p in (before_path, after_path):
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def main():
    port = int(os.getenv("CHANGE_WORKER_PORT", "8002"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
