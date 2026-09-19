"""Local change detection worker HTTP service.

Serves the trained Siamese CNN change detection model over HTTP
so the FastAPI backend stays torch-free. Requests are served
serially (single inference slot).

Endpoints:
  GET  /health   -> service/model status
  POST /change   -> multipart: before_image, after_image
"""

import os
import tempfile
import threading
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from model_provider import get_provider as _get_model_provider, MODEL_VERSION

APP_DIR = os.path.dirname(os.path.abspath(__file__))


class ChangeRequest(BaseModel):
    pass


class HealthStatus(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_version: str


class ChangeResponse(BaseModel):
    model_version: str
    latency_ms: int
    mask_path: str | None = None


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
        suffix_before = os.path.splitext(before_image.filename or "")[1] or ".png"
        suffix_after = os.path.splitext(after_image.filename or "")[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix_before, delete=False, dir=APP_DIR) as tmp:
            tmp.write(before_image.file.read())
            before_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=suffix_after, delete=False, dir=APP_DIR) as tmp:
            tmp.write(after_image.file.read())
            after_path = tmp.name

        with _change_lock:
            result = get_provider().detect(before_path, after_path)

        mask_path = None
        try:
            mask_out = os.path.join(APP_DIR, f"change_mask_{result['latency_ms']}.png")
            from PIL import Image as PILImage
            pil_mask = PILImage.fromarray(result["mask"] * 255, mode="L")
            pil_mask.save(mask_out)
            mask_path = mask_out
        except Exception:
            pass

        return ChangeResponse(
            model_version=result["model_version"],
            latency_ms=result["latency_ms"],
            mask_path=mask_path,
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
