"""Local VQA worker HTTP service.

Serves Base SmolVLM-256M-Instruct (and optionally the experimental
BigEarthNet Stage 3 LoRA specialist) over HTTP so the FastAPI backend
stays torch-free. Requests are served serially (single inference slot).

Endpoints:
  GET  /health   -> service/model status
  POST /vqa      -> multipart: image + question [+ use_specialist]
"""

import os
import tempfile

import threading
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from model_provider import SmolVLMProvider, should_use_specialist

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = "HuggingFaceTB/SmolVLM-256M-Instruct"


class VQARequest(BaseModel):
    question: str


class HealthStatus(BaseModel):
    status: str
    device: str
    model_loaded: bool
    specialist_available: bool
    specialist_errors: list


app = FastAPI(title="SatQuery VQA Worker")
_vqa_lock = threading.Lock()
_provider = None


def get_provider():
    global _provider
    if _provider is None:
        adapter_dir = os.getenv(
            "ADAPTER_DIR", os.path.join(APP_DIR, "..", "smolvlm", "lora_stage3_v2")
        )
        _provider = SmolVLMProvider(
            model_dir=os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
            adapter_dir=adapter_dir,
        )
        _provider.load()
    return _provider


@app.on_event("startup")
def _startup():
    try:
        get_provider()
    except Exception as exc:  # service must report why it failed rather than half-start
        raise RuntimeError(f"Failed to load VQA model: {exc}") from exc


@app.get("/health", response_model=HealthStatus)
def health():
    p = _provider
    if p is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthStatus(
        status="ok",
        device=p.device,
        model_loaded=True,
        specialist_available=p.specialist_available,
        specialist_errors=p.specialist_errors,
    )


@app.post("/vqa")
def vqa(image: UploadFile = File(...), question: str = Form(...), use_specialist: bool = Form(False)):
    if not (question or "").strip():
        raise HTTPException(status_code=400, detail="question is required")

    use_specialist = bool(use_specialist) or should_use_specialist(question)

    suffix = os.path.splitext(image.filename or "")[1] or ".png"
    if not suffix.lower().startswith("."):
        suffix = ".png"
    image_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=APP_DIR) as tmp:
            tmp.write(image.file.read())
            image_path = tmp.name
        with _vqa_lock:
            result = _provider.answer_question(image_path, question, use_specialist=use_specialist)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VQA inference failed: {exc}") from exc
    finally:
        if image_path and os.path.isfile(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

    return result


def main():
    port = int(os.getenv("VQA_WORKER_PORT", "8001"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()