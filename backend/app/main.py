import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.database import SessionLocal
from app.routes.images import router as images_router
from app.routes.query import router as query_router
from app.routes.report import router as report_router
from app.routes.specialists import router as specialists_router
from app.services.change_detection import MASK_DIR
from app.services.registry import _semantic_change_available, _vqa_worker_available

os.makedirs(MASK_DIR, exist_ok=True)

# Phase C9: the repo-root data/ directory (demo scenarios + manifest.json)
# is static, source-controlled, non-sensitive content — distinct from
# backend/data/ (uploaded_images/change_masks, real user data, never
# exposed this way). Served read-only so the frontend's "Load demo
# scenario" feature can fetch the real demo files and upload them through
# the normal POST /images/upload path, same as any user-selected file.
DEMO_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

app = FastAPI(title="SatQuery AI", version="0.1.0")

app.include_router(images_router)
app.include_router(query_router)
app.include_router(specialists_router)
app.include_router(report_router)

app.mount("/masks", StaticFiles(directory=MASK_DIR), name="masks")
if os.path.isdir(DEMO_DATA_DIR):
    app.mount("/demo-assets", StaticFiles(directory=DEMO_DATA_DIR), name="demo-assets")


# Phase A8: catch-all so an unhandled exception (e.g. DB down mid-request)
# returns a clean JSON error instead of a framework-default page, and never
# leaks the exception message/traceback to the client — full detail still
# goes to the server log for debugging.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _db_available() -> bool:
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:
        return False


@app.get("/health")
def health_check():
    db_ok = _db_available()
    vqa_ok = _vqa_worker_available()
    semantic_change_ok = _semantic_change_available()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "up" if db_ok else "down",
        "vqa_worker": "up" if vqa_ok else "down",
        "semantic_change_model": "available" if semantic_change_ok else "unavailable",
    }