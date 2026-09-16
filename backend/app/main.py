import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.images import router as images_router
from app.routes.query import router as query_router
from app.routes.specialists import router as specialists_router
from app.services.change_detection import MASK_DIR

os.makedirs(MASK_DIR, exist_ok=True)

app = FastAPI(title="SatQuery AI", version="0.1.0")

app.include_router(images_router)
app.include_router(query_router)
app.include_router(specialists_router)

app.mount("/masks", StaticFiles(directory=MASK_DIR), name="masks")


@app.get("/health")
def health_check():
    return {"status": "ok"}