from fastapi import FastAPI

from app.routes.images import router as images_router
from app.routes.query import router as query_router

app = FastAPI(title="SatQuery AI", version="0.1.0")

app.include_router(images_router)
app.include_router(query_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
