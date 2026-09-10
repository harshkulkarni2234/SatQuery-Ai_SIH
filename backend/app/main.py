from fastapi import FastAPI

app = FastAPI(title="SatQuery AI", version="0.1.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}
