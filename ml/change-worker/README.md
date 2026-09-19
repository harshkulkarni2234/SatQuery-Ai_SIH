# SatQuery Change Detection Worker

## 1. Purpose
A small local HTTP service that runs the trained Siamese CNN change
detection model on satellite image pairs for the SatQuery backend.
It keeps the heavy ML stack (`torch` / `numpy` / `torchvision`)
completely out of the FastAPI backend environment, which stays
dependency-light.

## 2. Architecture
```
React Frontend -> FastAPI Backend (backend/) --HTTP--> Change Worker (this dir) -> Siamese CNN
```
The backend never imports torch. Its `app/services/change_detection.py`
falls back to the deterministic pixel-diff approach if the worker is
unreachable; when the worker IS reachable, the backend calls
`POST /change` to get a learned change mask.

## 3. Create the separate venv
The worker uses its OWN virtual environment — never the backend venv.
```powershell
cd ml/change-worker
python -m venv venv
venv\Scripts\activate
```

## 4. Install requirements
```bash
# CUDA GPU (the original tested build) first installs the CUDA torch wheel:
pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
# then the rest (pinned to the versions verified in testing):
pip install -r requirements.change-worker.txt
```
No GPU at all (neither CUDA): install the regular CPU `torch` wheel —
the worker still runs, just far slower.

## 5. Start the worker
```powershell
.\run_worker.ps1            # port 8002
.\run_worker.ps1 -Port 8002
```
`run_worker.ps1` uses `$env:CHANGE_WORKER_PYTHON` if set, otherwise
`ml/change-worker/venv`. It honors `MODEL_PATH` env var (see
`.env.example`). The backend expects the worker at
`http://127.0.0.1:8002` (set `CHANGE_WORKER_URL` in `backend/.env`).

## 6. Worker endpoints
- `GET /health` — device/model/model_version status. Response:
```json
{
  "status": "ok",
  "device": "cuda",
  "model_loaded": true,
  "model_version": "siamese-cnn-v1"
}
```
- `POST /change` — multipart form: `before_image` (file), `after_image`
  (file). Response:
```json
{
  "model_version": "siamese-cnn-v1",
  "latency_ms": 1234,
  "mask_path": "change_mask_1234.png"
}
```
`mask_path` points to the saved binary change mask PNG (256x256, uint8
0/1). The mask is also returned as the `mask` field internally; the
`mask_path` is the on-disk location for the backend to serve.

## 7. Model
The model is a small Siamese CNN encoder + feature difference/fusion
module + decoder, trained on the SECOND-derived dataset
(ml/change_model/trained/data/). Weights are loaded from
`ml/change_model/trained/best_change_model.pt` (trained via
`ml/change_model/train_change.py` with fp16 mixed precision on the
RTX 2050). Model version: `siamese-cnn-v1`.

## 8. GPU requirement / limitations
- Needs a real GPU for reasonable latency: a CUDA GPU (fp16) or
  Apple Silicon via PyTorch's MPS backend. `model_provider.py`
  auto-detects CUDA, then falls back to CPU. On CPU it technically
  runs but is far slower.
- Requests are processed **serially** (single inference slot) — no
  parallelism.
- No retraining at request time, and this service will NOT fetch any
  external model.
