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
The backend never imports torch. When the worker is reachable and the
image pair is inside the model's training domain, the backend's
`change.siamese_binary_cnn` specialist (`backend/app/services/change_specialists.py`)
calls `POST /change` and stores the returned mask itself. Otherwise — worker
down, invalid response, size mismatch, coarse imagery, non-corresponding
pair — it runs the deterministic pixel-difference method and reports that
fallback in the result.

**Scope:** the model is a *binary* change/no-change segmenter. It says where
pixels changed, not what they changed to. See `ml/change_model/MODEL_CARD.md`.

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
No CUDA GPU: install the regular CPU `torch` wheel — the worker runs fine
(this model is small).

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
  "mask_png_base64": "<base64 PNG>",
  "mask_width": 256,
  "mask_height": 256
}
```
The mask is a single-channel PNG at the model's 256x256 working resolution,
values 0 (unchanged) / 255 (changed). The worker writes no files: uploaded
images go to the OS temp dir and are deleted after the request; the backend
saves the mask under its own served `/masks` directory.

## 7. Model
The model is a small Siamese CNN encoder + feature difference/fusion
module + decoder, trained on the SECOND-derived dataset
(ml/change_model/trained/data/). Weights are loaded from
`ml/change_model/trained/best_change_model.pt` (trained via
`ml/change_model/train_change.py` with fp16 mixed precision on the
RTX 2050). Model version: `siamese-cnn-v1`.

## 8. GPU requirement / limitations
- A GPU is optional. `model_provider.py` auto-detects CUDA (fp16) and
  otherwise uses CPU (fp32); the model is small, and one 256×256 pair took
  roughly 70–120 ms on an Apple-silicon CPU (one warm sample, not a
  benchmark). There is no MPS path.
- Requests are processed **serially** (single inference slot) — no
  parallelism.
- No retraining at request time, and this service will NOT fetch any
  external model.
