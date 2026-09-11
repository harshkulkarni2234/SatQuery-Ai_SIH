# SatQuery VQA Worker

## 1. Purpose
A small local HTTP service that runs VQA (visual question answering) on satellite
images for the SatQuery backend. It keeps the heavy ML stack
(`torch` / `transformers` / `peft`) completely out of the FastAPI backend
environment, which stays dependency-light.

## 2. Architecture
```
React Frontend -> FastAPI Backend (backend/) --HTTP--> VQA Worker (this dir) -> SmolVLM
```
The backend never imports torch. Its `app/services/vqa.py` calls the worker's
`POST /vqa` endpoint and returns an honest graceful message if the worker is
unreachable.

## 3. Create the separate venv
The worker uses its OWN virtual environment — never the backend venv.
```powershell
cd ml/vqa-worker
python -m venv venv
venv\Scripts\activate
```

## 4. Install requirements
```powershell
# CUDA GPU workers (the tested build) first install the CUDA torch wheel:
pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
# then the rest (pinned to the versions verified in Phase 7C Stage 3):
pip install -r requirements.vqa-worker.txt
```
CPU-only machines: install the regular CPU `torch` wheel instead of the `+cu130`
build.

## 5. Start the worker
```powershell
.\run_worker.ps1            # port 8001
.\run_worker.ps1 -Port 8001
```
`run_worker.ps1` uses `$env:VQA_WORKER_PYTHON` if set, otherwise
`ml/vqa-worker/venv`. It honors `MODEL_DIR` / `ADAPTER_DIR` env vars
(see `.env.example`). The backend expects the worker at
`http://127.0.0.1:8001` (set `VQA_WORKER_URL` in `backend/.env`).

## 6. Worker endpoint
- `GET /health` — model/device/specialist status.
- `POST /vqa` — multipart form: `image` (file), `question` (string),
  optional `use_specialist` (bool). Response:
```json
{
  "answer_text": "...",
  "model_version": "SmolVLM-256M-Instruct",
  "specialist": false,
  "confidence_score": null,
  "execution_time_ms": 1234
}
```
`confidence_score` is always `null` — there is no calibrated confidence method yet.

## 7. Base model
`HuggingFaceTB/SmolVLM-256M-Instruct` (Idefics3, 256M). This is the DEFAULT and
general VQA provider used for every query unless specialist mode is explicitly
selected for a supported narrow question. To avoid re-downloading, point
`MODEL_DIR` at the tested local snapshot:
`C:\Users\ihars\.cache\huggingface\hub\models--HuggingFaceTB--SmolVLM-256M-Instruct\snapshots\7e3e67edbbed1bf9888184d9df282b700a323964`

## 8. Experimental adapter
`ml/smolvlm/lora_stage3/` holds the Phase 7C **Stage 3 BigEarthNet-adapted LoRA**
(a durable copy of the training output, `r=8`, decoder-only). It attaches to the
same loaded base model in-place — no duplicated model copy — and the provider
toggles it with peft's `disable_adapter()` context manager.

## 9. Why the adapter is narrow / experimental
The Stage 3 LoRA was trained on a small (~64-example) BigEarthNet slice and is
**not** generally better than the base model. In controlled held-out evaluation
it improved some presence/count categories while captions regressed (became
empty/terse). Specialist mode is therefore selected only for obvious
presence/existence and simple counting questions; everything else uses the base
model, and any specialist failure falls back to the base model. It is adapted
BigEarthNet satellite data only — do not treat it as a general model.

## 10. GPU requirement / limitations
- Needs a CUDA GPU for reasonable latency (fp16, ~4 GB VRAM with the T600 used in
  development). On CPU it technically runs but is far slower.
- Requests are processed **serially** (single inference slot) — no parallelism.
- No retraining, no downloads at request time, and this service will NOT fetch
  GeoChat or any other model.