# Contracts

Shared shapes between backend, frontend, and ML workers. The source of truth
for the Python types is [`backend/app/contracts.py`](../backend/app/contracts.py)
— this document explains each one with a JSON example and lists the planned
API additions so the frontend can be built against mocks before the backend
ships them. All additions to existing endpoints are **additive/optional
fields only** — nothing here removes or renames a field that currently exists
in `ImageUploadResponse` or `QueryResponse`.

## RasterMetadata

Produced by `raster_ingest.extract_metadata()` (Phase A1), persisted on
`Image` (Phase A2).

```json
{
  "format": "GTiff",
  "width": 512,
  "height": 512,
  "band_count": 4,
  "dtype": "uint16",
  "crs": "EPSG:32643",
  "bounds": [500000.0, 3100000.0, 505120.0, 3105120.0],
  "bounds_wgs84": [76.90, 28.02, 76.95, 28.07],
  "resolution": [10.0, 10.0],
  "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 3105120.0],
  "nodata": 0.0,
  "acquisition_date": "2023-06-14",
  "acquisition_date_source": "file_metadata",
  "is_georeferenced": true,
  "file_size_bytes": 2097152,
  "warnings": []
}
```

Rules: `acquisition_date_source` is `"user"` only when the uploader supplied
`capture_date` explicitly (it always wins over file tags); `"file_metadata"`
when read from a TIFF tag; `"unknown"` — never guessed — otherwise. For
non-georeferenced formats (PNG/JPG/BMP), `crs`, `bounds`, `bounds_wgs84`,
`transform`, and `resolution` are `null` and `is_georeferenced` is `false`.

## CompatibilityReport

Produced by `services/compatibility.py` (Phase A3) before dispatching a
two-image task.

```json
{
  "ok": false,
  "checks": [
    {"name": "modality_match", "status": "PASS", "detail": "Both images are OPTICAL"},
    {"name": "date_distinct", "status": "PASS", "detail": "2021-03-01 vs 2023-06-14"},
    {"name": "overlap", "status": "FAIL", "detail": "Footprint overlap ratio 0.12 is below minimum 0.5"}
  ],
  "overlap_ratio": 0.12,
  "coregistration": null,
  "alignment_possible": false
}
```

On `ok: false` the API returns HTTP 422 with this report embedded — the
system never runs a comparison it can't justify.

## SpecialistResult

The canonical output shape every specialist (VQA, grounding, change
detection, cross-modal) should converge on (natively from Phase B7 onward;
via adapters in `services/registry_adapters.py` before that, Phase A4).

```json
{
  "answer": "Water is visible in the lower-left quadrant of the image.",
  "evidence": {
    "boxes": [[120, 340, 210, 410]],
    "labels": ["water"],
    "mask_path": null,
    "overlay_path": "results/abc123_overlay.png",
    "per_modality": null,
    "stats": {"changed_area_px": 4213}
  },
  "confidence": null,
  "confidence_source": "unavailable",
  "model_or_tool": "grounding.deterministic_cv",
  "model_version": "cv-grounding-v1",
  "used_fallback": false,
  "fallback_reason": null,
  "warnings": []
}
```

`confidence` is `null` unless there's a real, named quantity behind it — in
which case `confidence_source` says what that quantity is (e.g. `"box fill
ratio"`, `"agreement fraction between optical and SAR evidence"`). Never a
made-up score.

## SpecialistSpec (registry entry)

```json
{
  "id": "vqa.smolvlm_bigearthnet_lora_stage3",
  "task": "VQA",
  "name": "SmolVLM-256M + BigEarthNet LoRA (Stage 3)",
  "kind": "learned_model",
  "input_count": 1,
  "modalities": ["OPTICAL", "SAR"],
  "formats": ["tif", "tiff", "png", "jpg"],
  "required_metadata": [],
  "outputs": ["answer"],
  "confidence_available": false,
  "version": "smolvlm256m-ben-lora-s3-v1.0",
  "priority": 10,
  "is_fallback": false,
  "is_rs_adapted": true
}
```

## TraceEvent

Real, recorded execution steps (Phase A5 planner, Phase A6 persistence) —
never a simulated/animated sequence on the frontend.

```json
{
  "step": "SPECIALIST_SELECTED",
  "status": "COMPLETED",
  "detail": "Selected grounding.deterministic_cv (learned model unavailable: VQA worker down)",
  "data": {"spec_id": "grounding.deterministic_cv", "rejected": [{"id": "grounding.learned_v1", "reason": "not implemented"}]},
  "timestamp": "2026-09-16T10:15:32.104Z",
  "duration_ms": 3
}
```

## Planned API additions

All are additive/backward compatible.

| Endpoint | Addition | Phase |
|---|---|---|
| `POST /images/upload` | response gains `metadata: RasterMetadata \| null` | A2 |
| `GET /images/{id}` | new — returns image + metadata | A2 |
| `POST /query` | response gains `trace_events: TraceEvent[]`, `compatibility: CompatibilityReport \| null`, `confidence_source`, `warnings`, `used_fallback`, `metadata.plan` | A5/A6 |
| `GET /query/{id}` | returns persisted `trace_events`, `compatibility`, etc. | A6 |
| `GET /query/{id}/report.pdf` | new — PDF report (owned by Phase C) | C5 |
| `GET /query/{id}/report.json` | new — same content, machine-readable | C5 |
| `GET /specialists` | new — registry listing | A4 |

Frontend builds (Phase C2/C3/C4) against `frontend/src/mocks/` shaped exactly
like these examples, behind a `VITE_USE_MOCKS` flag, until each gate phase
merges.
