// Thin API client for the EXISTING SatQuery backend (schema-driven, no invented fields).

import { USE_MOCKS, mockUploadResponse } from "./mocks/metadata.js";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function uploadImage(file, modality, captureDate) {
  if (USE_MOCKS) {
    await new Promise((r) => setTimeout(r, 300));
    return mockUploadResponse(file, modality);
  }
  const form = new FormData();
  form.append("file", file);
  form.append("modality", modality);
  if (captureDate) form.append("capture_date", captureDate);
  const res = await fetch("/images/upload", { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(extractMessage(data, res), res.status, data.detail);
  }
  return data; // { image_id, filename, modality, crs, resolution_m, metadata: RasterMetadata|null }
}

async function getImage(imageId) {
  const res = await fetch(`/images/${imageId}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(extractMessage(data, res), res.status, data.detail);
  }
  return data; // ImageDetailResponse
}

async function runQuery(queryText, imageIds) {
  const res = await fetch("/query", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ query_text: queryText, image_ids: imageIds }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(extractMessage(data, res), res.status, data.detail);
  }
  return data; // QueryResponse
}

async function getSpecialists() {
  const res = await fetch("/specialists");
  const data = await res.json().catch(() => []);
  if (!res.ok) {
    throw new ApiError(extractMessage(data, res), res.status, data.detail);
  }
  return data; // SpecialistSpec[]
}

// Report PDF/JSON are plain GETs the browser can download directly — no
// fetch wrapper needed, just the URL (used as an <a href>).
function reportPdfUrl(queryId) {
  return `/query/${queryId}/report.pdf`;
}
function reportJsonUrl(queryId) {
  return `/query/${queryId}/report.json`;
}

// The backend's error `detail` is either a plain string, a FastAPI/pydantic
// validation array, or a structured {message, suggestion, compatibility?}
// object (services/planner.py + services/compatibility.py). extractMessage
// always returns a short human string for a plain banner; ApiError also
// keeps the raw `detail` so callers that want the structured form (the
// compatibility checklist, the suggestion) can render it properly.
function extractMessage(data, res) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  if (detail && typeof detail === "object" && typeof detail.message === "string") {
    return detail.message;
  }
  return `Request failed with status ${res.status}`;
}

class ApiError extends Error {
  constructor(message, status = 0, detail = null) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

export { uploadImage, getImage, runQuery, getSpecialists, reportPdfUrl, reportJsonUrl, ApiError };