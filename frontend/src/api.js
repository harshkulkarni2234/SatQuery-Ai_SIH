// Thin API client for the EXISTING SatQuery backend (schema-driven, no invented fields).

const JSON_HEADERS = { "Content-Type": "application/json" };

async function uploadImage(file, modality) {
  const form = new FormData();
  form.append("file", file);
  form.append("modality", modality);
  const res = await fetch("/images/upload", { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(extractDetail(data, res), res.status);
  }
  return data; // { image_id, filename, modality, crs, resolution_m }
}

async function runQuery(queryText, imageIds) {
  const res = await fetch("/query", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ query_text: queryText, image_ids: imageIds }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(extractDetail(data, res), res.status);
  }
  return data; // QueryResponse
}

function extractDetail(data, res) {
  if (data && typeof data.detail === "string") return data.detail;
  if (Array.isArray(data && data.detail)) {
    return data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return `Request failed with status ${res.status}`;
}

class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

export { uploadImage, runQuery, ApiError };