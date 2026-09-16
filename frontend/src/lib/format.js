export const fmtConfidence = (v) => (v === null || v === undefined ? null : v.toFixed(3));

// Matches backend/app/services/report.py's report_id exactly (both derive
// from the real query_id) so the ID shown here is the same one that
// appears on the downloadable PDF/JSON report — not a client-side guess.
export function reportId(result) {
  const queryId = result?.query_id;
  if (!queryId) return "Not yet generated";
  return `SQ-${queryId.slice(0, 8).toUpperCase()}`;
}

// "QUERY_RECEIVED" -> "Query received"
export function fmtStepLabel(step) {
  if (!step) return "";
  const words = step.toLowerCase().split("_");
  return words[0].charAt(0).toUpperCase() + words[0].slice(1) + (words.length > 1 ? " " + words.slice(1).join(" ") : "");
}

export function fmtSpecialistId(id) {
  if (!id) return "";
  const [, ...rest] = id.split(".");
  return (rest.join(".") || id).replace(/_/g, " ");
}

const NOT_AVAILABLE = "Not available";

export function fmtDimensions(meta) {
  if (!meta || meta.width == null || meta.height == null) return NOT_AVAILABLE;
  return `${meta.width} × ${meta.height} px`;
}

export function fmtBands(meta) {
  if (!meta || meta.band_count == null) return NOT_AVAILABLE;
  return `${meta.band_count}${meta.dtype ? ` (${meta.dtype})` : ""}`;
}

export function fmtCrs(meta) {
  return meta?.crs || NOT_AVAILABLE;
}

export function fmtResolution(meta) {
  if (!meta || !meta.resolution) return NOT_AVAILABLE;
  const [x, y] = meta.resolution;
  return x === y ? `${x} m` : `${x} × ${y} m`;
}

export function fmtExtent(meta) {
  if (!meta || !meta.bounds_wgs84) return NOT_AVAILABLE;
  const [minx, miny, maxx, maxy] = meta.bounds_wgs84;
  return `${minx.toFixed(3)}, ${miny.toFixed(3)} – ${maxx.toFixed(3)}, ${maxy.toFixed(3)}`;
}

export function fmtAcquisitionDate(meta) {
  if (!meta || !meta.acquisition_date) return NOT_AVAILABLE;
  const sourceLabel =
    { user: "user-entered", file_metadata: "from file", unknown: "unknown source" }[
      meta.acquisition_date_source
    ] || meta.acquisition_date_source;
  return `${meta.acquisition_date} (${sourceLabel})`;
}

export function fmtGeoreferenced(meta) {
  if (!meta) return NOT_AVAILABLE;
  return meta.is_georeferenced ? "Yes" : "No";
}

export function fmtFormat(meta) {
  return meta?.format || NOT_AVAILABLE;
}

export function pairSummary(images) {
  if (images.length !== 2) return null;
  const [a, b] = images;
  const metaA = a.metadata;
  const metaB = b.metadata;
  const sameModality = a.modality === b.modality;
  const datesKnown = !!metaA?.acquisition_date && !!metaB?.acquisition_date;
  const datesDiffer = datesKnown ? metaA.acquisition_date !== metaB.acquisition_date : null;
  const bothGeoreferenced = !!metaA?.is_georeferenced && !!metaB?.is_georeferenced;

  return {
    sameModality,
    datesKnown,
    datesDiffer,
    bothGeoreferenced,
  };
}

export function describeTarget(images) {
  if (!images.length) return "";
  const optical = images.filter((i) => i.modality === "OPTICAL").length;
  const sar = images.filter((i) => i.modality === "SAR").length;
  if (optical && sar) return "OPTICAL + SAR pair";
  const mod = optical ? "optical" : "SAR";
  const plural = images.length > 1 ? `${mod} images` : `${mod} image`;
  return `${images.length} ${plural}`;
}
