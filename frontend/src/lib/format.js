export const fmtConfidence = (v) => (v === null || v === undefined ? null : v.toFixed(3));

// Deterministic, stable report identifier rendered from the analysis output.
export function reportId(result) {
  const seed = `${result?.task_classified || ""}|${result?.answer_text || ""}`;
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return `SQ-2026-${String((Math.abs(h) % 9000) + 1000)}`;
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
