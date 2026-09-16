// `combo` tags which uploaded-image combination each example fits, so the
// input screen can show only the examples that would actually route
// somewhere (Phase C4) instead of a fixed list regardless of what's staged.
export const SCENARIOS = [
  {
    label: "Visual QA",
    description: "Describe or ask questions about a single image.",
    query: "What can you tell me about this image?",
    combo: "single",
  },
  {
    label: "Grounding",
    description: "Locate land-cover objects such as water or buildings.",
    query: "Where is the water located?",
    combo: "single",
  },
  {
    label: "Change Detection",
    description: "Compare two images captured at different dates.",
    query: "Show me the changes between these two images.",
    combo: "pair-same-modality",
  },
  {
    label: "Cross-Modal",
    description: "Analyze optical and SAR imagery together.",
    query: "Compare the optical and SAR images.",
    combo: "pair-optical-sar",
  },
];

// Which combo the current staged images match, or null before any upload.
export function imageCombo(images) {
  if (images.length === 0) return null;
  if (images.length === 1) return "single";
  const modalities = new Set(images.map((i) => i.modality));
  return modalities.size === 2 ? "pair-optical-sar" : "pair-same-modality";
}
