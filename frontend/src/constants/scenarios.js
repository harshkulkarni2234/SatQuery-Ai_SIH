// `combo` tags which uploaded-image combination each example fits, so the
// input screen can show only the examples that would actually route
// somewhere (Phase C4) instead of a fixed list regardless of what's staged.
export const SCENARIOS = [
  {
    label: "Visual QA",
    description: "Ask what is in one scene.",
    query: "What can you tell me about this image?",
    combo: "single",
  },
  {
    label: "Grounding",
    description: "Find water, vegetation or built-up areas.",
    query: "Where is the water located?",
    combo: "single",
  },
  {
    label: "Change Detection",
    description: "Compare two dates of the same place.",
    query: "Show me the changes between these two images.",
    combo: "pair-same-modality",
  },
  {
    label: "Cross-Modal",
    description: "Read optical and radar together.",
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
