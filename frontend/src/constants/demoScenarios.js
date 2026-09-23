// Phase C9: real demo files (Phase B1) served by the backend at
// /demo-assets/... (mounted read-only from the repo-root data/ directory,
// see backend/app/main.py). Dates/paths mirror data/manifest.json exactly —
// keep in sync if that manifest changes.
export const DEMO_SCENARIOS = [
  {
    id: "scenario_A_single",
    label: "Single image",
    description: "One optical scene of a city on a river.",
    files: [
      {
        url: "/demo-assets/demo/scenario_A_single/single_image.jpg",
        filename: "single_image.jpg",
        modality: "OPTICAL",
        captureDate: null,
      },
    ],
    queries: [
      "What can you tell me about this image?",
      // The grounding specialist is HSV-based. On this scene "water" honestly
      // returns no regions (the river is grey-brown, not blue), and
      // farmland/vegetation each collapse to a single full-frame box. Built-up
      // is the only target here that yields real, localised regions.
      // See docs/DEMO_RUNBOOK.md.
      "Show me the built-up area.",
    ],
  },
  {
    id: "scenario_B_temporal",
    label: "Temporal pair (Lake Mead / Hoover Dam)",
    description: "Two real Sentinel-2 scenes of Lake Mead, four years apart.",
    files: [
      {
        url: "/demo-assets/demo/scenario_B_temporal/before.tif",
        filename: "before.tif",
        modality: "OPTICAL",
        captureDate: "2018-08-24",
      },
      {
        url: "/demo-assets/demo/scenario_B_temporal/after.tif",
        filename: "after.tif",
        modality: "OPTICAL",
        captureDate: "2022-08-23",
      },
    ],
    queries: ["What changed between these two dates?"],
  },
  {
    id: "scenario_C_optical_sar",
    label: "Optical + SAR pair",
    description: "An optical scene and a radar scene of the same ground.",
    files: [
      {
        url: "/demo-assets/demo/scenario_C_optical_sar/optical.png",
        filename: "optical.png",
        modality: "OPTICAL",
        captureDate: "2017-08-08",
      },
      {
        url: "/demo-assets/demo/scenario_C_optical_sar/sar.png",
        filename: "sar.png",
        modality: "SAR",
        captureDate: "2017-08-08",
      },
    ],
    queries: [
      "Use the optical and SAR images together to identify built-up and water-covered regions.",
    ],
  },
];

// Flattened one-card-per-query view for the picker UI: scenario_A_single
// has 2 preset queries over the same file, so it renders as 2 cards
// sharing one `files` list rather than 1 card the user has to edit.
// A short, human title per card. Keyed by scenario id + query index so the
// single-image scenario can show its two very different questions separately.
const CARD_TITLES = {
  "scenario_A_single-0": "Describe a scene",
  "scenario_A_single-1": "Find land cover",
  "scenario_B_temporal-0": "Detect change over time",
  "scenario_C_optical_sar-0": "Combine optical and radar",
};

export const DEMO_SCENARIO_ITEMS = DEMO_SCENARIOS.flatMap((scenario) =>
  scenario.queries.map((query, i) => {
    const key = `${scenario.id}-${i}`;
    return {
      key,
      scenarioId: scenario.id,
      label: CARD_TITLES[key] || scenario.label,
      description: scenario.description,
      query,
      files: scenario.files,
    };
  })
);
