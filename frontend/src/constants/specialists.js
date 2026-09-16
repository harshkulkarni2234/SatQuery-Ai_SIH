import {
  IconEye,
  IconPin,
  IconCompare,
  IconLayers,
} from "../components/icons/index.jsx";

export const SPECIALISTS = [
  {
    id: "VQA",
    label: "Visual Question Answering",
    short: "Visual QA",
    desc: "Interpret and describe satellite imagery",
    icon: IconEye,
  },
  {
    id: "GROUNDING",
    label: "Land Cover / Object Grounding",
    short: "Grounding",
    desc: "Locate and map land-cover features",
    icon: IconPin,
  },
  {
    id: "CHANGE_DETECTION",
    label: "Change Detection",
    short: "Change Detection",
    desc: "Identify differences between images",
    icon: IconCompare,
  },
  {
    id: "CROSS_MODAL",
    label: "Optical + SAR Analysis",
    short: "Cross-Modal",
    desc: "Fuse multi-sensor data for analysis",
    icon: IconLayers,
  },
];

// Ring traversal order (single source of truth). The orbit dot rotates ONCE
// through exactly these specialist ids, one per SPECIALIST_DWELL_MS. Each
// specialist is assigned the ring position equal to its index in this sequence
// so the dot's single continuous rotation lands on every one in this order.
export const ORBIT_SEQUENCE = ["VQA", "CHANGE_DETECTION", "GROUNDING", "CROSS_MODAL"];
export const ORBIT_POS = Object.fromEntries(
  ORBIT_SEQUENCE.map((id, i) => [id, i])
);

// Display names shown in the winner reveal / "Model selected" transcript.
export const WINNER_NAMES = {
  VQA: "Visual QA",
  GROUNDING: "Spatial Grounding",
  CHANGE_DETECTION: "Change Detection",
  CROSS_MODAL: "Cross-Modal Fusion",
};

export const STEP_LABELS = [
  "Understanding query…",
  "Checking input…",
  "Evaluating specialists…",
  "Selecting best specialist…",
];

// Agent-selection traversal clock. Every specialist is visited once for a
// fixed dwell before the dot moves on; the total run is 4 × 1s = 4s and is
// intentionally independent of backend inference latency.
export const SPECIALIST_DWELL_MS = 1000;

// Grounding/VQA visual-evidence label prefixes, keyed by detected land-cover target.
export const LABEL_PREFIX = {
  water: "Water Body",
  vegetation: "Vegetation",
  "built-up": "Building",
  roads: "Road",
  farmland: "Farmland",
};
