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

// Display names shown in the winner reveal / "Model selected" transcript.
export const WINNER_NAMES = {
  VQA: "Visual QA",
  GROUNDING: "Spatial Grounding",
  CHANGE_DETECTION: "Change Detection",
  CROSS_MODAL: "Cross-Modal Fusion",
};

// Grounding/VQA visual-evidence label prefixes, keyed by detected land-cover target.
export const LABEL_PREFIX = {
  water: "Water Body",
  vegetation: "Vegetation",
  "built-up": "Building",
  roads: "Road",
  farmland: "Farmland",
};
