// The guided walkthrough. Every selector points at a real element in the
// running UI, so the tour describes the actual system rather than a mock.
export const TOUR_STEPS = [
  {
    sel: ".image-grid",
    title: "Real imagery, really uploaded",
    body:
      "These scenes went through the same upload endpoint you would use yourself. Nothing here is pre-baked.",
  },
  {
    sel: ".image-metadata",
    title: "Metadata read from the file",
    body:
      "Coordinate system, resolution and ground extent come straight out of the file header. That is what lets the system work in real metres later.",
  },
  {
    sel: ".query-input",
    title: "Ask in plain English",
    body:
      "No tool to choose and no parameters to set. The question alone decides which analysis runs.",
  },
  {
    sel: ".btn-analyze",
    title: "Run it",
    body:
      "Click Analyze. The agent classifies the question, checks the scenes can be compared, picks a specialist and runs it.",
    nextLabel: "Next",
  },
  {
    sel: ".trace-replay",
    wait: true,
    title: "Every step is recorded",
    body:
      "This is the backend replaying its own log with real timings. It also shows which specialist was chosen and, if they differ, which one actually ran.",
  },
  {
    sel: ".answer-text",
    wait: true,
    title: "The answer",
    body:
      "Written from what the specialist measured. Where a number is not justifiable the system says so instead of inventing one.",
  },
  {
    sel: ".report-downloads",
    wait: true,
    title: "Take it with you",
    body:
      "The whole result exports as a PDF or JSON, including the trace. That is the audit trail an operational decision would need.",
  },
];
