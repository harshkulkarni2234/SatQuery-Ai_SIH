import { useState } from "react";
import { IconChevron } from "../icons/index.jsx";

export default function TechnicalDetails({ result }) {
  const [open, setOpen] = useState(false);
  if (!result) return null;
  const trace = result.execution_trace || {};
  const meta = result.metadata || {};
  const task = result.task_classified;
  const toolName =
    task === "CROSS_MODAL"
      ? "optical_sar_analysis"
      : task === "VQA"
        ? meta.specialist_mode || "vqa"
        : task === "GROUNDING"
          ? "grounding"
          : task === "CHANGE_DETECTION"
            ? "change_detection"
            : undefined;
  const rows = [
    { label: "Task", value: task },
    { label: "Tool", value: toolName },
    { label: "Model", value: meta.model_version || trace.model_version },
    {
      label: "Execution time",
      value: trace.execution_time_ms != null ? `${trace.execution_time_ms} ms` : undefined,
    },
    {
      label: "Input modalities",
      value: trace.modalities_detected?.length
        ? trace.modalities_detected.join(", ")
        : undefined,
    },
    { label: "Router reason", value: trace.reason },
  ].filter((r) => r.value);
  if (!rows.length) return null;
  return (
    <div className="tech-details">
      <button className="tech-toggle" onClick={() => setOpen(!open)}>
        <span className={`tech-chevron${open ? " open" : ""}`}>
          <IconChevron size={14} />
        </span>
        Technical details
      </button>
      {open && (
        <div className="tech-body">
          {rows.map((r) => (
            <div className="tech-row" key={r.label}>
              <span className="tech-key">{r.label}</span>
              <span className="tech-val">{r.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
