import { IconSatellite } from "../icons/index.jsx";
import { SPECIALISTS } from "../../constants/specialists.js";
import { reportId, describeTarget } from "../../lib/format.js";
import ResultContent from "./ResultContent.jsx";
import TechnicalDetails from "./TechnicalDetails.jsx";

export default function AnalysisResult({ result, images, onBack }) {
  const task = result?.task_classified;
  const spec = SPECIALISTS.find((s) => s.id === task) || {
    label: task || "Unclassified",
    short: task,
    icon: IconSatellite,
  };
  const SpecIcon = spec.icon;
  const today = new Date().toLocaleDateString("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
  return (
    <div className="result-screen fade-in">
      <div className="result-topbar">
        <button className="btn-back" onClick={onBack}>
          {"←"} New Analysis
        </button>
        <span className="result-topbar-label">Intelligence Report</span>
      </div>
      <div className="result-card">
        <div className="report-letterhead">
          <div className="result-header">
            <div className="result-header-icon">
              <SpecIcon size={20} />
            </div>
            <div>
              <h2 className="result-header-title">{spec.label}</h2>
              <span className="result-header-sub">{describeTarget(images)}</span>
            </div>
          </div>
          <div className="report-meta-grid">
            <div className="report-meta-item">
              <span className="meta-key">Report</span>
              <span className="meta-val">{reportId(result)}</span>
            </div>
            <div className="report-meta-item">
              <span className="meta-key">Date</span>
              <span className="meta-val">{today}</span>
            </div>
            <div className="report-meta-item">
              <span className="meta-key">Specialist</span>
              <span className="meta-val">{spec.label}</span>
            </div>
            <div className="report-meta-item">
              <span className="meta-key">Classification</span>
              <span className="meta-val meta-class">UNCLASSIFIED</span>
            </div>
          </div>
        </div>
        <ResultContent result={result} images={images} />
      </div>
      <TechnicalDetails result={result} />
    </div>
  );
}
