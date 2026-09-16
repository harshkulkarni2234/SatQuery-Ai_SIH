import { IconSatellite } from "../icons/index.jsx";
import { SPECIALISTS } from "../../constants/specialists.js";
import { reportId, describeTarget } from "../../lib/format.js";
import { useSpecialists, findSpecialist, specialistBadgeLabel } from "../../lib/useSpecialists.js";
import { reportPdfUrl, reportJsonUrl } from "../../api.js";
import ResultContent from "./ResultContent.jsx";
import TechnicalDetails from "./TechnicalDetails.jsx";
import WarningsBanner from "../common/WarningsBanner.jsx";

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

  const specialists = useSpecialists();
  const specialistId = result?.metadata?.specialist_id;
  const registrySpec = findSpecialist(specialists, specialistId);
  const badgeLabel = specialistBadgeLabel(registrySpec);

  const queryId = result?.query_id;

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
            {badgeLabel && <span className="specialist-badge">{badgeLabel}</span>}
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
          {queryId && (
            <div className="report-downloads">
              <a
                className="btn-download"
                href={reportPdfUrl(queryId)}
                target="_blank"
                rel="noreferrer"
              >
                Download report (PDF)
              </a>
              <a
                className="btn-download btn-download-secondary"
                href={reportJsonUrl(queryId)}
                target="_blank"
                rel="noreferrer"
              >
                Download JSON
              </a>
            </div>
          )}
        </div>
        <WarningsBanner warnings={result?.warnings} />
        <ResultContent result={result} images={images} />
      </div>
      <TechnicalDetails result={result} />
    </div>
  );
}
