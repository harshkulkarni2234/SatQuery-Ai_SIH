const STATUS_LABEL = { PASS: "Pass", WARN: "Warning", FAIL: "Fail", SKIPPED: "Skipped" };

const COREGISTRATION_LABEL = {
  verified: "Verified",
  assumed: "Assumed (unverified)",
  unverified: "Unverified",
  failed: "Failed",
};

export default function CompatibilityChecklist({ report }) {
  if (!report) return null;
  return (
    <div className="compat-checklist">
      <div className="compat-summary">
        {report.overlap_ratio != null && (
          <div className="meta-item">
            <span className="meta-key">Overlap ratio</span>
            <span className="meta-val">{report.overlap_ratio.toFixed(2)}</span>
          </div>
        )}
        {report.coregistration && (
          <div className="meta-item">
            <span className="meta-key">Co-registration</span>
            <span className="meta-val">{COREGISTRATION_LABEL[report.coregistration] || report.coregistration}</span>
          </div>
        )}
      </div>
      <ul className="compat-list">
        {report.checks.map((c, i) => (
          <li className={`compat-item compat-${c.status.toLowerCase()}`} key={i}>
            <span className="compat-status-dot" aria-hidden="true" />
            <div>
              <div className="compat-item-top">
                <span className="compat-name">{c.name.replace(/_/g, " ")}</span>
                <span className="compat-status-label">{STATUS_LABEL[c.status] || c.status}</span>
              </div>
              <div className="compat-detail">{c.detail}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
