import CompatibilityChecklist from "./CompatibilityChecklist.jsx";

// Renders the backend's real validation/compatibility failure — never a
// generic "something went wrong". `detail` is the raw error.detail from
// ApiError: a plain string, or {message, suggestion, compatibility?} from
// services/planner.py / services/compatibility.py.
export default function ErrorPanel({ message, detail }) {
  const structured = detail && typeof detail === "object" ? detail : null;
  const text = structured?.message || message;
  const suggestion = structured?.suggestion;
  const compatibility = structured?.compatibility;

  return (
    <div className="error-panel">
      <div className="error-panel-message">{text}</div>
      {suggestion && <div className="error-panel-suggestion">{suggestion}</div>}
      {compatibility && <CompatibilityChecklist report={compatibility} />}
    </div>
  );
}
