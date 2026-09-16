import { IconSatellite } from "../icons/index.jsx";

// Deliberately neutral: the backend hasn't responded yet, so nothing here
// may claim or imply that any specific specialist is running, has been
// tried, or has been ruled out. Once the real response arrives, the
// TraceReplay screen takes over and shows exactly what happened, from the
// backend's own recorded trace.
export default function AgentSelection() {
  return (
    <div className="agent-screen fade-in">
      <div className="waiting-panel">
        <div className="waiting-icon">
          <span className="spinner-accent spinner-large" />
          <IconSatellite size={22} />
        </div>
        <div className="waiting-title">Analyzing your query{"…"}</div>
        <p className="waiting-sub">
          Classifying the request and checking the uploaded imagery.
        </p>
      </div>
    </div>
  );
}
