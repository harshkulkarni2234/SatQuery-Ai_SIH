import { IconCheck, IconArrow } from "../icons/index.jsx";
import { SPECIALISTS } from "../../constants/specialists.js";

export default function AnalysisReady({ specialistId, onView }) {
  const spec = SPECIALISTS.find((s) => s.id === specialistId) || SPECIALISTS[0];
  const ReadyIcon = spec.icon;
  return (
    <div className="ready-screen fade-in">
      <div className="ready-check">
        <IconCheck size={30} />
      </div>
      <h2 className="ready-title">Analysis Ready</h2>
      <div className="ready-specialist">
        <ReadyIcon size={16} /> {spec.short} selected
      </div>
      <p className="ready-text">Your satellite imagery has been analyzed.</p>
      <button className="btn-view" onClick={onView}>
        View Analysis <IconArrow size={15} />
      </button>
    </div>
  );
}
