import { fmtConfidence } from "../../lib/format.js";
import { LABEL_PREFIX } from "../../constants/specialists.js";
import ConfidenceMeta from "../common/ConfidenceMeta.jsx";
import GroundingImage from "./GroundingImage.jsx";

export default function GroundingResult({ result, images }) {
  const boxes = result.bounding_boxes || [];
  const meta = result.metadata || {};
  const regions = meta.regions || [];
  const target = meta.object_type || result.task_classified?.toLowerCase();
  const labelPrefix = LABEL_PREFIX[target] || "Region";
  const regionLabels = regions.map((r) => r.label || labelPrefix);
  const optical = images.find((i) => i.modality === "OPTICAL") || images[0];
  const confidence = fmtConfidence(result.confidence_score);
  return (
    <div className="result-block">
      {optical && boxes.length > 0 && (
        <GroundingImage src={optical.url} boxes={boxes} labels={regionLabels} labelPrefix={labelPrefix} />
      )}
      <div className="result-section">
        <h3 className="result-section-title">Grounding result</h3>
        <div className="answer-text">{result.answer_text}</div>
      </div>
      {boxes.length > 0 && (
        <>
          <h4 className="sub-title">Detected regions</h4>
          <table className="box-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Label</th>
                <th>Area</th>
                <th>x min</th>
                <th>y min</th>
                <th>x max</th>
                <th>y max</th>
              </tr>
            </thead>
            <tbody>
              {boxes.map((b, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>{regionLabels[i] || `${labelPrefix} ${i + 1}`}</td>
                  <td>{regions[i]?.area_pct != null ? `${regions[i].area_pct}%` : "—"}</td>
                  {b.map((v, j) => (
                    <td key={j}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note evidence-note">
            Bounding boxes are deterministic visual detections from the grounding
            specialist. Confidence reflects the bbox fill ratio — a real,
            measured quantity.
          </p>
        </>
      )}
      <div className="result-meta">
        <ConfidenceMeta value={confidence} />
      </div>
    </div>
  );
}
