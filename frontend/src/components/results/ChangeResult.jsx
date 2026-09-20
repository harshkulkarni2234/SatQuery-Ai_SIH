import { useState } from "react";
import { fmtConfidence } from "../../lib/format.js";
import { useSpecialists, findSpecialist } from "../../lib/useSpecialists.js";
import ConfidenceMeta from "../common/ConfidenceMeta.jsx";
import ViewerToggles from "../common/ViewerToggles.jsx";
import OverlayImage from "../common/OverlayImage.jsx";
import Stat from "../common/Stat.jsx";

const LEARNED_CHANGE_ID = "change.siamese_binary_cnn";

const ALIGNMENT_LABEL = {
  geographic_reprojection: "Geographic reprojection (real CRS/grid alignment)",
  phase_correlation: "Phase-correlation translation alignment",
  none: "None (images already matched, or alignment was not attempted)",
};

export default function ChangeResult({ result, images }) {
  const meta = result.metadata || {};
  const boxes = result.bounding_boxes || [];
  const regions = meta.regions || [];
  const [viewer, setViewer] = useState("overlay");
  const [before, after] =
    images.length >= 2 ? [images[0], images[1]] : [images[0], images[0]];
  const confidence = fmtConfidence(result.confidence_score);
  const unavailable = meta.validation_failed === true;

  const specialists = useSpecialists();
  const spec = findSpecialist(specialists, meta.specialist_id);
  // The specialist that actually EXECUTED (metadata.specialist_id), not the one
  // that was planned — a runtime fallback lands on the deterministic method.
  const learned = spec
    ? spec.kind === "learned_model"
    : meta.specialist_id === LEARNED_CHANGE_ID;
  const methodLabel = learned
    ? "Learned Siamese CNN · binary change / no-change"
    : "Pixel-level deterministic method";

  if (unavailable) {
    return (
      <div className="result-block">
        <div className="result-section">
          <h3 className="result-section-title">Analysis</h3>
          <div className="answer-text">{result.answer_text}</div>
        </div>
        <div className="notice-block">
          The change-detection pipeline refused a percentage here rather than
          fabricate one: reliable comparison requires the two images to cover
          the same geographic area. Upload a co-registered before/after pair.
        </div>
        <div className="result-meta">
          <ConfidenceMeta value={confidence} source={result.confidence_source} />
        </div>
      </div>
    );
  }

  return (
    <div className="result-block">
      <div className="result-section">
        <h3 className="result-section-title">Analysis</h3>
        <div className="answer-text">{result.answer_text}</div>
      </div>

      <div className="report-section">
        <h3 className="result-section-title">Spatial evidence</h3>
        <ViewerToggles
          label="Layers"
          value={viewer}
          onChange={setViewer}
          options={[
            ...(result.overlay_url
              ? [{ value: "overlay", label: "Change overlay" }]
              : []),
            // The learned model produces a mask only — no region boxes exist.
            ...(learned ? [] : [{ value: "annotated", label: "Changed regions" }]),
            { value: "plain", label: "Before / After" },
            { value: "mask", label: "Change mask" },
          ]}
        />
        {viewer === "mask" ? (
          <div className="mask-view">
            {result.change_mask_url && (
              <figure className="evidence-figure mask">
                <figcaption>Change mask</figcaption>
                <img src={result.change_mask_url} alt="Change mask" />
              </figure>
            )}
            <p className="mask-explainer">
              {learned
                ? "Highlighted pixels are where the learned model predicts change. The mask says where, not what: it does not identify the type of change."
                : "Highlighted pixels in the change mask represent detected differences according to the deterministic pixel-difference method."}
            </p>
          </div>
        ) : viewer === "overlay" && result.overlay_url ? (
          <figure className="evidence-figure">
            <figcaption>After image · changed pixels highlighted</figcaption>
            <img
              src={result.overlay_url}
              alt="Change overlay on the after image"
              className="evidence-img"
            />
          </figure>
        ) : (
          <div className="evidence-grid">
            <figure className="evidence-figure">
              <figcaption>Before</figcaption>
              <OverlayImage src={before.url} alt="Image before" boxes={[]} />
            </figure>
            <figure className="evidence-figure">
              <figcaption>After</figcaption>
              <OverlayImage
                src={after.url}
                alt="Image after"
                boxes={boxes}
                labelPrefix="Changed Region"
                showBoxes={
                  viewer === "annotated" ||
                  (viewer === "overlay" && !result.overlay_url)
                }
              />
            </figure>
          </div>
        )}
      </div>

      <div className="report-section">
        <h3 className="result-section-title">Change statistics</h3>
        <div className="method-badge">{methodLabel}</div>
        {meta.execution_note && <p className="note evidence-note">{meta.execution_note}</p>}
        <div className="stat-grid">
          <Stat
            label="Changed area"
            value={meta.change_percentage != null ? `${meta.change_percentage}%` : undefined}
          />
          <Stat
            label="Changed area (m²)"
            value={meta.changed_area_m2 != null ? meta.changed_area_m2.toLocaleString() : undefined}
          />
          <Stat label="Changed pixels" value={meta.changed_pixels} />
          <Stat label="Total pixels" value={meta.total_pixels} />
          <Stat label="Regions" value={meta.num_regions} />
          <Stat
            label="Alignment method"
            value={meta.alignment_method ? ALIGNMENT_LABEL[meta.alignment_method] || meta.alignment_method : undefined}
          />
        </div>
        {regions.length > 0 && (
          <>
            <h4 className="sub-title">Changed regions</h4>
            <table className="box-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Area</th>
                  <th>x min</th>
                  <th>y min</th>
                  <th>x max</th>
                  <th>y max</th>
                </tr>
              </thead>
              <tbody>
                {regions.map((r, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{r.area_pct != null ? `${r.area_pct}%` : "—"}</td>
                    {r.box.map((v, j) => (
                      <td key={j}>{v}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        <p className="note evidence-note">
          {learned
            ? "Change statistics are measured from the model's binary mask at its 256×256 working resolution. The model marks where pixels changed, not what they changed to, and was trained on 0.5–3 m aerial imagery, so results on other imagery are unvalidated. Field verification is recommended before operational use."
            : "Change statistics are measured from the cleaned pixel-difference mask; they describe pixel-level visual differences, not semantic change. Field verification is recommended before operational use."}
        </p>
      </div>

      <div className="result-meta">
        <ConfidenceMeta value={confidence} source={result.confidence_source} />
      </div>
    </div>
  );
}
