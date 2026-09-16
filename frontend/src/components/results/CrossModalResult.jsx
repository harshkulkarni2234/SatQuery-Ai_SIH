import { useState } from "react";
import { fmtConfidence } from "../../lib/format.js";
import { IconImage, IconSatellite, IconLayer2 } from "../icons/index.jsx";
import ConfidenceMeta from "../common/ConfidenceMeta.jsx";
import ViewerToggles from "../common/ViewerToggles.jsx";
import OverlayImage from "../common/OverlayImage.jsx";
import Stat from "../common/Stat.jsx";
import StatGrid from "../common/StatGrid.jsx";

export default function CrossModalResult({ result, images }) {
  const meta = result.metadata || {};
  const evidence = meta.evidence || {};
  const combined = evidence.combined || {};
  const confidence = fmtConfidence(result.confidence_score);
  const [viewer, setViewer] = useState("combined");
  const optical = images.find((i) => i.modality === "OPTICAL");
  const sar = images.find((i) => i.modality === "SAR");

  const statRows = (obj) =>
    Object.entries(obj || {})
      .filter(
        ([, v]) =>
          v === null ||
          typeof v === "number" ||
          typeof v === "string" ||
          typeof v === "boolean"
      )
      .map(([k, v]) => ({
        label: k.replace(/_/g, " "),
        value: typeof v === "number" ? Number(v.toFixed(3)) : String(v),
      }));

  const regionBoxes = (regions) => (regions || []).map((r) => r.box).filter(Boolean);
  const regionLabels = (regions) =>
    (regions || []).map((r) => r.label).filter(Boolean);
  const opticalRegions = evidence.optical?.regions || [];
  const sarRegions = evidence.sar?.regions || [];

  const opticalStats = statRows(evidence.optical);
  const sarStats = statRows(evidence.sar);
  const opticalVisible = viewer !== "sar";
  const sarVisible = viewer !== "optical";

  return (
    <div className="result-block">
      <div className="report-section">
        <h3 className="result-section-title">Spatial evidence</h3>
        <ViewerToggles
          label="Layers"
          value={viewer}
          onChange={setViewer}
          options={[
            { value: "combined", label: "Optical + SAR" },
            { value: "optical", label: "Optical" },
            { value: "sar", label: "SAR" },
          ]}
        />
        {((opticalVisible && optical) || (sarVisible && sar)) && (
          <div className="evidence-grid">
            {opticalVisible && optical && (
              <figure className="evidence-figure">
                <figcaption>Optical</figcaption>
                <OverlayImage
                  src={optical.url}
                  alt="Optical image"
                  boxes={regionBoxes(opticalRegions)}
                  labels={regionLabels(opticalRegions)}
                  labelPrefix="Optical"
                  showBoxes
                />
              </figure>
            )}
            {sarVisible && sar && (
              <figure className="evidence-figure">
                <figcaption>SAR</figcaption>
                <OverlayImage
                  src={sar.url}
                  alt="SAR image"
                  boxes={regionBoxes(sarRegions)}
                  labels={regionLabels(sarRegions)}
                  labelPrefix="SAR"
                  showBoxes
                />
              </figure>
            )}
          </div>
        )}
      </div>

      <div className="result-section">
        <h3 className="result-section-title">Combined result</h3>
        <div className="answer-text">{result.answer_text}</div>
      </div>

      {((opticalVisible && opticalStats.length > 0) ||
        (sarVisible && sarStats.length > 0)) && (
        <div className="evidence-columns">
          {opticalVisible && opticalStats.length > 0 && (
            <section className="evidence-section">
              <h4 className="sub-title">
                <IconImage size={14} /> Optical
                <span className="evidence-caption">Spectral and colour evidence</span>
              </h4>
              <StatGrid>
                {opticalStats.map((s) => (
                  <Stat key={s.label} label={s.label} value={s.value} />
                ))}
              </StatGrid>
            </section>
          )}
          {sarVisible && sarStats.length > 0 && (
            <section className="evidence-section">
              <h4 className="sub-title">
                <IconSatellite size={14} /> SAR
                <span className="evidence-caption">Radar and backscatter evidence</span>
              </h4>
              <StatGrid>
                {sarStats.map((s) => (
                  <Stat key={s.label} label={s.label} value={s.value} />
                ))}
              </StatGrid>
            </section>
          )}
        </div>
      )}

      {viewer === "combined" &&
        (combined.combined_readings?.length > 0 ||
          meta.modality_contribution_note) && (
          <section className="evidence-section">
            <h4 className="sub-title">
              <IconLayer2 size={14} /> Combined interpretation
              <span className="evidence-caption">Joint reading of both modalities</span>
            </h4>
            {combined.combined_readings?.length > 0 && (
              <ul className="reading-list">
                {combined.combined_readings.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            )}
            {meta.modality_contribution_note && (
              <p className="note">{meta.modality_contribution_note}</p>
            )}
          </section>
        )}

      <p className="note evidence-note">
        Regions are reported per sensor with their own coordinates and are not
        registered to one another.
        {meta.spatial_correspondence_note
          ? ` ${meta.spatial_correspondence_note}`
          : " Spatial correspondence between the two sensors could not be verified."}
      </p>
      <div className="result-meta">
        <ConfidenceMeta value={confidence} source={result.confidence_source} />
      </div>
    </div>
  );
}
