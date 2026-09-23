import { useRef } from "react";
import { IconUpload, IconPlay, IconPlus } from "../icons/index.jsx";
import { SCENARIOS, imageCombo } from "../../constants/scenarios.js";
import { pairSummary } from "../../lib/format.js";
import ImageCard from "./ImageCard.jsx";
import ErrorPanel from "../common/ErrorPanel.jsx";
import RegistryPanel from "./RegistryPanel.jsx";

const MAX_IMAGES = 2;

function PairSummary({ images }) {
  const summary = pairSummary(images);
  if (!summary) return null;
  const { sameModality, datesKnown, datesDiffer, bothGeoreferenced } = summary;
  return (
    <div className="pair-summary">
      <span className="pair-summary-label">Pair summary</span>
      <span className="pair-summary-item">
        Same modality: <strong>{sameModality ? "Yes" : "No"}</strong>
      </span>
      <span className="pair-summary-item">
        Dates differ:{" "}
        <strong>{!datesKnown ? "Unknown" : datesDiffer ? "Yes" : "No"}</strong>
      </span>
      <span className="pair-summary-item">
        Both georeferenced: <strong>{bothGeoreferenced ? "Yes" : "No"}</strong>
      </span>
      <p className="note">
        A quick look only. The backend compatibility check decides whether these
        two scenes can actually be analysed together.
      </p>
    </div>
  );
}

export default function InputWorkspace({
  images,
  onAddFiles,
  onUpdateImage,
  onRemoveImage,
  queryText,
  setQueryText,
  onAnalyze,
  error,
  errorDetail,
}) {
  const fileRef = useRef(null);
  const pickFiles = () => fileRef.current && fileRef.current.click();
  const combo = imageCombo(images);
  const examples = combo ? SCENARIOS.filter((s) => s.combo === combo) : SCENARIOS;
  const canSubmit = queryText.trim().length > 0 && !images.some((it) => it.uploading);
  return (
    <div className="input-workspace fade-in">
      <div className="workspace-hero">
        <h1 className="hero-title">Multimodal remote sensing analysis</h1>
        <p className="hero-sub">
          Upload one or two scenes and ask a question. You get an answer, the
          evidence behind it, and a record of how it was produced.
        </p>
      </div>

      <div className="workspace-grid">
      <div className="workspace-main">
      <div className="input-card">
        <section className="field">
          <div className="field-label">Remote Sensing Imagery</div>
          <div
            className={`dropzone${error && !images.length ? " dz-error" : ""}`}
            onClick={(e) => {
              if (e.target.closest(".image-remove")) return;
              pickFiles();
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.currentTarget.classList.add("dz-drag");
            }}
            onDragLeave={(e) => e.currentTarget.classList.remove("dz-drag")}
            onDrop={(e) => {
              e.preventDefault();
              e.currentTarget.classList.remove("dz-drag");
              onAddFiles(Array.from(e.dataTransfer.files || []));
            }}
          >
            <IconUpload size={26} />
            <span className="dz-title">
              {images.length >= MAX_IMAGES
                ? `Maximum ${MAX_IMAGES} images selected`
                : "Drop imagery here or click to browse"}
            </span>
            <span className="dz-formats">
              Sentinel-2 {"·"} Landsat {"·"} SAR {"·"} TIF {"·"} PNG {"·"} JPG
            </span>
            <input
              ref={fileRef}
              type="file"
              hidden
              multiple
              accept=".tif,.tiff,.png,.jpg,.jpeg,.bmp,.jp2,image/*"
              onChange={(e) => onAddFiles(Array.from(e.target.files || []))}
            />
          </div>
        </section>

        {images.length > 0 && (
          <section className="field">
            <div className="field-label stage-label">Stage Image(s)</div>
            <div className="image-grid">
              {images.map((item, i) => (
                <ImageCard
                  key={item.key}
                  item={item}
                  index={i}
                  onChange={onUpdateImage}
                  onRemove={() => onRemoveImage(item.key)}
                />
              ))}
            </div>
            {images.length === 1 && (
              <button className="add-compare" onClick={pickFiles}>
                <IconPlus size={14} /> Add comparison image
              </button>
            )}
            <PairSummary images={images} />
          </section>
        )}

        <section className="field">
          <div className="field-label">Natural Language Query</div>
          <textarea
            className="query-input"
            rows={3}
            placeholder={"Ask about the imagery, e.g. “Where is the water located?”"}
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
          />
        </section>

        {error && (
          errorDetail && typeof errorDetail === "object" ? (
            <ErrorPanel message={error} detail={errorDetail} />
          ) : (
            <div className="error-banner">{error}</div>
          )
        )}

        <button
          className="btn-analyze"
          onClick={onAnalyze}
          disabled={!canSubmit}
        >
          <IconPlay size={15} />{" "}
          {images.some((it) => it.uploading) ? "Uploading…" : "Analyze"}
        </button>
      </div>

      </div>

      <aside className="workspace-side">
      <RegistryPanel />

      <div className="demo-queries panel">
        <span className="demo-label">Try an example</span>
        <div className="preset-row">
          {examples.map((s) => (
            <button
              key={s.label}
              className="preset-btn"
              onClick={() => setQueryText(s.query)}
            >
              <span className="preset-title">{s.label}</span>
              <span className="preset-desc">{s.description}</span>
            </button>
          ))}
        </div>
      </div>

      </aside>
      </div>
    </div>
  );
}

export { MAX_IMAGES };
