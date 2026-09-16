import { useRef } from "react";
import { IconUpload, IconPlay, IconPlus } from "../icons/index.jsx";
import { SCENARIOS } from "../../constants/scenarios.js";
import ImageCard from "./ImageCard.jsx";

const MAX_IMAGES = 2;

export default function InputWorkspace({
  images,
  onAddFiles,
  onUpdateImage,
  onRemoveImage,
  queryText,
  setQueryText,
  onAnalyze,
  error,
}) {
  const fileRef = useRef(null);
  const pickFiles = () => fileRef.current && fileRef.current.click();
  return (
    <div className="input-workspace fade-in">
      <div className="workspace-hero">
        <h1 className="hero-title">SatQuery AI</h1>
        <p className="hero-sub">Multimodal Remote Sensing Intelligence Platform</p>
      </div>

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

        {error && <div className="error-banner">{error}</div>}

        <button className="btn-analyze" onClick={onAnalyze}>
          <IconPlay size={15} /> Analyze
        </button>
      </div>

      {images.length === 0 && (
        <div className="demo-queries">
          <span className="demo-label">Try an example</span>
          <div className="preset-row">
            {SCENARIOS.map((s) => (
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
      )}
    </div>
  );
}

export { MAX_IMAGES };
