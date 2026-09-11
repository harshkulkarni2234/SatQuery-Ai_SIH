import { useRef, useState } from "react";
import { uploadImage, runQuery, ApiError } from "./api.js";

const MAX_IMAGES = 2;

/* ═══════════════════════════════════════════════════════════════════════
   SVG Icons
   ═══════════════════════════════════════════════════════════════════════ */

const ic = (paths) =>
  function Icon({ size = 16 }) {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {paths}
      </svg>
    );
  };

const IconSatellite = ic(
  <>
    <rect x="8" y="8" width="8" height="8" rx="1.5" />
    <path d="M4 8h4M16 8h4M4 16h4M16 16h4" />
    <path d="M8 8 12 4h3" />
  </>
);
const IconUpload = ic(
  <>
    <path d="M12 15V4" />
    <path d="m7 9 5-5 5 5" />
    <path d="M4 18v1a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-1" />
  </>
);
const IconTrash = ic(
  <>
    <path d="M4 7h16" />
    <path d="M10 11v6M14 11v6" />
    <path d="M6 7l1 14h10l1-14" />
    <path d="M9 7V4h6v3" />
  </>
);
const IconCheck = ic(<path d="m4 12 5 5L20 6" />);
const IconPlay = ic(<path d="m7 5 13 7-13 7V5Z" />);
const IconArrow = ic(
  <>
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </>
);
const IconChevron = ic(<path d="m6 9 6 6 6-6" />);
const IconEye = ic(
  <>
    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12Z" />
    <circle cx="12" cy="12" r="3" />
  </>
);
const IconPin = ic(
  <>
    <path d="M12 17v5" />
    <path d="M9 2h6l-1 7H10Z" />
    <circle cx="12" cy="9" r="5" />
  </>
);
const IconCompare = ic(
  <>
    <rect x="2" y="4" width="8" height="16" rx="1" />
    <rect x="14" y="4" width="8" height="16" rx="1" />
    <path d="M10 10h4M10 14h4" />
  </>
);
const IconLayers = ic(
  <>
    <path d="m12 3 9 5-9 5-9-5 9-5Z" />
    <path d="m3 13 9 5 9-5" />
  </>
);
const IconImage = ic(
  <>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="9" cy="9" r="2" />
    <path d="m21 15-4.5-4.5L6 21" />
  </>
);
const IconLayer2 = ic(
  <>
    <rect x="3" y="3" width="7" height="7" rx="1" />
    <path d="m14 3 7 7" />
    <path d="M3 14l7 7" />
    <rect x="14" y="14" width="7" height="7" rx="1" />
  </>
);
const IconPlus = ic(
  <>
    <path d="M12 5v14M5 12h14" />
  </>
);

/* ═══════════════════════════════════════════════════════════════════════
   Specialists, Scenarios, Orbit labels
   ═══════════════════════════════════════════════════════════════════════ */

const SPECIALISTS = [
  {
    id: "VQA",
    label: "Visual Question Answering",
    short: "Visual QA",
    desc: "Interpret and describe satellite imagery",
    icon: IconEye,
  },
  {
    id: "GROUNDING",
    label: "Land Cover / Object Grounding",
    short: "Grounding",
    desc: "Locate and map land-cover features",
    icon: IconPin,
  },
  {
    id: "CHANGE_DETECTION",
    label: "Change Detection",
    short: "Change Detection",
    desc: "Identify differences between images",
    icon: IconCompare,
  },
  {
    id: "CROSS_MODAL",
    label: "Optical + SAR Analysis",
    short: "Cross-Modal",
    desc: "Fuse multi-sensor data for analysis",
    icon: IconLayers,
  },
];

// Ring traversal order (single source of truth). The orbit dot rotates ONCE
// through exactly these specialist ids, one per SPECIALIST_DWELL_MS. Each
// specialist is assigned the ring position equal to its index in this sequence
// so the dot's single continuous rotation lands on every one in this order.
const ORBIT_SEQUENCE = ["VQA", "CHANGE_DETECTION", "GROUNDING", "CROSS_MODAL"];
const ORBIT_POS = Object.fromEntries(
  ORBIT_SEQUENCE.map((id, i) => [id, i])
);

// Display names shown in the winner reveal / "Model selected" transcript.
const WINNER_NAMES = {
  VQA: "Visual QA",
  GROUNDING: "Spatial Grounding",
  CHANGE_DETECTION: "Change Detection",
  CROSS_MODAL: "Cross-Modal Fusion",
};

const STEP_LABELS = [
  "Understanding query\u2026",
  "Checking input\u2026",
  "Evaluating specialists\u2026",
  "Selecting best specialist\u2026",
];

// Agent-selection traversal clock. Every specialist is visited once for a
// fixed dwell before the dot moves on; the total run is 4 × 1s = 4s and is
// intentionally independent of backend inference latency.
const SPECIALIST_DWELL_MS = 1000;

const SCENARIOS = [
  {
    label: "Visual QA",
    description: "Describe or ask questions about a single image.",
    query: "What can you tell me about this image?",
  },
  {
    label: "Grounding",
    description: "Locate land-cover objects such as water or buildings.",
    query: "Where is the water located?",
  },
  {
    label: "Change Detection",
    description: "Compare two images captured at different dates.",
    query: "Show me the changes between these two images.",
  },
  {
    label: "Cross-Modal",
    description: "Analyze optical and SAR imagery together.",
    query: "Compare the optical and SAR images.",
  },
];

/* ═══════════════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════════════ */

const fmtConfidence = (v) => (v === null || v === undefined ? null : v.toFixed(3));

// Deterministic, stable report identifier rendered from the analysis output.
function reportId(result) {
  const seed = `${result?.task_classified || ""}|${result?.answer_text || ""}`;
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return `SQ-2026-${String((Math.abs(h) % 9000) + 1000)}`;
}

function describeTarget(images) {
  if (!images.length) return "";
  const optical = images.filter((i) => i.modality === "OPTICAL").length;
  const sar = images.filter((i) => i.modality === "SAR").length;
  if (optical && sar) return "OPTICAL + SAR pair";
  const mod = optical ? "optical" : "SAR";
  const plural = images.length > 1 ? `${mod} images` : `${mod} image`;
  return `${images.length} ${plural}`;
}

/* ═══════════════════════════════════════════════════════════════════════
   Shared UI
   ═══════════════════════════════════════════════════════════════════════ */

function Stat({ label, value }) {
  if (value === undefined || value === null || value === "") return null;
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

function StatGrid({ children }) {
  return <div className="stat-grid">{children}</div>;
}

/* ═══════════════════════════════════════════════════════════════════════
   Image Card
   ═══════════════════════════════════════════════════════════════════════ */

function ImageCard({ item, index, onChange, onRemove }) {
  return (
    <div className="image-card">
      <div className="image-preview">
        <img src={item.url} alt={item.file.name} />
        {item.imageId && (
          <span className="image-uploaded">
            <IconCheck size={11} /> uploaded
          </span>
        )}
        <button
          className="image-remove"
          onClick={onRemove}
          title="Remove image"
          aria-label="Remove image"
        >
          <IconTrash size={13} />
        </button>
      </div>
      <div className="image-body">
        <div className="image-name" title={item.file.name}>
          {item.file.name}
        </div>
        <div className="image-controls">
          <label className="control-group">
            <span className="control-label">Modality</span>
            <select
              aria-label="Modality"
              value={item.modality}
              onChange={(e) => onChange(index, { modality: e.target.value })}
            >
              <option value="OPTICAL">OPTICAL</option>
              <option value="SAR">SAR</option>
            </select>
          </label>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   SCREEN 1 — Input Workspace
   ═══════════════════════════════════════════════════════════════════════ */

function InputWorkspace({
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
              Sentinel-2 {"\u00b7"} Landsat {"\u00b7"} SAR {"\u00b7"} TIF {"\u00b7"} PNG {"\u00b7"} JPG
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
            placeholder={"Ask about the imagery, e.g. \u201cWhere is the water located?\u201d"}
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

/* ═══════════════════════════════════════════════════════════════════════
   SCREEN 2 — Agent Selection (orbit animation)
   phase: "traversing" | "selected" | "winner" | "preparing"
      traversing – moving-dot cycles the orbit (unchanged).
      selected   – lock-on: winning node highlights, checkmark, label.
      winner     – others fade, winner enlarges and moves to centre,
                   "Model selected: <name>" + "Preparing analysis report…".
      preparing  – winner held, "Specialist execution complete ✓".
   ═══════════════════════════════════════════════════════════════════════ */

function AgentSelection({ orbitStep, phase, selectedId, animLabel }) {
  const isTraversing = phase === "traversing";
  const isSelected = phase === "selected";
  const isWinner = phase === "winner";
  const isPreparing = phase === "preparing";
  const isSettled = isSelected || isWinner || isPreparing;

  const selectedSpec = selectedId
    ? SPECIALISTS.find((s) => s.id === selectedId) || null
    : null;
  const WinnerIcon = selectedSpec?.icon || IconSatellite;
  const winnerName = selectedId
    ? WINNER_NAMES[selectedId] || selectedSpec?.label || selectedId
    : "";

  return (
    <div className="agent-screen fade-in">
      <div
        className={`orbit-container${isWinner || isPreparing ? " winner" : ""}${
          isSettled ? " settled" : ""
        }`}
      >
        <div className="orbit-ring" />
        <div className="orbit-hint">
          {isTraversing
            ? "Selecting the best specialist\u2026"
            : isSelected
              ? "Specialist target locked"
              : ""}
        </div>

        <div className="orbit-center">
          <span className="orbit-brand">
            <IconSatellite size={20} />
          </span>
          <div className="orbit-title">SatQuery AI</div>
          <div className={`orbit-status ${isSettled ? "orbit-settled" : ""}`}>
            {isSelected ? "Specialist selected \u2713" : animLabel}
          </div>
        </div>

        {SPECIALISTS.map((spec) => {
          const ringPos = ORBIT_POS[spec.id] ?? 0;
          const isActive = isTraversing && orbitStep === ringPos;
          const isNodeSelected = isSelected && selectedId === spec.id;
          const isNodeWinner = (isWinner || isPreparing) && selectedId === spec.id;
          const isNodeFaded =
            (isWinner || isPreparing) && selectedId !== spec.id;
          const SpecIcon = spec.icon;
          return (
            <div
              key={spec.id}
              className={`orbit-node${isActive ? " node-active" : ""}${
                isNodeSelected ? " node-selected" : ""
              }${isNodeWinner ? " node-winner" : ""}${
                isNodeFaded ? " node-faded" : ""
              }`}
              data-pos={ringPos}
            >
              <div className="node-icon">
                <SpecIcon size={22} />
                {isNodeSelected && (
                  <span className="node-check">
                    <IconCheck size={12} />
                  </span>
                )}
              </div>
              {isNodeWinner ? (
                <div className="node-label node-label-winner">{spec.label}</div>
              ) : (
                <div className="node-label">{spec.short}</div>
              )}
            </div>
          );
        })}

        <div
          className="orbit-dot"
          style={{ transform: `rotate(${orbitStep * 90}deg) translateY(-145px)` }}
        >
          <span className="orbit-dot-core" />
        </div>
      </div>

      {(isWinner || isPreparing) && (
        <div className={`winner-hero${isPreparing ? " winner-done" : ""}`}>
          <div className="winner-icon">
            <WinnerIcon size={24} />
          </div>
          <div className="winner-model">
            Model selected: <strong>{winnerName}</strong>
          </div>
          {isPreparing ? (
            <div className="winner-preparing winner-preparing-done">
              <IconCheck size={15} />
              {"Analysis ready \u2713"}
            </div>
          ) : (
            <div className="winner-preparing">
              <span className="spinner-accent" />
              {"Preparing analysis report\u2026"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   SCREEN 3 — Analysis Ready
   ═══════════════════════════════════════════════════════════════════════ */

function AnalysisReady({ specialistId, onView }) {
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

/* ═══════════════════════════════════════════════════════════════════════
   Technical Details — collapsible
   ═══════════════════════════════════════════════════════════════════════ */

function TechnicalDetails({ result }) {
  const [open, setOpen] = useState(false);
  if (!result) return null;
  const trace = result.execution_trace || {};
  const meta = result.metadata || {};
  const task = result.task_classified;
  const toolName =
    task === "CROSS_MODAL"
      ? "optical_sar_analysis"
      : task === "VQA"
        ? meta.specialist_mode || "vqa"
        : task === "GROUNDING"
          ? "grounding"
          : task === "CHANGE_DETECTION"
            ? "change_detection"
            : undefined;
  const rows = [
    { label: "Task", value: task },
    { label: "Tool", value: toolName },
    { label: "Model", value: meta.model_version || trace.model_version },
    {
      label: "Execution time",
      value: trace.execution_time_ms != null ? `${trace.execution_time_ms} ms` : undefined,
    },
    {
      label: "Input modalities",
      value: trace.modalities_detected?.length
        ? trace.modalities_detected.join(", ")
        : undefined,
    },
    { label: "Router reason", value: trace.reason },
  ].filter((r) => r.value);
  if (!rows.length) return null;
  return (
    <div className="tech-details">
      <button className="tech-toggle" onClick={() => setOpen(!open)}>
        <span className={`tech-chevron${open ? " open" : ""}`}>
          <IconChevron size={14} />
        </span>
        Technical details
      </button>
      {open && (
        <div className="tech-body">
          {rows.map((r) => (
            <div className="tech-row" key={r.label}>
              <span className="tech-key">{r.label}</span>
              <span className="tech-val">{r.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   Result renderers
   ═══════════════════════════════════════════════════════════════════════ */

function ToggleChip({ active, label, onClick }) {
  return (
    <button
      className={`toggle-chip${active ? " active" : ""}`}
      onClick={onClick}
      aria-pressed={active}
    >
      {label}
    </button>
  );
}

function ViewerToggles({ options, value, onChange, label }) {
  return (
    <div className="viewer-toggles">
      <span className="viewer-toggles-label">{label}</span>
      <div className="viewer-toggles-group">
        {options.map((o) => (
          <ToggleChip
            key={o.value}
            active={value === o.value}
            label={o.label}
            onClick={() => onChange(o.value)}
          />
        ))}
      </div>
    </div>
  );
}

function OverlayImage({ src, boxes, labels, labelPrefix, showBoxes, alt }) {
  const [size, setSize] = useState(null);
  const labelText = (i) => (labels && labels[i]) || `${labelPrefix || "Region"} ${i + 1}`;
  return (
    <div className="overlay-img">
      <img
        src={src}
        alt={alt}
        onLoad={(e) =>
          setSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })
        }
      />
      {size &&
        showBoxes &&
        boxes.map((b, i) => (
          <span
            key={i}
            className="overlay-box"
            style={{
              left: `${(b[0] / size.w) * 100}%`,
              top: `${(b[1] / size.h) * 100}%`,
              width: `${((b[2] - b[0]) / size.w) * 100}%`,
              height: `${((b[3] - b[1]) / size.h) * 100}%`,
            }}
          >
            <span className="overlay-box-label">
              {labelText(i)}
            </span>
          </span>
        ))}
    </div>
  );
}

function ConfidenceMeta({ value }) {
  return (
    <div className="meta-item">
      <span className="meta-key">Confidence</span>
      {value !== null ? (
        <span className="meta-val">{value}</span>
      ) : (
        <span className="meta-val meta-unavail">Unavailable for this analysis</span>
      )}
    </div>
  );
}

function VQAResult({ result, images }) {
  const meta = result.metadata || {};
  const model = meta.model_version || result.execution_trace?.model_version;
  const specialist = meta.specialist_mode === "experimental";
  const confidence = fmtConfidence(result.confidence_score);
  const optical = images.find((i) => i.modality === "OPTICAL") || images[0];
  const ve = meta.visual_evidence || null;
  const boxes = result.bounding_boxes || [];
  const { detected_cues = [], dominant_cue } = ve || {};
  const target = ve?.target;
  const labelPrefix = LABEL_PREFIX[target] || "Region";
  return (
    <div className="result-block">
      {optical && boxes.length > 0 && (
        <div className="report-section">
          <h3 className="result-section-title">Visual evidence</h3>
          <GroundingImage src={optical.url} boxes={boxes} labelPrefix={labelPrefix} />
          <p className="note evidence-note">
            These bounding boxes are real deterministic detections of the
            object named in the question — they support the answer rather than
            being generated by the language model.
          </p>
        </div>
      )}
      {optical && boxes.length === 0 && (
        <div className="report-section">
          <h3 className="result-section-title">Visual evidence</h3>
          <figure className="result-image-figure">
            <img src={optical.url} alt="Analyzed image" />
          </figure>
          {detected_cues.length > 0 ? (
            <div className="cue-list">
              {detected_cues.map((c, i) => (
                <span className="cue-chip" key={i}>
                  {c.cue} · {(c.fraction * 100).toFixed(1)}% of frame
                </span>
              ))}
              {dominant_cue && <p className="note">{dominant_cue}.</p>}
            </div>
          ) : (
            <p className="note evidence-note">
              {ve?.message ||
                "Spatial localization unavailable for this analysis. The answer is based on the analyzed satellite frame without claimed coordinates."}
            </p>
          )}
        </div>
      )}
      <div className="result-section">
        <h3 className="result-section-title">Answer</h3>
        <div className="answer-text">{result.answer_text}</div>
      </div>
      <p className="note evidence-note">
        Answer generated from the analyzed satellite frame.
      </p>
      <div className="result-meta">
        <ConfidenceMeta value={confidence} />
        {model && (
          <div className="meta-item">
            <span className="meta-key">Model</span>
            <span className="meta-val">
              {model}
              {specialist ? " (RS-adapted)" : ""}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function GroundingImage({ src, boxes, labels, labelPrefix }) {
  const [size, setSize] = useState(null);
  const labelText = (i) => (labels && labels[i]) || `${labelPrefix} ${i + 1}`;
  return (
    <div className="geo-wrap">
      <div className="geo-image">
        <img
          src={src}
          alt="Analysis image"
          onLoad={(e) =>
            setSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })
          }
        />
        {size &&
          boxes.map((b, i) => (
            <span
              key={i}
              className="geo-box"
              style={{
                left: `${(b[0] / size.w) * 100}%`,
                top: `${(b[1] / size.h) * 100}%`,
                width: `${((b[2] - b[0]) / size.w) * 100}%`,
                height: `${((b[3] - b[1]) / size.h) * 100}%`,
              }}
              title={labelText(i)}
            >
              <span className="geo-box-label">
                {labelText(i)}
              </span>
            </span>
          ))}
      </div>
    </div>
  );
}

const LABEL_PREFIX = {
  water: "Water Body",
  vegetation: "Vegetation",
  "built-up": "Building",
  roads: "Road",
  farmland: "Farmland",
};

function GroundingResult({ result, images }) {
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

function ChangeResult({ result, images }) {
  const meta = result.metadata || {};
  const boxes = result.bounding_boxes || [];
  const regions = meta.regions || [];
  const [viewer, setViewer] = useState("overlay");
  const [before, after] =
    images.length >= 2 ? [images[0], images[1]] : [images[0], images[0]];
  const confidence = fmtConfidence(result.confidence_score);
  const unavailable = meta.validation_failed === true;

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
          <ConfidenceMeta value={confidence} />
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
            { value: "annotated", label: "Changed regions" },
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
              Highlighted pixels in the change mask represent detected
              differences according to the current deterministic method.
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
        <div className="stat-grid">
          <Stat
            label="Changed area"
            value={meta.change_percentage != null ? `${meta.change_percentage}%` : undefined}
          />
          <Stat label="Changed pixels" value={meta.changed_pixels} />
          <Stat label="Total pixels" value={meta.total_pixels} />
          <Stat label="Regions" value={meta.num_regions} />
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
        {meta.registration_applied && (
          <p className="note">
            Images were translation-aligned before differencing to remove small
            spatial offsets.
          </p>
        )}
        <p className="note evidence-note">
          Change statistics are measured from the cleaned pixel-difference mask;
          they describe pixel-level visual differences, not semantic change.
          Field verification is recommended before operational use.
        </p>
      </div>

      <div className="result-meta">
        <ConfidenceMeta value={confidence} />
      </div>
    </div>
  );
}

function CrossModalResult({ result, images }) {
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
        <ConfidenceMeta value={confidence} />
      </div>
    </div>
  );
}

function ResultContent({ result, images }) {
  const task = result.task_classified;
  if (task === "VQA") return <VQAResult result={result} images={images} />;
  if (task === "GROUNDING")
    return <GroundingResult result={result} images={images} />;
  if (task === "CHANGE_DETECTION")
    return <ChangeResult result={result} images={images} />;
  if (task === "CROSS_MODAL")
    return <CrossModalResult result={result} images={images} />;
  return (
    <div className="result-block">
      <div className="answer-text">
        {result.answer_text || "Analysis returned no answer."}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   SCREEN 4 — Result
   ═══════════════════════════════════════════════════════════════════════ */

function AnalysisResult({ result, images, onBack }) {
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
  return (
    <div className="result-screen fade-in">
      <div className="result-topbar">
        <button className="btn-back" onClick={onBack}>
          {"\u2190"} New Analysis
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
        </div>
        <ResultContent result={result} images={images} />
      </div>
      <TechnicalDetails result={result} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   App Root
   ═══════════════════════════════════════════════════════════════════════ */

export default function App() {
  const [images, setImages] = useState([]);
  const [queryText, setQueryText] = useState("");
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [view, setView] = useState("input"); // input | analyzing | ready | result

  const [orbitStep, setOrbitStep] = useState(0);
  const [phase, setPhase] = useState("traversing");
  const [selectedTask, setSelectedTask] = useState(null);
  const [animLabel, setAnimLabel] = useState(STEP_LABELS[0]);

  function addFiles(fileList) {
    setError(null);
    const files = fileList.filter((f) => f && f.size > 0);
    if (!files.length) return;
    const newItems = files.map((file) => ({
      key: `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      file,
      url: URL.createObjectURL(file),
      modality: "OPTICAL",
      imageId: null,
    }));
    setImages((prev) => [...prev, ...newItems].slice(0, MAX_IMAGES));
  }

  function updateImage(index, patch) {
    setImages((prev) =>
      prev.map((it, i) => (i === index ? { ...it, ...patch } : it))
    );
  }

  function removeImage(key) {
    setImages((prev) => prev.filter((it) => it.key !== key));
  }

  async function handleAnalyze() {
    setError(null);
    if (!queryText.trim()) {
      setError("Please enter a question before analyzing.");
      return;
    }
    if (!images.length) {
      setError("Please upload at least one image.");
      return;
    }
    setView("analyzing");
    setPhase("traversing");
    setSelectedTask(null);
    setResult(null);

    const delay = (ms) => new Promise((r) => setTimeout(r, ms));

    // Backend work is requested immediately but runs in the background. The
    // UI traversal below advances on a FIXED 1s-per-specialist clock (4 × 1s
    // = 4s total) and is completely independent of upload / VQA inference
    // latency. Step boundaries are anchored to an absolute start instant so
    // the sequence cannot drift even when the event loop is busy.
    const finish = (async () => {
      const withIds = [...images];
      for (let i = 0; i < withIds.length; i++) {
        if (!withIds[i].imageId) {
          const up = await uploadImage(withIds[i].file, withIds[i].modality);
          withIds[i] = { ...withIds[i], imageId: up.image_id };
        }
      }
      setImages(withIds);
      return runQuery(queryText.trim(), withIds.map((it) => it.imageId));
    })();

    try {
      const data = await new Promise((resolve, reject) => {
        let settled = false;
        let earlyData = null;
        const cancel = () => { settled = true; };

        finish.then(
          (d) => { earlyData = d; },
          (err) => {
            if (!settled) { settled = true; reject(err); }
          }
        );

        (async () => {
          const stepTimes = Array.from(
            { length: ORBIT_SEQUENCE.length },
            (_, k) => performance.now() + (k + 1) * SPECIALIST_DWELL_MS
          );
          setOrbitStep(0);
          setAnimLabel(STEP_LABELS[0]);
          for (let k = 0; k < stepTimes.length; k++) {
            await delay(Math.max(0, stepTimes[k] - performance.now()));
            if (settled) return; // error path — stop advancing
            const next = k + 1;
            if (next < stepTimes.length) {
              setOrbitStep(next);
              setAnimLabel(STEP_LABELS[next]);
            }
          }
          if (settled) return;
          cancel();
          try {
            const d = earlyData !== null ? earlyData : await finish;
            resolve(d);
          } catch (err) {
            reject(err);
          }
        })();
      });

      const task = data.task_classified;
      // Phased winner reveal, driven by the REAL backend selection:
      //   A) lock-on (~520ms)  B+C) fade + centre move (~900ms)
      //   D) preparing report (~900ms) then hand off to the ready screen.
      setPhase("selected");
      setSelectedTask(task);
      setOrbitStep(ORBIT_POS[task] ?? 0);
      setAnimLabel("Specialist selected \u2713");
      setResult(data);
      await delay(520);
      setPhase("winner");
      await delay(900);
      setPhase("preparing");
      await delay(900);
      setView("ready");
    } catch (err) {
      setPhase("traversing");
      setSelectedTask(null);
      setView("input");
      setError(
        err instanceof ApiError ? err.message : `Unexpected error: ${err.message}`
      );
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark">
            <IconSatellite size={18} />
          </span>
          <div className="brand-text">
            <h1>SatQuery AI</h1>
          </div>
        </div>
        <span className="proto-badge">SIH 2026</span>
      </header>

      <main className="main">
        {view === "input" && (
          <InputWorkspace
            key="input"
            images={images}
            onAddFiles={addFiles}
            onUpdateImage={updateImage}
            onRemoveImage={removeImage}
            queryText={queryText}
            setQueryText={setQueryText}
            onAnalyze={handleAnalyze}
            error={error}
          />
        )}
        {view === "analyzing" && (
          <AgentSelection
            key="analyzing"
            orbitStep={orbitStep}
            phase={phase}
            selectedId={selectedTask}
            animLabel={animLabel}
          />
        )}
        {view === "ready" && (
          <AnalysisReady
            key="ready"
            specialistId={selectedTask}
            onView={() => setView("result")}
          />
        )}
        {view === "result" && (
          <AnalysisResult
            key="result"
            result={result}
            images={images}
            onBack={() => {
              setView("input");
              setResult(null);
              setPhase("traversing");
              setSelectedTask(null);
              setError(null);
            }}
          />
        )}
      </main>

      <footer className="footer">
        SatQuery AI {"\u00b7"} SIH 2026 Prototype
      </footer>
    </div>
  );
}