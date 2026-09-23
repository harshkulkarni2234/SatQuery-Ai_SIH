import { useState } from "react";
import { uploadImage, runQuery, ApiError } from "./api.js";
import { IconSatellite } from "./components/icons/index.jsx";
import InputWorkspace, { MAX_IMAGES } from "./components/input/InputWorkspace.jsx";
import AgentSelection from "./components/agent/AgentSelection.jsx";
import TraceReplay from "./components/agent/TraceReplay.jsx";
import SystemStatus from "./components/common/SystemStatus.jsx";
import DemoLauncher from "./components/demo/DemoLauncher.jsx";
import DemoTour from "./components/demo/DemoTour.jsx";
import { TOUR_STEPS } from "./constants/tourSteps.js";
import AnalysisReady from "./components/agent/AnalysisReady.jsx";
import AnalysisResult from "./components/results/AnalysisResult.jsx";

export default function App() {
  const [images, setImages] = useState([]);
  const [queryText, setQueryText] = useState("");
  const [error, setError] = useState(null);
  const [errorDetail, setErrorDetail] = useState(null);
  const [result, setResult] = useState(null);
  const [view, setView] = useState("input"); // input | waiting | trace | ready | result
  const [demoOpen, setDemoOpen] = useState(false);
  const [tourStep, setTourStep] = useState(-1); // -1 = tour not running
  const [selectedTask, setSelectedTask] = useState(null);

  // Images upload to the backend as soon as they're added (not deferred to
  // Analyze) so real metadata (CRS, bounds, resolution, bands — Phase A2)
  // can be shown on the card right away. Re-uploads on modality/date change
  // create a new backend Image row rather than mutating the old one — there
  // is no update/delete endpoint yet, so the previous row is simply orphaned.
  async function uploadItem(key, file, modality, captureDate) {
    setImages((prev) =>
      prev.map((it) => (it.key === key ? { ...it, uploading: true, uploadError: null } : it))
    );
    try {
      const up = await uploadImage(file, modality, captureDate);
      setImages((prev) =>
        prev.map((it) =>
          it.key === key
            ? { ...it, uploading: false, imageId: up.image_id, metadata: up.metadata || null }
            : it
        )
      );
    } catch (err) {
      setImages((prev) =>
        prev.map((it) =>
          it.key === key
            ? {
                ...it,
                uploading: false,
                imageId: null,
                metadata: null,
                uploadError:
                  err instanceof ApiError ? err.message : `Upload failed: ${err.message}`,
              }
            : it
        )
      );
    }
  }

  function addFiles(fileList) {
    setError(null);
    setErrorDetail(null);
    const files = fileList.filter((f) => f && f.size > 0);
    if (!files.length) return;
    const newItems = files.map((file) => ({
      key: `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      file,
      url: URL.createObjectURL(file),
      modality: "OPTICAL",
      captureDate: null,
      imageId: null,
      uploading: false,
      uploadError: null,
      metadata: null,
    }));
    setImages((prev) => [...prev, ...newItems].slice(0, MAX_IMAGES));
    for (const item of newItems) {
      uploadItem(item.key, item.file, item.modality, item.captureDate);
    }
  }

  function updateImage(index, patch) {
    // Read the current item from the `images` closure (fresh each render)
    // rather than trying to capture it from inside the setImages updater —
    // that updater callback runs asynchronously, not synchronously within
    // this call, so a variable set inside it isn't populated yet by the
    // time code right after setImages() runs.
    const current = images[index];
    if (!current) return;
    const updated = { ...current, ...patch };
    setImages((prev) => prev.map((it, i) => (i === index ? updated : it)));
    if ("modality" in patch || "captureDate" in patch) {
      uploadItem(updated.key, updated.file, updated.modality, updated.captureDate);
    }
  }

  function removeImage(key) {
    setImages((prev) => prev.filter((it) => it.key !== key));
  }

  // Phase C9: fetches the real demo files (served by the backend at
  // /demo-assets/..., see main.py) and uploads them through the exact same
  // uploadItem() -> POST /images/upload path as a user-picked file — no
  // pre-baked results, the specialist genuinely runs on these bytes.
  async function loadDemoScenario(item) {
    setError(null);
    setErrorDetail(null);
    setImages([]);
    setQueryText(item.query);

    const newItems = [];
    for (const f of item.files) {
      const resp = await fetch(f.url);
      if (!resp.ok) {
        throw new Error(`Failed to fetch demo file ${f.filename} (HTTP ${resp.status})`);
      }
      const blob = await resp.blob();
      const file = new File([blob], f.filename, { type: blob.type });
      newItems.push({
        key: `${f.filename}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        file,
        url: URL.createObjectURL(file),
        modality: f.modality,
        captureDate: f.captureDate,
        imageId: null,
        uploading: false,
        uploadError: null,
        metadata: null,
      });
    }
    setImages(newItems);
    for (const it of newItems) {
      uploadItem(it.key, it.file, it.modality, it.captureDate);
    }
  }

  async function handleAnalyze() {
    setError(null);
    setErrorDetail(null);
    if (!queryText.trim()) {
      setError("Please enter a question before analyzing.");
      return;
    }
    if (!images.length) {
      setError("Please upload at least one image.");
      return;
    }
    if (images.some((it) => it.uploading)) {
      setError("Please wait for all images to finish uploading.");
      return;
    }
    if (images.some((it) => !it.imageId)) {
      setError("One or more images failed to upload. Remove or retry them before analyzing.");
      return;
    }
    setView("waiting");
    setSelectedTask(null);
    setResult(null);

    // No fixed-clock animation here — the "waiting" screen (AgentSelection)
    // stays neutral for however long the real request actually takes, and
    // once it responds we replay the backend's OWN recorded trace_events
    // (TraceReplay) rather than a client-side guess at what happened.
    try {
      const data = await runQuery(queryText.trim(), images.map((it) => it.imageId));
      setSelectedTask(data.task_classified);
      setResult(data);
      setView("trace");
    } catch (err) {
      setSelectedTask(null);
      setView("input");
      setError(
        err instanceof ApiError ? err.message : `Unexpected error: ${err.message}`
      );
      setErrorDetail(err instanceof ApiError ? err.detail : null);
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
        <div className="header-right">
          <SystemStatus />
          <button
            className="demo-mode-btn"
            onClick={() => setDemoOpen(true)}
            disabled={tourStep >= 0}
          >
            Demo mode
          </button>
          <span className="proto-badge">SIH 2026</span>
        </div>
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
            errorDetail={errorDetail}
          />
        )}
        {view === "waiting" && <AgentSelection key="waiting" />}
        {view === "trace" && (
          <TraceReplay
            key="trace"
            result={result}
            hold={tourStep >= 0}
            onDone={() => setView("ready")}
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
              setSelectedTask(null);
              setError(null);
              setErrorDetail(null);
            }}
          />
        )}
      </main>

      <DemoLauncher
        open={demoOpen}
        onClose={() => setDemoOpen(false)}
        onPick={async (item) => {
          await loadDemoScenario(item);
          setDemoOpen(false);
          setTourStep(0);
        }}
      />

      {tourStep >= 0 && (
        <DemoTour
          steps={TOUR_STEPS}
          index={tourStep}
          onExit={() => setTourStep(-1)}
          onBack={() => setTourStep((i) => Math.max(0, i - 1))}
          onNext={() => {
            const step = TOUR_STEPS[tourStep];
            // "Run it" actually runs the query. Leaving the trace goes straight
            // to the result: the Analysis Ready interstitial would otherwise
            // leave the next step waiting for an element that is not on screen.
            if (step.sel === ".btn-analyze") handleAnalyze();
            if (step.sel === ".trace-replay") setView("result");
            if (tourStep === TOUR_STEPS.length - 1) setTourStep(-1);
            else setTourStep((i) => i + 1);
          }}
        />
      )}

      <footer className="footer">
        SatQuery AI {"·"} SIH 2026 Prototype
      </footer>
    </div>
  );
}
