import { useState } from "react";
import { uploadImage, runQuery, ApiError } from "./api.js";
import { IconSatellite } from "./components/icons/index.jsx";
import { ORBIT_SEQUENCE, ORBIT_POS, STEP_LABELS, SPECIALIST_DWELL_MS } from "./constants/specialists.js";
import InputWorkspace, { MAX_IMAGES } from "./components/input/InputWorkspace.jsx";
import AgentSelection from "./components/agent/AgentSelection.jsx";
import AnalysisReady from "./components/agent/AnalysisReady.jsx";
import AnalysisResult from "./components/results/AnalysisResult.jsx";

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
      setAnimLabel("Specialist selected ✓");
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
        SatQuery AI {"·"} SIH 2026 Prototype
      </footer>
    </div>
  );
}
