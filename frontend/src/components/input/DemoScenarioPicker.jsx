import { useState } from "react";
import { DEMO_SCENARIO_ITEMS } from "../../constants/demoScenarios.js";

// Phase C9: gated behind a dev/demo flag — visible during `npm run dev`
// (import.meta.env.DEV) and in a built demo deploy that explicitly opts in
// via VITE_ENABLE_DEMO_SCENARIOS=true, hidden in a plain production build.
const DEMO_SCENARIOS_ENABLED =
  import.meta.env.DEV || import.meta.env.VITE_ENABLE_DEMO_SCENARIOS === "true";

export default function DemoScenarioPicker({ onLoad }) {
  const [loadingId, setLoadingId] = useState(null);
  const [loadError, setLoadError] = useState(null);

  if (!DEMO_SCENARIOS_ENABLED) return null;

  async function handleClick(item) {
    setLoadError(null);
    setLoadingId(item.key);
    try {
      await onLoad(item);
    } catch (err) {
      setLoadError(`Could not load "${item.label}": ${err.message}`);
    } finally {
      setLoadingId(null);
    }
  }

  return (
    <div className="demo-queries demo-scenario-picker">
      <span className="demo-label">Load demo scenario (real files, real upload)</span>
      <div className="preset-row">
        {DEMO_SCENARIO_ITEMS.map((item) => (
          <button
            key={item.key}
            className="preset-btn"
            onClick={() => handleClick(item)}
            disabled={loadingId !== null}
          >
            <span className="preset-title">
              {loadingId === item.key ? "Loading…" : item.label}
            </span>
            <span className="preset-desc">{item.description}</span>
          </button>
        ))}
      </div>
      {loadError && <div className="error-banner">{loadError}</div>}
    </div>
  );
}
