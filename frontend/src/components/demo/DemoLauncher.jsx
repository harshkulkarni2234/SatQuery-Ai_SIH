import { useState } from "react";
import { DEMO_SCENARIO_ITEMS } from "../../constants/demoScenarios.js";

// Full-screen scenario chooser. Opens from the Demo mode button in the header.
// Every card loads the real file through the normal upload API, so nothing
// shown afterwards is pre-baked.
export default function DemoLauncher({ open, onClose, onPick }) {
  const [loadingKey, setLoadingKey] = useState(null);
  const [error, setError] = useState(null);

  if (!open) return null;

  async function pick(item) {
    setError(null);
    setLoadingKey(item.key);
    try {
      await onPick(item);
    } catch (err) {
      setError(`Could not load ${item.label}. ${err.message}`);
      setLoadingKey(null);
    }
  }

  return (
    <div className="demo-overlay" onClick={onClose}>
      <div
        className="demo-modal glass"
        role="dialog"
        aria-modal="true"
        aria-label="Choose a guided demo"
        onClick={(e) => e.stopPropagation()}
      >
        <button className="demo-close" onClick={onClose} aria-label="Close">✕</button>

        <h2 className="demo-modal-title">Guided demo</h2>
        <p className="demo-modal-sub">
          Pick a scenario. Real files load through the normal upload API, then
          the walkthrough takes you through the system one step at a time.
        </p>

        <div className="demo-cards">
          {DEMO_SCENARIO_ITEMS.map((item, i) => (
            <button
              key={item.key}
              className="demo-card glass-soft"
              disabled={loadingKey !== null}
              onClick={() => pick(item)}
            >
              <span className="demo-card-num">{String(i + 1).padStart(2, "0")}</span>
              <span className="demo-card-body">
                <span className="demo-card-title">
                  {loadingKey === item.key ? "Loading…" : item.label}
                </span>
                <span className="demo-card-desc">{item.description}</span>
                <span className="demo-card-query">“{item.query}”</span>
              </span>
            </button>
          ))}
        </div>

        {error && <div className="error-banner">{error}</div>}
      </div>
    </div>
  );
}
