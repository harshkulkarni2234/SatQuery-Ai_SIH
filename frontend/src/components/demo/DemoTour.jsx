import { useEffect, useLayoutEffect, useState } from "react";

// A spotlight walkthrough. Each step points at a real element in the running
// UI, so the tour explains the actual product rather than a mock of it.
//
// A step is { sel, title, body, wait?, nextLabel? }:
//   sel   CSS selector for the element to highlight
//   wait  hold the step until `sel` exists (the result screen appears late)
function useRect(sel, active, tick) {
  const [rect, setRect] = useState(null);
  useLayoutEffect(() => {
    if (!active) return undefined;
    let raf = 0;
    const measure = () => {
      const el = sel ? document.querySelector(sel) : null;
      if (el) {
        const r = el.getBoundingClientRect();
        setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
      } else {
        setRect(null);
      }
      raf = requestAnimationFrame(measure);
    };
    measure();
    return () => cancelAnimationFrame(raf);
  }, [sel, active, tick]);
  return rect;
}

export default function DemoTour({ steps, index, onNext, onBack, onExit }) {
  const step = steps[index];
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!step) return undefined;
    if (!step.wait) {
      setReady(true);
      return undefined;
    }
    setReady(false);
    const t = setInterval(() => {
      if (document.querySelector(step.sel)) setReady(true);
    }, 150);
    return () => clearInterval(t);
  }, [step]);

  const rect = useRect(step && step.sel, Boolean(step) && ready, index);

  useEffect(() => {
    if (!step || !ready) return;
    const el = document.querySelector(step.sel);
    if (el) el.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [step, ready]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onExit();
      else if (e.key === "ArrowRight") onNext();
      else if (e.key === "ArrowLeft") onBack();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onNext, onBack, onExit]);

  if (!step) return null;

  const pad = 8;
  const box = rect
    ? {
        top: rect.top - pad,
        left: rect.left - pad,
        width: rect.width + pad * 2,
        height: rect.height + pad * 2,
      }
    : null;

  let cardStyle = { top: "50%", left: "50%", transform: "translate(-50%, -50%)" };
  if (box) {
    const cardW = 380;
    const below = box.top + box.height + 16;
    const left = Math.min(
      Math.max(16, box.left + box.width / 2 - cardW / 2),
      Math.max(16, window.innerWidth - cardW - 16)
    );
    cardStyle =
      window.innerHeight - below > 240
        ? { top: below, left, width: cardW }
        : { top: Math.max(16, box.top - 240), left, width: cardW };
  }

  return (
    <div className="tour-layer">
      {box ? <div className="tour-spot" style={box} /> : <div className="tour-dim" />}

      <div className="tour-card glass" style={cardStyle}>
        <div className="tour-progress">
          Step {index + 1} of {steps.length}
        </div>
        <h3 className="tour-title">{step.title}</h3>
        <p className="tour-body">{step.body}</p>
        {!ready && <p className="tour-waiting">Waiting for the system…</p>}
        <div className="tour-actions">
          <button className="tour-skip" onClick={onExit}>Exit</button>
          <span className="tour-spacer" />
          {index > 0 && <button className="tour-back" onClick={onBack}>Back</button>}
          <button className="tour-next" onClick={onNext} disabled={!ready}>
            {index === steps.length - 1 ? "Finish" : step.nextLabel || "Next"}
          </button>
        </div>
      </div>
    </div>
  );
}
