import { useEffect, useRef, useState } from "react";
import { IconSatellite } from "../icons/index.jsx";
import { SPECIALISTS, WINNER_NAMES } from "../../constants/specialists.js";
import { fmtSpecialistId, fmtStepLabel } from "../../lib/format.js";

const STEP_REVEAL_MS = 220;
const HOLD_AFTER_MS = 500;

const STATUS_ICON = {
  COMPLETED: "✓",
  PASSED: "✓",
  SKIPPED: "–",
  FAILED: "✕",
  STARTED: "•",
};

// Replays the backend's OWN recorded trace_events, one at a time, labeled as
// a replay (never presented as live execution) — see AGENTS.md rule 4 ("UI
// must never imply a model executed unless the backend says it did") and
// rule 2/3 (never fabricate results). Every detail/duration shown here comes
// straight from the /query response; nothing is invented client-side.
export default function TraceReplay({ result, onDone }) {
  const events = result?.trace_events || [];
  const [revealCount, setRevealCount] = useState(0);
  const doneCalled = useRef(false);

  useEffect(() => {
    if (events.length === 0) {
      if (!doneCalled.current) {
        doneCalled.current = true;
        onDone();
      }
      return;
    }
    if (revealCount >= events.length) {
      const t = setTimeout(() => {
        if (!doneCalled.current) {
          doneCalled.current = true;
          onDone();
        }
      }, HOLD_AFTER_MS);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => setRevealCount((c) => c + 1), STEP_REVEAL_MS);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [revealCount, events.length]);

  const task = result?.task_classified;
  const spec = SPECIALISTS.find((s) => s.id === task);
  const WinnerIcon = spec?.icon || IconSatellite;
  const winnerName = task ? WINNER_NAMES[task] || spec?.label || task : "";

  const plan = result?.metadata?.plan || {};
  const rejected = plan.rejected_specialists || [];
  // Show the specialist that actually ran; the plan only records what was
  // selected up front. They differ when a planned learned model was skipped.
  const specialistId = result?.metadata?.specialist_id || plan.selected_specialist_id;
  const plannedId = result?.metadata?.planned_specialist_id;
  const executionNote = result?.metadata?.execution_note;
  const usedFallback = result?.used_fallback;

  const visible = events.slice(0, revealCount);

  return (
    <div className="agent-screen fade-in">
      <div className="trace-replay">
        <div className="trace-replay-header">
          <div className="trace-winner">
            <span className="trace-winner-icon">
              <WinnerIcon size={22} />
            </span>
            <div>
              <div className="trace-winner-label">
                Specialist selected: <strong>{winnerName}</strong>
              </div>
              {specialistId && (
                <div className="trace-winner-sub">
                  {plannedId
                    ? `${fmtSpecialistId(specialistId)} (planned: ${fmtSpecialistId(plannedId)})`
                    : fmtSpecialistId(specialistId)}
                </div>
              )}
            </div>
          </div>

          {usedFallback && (
            <div className="fallback-badge">
              Used fallback
              {executionNote
                ? `: ${executionNote}`
                : plan.selection_reason
                  ? `: ${plan.selection_reason}`
                  : ""}
            </div>
          )}

          {rejected.length > 0 && (
            <div className="rejected-list">
              <span className="rejected-list-label">Not selected</span>
              {rejected.map((r) => (
                <span className="rejected-chip" key={r.id} title={r.reason}>
                  {fmtSpecialistId(r.id)}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="trace-timeline">
          <div className="trace-timeline-caption">Replay of recorded execution</div>
          {visible.map((e, i) => (
            <div
              className={`trace-step trace-step-${(e.status || "").toLowerCase()}`}
              key={`${e.step}-${i}`}
            >
              <span className="trace-step-icon">{STATUS_ICON[e.status] || "•"}</span>
              <div className="trace-step-body">
                <div className="trace-step-top">
                  <span className="trace-step-label">{fmtStepLabel(e.step)}</span>
                  <span className="trace-step-duration">
                    {e.duration_ms != null ? `${e.duration_ms} ms` : ""}
                  </span>
                </div>
                {e.detail && <div className="trace-step-detail">{e.detail}</div>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
