import { IconSatellite, IconCheck } from "../icons/index.jsx";
import { SPECIALISTS, ORBIT_POS, WINNER_NAMES } from "../../constants/specialists.js";

/* phase: "traversing" | "selected" | "winner" | "preparing"
   traversing – moving-dot cycles the orbit (unchanged).
   selected   – lock-on: winning node highlights, checkmark, label.
   winner     – others fade, winner enlarges and moves to centre,
                "Model selected: <name>" + "Preparing analysis report…".
   preparing  – winner held, "Specialist execution complete ✓". */

export default function AgentSelection({ orbitStep, phase, selectedId, animLabel }) {
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
            ? "Selecting the best specialist…"
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
            {isSelected ? "Specialist selected ✓" : animLabel}
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
              {"Analysis ready ✓"}
            </div>
          ) : (
            <div className="winner-preparing">
              <span className="spinner-accent" />
              {"Preparing analysis report…"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
