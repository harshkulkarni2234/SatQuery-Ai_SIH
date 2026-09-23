import { useEffect, useState } from "react";
import { getHealth } from "../../api.js";

// Live system status read from the backend's own /health endpoint.
// Nothing here is assumed: until a real response arrives every service shows
// "unknown", and a failed request keeps it that way.
const SERVICES = [
  { key: "database", label: "Database", up: (v) => v === "up" },
  { key: "vqa_worker", label: "VQA model", up: (v) => v === "up" },
  { key: "learned_change_model", label: "Change model", up: (v) => v === "available" },
];

export default function SystemStatus() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const poll = () =>
      getHealth()
        .then((h) => !cancelled && setHealth(h))
        .catch(() => !cancelled && setHealth(null));
    poll();
    const t = setInterval(poll, 20000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  return (
    <div className="sys-status" title="Live status from the backend /health endpoint">
      {SERVICES.map(({ key, label, up }) => {
        const value = health ? health[key] : undefined;
        const state = value === undefined ? "unknown" : up(value) ? "up" : "down";
        return (
          <span key={key} className={`sys-pill sys-${state}`}>
            <span className="sys-dot" />
            {label}
            <span className="sys-state">
              {state === "unknown" ? "unknown" : state === "up" ? "online" : "offline"}
            </span>
          </span>
        );
      })}
    </div>
  );
}
