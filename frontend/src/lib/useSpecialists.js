import { useEffect, useState } from "react";
import { getSpecialists } from "../api.js";

// The registry rarely changes within a session, so cache the one successful
// fetch at module scope instead of re-requesting per result screen.
let cache = null;

export function useSpecialists() {
  const [specialists, setSpecialists] = useState(cache || []);

  useEffect(() => {
    if (cache) return;
    let cancelled = false;
    getSpecialists()
      .then((data) => {
        if (cancelled) return;
        cache = data;
        setSpecialists(data);
      })
      .catch(() => {
        // Registry lookup is enrichment only (badge label) — never block
        // or error the result screen if it fails.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return specialists;
}

export function findSpecialist(specialists, id) {
  return specialists.find((s) => s.id === id) || null;
}

const KIND_LABEL = {
  learned_model: "Learned model",
  deterministic_cv: "Deterministic CV",
  rules: "Rules-based",
};

export function specialistBadgeLabel(spec) {
  if (!spec) return null;
  if (spec.is_rs_adapted) return `RS-adapted (${spec.name}, narrow)`;
  return KIND_LABEL[spec.kind] || spec.kind;
}
