// Always shown when non-empty (Phase C6) — never hidden behind a toggle,
// since a warning is exactly the kind of thing a user shouldn't have to
// go looking for.
export default function WarningsBanner({ warnings }) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <div className="warnings-banner">
      <span className="warnings-banner-label">Warnings</span>
      <ul>
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </div>
  );
}
