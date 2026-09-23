export default function Stat({ label, value }) {
  if (value === undefined || value === null || value === "") return null;
  // Short values (counts, percentages, areas) read as telemetry in monospace.
  // Long prose values — e.g. "Geographic reprojection (real CRS/grid
  // alignment)" — need normal type, or they wrap into an unreadable column.
  const isProse = String(value).length > 16;
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={isProse ? "stat-value stat-value-prose" : "stat-value"}>{value}</span>
    </div>
  );
}
