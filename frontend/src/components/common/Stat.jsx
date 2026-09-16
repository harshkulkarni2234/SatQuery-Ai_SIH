export default function Stat({ label, value }) {
  if (value === undefined || value === null || value === "") return null;
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}
