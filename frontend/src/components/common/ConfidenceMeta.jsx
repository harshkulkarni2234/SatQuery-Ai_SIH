export default function ConfidenceMeta({ value }) {
  return (
    <div className="meta-item">
      <span className="meta-key">Confidence</span>
      {value !== null ? (
        <span className="meta-val">{value}</span>
      ) : (
        <span className="meta-val meta-unavail">Unavailable for this analysis</span>
      )}
    </div>
  );
}
