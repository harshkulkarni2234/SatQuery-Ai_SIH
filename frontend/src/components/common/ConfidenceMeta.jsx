export default function ConfidenceMeta({ value, source }) {
  return (
    <div className="meta-item">
      <span className="meta-key">Confidence</span>
      {value !== null && value !== undefined ? (
        <span className="meta-val">
          {value}
          {source && source !== "unavailable" && (
            <span className="meta-val-source"> {"—"} source: {source}</span>
          )}
        </span>
      ) : (
        <span className="meta-val meta-unavail">Unavailable</span>
      )}
    </div>
  );
}
