import ToggleChip from "./ToggleChip.jsx";

export default function ViewerToggles({ options, value, onChange, label }) {
  return (
    <div className="viewer-toggles">
      <span className="viewer-toggles-label">{label}</span>
      <div className="viewer-toggles-group">
        {options.map((o) => (
          <ToggleChip
            key={o.value}
            active={value === o.value}
            label={o.label}
            onClick={() => onChange(o.value)}
          />
        ))}
      </div>
    </div>
  );
}
