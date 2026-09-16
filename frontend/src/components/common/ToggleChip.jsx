export default function ToggleChip({ active, label, onClick }) {
  return (
    <button
      className={`toggle-chip${active ? " active" : ""}`}
      onClick={onClick}
      aria-pressed={active}
    >
      {label}
    </button>
  );
}
