import { IconCheck, IconTrash } from "../icons/index.jsx";

export default function ImageCard({ item, index, onChange, onRemove }) {
  return (
    <div className="image-card">
      <div className="image-preview">
        <img src={item.url} alt={item.file.name} />
        {item.imageId && (
          <span className="image-uploaded">
            <IconCheck size={11} /> uploaded
          </span>
        )}
        <button
          className="image-remove"
          onClick={onRemove}
          title="Remove image"
          aria-label="Remove image"
        >
          <IconTrash size={13} />
        </button>
      </div>
      <div className="image-body">
        <div className="image-name" title={item.file.name}>
          {item.file.name}
        </div>
        <div className="image-controls">
          <label className="control-group">
            <span className="control-label">Modality</span>
            <select
              aria-label="Modality"
              value={item.modality}
              onChange={(e) => onChange(index, { modality: e.target.value })}
            >
              <option value="OPTICAL">OPTICAL</option>
              <option value="SAR">SAR</option>
            </select>
          </label>
        </div>
      </div>
    </div>
  );
}
