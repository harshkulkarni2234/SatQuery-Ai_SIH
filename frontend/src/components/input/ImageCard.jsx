import { IconCheck, IconTrash } from "../icons/index.jsx";
import {
  fmtDimensions,
  fmtBands,
  fmtCrs,
  fmtResolution,
  fmtExtent,
  fmtAcquisitionDate,
  fmtGeoreferenced,
  fmtFormat,
} from "../../lib/format.js";

export default function ImageCard({ item, index, onChange, onRemove }) {
  const meta = item.metadata;
  return (
    <div className="image-card">
      <div className="image-preview">
        <img src={item.url} alt={item.file.name} />
        {item.uploading && (
          <span className="image-uploaded image-uploading">
            <span className="spinner-accent" /> uploading
          </span>
        )}
        {!item.uploading && item.imageId && (
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
          <label className="control-group">
            <span className="control-label">Capture date (optional)</span>
            <input
              type="date"
              aria-label="Capture date"
              value={item.captureDate || ""}
              onChange={(e) => onChange(index, { captureDate: e.target.value || null })}
            />
          </label>
        </div>

        {item.uploadError && (
          <div className="error-banner image-error">{item.uploadError}</div>
        )}

        {meta && (
          <div className="image-metadata">
            <div className="meta-item">
              <span className="meta-key">Format</span>
              <span className="meta-val">{fmtFormat(meta)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Dimensions</span>
              <span className="meta-val">{fmtDimensions(meta)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Bands</span>
              <span className="meta-val">{fmtBands(meta)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Capture date</span>
              <span className="meta-val">{fmtAcquisitionDate(meta)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Georeferenced</span>
              <span className="meta-val">{fmtGeoreferenced(meta)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">CRS</span>
              <span className="meta-val">{fmtCrs(meta)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Resolution</span>
              <span className="meta-val">{fmtResolution(meta)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Extent (WGS84)</span>
              <span className="meta-val">{fmtExtent(meta)}</span>
            </div>
            {meta.warnings && meta.warnings.length > 0 && (
              <div className="meta-warnings">
                {meta.warnings.map((w, i) => (
                  <div className="meta-warning" key={i}>
                    {w}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
