import {
  fmtCrs,
  fmtResolution,
  fmtExtent,
  fmtAcquisitionDate,
} from "../../lib/format.js";

// Phase C7 item 3 — a compact per-input metadata panel on the results
// page itself (previously this only ever showed on the upload/staging
// screen, not carried through to the report view).
export default function InputMetadataPanel({ images }) {
  if (!images || images.length === 0) return null;
  return (
    <div className="input-metadata-panel">
      {images.map((img) => (
        <div className="input-metadata-card" key={img.key || img.imageId}>
          <div className="input-metadata-name">{img.file?.name || img.filename}</div>
          <div className="input-metadata-grid">
            <div className="meta-item">
              <span className="meta-key">Modality</span>
              <span className="meta-val">{img.modality}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Capture date</span>
              <span className="meta-val">{fmtAcquisitionDate(img.metadata)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">CRS</span>
              <span className="meta-val">{fmtCrs(img.metadata)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Resolution</span>
              <span className="meta-val">{fmtResolution(img.metadata)}</span>
            </div>
            <div className="meta-item input-metadata-extent">
              <span className="meta-key">Extent (WGS84)</span>
              <span className="meta-val">{fmtExtent(img.metadata)}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
