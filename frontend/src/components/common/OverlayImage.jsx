import { useState } from "react";

export default function OverlayImage({ src, boxes, labels, labelPrefix, showBoxes, alt }) {
  const [size, setSize] = useState(null);
  const labelText = (i) => (labels && labels[i]) || `${labelPrefix || "Region"} ${i + 1}`;
  return (
    <div className="overlay-img">
      <img
        src={src}
        alt={alt}
        onLoad={(e) =>
          setSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })
        }
      />
      {size &&
        showBoxes &&
        boxes.map((b, i) => (
          <span
            key={i}
            className="overlay-box"
            style={{
              left: `${(b[0] / size.w) * 100}%`,
              top: `${(b[1] / size.h) * 100}%`,
              width: `${((b[2] - b[0]) / size.w) * 100}%`,
              height: `${((b[3] - b[1]) / size.h) * 100}%`,
            }}
          >
            <span className="overlay-box-label">
              {labelText(i)}
            </span>
          </span>
        ))}
    </div>
  );
}
