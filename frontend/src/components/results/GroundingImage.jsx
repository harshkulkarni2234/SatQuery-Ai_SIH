import { useState } from "react";

export default function GroundingImage({ src, boxes, labels, labelPrefix }) {
  const [size, setSize] = useState(null);
  const labelText = (i) => (labels && labels[i]) || `${labelPrefix} ${i + 1}`;
  return (
    <div className="geo-wrap">
      <div className="geo-image">
        <img
          src={src}
          alt="Analysis image"
          onLoad={(e) =>
            setSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })
          }
        />
        {size &&
          boxes.map((b, i) => (
            <span
              key={i}
              className="geo-box"
              style={{
                left: `${(b[0] / size.w) * 100}%`,
                top: `${(b[1] / size.h) * 100}%`,
                width: `${((b[2] - b[0]) / size.w) * 100}%`,
                height: `${((b[3] - b[1]) / size.h) * 100}%`,
              }}
              title={labelText(i)}
            >
              <span className="geo-box-label">
                {labelText(i)}
              </span>
            </span>
          ))}
      </div>
    </div>
  );
}
