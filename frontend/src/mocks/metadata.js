// Mock RasterMetadata shapes matching docs/CONTRACTS.md, for local UI dev
// without a running backend. Real data (Gate G1, Phase A2) has been merged,
// so these are opt-in only via VITE_USE_MOCKS=true — never used by default.

export const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === "true";

export const MOCK_GEOREFERENCED_METADATA = {
  format: "GTiff",
  width: 512,
  height: 512,
  band_count: 4,
  dtype: "uint16",
  crs: "EPSG:32643",
  bounds: [500000.0, 3100000.0, 505120.0, 3105120.0],
  bounds_wgs84: [76.9, 28.02, 76.95, 28.07],
  resolution: [10.0, 10.0],
  transform: [10.0, 0.0, 500000.0, 0.0, -10.0, 3105120.0],
  nodata: 0.0,
  acquisition_date: "2023-06-14",
  acquisition_date_source: "file_metadata",
  is_georeferenced: true,
  file_size_bytes: 2097152,
  warnings: [],
};

export const MOCK_NON_GEOREFERENCED_METADATA = {
  format: "PNG",
  width: 512,
  height: 384,
  band_count: 3,
  dtype: "uint8",
  crs: null,
  bounds: null,
  bounds_wgs84: null,
  resolution: null,
  transform: null,
  nodata: null,
  acquisition_date: null,
  acquisition_date_source: "unknown",
  is_georeferenced: false,
  file_size_bytes: 184320,
  warnings: [
    "Image format has no georeferencing; spatial checks are skipped",
    "No acquisition date available (format carries no date metadata)",
  ],
};

export function mockUploadResponse(file, modality) {
  const isGeoTiff = /\.tif?f$/i.test(file.name);
  return {
    image_id: `mock-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    filename: file.name,
    modality,
    crs: isGeoTiff ? MOCK_GEOREFERENCED_METADATA.crs : null,
    resolution_m: isGeoTiff ? MOCK_GEOREFERENCED_METADATA.resolution[0] : null,
    metadata: isGeoTiff ? MOCK_GEOREFERENCED_METADATA : MOCK_NON_GEOREFERENCED_METADATA,
  };
}
