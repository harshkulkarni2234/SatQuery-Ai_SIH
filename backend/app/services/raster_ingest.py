"""Raster ingestion: real metadata extraction from uploaded image files.

Supports GeoTIFF/TIFF (via rasterio, full georeferencing) and PNG/JPG/BMP
(via Pillow, no georeferencing). JP2 is attempted via rasterio and degrades
gracefully with a warning if unsupported by the local GDAL build. Never
guesses values it can't verify from the file: missing CRS/bounds/dates stay
None with an explicit warning, per AGENTS.md rule 2.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime
from typing import Optional

import numpy as np
import rasterio
from PIL import Image as PILImage, UnidentifiedImageError
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds

from app.contracts import RasterMetadata

GEOSPATIAL_EXTENSIONS = {".tif", ".tiff", ".jp2"}
PLAIN_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

MAX_RASTER_PIXELS = int(os.environ.get("MAX_RASTER_PIXELS", 100_000_000))  # ~100 MP

# Tag keys different pipelines use for acquisition date; checked in order.
# Not exhaustive — if a producer uses something else, the date stays "unknown"
# rather than being guessed.
_DATE_TAG_KEYS = (
    "TIFFTAG_DATETIME",
    "ACQUISITION_DATE",
    "ACQUISITIONDATETIME",
    "DATE_ACQUIRED",
    "SENSING_TIME",
)

# TIFF standard datetime format is "YYYY:MM:DD HH:MM:SS"; also accept ISO dates.
_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y:%m:%d")


class RasterIngestError(ValueError):
    """Raised for any file that can't be read as a valid image/raster."""


def _parse_tag_date(raw: str) -> Optional[date]:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _find_file_metadata_date(tags: dict) -> Optional[date]:
    for key in _DATE_TAG_KEYS:
        raw = tags.get(key)
        if raw:
            parsed = _parse_tag_date(str(raw))
            if parsed:
                return parsed
    return None


def _extract_geospatial_metadata(
    path: str, ext: str, user_capture_date: Optional[date], warnings: list[str]
) -> RasterMetadata:
    try:
        with rasterio.open(path) as ds:
            width, height = ds.width, ds.height
            band_count = ds.count
            dtype = ds.dtypes[0] if ds.dtypes else None
            nodata = ds.nodata
            crs = ds.crs.to_string() if ds.crs else None
            transform = tuple(ds.transform)[:6] if ds.transform else None
            bounds = tuple(ds.bounds) if ds.crs else None
            resolution = (abs(ds.transform.a), abs(ds.transform.e)) if ds.transform else None

            bounds_wgs84 = None
            if ds.crs and ds.bounds:
                try:
                    bounds_wgs84 = tuple(
                        transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
                    )
                except Exception:
                    warnings.append("Could not reproject bounds to WGS84 from source CRS")

            if not ds.crs:
                warnings.append("File has no CRS; treating as non-georeferenced for spatial checks")

            tiff_tags = ds.tags()
            file_date = _find_file_metadata_date(tiff_tags)

            if user_capture_date is not None:
                acquisition_date = user_capture_date
                acquisition_date_source = "user"
            elif file_date is not None:
                acquisition_date = file_date
                acquisition_date_source = "file_metadata"
            else:
                acquisition_date = None
                acquisition_date_source = "unknown"
                warnings.append("No acquisition date found in file metadata or user input")

            pixel_count = width * height
            if pixel_count > MAX_RASTER_PIXELS:
                warnings.append(
                    f"Raster has {pixel_count:,} pixels, exceeding MAX_RASTER_PIXELS "
                    f"({MAX_RASTER_PIXELS:,}); only metadata was read, no pixel data"
                )

            return RasterMetadata(
                format=ds.driver or ext.lstrip(".").upper(),
                width=width,
                height=height,
                band_count=band_count,
                dtype=dtype,
                crs=crs,
                bounds=bounds,
                bounds_wgs84=bounds_wgs84,
                resolution=resolution,
                transform=transform,
                nodata=nodata,
                acquisition_date=acquisition_date,
                acquisition_date_source=acquisition_date_source,
                is_georeferenced=bool(ds.crs and ds.transform and not ds.transform.is_identity),
                file_size_bytes=os.path.getsize(path),
                warnings=warnings,
            )
    except RasterioIOError as exc:
        if ext == ".jp2":
            raise RasterIngestError(
                "JPEG2000 file could not be read by the local GDAL build "
                "(JP2 driver may be unavailable). Try a GeoTIFF instead."
            ) from exc
        # GDAL's own error text embeds the server-side file path; keep that
        # out of the client-facing message and log it server-side instead.
        logging.getLogger(__name__).warning("Raster read failed for %s: %s", path, exc)
        raise RasterIngestError(
            f"File could not be read as a valid {ext.lstrip('.').upper()} raster. "
            "It may be corrupt or in an unsupported format."
        ) from exc


def _extract_plain_image_metadata(path: str, ext: str, user_capture_date: Optional[date]) -> RasterMetadata:
    try:
        with PILImage.open(path) as img:
            img.verify()
        with PILImage.open(path) as img:
            width, height = img.size
            mode = img.mode
            band_count = len(mode) if mode not in ("P",) else 1
            dtype = "uint8"
    except (UnidentifiedImageError, OSError) as exc:
        logging.getLogger(__name__).warning("Image read failed for %s: %s", path, exc)
        raise RasterIngestError(
            f"File could not be read as a valid {ext.lstrip('.').upper()} image. "
            "It may be corrupt or in an unsupported format."
        ) from exc

    warnings: list[str] = ["Image format has no georeferencing; spatial checks are skipped"]
    if user_capture_date is not None:
        acquisition_date = user_capture_date
        acquisition_date_source = "user"
    else:
        acquisition_date = None
        acquisition_date_source = "unknown"
        warnings.append("No acquisition date available (format carries no date metadata)")

    return RasterMetadata(
        format=ext.lstrip(".").upper(),
        width=width,
        height=height,
        band_count=band_count,
        dtype=dtype,
        crs=None,
        bounds=None,
        bounds_wgs84=None,
        resolution=None,
        transform=None,
        nodata=None,
        acquisition_date=acquisition_date,
        acquisition_date_source=acquisition_date_source,
        is_georeferenced=False,
        file_size_bytes=os.path.getsize(path),
        warnings=warnings,
    )


def extract_metadata(path: str, user_capture_date: Optional[date] = None) -> RasterMetadata:
    """Extract real metadata from an image/raster file. Raises RasterIngestError
    for anything that can't be read; never fabricates a value it can't verify."""
    if not os.path.isfile(path):
        raise RasterIngestError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in GEOSPATIAL_EXTENSIONS:
        return _extract_geospatial_metadata(path, ext, user_capture_date, [])
    if ext in PLAIN_IMAGE_EXTENSIONS:
        return _extract_plain_image_metadata(path, ext, user_capture_date)

    raise RasterIngestError(f"Unsupported file extension: {ext}")


def load_rgb_preview(path: str, max_side: int = 2048) -> np.ndarray:
    """Load a uint8 RGB (or grayscale) preview array for specialists to consume.

    Handles 16-bit Sentinel-style data with a percentile stretch, downsamples
    to max_side, and for multispectral rasters (band_count >= 4) prefers
    bands 4,3,2 (1-indexed, i.e. red/green/blue in typical Sentinel-2 band
    ordering) — documented assumption, not verified per-file band semantics.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in GEOSPATIAL_EXTENSIONS:
        with rasterio.open(path) as ds:
            band_count = ds.count
            if band_count >= 4:
                band_indices = (4, 3, 2)
            elif band_count == 3:
                band_indices = (1, 2, 3)
            else:
                band_indices = (1,)

            scale = min(1.0, max_side / max(ds.width, ds.height))
            out_width = max(1, int(ds.width * scale))
            out_height = max(1, int(ds.height * scale))

            arr = ds.read(
                list(band_indices),
                out_shape=(len(band_indices), out_height, out_width),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            arr = np.moveaxis(arr, 0, -1)  # (H, W, C)
    else:
        with PILImage.open(path) as img:
            img = img.convert("RGB")
            scale = min(1.0, max_side / max(img.width, img.height))
            if scale < 1.0:
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
            arr = np.array(img)

    if arr.dtype != np.uint8:
        arr = _percentile_stretch_to_uint8(arr)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    return arr


def _percentile_stretch_to_uint8(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float64)
    out = np.zeros_like(arr, dtype=np.uint8)
    channels = arr.shape[-1] if arr.ndim == 3 else 1
    for c in range(channels):
        band = arr[..., c] if arr.ndim == 3 else arr
        lo, hi = np.percentile(band, [low, high])
        if hi <= lo:
            stretched = np.zeros_like(band)
        else:
            stretched = np.clip((band - lo) / (hi - lo), 0, 1) * 255
        if arr.ndim == 3:
            out[..., c] = stretched.astype(np.uint8)
        else:
            out = stretched.astype(np.uint8)
    return out
