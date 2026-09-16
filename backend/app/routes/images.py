import os
import uuid
from datetime import date

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.contracts import RasterMetadata
from app.database import get_db
from app.models import Image
from app.schemas import ImageDetailResponse, ImageUploadResponse
from app.services.raster_ingest import RasterIngestError, extract_metadata

router = APIRouter(prefix="/images", tags=["images"])

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".jp2"}
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploaded_images")

# Phase A8: previously unbounded — file.file.read() would load the entire
# upload into memory regardless of size. Read in chunks and reject (413)
# once this many bytes have been seen, rather than trusting a possibly-
# absent Content-Length header.
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", 200 * 1024 * 1024))  # 200 MB
_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _image_to_metadata(image: Image) -> RasterMetadata | None:
    if image.file_format is None:
        return None

    # resolution (x, y) is derived from the stored transform when available,
    # since only a single resolution_m (x) column exists on the images table.
    if image.transform:
        resolution = (abs(image.transform[0]), abs(image.transform[4]))
    elif image.resolution_m is not None:
        resolution = (image.resolution_m, image.resolution_m)
    else:
        resolution = None

    return RasterMetadata(
        format=image.file_format,
        width=image.width,
        height=image.height,
        band_count=image.band_count,
        dtype=image.dtype,
        crs=image.crs,
        bounds=tuple(image.bounds) if image.bounds else None,
        bounds_wgs84=tuple(image.bounds_wgs84) if image.bounds_wgs84 else None,
        resolution=resolution,
        transform=tuple(image.transform) if image.transform else None,
        nodata=image.nodata,
        acquisition_date=image.capture_date,
        acquisition_date_source=image.acquisition_date_source or "unknown",
        is_georeferenced=bool(image.is_georeferenced),
        file_size_bytes=image.file_size_bytes,
        warnings=image.metadata_warnings or [],
    )


@router.post("/upload", response_model=ImageUploadResponse)
def upload_image(
    file: UploadFile = File(...),
    modality: str = Form(...),
    capture_date: date | None = Form(None),
    db: Session = Depends(get_db),
):
    modality_upper = modality.upper()
    if modality_upper not in ("OPTICAL", "SAR"):
        raise HTTPException(status_code=400, detail="modality must be 'OPTICAL' or 'SAR'")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' not allowed. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    _ensure_data_dir()
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(DATA_DIR, stored_name)
    total_bytes = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := file.file.read(_UPLOAD_CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds maximum upload size of "
                            f"{MAX_UPLOAD_SIZE_BYTES:,} bytes"
                        ),
                    )
                f.write(chunk)
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    try:
        metadata = extract_metadata(file_path, user_capture_date=capture_date)
    except RasterIngestError as exc:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(exc))

    image = Image(
        filename=file.filename,
        modality=modality_upper,
        capture_date=metadata.acquisition_date,
        file_path=file_path,
        crs=metadata.crs,
        bbox_coords=list(metadata.bounds) if metadata.bounds else None,
        resolution_m=metadata.resolution[0] if metadata.resolution else None,
        width=metadata.width,
        height=metadata.height,
        band_count=metadata.band_count,
        dtype=metadata.dtype,
        bounds=list(metadata.bounds) if metadata.bounds else None,
        bounds_wgs84=list(metadata.bounds_wgs84) if metadata.bounds_wgs84 else None,
        transform=list(metadata.transform) if metadata.transform else None,
        nodata=metadata.nodata,
        file_format=metadata.format,
        is_georeferenced=metadata.is_georeferenced,
        acquisition_date_source=metadata.acquisition_date_source,
        file_size_bytes=metadata.file_size_bytes,
        metadata_warnings=metadata.warnings or None,
    )
    db.add(image)
    db.commit()
    db.refresh(image)

    return ImageUploadResponse(
        image_id=image.id,
        filename=image.filename,
        modality=image.modality,
        crs=image.crs,
        resolution_m=image.resolution_m,
        metadata=metadata,
    )


@router.get("/{image_id}", response_model=ImageDetailResponse)
def get_image(image_id: uuid.UUID, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

    return ImageDetailResponse(
        image_id=image.id,
        filename=image.filename,
        modality=image.modality,
        capture_date=image.capture_date,
        crs=image.crs,
        resolution_m=image.resolution_m,
        metadata=_image_to_metadata(image),
    )
