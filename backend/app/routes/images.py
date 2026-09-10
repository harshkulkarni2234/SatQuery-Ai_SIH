import os
import uuid
from datetime import date

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Image
from app.schemas import ImageUploadResponse

router = APIRouter(prefix="/images", tags=["images"])

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".jp2"}
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploaded_images")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


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
    with open(file_path, "wb") as f:
        content = file.file.read()
        f.write(content)

    image = Image(
        filename=file.filename,
        modality=modality_upper,
        capture_date=capture_date,
        file_path=file_path,
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
    )
