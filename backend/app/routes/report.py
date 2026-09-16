import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.report import build_report_data, render_report_pdf

router = APIRouter(tags=["report"])


@router.get("/query/{query_id}/report.pdf")
def get_report_pdf(query_id: uuid.UUID, db: Session = Depends(get_db)):
    data = build_report_data(db, query_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Query not found")
    pdf_bytes = render_report_pdf(data)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{data["report_id"]}.pdf"'},
    )


@router.get("/query/{query_id}/report.json")
def get_report_json(query_id: uuid.UUID, db: Session = Depends(get_db)):
    data = build_report_data(db, query_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return data
