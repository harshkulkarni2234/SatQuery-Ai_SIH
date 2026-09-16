"""Downloadable analysis report (Phase C5).

build_report_data() reconstructs everything from the STORED query/result/
image rows — it never re-runs a specialist. render_report_pdf() renders
that same data as a PDF; GET /query/{id}/report.json returns it directly,
so the two endpoints are guaranteed to agree (see routes/report.py).
"""

from __future__ import annotations

import io
import os
import uuid
from datetime import datetime, timezone

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy.orm import Session

from app.models import Image, Query, QueryResult
from app.routes.images import _image_to_metadata
from app.services.raster_ingest import load_rgb_preview
from app.services.registry import list_specialists

LIMITATIONS = [
    "Pixel-level change detection measures visual differences between pixels; it "
    "does not by itself constitute verified semantic land-cover change.",
    "Optical/SAR co-registration is only ever reported as \"verified\" when the "
    "backend can check matching CRS and pixel grids from the files themselves; "
    "otherwise it is reported as \"assumed\" or \"unverified\", never assumed silently.",
    "Deterministic computer-vision specialists (grounding, change detection, "
    "cross-modal fusion) are not trained/learned models. Where a learned model "
    "was actually used, that is stated explicitly below along with its version.",
]


def build_report_data(db: Session, query_id: uuid.UUID) -> dict | None:
    query_record = db.query(Query).filter(Query.id == query_id).first()
    if not query_record:
        return None

    result_record = db.query(QueryResult).filter(QueryResult.query_id == query_id).first()

    images = db.query(Image).filter(Image.id.in_(query_record.image_ids)).all()
    by_id = {img.id: img for img in images}
    ordered_images = [by_id[i] for i in query_record.image_ids if i in by_id]

    inputs = []
    for img in ordered_images:
        meta = _image_to_metadata(img)
        inputs.append(
            {
                "filename": img.filename,
                "modality": img.modality,
                "capture_date": str(img.capture_date) if img.capture_date else None,
                "capture_date_source": img.acquisition_date_source or "unknown",
                "crs": img.crs,
                "resolution_m": meta.resolution[0] if meta and meta.resolution else None,
                "bounds_wgs84": list(meta.bounds_wgs84) if meta and meta.bounds_wgs84 else None,
                "band_count": img.band_count,
                "is_georeferenced": bool(img.is_georeferenced),
                "file_path": img.file_path,
                "file_exists": bool(img.file_path and os.path.isfile(img.file_path)),
            }
        )

    meta = (result_record.metadata_ or {}) if result_record else {}
    plan = meta.get("plan") or {}
    specialist_id = query_record.specialist_id or plan.get("selected_specialist_id")
    spec = next((s for s in list_specialists() if s.id == specialist_id), None)

    overlay_path = meta.get("overlay_path")
    change_mask_path = result_record.change_mask_path if result_record else None

    trace_events = query_record.trace_events or meta.get("trace_events") or []
    compatibility = query_record.compatibility or plan.get("compatibility")

    return {
        "report_id": f"SQ-{str(query_record.id)[:8].upper()}",
        "query_id": str(query_record.id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_text": query_record.query_text,
        "inputs": inputs,
        "compatibility": compatibility,
        "task": query_record.task_classified,
        "specialist": (
            {
                "id": specialist_id,
                "name": spec.name if spec else None,
                "kind": spec.kind if spec else None,
                "version": (spec.version if spec else None) or query_record.model_version,
                "is_rs_adapted": spec.is_rs_adapted if spec else None,
            }
            if specialist_id
            else None
        ),
        "used_fallback": bool(query_record.used_fallback),
        "fallback_reason": plan.get("selection_reason") if query_record.used_fallback else None,
        "answer_text": result_record.answer_text if result_record else None,
        "evidence": {
            "bounding_boxes": result_record.bounding_boxes if result_record else None,
            "change_mask_path": change_mask_path,
            "change_mask_exists": bool(change_mask_path and os.path.isfile(change_mask_path)),
            "overlay_path": overlay_path,
            "overlay_exists": bool(overlay_path and os.path.isfile(overlay_path)),
        },
        "confidence_score": query_record.confidence_score,
        "confidence_source": query_record.confidence_source or "unavailable",
        "execution_trace": trace_events,
        "warnings": query_record.warnings or [],
        "limitations": LIMITATIONS,
    }


def _thumbnail_reader(path: str, max_side: int = 320):
    """Best-effort small PNG thumbnail for embedding in the PDF. Returns None
    (never raises) if the file is missing or unreadable — the caller notes
    "evidence image not available" instead."""
    if not path or not os.path.isfile(path):
        return None
    try:
        arr = load_rgb_preview(path, max_side=max_side)
        img = PILImage.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:
        return None


def render_report_pdf(data: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
    )
    styles = getSampleStyleSheet()
    h1 = styles["Title"]
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=4)
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.5, textColor=colors.grey)
    cell = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=7.5, leading=9)

    def _c(text):
        """Wrap table-cell text in a Paragraph so long text wraps inside the
        cell instead of overflowing into the next column."""
        return Paragraph(str(text).replace("&", "&amp;").replace("<", "&lt;"), cell)

    story = []
    story.append(Paragraph("SatQuery AI Analysis Report", h1))
    story.append(
        Paragraph(
            f"Report ID: {data['report_id']} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Generated: {data['generated_at']} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Query ID: {data['query_id']}",
            small,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Query", h2))
    story.append(Paragraph(data["query_text"] or "(none)", body))

    story.append(Paragraph("Inputs", h2))
    if data["inputs"]:
        rows = [["Filename", "Modality", "Capture date", "CRS", "Resolution (m)", "Bands", "Extent (WGS84)"]]
        for inp in data["inputs"]:
            rows.append(
                [
                    _c(inp["filename"]),
                    _c(inp["modality"]),
                    _c(f"{inp['capture_date'] or 'Unknown'} ({inp['capture_date_source']})"),
                    _c(inp["crs"] or "N/A"),
                    _c(f"{inp['resolution_m']:.4g}" if inp["resolution_m"] else "N/A"),
                    _c(str(inp["band_count"] or "N/A")),
                    _c(
                        ", ".join(f"{v:.3f}" for v in inp["bounds_wgs84"])
                        if inp["bounds_wgs84"]
                        else "N/A"
                    ),
                ]
            )
        table = Table(rows, hAlign="LEFT", colWidths=[3.3 * cm, 2 * cm, 3.2 * cm, 2.2 * cm, 2.2 * cm, 1.5 * cm, 4 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)

        thumbs = []
        for inp in data["inputs"]:
            reader = _thumbnail_reader(inp["file_path"]) if inp["file_exists"] else None
            if reader:
                thumbs.append(RLImage(reader, width=4 * cm, height=4 * cm))
            else:
                thumbs.append(Paragraph("(evidence image not available)", small))
        if thumbs:
            story.append(Spacer(1, 6))
            thumb_table = Table([thumbs])
            story.append(thumb_table)
    else:
        story.append(Paragraph("No input images recorded.", body))

    if data["compatibility"]:
        story.append(Paragraph("Compatibility", h2))
        compat = data["compatibility"]
        rows = [["Check", "Status", "Detail"]]
        for c in compat.get("checks", []):
            rows.append([_c(c["name"].replace("_", " ")), _c(c["status"]), _c(c["detail"])])
        table = Table(rows, hAlign="LEFT", colWidths=[3.5 * cm, 2 * cm, 10 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ]
            )
        )
        story.append(table)
        if compat.get("overlap_ratio") is not None:
            story.append(Paragraph(f"Overlap ratio: {compat['overlap_ratio']:.2f}", small))
        if compat.get("coregistration"):
            story.append(Paragraph(f"Co-registration: {compat['coregistration']}", small))

    story.append(Paragraph("Task &amp; Specialist", h2))
    story.append(Paragraph(f"Task: {data['task'] or 'N/A'}", body))
    if data["specialist"]:
        s = data["specialist"]
        kind_label = {"learned_model": "Learned model", "deterministic_cv": "Deterministic CV", "rules": "Rules-based"}.get(
            s["kind"], s["kind"] or "Unknown"
        )
        story.append(
            Paragraph(
                f"Specialist: {s['name'] or s['id']} ({kind_label}, version {s['version'] or 'unknown'}"
                f"{', RS-adapted' if s['is_rs_adapted'] else ''})",
                body,
            )
        )
    if data["used_fallback"]:
        story.append(Paragraph(f"<b>Fallback used:</b> {data['fallback_reason'] or 'reason not recorded'}", body))

    story.append(Paragraph("Answer", h2))
    story.append(Paragraph(data["answer_text"] or "(no answer recorded)", body))

    ev = data["evidence"]
    if ev.get("bounding_boxes") or ev.get("change_mask_exists") or ev.get("overlay_exists"):
        story.append(Paragraph("Visual Evidence", h2))
        if ev.get("overlay_exists"):
            reader = _thumbnail_reader(ev["overlay_path"], max_side=400)
            if reader:
                story.append(Paragraph("Change overlay:", small))
                story.append(RLImage(reader, width=8 * cm, height=8 * cm))
            else:
                story.append(Paragraph("(evidence image not available)", small))
        elif ev.get("overlay_path"):
            story.append(Paragraph("Change overlay: (evidence image not available)", small))
        if ev.get("change_mask_exists"):
            reader = _thumbnail_reader(ev["change_mask_path"], max_side=400)
            if reader:
                story.append(Paragraph("Change mask:", small))
                story.append(RLImage(reader, width=8 * cm, height=8 * cm))
        if ev.get("bounding_boxes"):
            story.append(Paragraph(f"Bounding boxes: {ev['bounding_boxes']}", small))

    story.append(Paragraph("Confidence", h2))
    if data["confidence_score"] is not None:
        story.append(Paragraph(f"{data['confidence_score']:.3f} (source: {data['confidence_source']})", body))
    else:
        story.append(Paragraph("Unavailable", body))

    if data["execution_trace"]:
        story.append(Paragraph("Execution Trace", h2))
        rows = [["Step", "Status", "Detail", "Duration"]]
        for e in data["execution_trace"]:
            rows.append(
                [
                    _c(e.get("step", "")),
                    _c(e.get("status", "")),
                    _c(e.get("detail", "") or ""),
                    _c(f"{e.get('duration_ms', 0)} ms"),
                ]
            )
        table = Table(rows, hAlign="LEFT", colWidths=[3.2 * cm, 2 * cm, 8 * cm, 2 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)

    if data["warnings"]:
        story.append(Paragraph("Warnings", h2))
        for w in data["warnings"]:
            story.append(Paragraph(f"• {w}", body))

    story.append(Paragraph("Limitations", h2))
    for note in data["limitations"]:
        story.append(Paragraph(f"• {note}", small))

    doc.build(story)
    return buf.getvalue()
