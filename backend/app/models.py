import uuid
from datetime import date, datetime
from sqlalchemy import Boolean, Column, String, Text, Float, Integer, Date, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from app.database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    filename = Column(String(255), nullable=False)
    modality = Column(String(20), nullable=False)
    capture_date = Column(Date)
    file_path = Column(Text, nullable=False)
    crs = Column(String(255))
    bbox_coords = Column(JSONB)
    resolution_m = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Added in A2 — real raster metadata (see app/contracts.py RasterMetadata)
    width = Column(Integer)
    height = Column(Integer)
    band_count = Column(Integer)
    dtype = Column(String(20))
    bounds = Column(JSONB)
    bounds_wgs84 = Column(JSONB)
    transform = Column(JSONB)
    nodata = Column(Float)
    file_format = Column(String(20))
    is_georeferenced = Column(Boolean, default=False)
    acquisition_date_source = Column(String(20))
    file_size_bytes = Column(Integer)
    metadata_warnings = Column(JSONB)


class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    query_text = Column(Text, nullable=False)
    image_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    task_classified = Column(String(50))
    selected_tool = Column(String(100))
    model_version = Column(String(50))
    confidence_score = Column(Float)
    execution_time_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Added in A6 — real execution trace, persisted for BOTH successful and
    # failed queries (see app/services/trace.py) so failures stay auditable.
    trace_events = Column(JSONB)
    compatibility = Column(JSONB)
    confidence_source = Column(String(50))
    warnings = Column(JSONB)
    used_fallback = Column(Boolean)
    specialist_id = Column(String(100))


class QueryResult(Base):
    __tablename__ = "query_results"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"))
    answer_text = Column(Text, nullable=False)
    bounding_boxes = Column(JSONB)
    change_mask_path = Column(Text)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
