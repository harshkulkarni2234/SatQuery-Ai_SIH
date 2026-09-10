import uuid
from datetime import date, datetime
from sqlalchemy import Column, String, Text, Float, Integer, Date, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from app.database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    filename = Column(String(255), nullable=False)
    modality = Column(String(20), nullable=False)
    capture_date = Column(Date)
    file_path = Column(Text, nullable=False)
    crs = Column(String(50))
    bbox_coords = Column(JSONB)
    resolution_m = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


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


class QueryResult(Base):
    __tablename__ = "query_results"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"))
    answer_text = Column(Text, nullable=False)
    bounding_boxes = Column(JSONB)
    change_mask_path = Column(Text)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
