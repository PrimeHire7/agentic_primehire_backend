from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableList, MutableDict
from datetime import datetime, timezone
from app.db import Base

class InterviewAttempts(Base):
    __tablename__ = "interview_attempts"

    attempt_id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String, nullable=False)
    jd_id = Column(Integer, nullable=True)

    interview_token = Column(String, nullable=False, unique=True)
    token_used_at = Column(DateTime(timezone=True), nullable=True)

    slot_start = Column(DateTime(timezone=True), nullable=False)
    slot_end   = Column(DateTime(timezone=True), nullable=False)

    slot_duration_min = Column(Integer, nullable=True)
    scheduled_date = Column(Date, nullable=True)
    interview_round = Column(Integer, nullable=True)

    status = Column(String, default="SCHEDULED")
    progress = Column(String, nullable=True)

    completed_at = Column(DateTime(timezone=True), nullable=True)

    interview_score = Column(Float, default=0)
    ai_score = Column(Float, default=0)
    manual_score = Column(Float, default=0)
    skill_score = Column(Float, default=0)

    anomalies = Column(MutableList.as_mutable(JSONB), default=list)
    anomaly_summary = Column(MutableDict.as_mutable(JSONB), nullable=True)

    mcq = Column(MutableDict.as_mutable(JSONB), nullable=True)
    coding = Column(MutableDict.as_mutable(JSONB), nullable=True)
    per_question = Column(MutableList.as_mutable(JSONB), nullable=True)
    feedback = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    technical_reason = Column(Text, nullable=True)
    communication_reason = Column(Text, nullable=True)
    behaviour_reason = Column(Text, nullable=True)