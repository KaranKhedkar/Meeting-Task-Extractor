import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from ..core.database import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=True, index=True)
    description = Column(Text, nullable=False)
    assignee = Column(String(150), nullable=True)
    deadline_text = Column(String(150), nullable=True)
    deadline = Column(String(50), nullable=True)  # Stored as ISO 8601 string (YYYY-MM-DD)
    status = Column(String(50), default="pending", nullable=False)
    classification = Column(String(50), default="ACTION_ITEM", nullable=False)
    confidence = Column(Float, nullable=False, default=0.8)
    evidence = Column(Text, nullable=False)
    needs_review = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    meeting = relationship("Meeting", back_populates="tasks")
