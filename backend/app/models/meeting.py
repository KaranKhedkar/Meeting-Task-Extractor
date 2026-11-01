import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, Date, DateTime, Text
from sqlalchemy.orm import relationship
from ..core.database import Base


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    meeting_date = Column(Date, nullable=False, default=date.today)
    transcript = Column(Text, nullable=False)
    extraction_mode = Column(String(50), nullable=True, default="hybrid")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    tasks = relationship("Task", back_populates="meeting", cascade="all, delete-orphan", lazy="joined")
