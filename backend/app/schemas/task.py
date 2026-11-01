from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class TaskClassification(str, Enum):
    ACTION_ITEM = "ACTION_ITEM"
    NON_ACTIONABLE = "NON_ACTIONABLE"
    COMPLETED_ACTION = "COMPLETED_ACTION"
    SUGGESTION = "SUGGESTION"


class TaskStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskBase(BaseModel):
    description: str = Field(..., description="Normalized task action description")
    assignee: Optional[str] = Field(None, description="Person responsible for the task")
    deadline_text: Optional[str] = Field(None, description="Extracted raw deadline expression")
    deadline: Optional[str] = Field(None, description="Normalized ISO 8601 date string (YYYY-MM-DD)")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Workflow confirmation status")
    classification: TaskClassification = Field(TaskClassification.ACTION_ITEM, description="Utterance classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score between 0.0 and 1.0")
    evidence: str = Field(..., description="Original evidence sentence or utterance from transcript")
    needs_review: bool = Field(False, description="Flag indicating uncertainty or ambiguity requiring human review")


class TaskCreate(TaskBase):
    id: Optional[str] = None
    meeting_id: Optional[str] = None


class TaskUpdate(BaseModel):
    description: Optional[str] = None
    assignee: Optional[str] = None
    deadline_text: Optional[str] = None
    deadline: Optional[str] = None
    status: Optional[TaskStatus] = None
    needs_review: Optional[bool] = None


class TaskResponse(TaskBase):
    id: str
    meeting_id: Optional[str] = None

    model_config = {"from_attributes": True}
