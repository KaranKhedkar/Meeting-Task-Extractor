from typing import List, Optional
from datetime import date, datetime
from pydantic import BaseModel, Field
from .task import TaskCreate, TaskResponse


class ExtractRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Raw transcript content (TXT or WebVTT)")
    meeting_date: date = Field(..., description="Reference meeting date used to resolve relative temporal expressions")
    extraction_mode: Optional[str] = Field("hybrid", description="Extraction engine mode: 'rule_based', 'ml', or 'hybrid'")


class ExtractResponse(BaseModel):
    tasks: List[TaskResponse]
    extraction_mode: str
    utterance_count: int
    action_item_count: int
    review_needed_count: int


class MeetingCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Title of the meeting")
    meeting_date: date = Field(..., description="Meeting date")
    transcript: str = Field(..., description="Original raw transcript text")
    extraction_mode: Optional[str] = Field("hybrid", description="Mode used during extraction")
    tasks: List[TaskCreate] = Field(default_factory=list, description="Extracted tasks to persist")


class MeetingSummary(BaseModel):
    id: str
    title: str
    meeting_date: date
    created_at: datetime
    task_count: int
    pending_count: int
    review_needed_count: int
    extraction_mode: Optional[str] = None

    model_config = {"from_attributes": True}


class MeetingListResponse(BaseModel):
    meetings: List[MeetingSummary]
    total: int


class MeetingResponse(BaseModel):
    id: str
    title: str
    meeting_date: date
    transcript: str
    extraction_mode: Optional[str] = None
    created_at: datetime
    tasks: List[TaskResponse]

    model_config = {"from_attributes": True}
