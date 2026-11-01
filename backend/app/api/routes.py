import io
import csv
import json
import uuid
from typing import List, Optional
from datetime import datetime, timezone, date
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..core.database import get_db
from ..core.config import settings
from ..models.meeting import Meeting
from ..models.task import Task
from ..schemas.task import TaskResponse, TaskCreate, TaskUpdate, TaskStatus
from ..schemas.meeting import (
    ExtractRequest,
    ExtractResponse,
    MeetingCreate,
    MeetingResponse,
    MeetingSummary,
    MeetingListResponse,
)
from ..services.pipeline import process_meeting_transcript
from ..services.audio_transcriber import transcribe_meeting_audio
from ..services.transformer_extractor import is_v2_available

router = APIRouter(prefix="/api", tags=["meetings"])


@router.post(
    "/meetings/extract",
    response_model=ExtractResponse,
    summary="Extract actionable tasks from transcript",
)
def extract_tasks(payload: ExtractRequest):
    """
    Runs the multi-stage classical NLP and ML pipeline on the transcript text.
    Uses reference meeting date to resolve relative deadlines.
    """
    transcript = payload.transcript.strip()
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript text cannot be empty.",
        )

    tasks, effective_mode, utterance_count, review_needed_count = process_meeting_transcript(
        transcript=transcript,
        meeting_date=payload.meeting_date,
        extraction_mode=payload.extraction_mode or "hybrid",
    )

    return ExtractResponse(
        tasks=tasks,
        extraction_mode=effective_mode,
        utterance_count=utterance_count,
        action_item_count=len(tasks),
        review_needed_count=review_needed_count,
    )


@router.post(
    "/meetings/upload",
    summary="Upload transcript file (.txt or .vtt)",
)
async def upload_transcript(file: UploadFile = File(...)):
    """
    Accepts .txt or .vtt file, validates file size and extension,
    and returns decoded text content and metadata.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    content_bytes = await file.read()
    if len(content_bytes) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_BYTES / (1024*1024):.1f}MB.",
        )

    try:
        content_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content_text = content_bytes.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to decode file content as text.")

    if not content_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return {
        "filename": file.filename,
        "format": ext,
        "character_count": len(content_text),
        "content": content_text,
    }


@router.post(
    "/meetings/transcribe-audio",
    summary="Transcribe meeting audio using local Faster-Whisper (V2 Audio Pipeline)",
)
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """
    Accepts meeting audio (.mp3, .wav, .m4a, .ogg, .webm), runs local Faster-Whisper
    speech-to-text, and returns formatted dialogue with speaker tags and timestamps.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file selected.")

    ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(settings.ALLOWED_AUDIO_EXTENSIONS)}",
        )

    audio_bytes = await file.read()
    if len(audio_bytes) > settings.MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds maximum allowed size of {settings.MAX_AUDIO_SIZE_BYTES / (1024*1024):.1f}MB.",
        )

    try:
        transcription_result = transcribe_meeting_audio(audio_bytes, file.filename)
        return transcription_result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Local audio transcription failed: {str(e)}",
        )


@router.get(
    "/benchmark",
    summary="Get comprehensive V1 vs V2 model comparison and benchmark report",
)
def get_benchmark_report():
    """Returns the multi-model evaluation metrics and AMI corpus benchmark report."""
    report_file = settings.BASE_DIR / "training" / "benchmark_report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark report not found.")

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check V2 status
    data["v2_transformer_available"] = is_v2_available()
    v2_meta_path = settings.BASE_DIR / "training" / "v2_distilbert" / "v2_metadata.json"
    if v2_meta_path.exists():
        with open(v2_meta_path, "r", encoding="utf-8") as vf:
            data["v2_transformer_metadata"] = json.load(vf)

    return data


@router.post(
    "/meetings",
    response_model=MeetingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a meeting and its extracted/edited tasks",
)
def save_meeting(payload: MeetingCreate, db: Session = Depends(get_db)):
    """
    Persists a meeting, its raw transcript, and all associated extracted tasks into SQLite.
    """
    meeting_id = str(uuid.uuid4())
    meeting = Meeting(
        id=meeting_id,
        title=payload.title,
        meeting_date=payload.meeting_date,
        transcript=payload.transcript,
        extraction_mode=payload.extraction_mode,
        created_at=datetime.now(timezone.utc),
    )
    db.add(meeting)

    for task_data in payload.tasks:
        task = Task(
            id=task_data.id or str(uuid.uuid4()),
            meeting_id=meeting_id,
            description=task_data.description,
            assignee=task_data.assignee,
            deadline_text=task_data.deadline_text,
            deadline=task_data.deadline,
            status=task_data.status.value if isinstance(task_data.status, TaskStatus) else task_data.status,
            classification=task_data.classification.value if hasattr(task_data.classification, "value") else str(task_data.classification),
            confidence=task_data.confidence,
            evidence=task_data.evidence,
            needs_review=task_data.needs_review,
            created_at=datetime.now(timezone.utc),
        )
        db.add(task)

    db.commit()
    db.refresh(meeting)
    return meeting


@router.get(
    "/meetings",
    response_model=MeetingListResponse,
    summary="List all saved meetings with summary metrics",
)
def list_meetings(db: Session = Depends(get_db)):
    """
    Lists all saved meetings with calculated task counts, pending counts, and review needed counts.
    """
    meetings = db.query(Meeting).order_by(Meeting.created_at.desc()).all()
    summaries = []

    for m in meetings:
        tasks = m.tasks
        summaries.append(
            MeetingSummary(
                id=m.id,
                title=m.title,
                meeting_date=m.meeting_date,
                created_at=m.created_at,
                task_count=len(tasks),
                pending_count=sum(1 for t in tasks if t.status == "pending"),
                review_needed_count=sum(1 for t in tasks if t.needs_review),
                extraction_mode=m.extraction_mode,
            )
        )

    return MeetingListResponse(meetings=summaries, total=len(summaries))


@router.get(
    "/meetings/{meeting_id}",
    response_model=MeetingResponse,
    summary="Get single meeting with transcript and tasks",
)
def get_meeting(meeting_id: str, db: Session = Depends(get_db)):
    """Retrieves an existing meeting by ID."""
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting with ID '{meeting_id}' not found.",
        )
    return meeting


@router.delete(
    "/meetings/{meeting_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a meeting and its tasks",
)
def delete_meeting(meeting_id: str, db: Session = Depends(get_db)):
    """Deletes a meeting and cascades to its tasks."""
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting with ID '{meeting_id}' not found.",
        )
    db.delete(meeting)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="Update task fields or confirmation status",
)
def update_task(task_id: str, payload: TaskUpdate, db: Session = Depends(get_db)):
    """Updates editable task fields (description, assignee, deadline, status, needs_review)."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found.",
        )

    if payload.description is not None:
        task.description = payload.description.strip()
    if payload.assignee is not None:
        task.assignee = payload.assignee.strip() if payload.assignee.strip() else None
    if payload.deadline_text is not None:
        task.deadline_text = payload.deadline_text.strip() if payload.deadline_text.strip() else None
    if payload.deadline is not None:
        task.deadline = payload.deadline.strip() if payload.deadline.strip() else None
    if payload.status is not None:
        task.status = payload.status.value if isinstance(payload.status, TaskStatus) else payload.status
    if payload.needs_review is not None:
        task.needs_review = payload.needs_review

    db.commit()
    db.refresh(task)
    return task


@router.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a task",
)
def delete_task(task_id: str, db: Session = Depends(get_db)):
    """Permanently deletes a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found.",
        )
    db.delete(task)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/meetings/{meeting_id}/export",
    summary="Export tasks as CSV or JSON",
)
def export_tasks(
    meeting_id: str,
    format: str = Query("csv", pattern="^(csv|json)$"),
    db: Session = Depends(get_db),
):
    """
    Exports all tasks of a meeting into structured CSV or JSON format.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting with ID '{meeting_id}' not found.",
        )

    tasks = meeting.tasks

    if format == "json":
        export_data = [
            {
                "id": t.id,
                "meeting_id": t.meeting_id,
                "description": t.description,
                "assignee": t.assignee,
                "deadline_text": t.deadline_text,
                "deadline": t.deadline,
                "status": t.status,
                "classification": t.classification,
                "confidence": t.confidence,
                "evidence": t.evidence,
                "needs_review": t.needs_review,
            }
            for t in tasks
        ]
        return Response(
            content=json.dumps(export_data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=meeting_{meeting_id[:8]}_tasks.json"},
        )

    # Default CSV Export
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Description",
        "Assignee",
        "Deadline",
        "Deadline Text",
        "Status",
        "Confidence",
        "Needs Review",
        "Evidence",
    ])
    for t in tasks:
        writer.writerow([
            t.id,
            t.description,
            t.assignee or "Unassigned",
            t.deadline or "",
            t.deadline_text or "",
            t.status,
            f"{t.confidence:.2f}",
            "Yes" if t.needs_review else "No",
            t.evidence,
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=meeting_{meeting_id[:8]}_tasks.csv"},
    )
