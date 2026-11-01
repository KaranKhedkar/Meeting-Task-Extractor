import uuid
from datetime import date
from typing import List, Tuple
from ..schemas.task import TaskResponse, TaskStatus, TaskClassification
from .preprocessing import preprocess_transcript, ProcessedSentence
from .action_classifier import ActionClassifier
from .task_extractor import extract_task_description
from .assignee_extractor import extract_assignee
from .deadline_extractor import extract_deadline
from .validation import calculate_calibrated_confidence, deduplicate_tasks

# Singleton instance of the action classifier
_classifier_instance = None


def get_action_classifier() -> ActionClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ActionClassifier()
    return _classifier_instance


def process_meeting_transcript(
    transcript: str,
    meeting_date: date,
    extraction_mode: str = "hybrid",
    meeting_id: str = None,
) -> Tuple[List[TaskResponse], str, int, int]:
    """
    Executes the complete End-to-End NLP pipeline:
    1. Preprocesses and segments transcript utterances/sentences.
    2. Classifies each sentence into ACTION_ITEM, NON_ACTIONABLE, COMPLETED_ACTION, SUGGESTION.
    3. For ACTION_ITEMs: extracts task description, assignee, and normalized deadline.
    4. Calculates calibrated confidence and review flags.
    5. Deduplicates tasks across utterances.
    
    Returns:
      (tasks, effective_mode, utterance_count, review_needed_count)
    """
    processed_sentences: List[ProcessedSentence] = preprocess_transcript(transcript)
    classifier = get_action_classifier()

    raw_tasks: List[TaskResponse] = []
    modes_used = set()

    for sent in processed_sentences:
        cls_res = classifier.classify(sent, mode=extraction_mode)
        modes_used.add(cls_res.method_used)

        # Only extract tasks from actionable utterances
        if cls_res.classification == TaskClassification.ACTION_ITEM:
            task_desc_res = extract_task_description(sent)
            assignee_res = extract_assignee(sent)
            deadline_res = extract_deadline(sent, meeting_date)

            calibrated_conf = calculate_calibrated_confidence(
                base_confidence=cls_res.confidence,
                task_penalty=task_desc_res.confidence_penalty,
                assignee_penalty=assignee_res.confidence_penalty,
                deadline_penalty=deadline_res.confidence_penalty,
            )

            # A task needs review if:
            # - Explicitly flagged by any extractor (ambiguous deadline/assignee/predicate)
            # - Assignee is unknown
            # - Calibrated confidence is under 0.70
            needs_review = (
                task_desc_res.needs_review
                or assignee_res.needs_review
                or deadline_res.needs_review
                or (assignee_res.assignee is None)
                or (calibrated_conf < 0.70)
            )

            # Evidence: include speaker attribution if present
            evidence_line = f"{sent.speaker}: {sent.text}" if sent.speaker else sent.text

            task_obj = TaskResponse(
                id=str(uuid.uuid4()),
                meeting_id=meeting_id,
                description=task_desc_res.description,
                assignee=assignee_res.assignee,
                deadline_text=deadline_res.deadline_text,
                deadline=deadline_res.deadline,
                status=TaskStatus.PENDING,
                classification=cls_res.classification,
                confidence=calibrated_conf,
                evidence=evidence_line,
                needs_review=needs_review,
            )
            raw_tasks.append(task_obj)

    # Stage F: Deduplication
    final_tasks = deduplicate_tasks(raw_tasks)

    # Determine reporting mode
    effective_mode = "hybrid"
    if "v2_transformer" in modes_used:
        effective_mode = "v2_transformer"
    elif "hybrid" in modes_used:
        effective_mode = "hybrid"
    elif "ml" in modes_used:
        effective_mode = "ml"
    else:
        effective_mode = "rule_based"

    review_count = sum(1 for t in final_tasks if t.needs_review)

    return final_tasks, effective_mode, len(processed_sentences), review_count
