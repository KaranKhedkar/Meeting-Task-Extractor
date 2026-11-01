import re
import uuid
from typing import List, Set
from ..schemas.task import TaskResponse, TaskStatus, TaskClassification


def normalize_description_for_dedup(text: str) -> str:
    """Produces a clean canonical string to compare duplicate tasks."""
    clean = re.sub(r"[^\w\s]", "", text.lower())
    words = sorted(clean.split())
    return " ".join(words)


def calculate_calibrated_confidence(
    base_confidence: float,
    task_penalty: float,
    assignee_penalty: float,
    deadline_penalty: float,
) -> float:
    """
    Explicit calibrated confidence scoring:
    Calculates final confidence by subtracting documented penalties for
    ambiguous elements from the base classifier score.
    """
    raw_score = base_confidence - task_penalty - assignee_penalty - deadline_penalty
    calibrated = max(0.10, min(0.98, raw_score))
    return round(calibrated, 2)


def deduplicate_tasks(tasks: List[TaskResponse]) -> List[TaskResponse]:
    """
    Deduplicates tasks that share the same assignee and semantically equivalent description.
    Preserves tasks with different assignees or different core actions.
    """
    unique_tasks: List[TaskResponse] = []
    seen_keys: Set[str] = set()

    for task in tasks:
        assignee_key = (task.assignee or "unassigned").lower()
        desc_key = normalize_description_for_dedup(task.description)
        combo_key = f"{assignee_key}::{desc_key}"

        if combo_key not in seen_keys:
            seen_keys.add(combo_key)
            unique_tasks.append(task)
        else:
            # If current duplicate has a valid deadline and previous did not, update previous
            for idx, existing in enumerate(unique_tasks):
                ex_key = f"{(existing.assignee or 'unassigned').lower()}::{normalize_description_for_dedup(existing.description)}"
                if ex_key == combo_key:
                    if not existing.deadline and task.deadline:
                        unique_tasks[idx] = task
                    break

    return unique_tasks
