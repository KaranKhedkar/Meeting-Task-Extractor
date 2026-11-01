import re
from datetime import date, datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass
import dateparser
from .preprocessing import ProcessedSentence

# Number words mapping for expressions like "within three days"
WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1, "couple of": 2, "few": 3
}

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}

# Regex patterns for deadlines
DEADLINE_PATTERNS = [
    # "before the next meeting", "before next standup"
    (r"\b(before (?:the )?next (?:meeting|standup|sync|call))\b", "event_relative"),
    # "by end of day", "by eod", "tonight", "this afternoon"
    (r"\b((?:by )?(?:end of (?:the )?day|eod|tonight|this afternoon|by today))\b", "today"),
    # "day after tomorrow"
    (r"\b((?:by |before )?day after tomorrow)\b", "day_after_tomorrow"),
    # "tomorrow"
    (r"\b((?:by |before |on )?tomorrow(?: morning| afternoon| evening| at \d+(?::\d+)?\s*(?:am|pm)?)?)\b", "tomorrow"),
    # "within X days", "in X days"
    (r"\b((?:within|in)\s+(\w+|\d+)\s+days?)\b", "within_n_days"),
    # "next week", "by next week", "within a week"
    (r"\b((?:by |before |within |for )?next week|(?:within )?a week)\b", "next_week"),
    # "next Friday", "next Monday"
    (r"\b((?:by |before |on )?next (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", "next_weekday"),
    # "by Friday", "before Monday", "this Thursday"
    (r"\b((?:by |before |on |this )(?:this )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", "this_weekday"),
    # Explicit dates: "by Oct 30th", "before November 5", "by 15th November"
    (r"\b((?:by |before |on )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\b", "explicit_month_day"),
    (r"\b((?:by |before |on )?\d{1,2}(?:st|nd|rd|th)? (?:of )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s*,?\s*\d{4})?)\b", "explicit_day_month"),
]


@dataclass
class DeadlineResult:
    deadline_text: Optional[str]
    deadline: Optional[str]  # ISO 8601 YYYY-MM-DD
    is_ambiguous: bool
    confidence_penalty: float
    needs_review: bool


def resolve_weekday_target(ref_date: date, target_weekday_name: str, is_next_week: bool = False) -> date:
    """
    Calculates the target date for a weekday.
    Consistent policy:
      - 'by Friday' (when ref is Tue): Friday of same week (+3 days).
      - 'by Tuesday' (when ref is Tue): next week (+7 days).
      - 'next Friday' (when ref is Tue): Friday of the following week (+10 days).
    """
    target_idx = WEEKDAYS[target_weekday_name.lower()]
    current_idx = ref_date.weekday()
    
    days_ahead = target_idx - current_idx
    if days_ahead <= 0:  # Target day already passed this week or is today
        days_ahead += 7

    if is_next_week:
        days_ahead += 7

    return ref_date + timedelta(days=days_ahead)


def extract_deadline(sentence: ProcessedSentence, reference_date: date) -> DeadlineResult:
    """
    Extracts deadline phrase and normalizes it to ISO 8601 date using reference meeting date.
    """
    text = sentence.text
    text_lower = text.lower()

    for pattern, p_type in DEADLINE_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            raw_matched_phrase = match.group(1).strip()
            
            # Case 1: Event relative ("before the next meeting") -> Cannot resolve to absolute date
            if p_type == "event_relative":
                return DeadlineResult(
                    deadline_text=raw_matched_phrase,
                    deadline=None,
                    is_ambiguous=True,
                    confidence_penalty=0.08,
                    needs_review=True,
                )

            # Case 2: Today / Tonight / EOD
            if p_type == "today":
                return DeadlineResult(
                    deadline_text=raw_matched_phrase,
                    deadline=reference_date.isoformat(),
                    is_ambiguous=False,
                    confidence_penalty=0.0,
                    needs_review=False,
                )

            # Case 3: Tomorrow
            if p_type == "tomorrow":
                resolved = reference_date + timedelta(days=1)
                return DeadlineResult(
                    deadline_text=raw_matched_phrase,
                    deadline=resolved.isoformat(),
                    is_ambiguous=False,
                    confidence_penalty=0.0,
                    needs_review=False,
                )

            # Case 4: Day after tomorrow
            if p_type == "day_after_tomorrow":
                resolved = reference_date + timedelta(days=2)
                return DeadlineResult(
                    deadline_text=raw_matched_phrase,
                    deadline=resolved.isoformat(),
                    is_ambiguous=False,
                    confidence_penalty=0.0,
                    needs_review=False,
                )

            # Case 5: Within N days
            if p_type == "within_n_days":
                num_str = match.group(2).lower()
                n_days = WORD_TO_NUM.get(num_str)
                if n_days is None:
                    try:
                        n_days = int(num_str)
                    except ValueError:
                        n_days = 3  # Default fallback
                resolved = reference_date + timedelta(days=n_days)
                return DeadlineResult(
                    deadline_text=raw_matched_phrase,
                    deadline=resolved.isoformat(),
                    is_ambiguous=False,
                    confidence_penalty=0.0,
                    needs_review=False,
                )

            # Case 6: Next week
            if p_type == "next_week":
                # Standard policy: 7 days from reference date
                resolved = reference_date + timedelta(days=7)
                return DeadlineResult(
                    deadline_text=raw_matched_phrase,
                    deadline=resolved.isoformat(),
                    is_ambiguous=True,  # "next week" is broad, flag for user confirmation
                    confidence_penalty=0.05,
                    needs_review=True,
                )

            # Case 7: Next weekday (e.g. "next Friday")
            if p_type == "next_weekday":
                day_name = match.group(2)
                resolved = resolve_weekday_target(reference_date, day_name, is_next_week=True)
                return DeadlineResult(
                    deadline_text=raw_matched_phrase,
                    deadline=resolved.isoformat(),
                    is_ambiguous=True,  # "next Friday" can colloquially mean this week's or next week's
                    confidence_penalty=0.05,
                    needs_review=True,
                )

            # Case 8: This weekday (e.g. "by Friday", "before Monday")
            if p_type == "this_weekday":
                day_name = match.group(2)
                resolved = resolve_weekday_target(reference_date, day_name, is_next_week=False)
                return DeadlineResult(
                    deadline_text=raw_matched_phrase,
                    deadline=resolved.isoformat(),
                    is_ambiguous=False,
                    confidence_penalty=0.0,
                    needs_review=False,
                )

            # Case 9: Explicit date phrases
            if p_type in ("explicit_month_day", "explicit_day_month"):
                clean_date_str = re.sub(r"^(by|before|on)\s+", "", raw_matched_phrase, flags=re.IGNORECASE).strip()
                ref_dt = datetime.combine(reference_date, datetime.min.time())
                parsed = dateparser.parse(
                    clean_date_str,
                    settings={
                        "RELATIVE_BASE": ref_dt,
                        "PREFER_DATES_FROM": "future",
                        "REQUIRE_PARTS": ["day", "month"],
                    }
                )
                if parsed:
                    return DeadlineResult(
                        deadline_text=raw_matched_phrase,
                        deadline=parsed.date().isoformat(),
                        is_ambiguous=False,
                        confidence_penalty=0.0,
                        needs_review=False,
                    )

    # Secondary fallback using dateparser on preposition phrases if regex missed
    prep_matches = re.findall(r"\b(by|before|until|due)\s+([A-Za-z0-9\s,]{3,25})\b", text, re.IGNORECASE)
    for prep, phrase in prep_matches:
        full_phrase = f"{prep} {phrase}".strip()
        ref_dt = datetime.combine(reference_date, datetime.min.time())
        parsed = dateparser.parse(
            phrase.strip(),
            settings={"RELATIVE_BASE": ref_dt, "PREFER_DATES_FROM": "future"}
        )
        if parsed and parsed.date() >= reference_date:
            return DeadlineResult(
                deadline_text=full_phrase,
                deadline=parsed.date().isoformat(),
                is_ambiguous=False,
                confidence_penalty=0.04,
                needs_review=False,
            )

    # No deadline found
    return DeadlineResult(
        deadline_text=None,
        deadline=None,
        is_ambiguous=False,
        confidence_penalty=0.0,
        needs_review=False,
    )
