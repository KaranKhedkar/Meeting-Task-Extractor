import re
from typing import Optional, List
from dataclasses import dataclass
from spacy.tokens import Doc
from .preprocessing import ProcessedSentence

INDEFINITE_PRONOUNS = {"someone", "somebody", "anyone", "anybody", "everyone", "we", "team"}


@dataclass
class AssigneeResult:
    assignee: Optional[str]
    source: str  # "speaker_first_person", "vocative_request", "named_entity_subject", "unknown"
    confidence_penalty: float
    needs_review: bool


def extract_person_entities(doc: Doc) -> List[str]:
    """Extracts PERSON entities recognized by spaCy."""
    return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]


def extract_assignee(sentence: ProcessedSentence) -> AssigneeResult:
    """
    Extracts the responsible person using speaker labels, NER, vocative syntax,
    and first-person pronoun resolution.
    """
    text = sentence.text
    text_lower = text.lower()
    doc = sentence.doc
    speaker = sentence.speaker

    # 1. Indefinite / Collective expressions: explicitly unknown
    if re.search(r"\b(someone|somebody|anyone|anybody)\b", text_lower) or re.search(r"\blet's assign someone\b", text_lower):
        return AssigneeResult(
            assignee=None,
            source="indefinite_expression",
            confidence_penalty=0.10,
            needs_review=True,
        )

    # 2. Directed Imperative / Vocative Request (e.g., "Amit, please review...", "Amit, could you...")
    # Check leading vocative pattern: "[Name], please..."
    vocative_match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+(?:please|can you|could you|would you|review|check|update|prepare|fix)", text)
    if vocative_match:
        candidate_name = vocative_match.group(1).strip()
        return AssigneeResult(
            assignee=candidate_name,
            source="vocative_request",
            confidence_penalty=0.0,
            needs_review=False,
        )

    # Check question-style directed requests: "Could Amit review...", "Can Rahul prepare..."
    modal_req_match = re.match(r"^(?:could|can|would)\s+([A-Z][a-z]+)\s+(?:please\s+)?([a-z]+)", text, re.IGNORECASE)
    if modal_req_match:
        candidate_name = modal_req_match.group(1).strip()
        if candidate_name.lower() not in ("you", "we", "someone", "anyone"):
            return AssigneeResult(
                assignee=candidate_name.capitalize(),
                source="modal_directed_request",
                confidence_penalty=0.0,
                needs_review=False,
            )

    # 3. First-Person Commitment Resolution: "I will...", "I'll...", "I'm going to..."
    has_first_person = bool(re.search(r"\b(i will|i'll|i am going to|i'm going to|i'll make sure|i can handle)\b", text_lower))
    if has_first_person:
        if speaker and speaker.strip():
            return AssigneeResult(
                assignee=speaker.strip(),
                source="speaker_first_person",
                confidence_penalty=0.0,
                needs_review=False,
            )
        else:
            # First person commitment but no speaker label in transcript
            return AssigneeResult(
                assignee=None,
                source="unresolved_first_person",
                confidence_penalty=0.15,
                needs_review=True,
            )

    # 4. Third-Person Explicit Subject (e.g., "Rahul will prepare...", "Rahul to prepare...")
    # Look for nominal subject in spaCy dependency parse
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            # Check if token is a proper noun or recognized as PERSON
            if token.pos_ == "PROPN" or token.ent_type_ == "PERSON":
                cand = token.text.strip()
                if cand.lower() not in INDEFINITE_PRONOUNS:
                    return AssigneeResult(
                        assignee=cand,
                        source="named_entity_subject",
                        confidence_penalty=0.02,
                        needs_review=False,
                    )
            elif token.lower_ in ("we", "team"):
                return AssigneeResult(
                    assignee=None,  # Leave unknown rather than inventing a person
                    source="collective_team",
                    confidence_penalty=0.08,
                    needs_review=True,
                )

    # 5. Check if any PERSON entity is mentioned before the main verb
    person_ents = extract_person_entities(doc)
    if person_ents:
        # If exactly one person entity in utterance and not speaker
        if len(person_ents) == 1 and person_ents[0] != speaker:
            return AssigneeResult(
                assignee=person_ents[0],
                source="single_person_entity",
                confidence_penalty=0.05,
                needs_review=False,
            )

    # 6. Fallback: Unknown assignee
    return AssigneeResult(
        assignee=None,
        source="unknown",
        confidence_penalty=0.12,
        needs_review=True,
    )
