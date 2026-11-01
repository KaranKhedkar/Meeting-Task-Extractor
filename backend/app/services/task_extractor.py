import re
from typing import Optional, Tuple, Set
from dataclasses import dataclass
from spacy.tokens import Doc, Token
from .preprocessing import ProcessedSentence

# Prepositions typically introducing deadlines or times to exclude from the core action description
TEMPORAL_PREPOSITIONS = {"by", "before", "until", "till", "within", "around"}
TEMPORAL_WORDS = {
    "today", "tomorrow", "tonight", "yesterday", "morning", "afternoon", "evening",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "week", "month", "standup", "meeting", "eod", "sprint"
}

# Conversational preambles to strip from beginning
PREAMBLE_PATTERNS = [
    r"^(could|can|would)\s+(you|amit|rahul|priya|someone)\s+(please\s+)?",
    r"^(please\s+)",
    r"^(i('ll|'m going to|\s+will|\s+am going to)\s+)",
    r"^(we('ll|\s+will|\s+need to|\s+have to|\s+must)\s+)",
    r"^(let's\s+(assign\s+someone\s+to\s+|make\s+sure\s+to\s+|ensure\s+that\s+)?)",
    r"^(someone\s+should\s+)",
    r"^(make\s+sure\s+to\s+)",
    r"^[A-Z][a-z]+,\s+(please\s+)?",
]


@dataclass
class TaskDescriptionResult:
    description: str
    action_verb: str
    confidence_penalty: float
    needs_review: bool


def is_temporal_token(token: Token) -> bool:
    """Checks if a token or its children represent temporal/deadline info."""
    if token.lower_ in TEMPORAL_WORDS:
        return True
    if token.ent_type_ in ("DATE", "TIME"):
        return True
    return False


def find_main_action_verb(doc: Doc) -> Optional[Token]:
    """
    Finds the semantic action verb in the sentence, handling modals,
    auxiliaries, clausal complements (xcomp), and imperatives.
    """
    # 1. Look for roots and xcomp
    roots = [t for t in doc if t.dep_ == "ROOT"]
    if not roots:
        return None

    root = roots[0]

    # If root is 'need' or 'have' with 'to + verb' (e.g. "We need to submit the report")
    if root.lemma_ in ("need", "have", "want", "plan", "assign"):
        for child in root.children:
            if child.dep_ in ("xcomp", "ccomp") and child.pos_ == "VERB":
                return child

    # If root is modal/aux or verb
    if root.pos_ in ("VERB", "AUX"):
        # If root itself is an action verb
        if root.pos_ == "VERB" and root.lemma_ not in ("be", "have"):
            return root
        # Check xcomp or ccomp children
        for child in root.children:
            if child.dep_ in ("xcomp", "ccomp", "conj") and child.pos_ == "VERB":
                return child

    # Check for any verb with direct object
    for token in doc:
        if token.pos_ == "VERB" and any(c.dep_ in ("dobj", "prt", "prep") for c in token.children):
            return token

    return root if root.pos_ == "VERB" else None


def extract_task_description(sentence: ProcessedSentence) -> TaskDescriptionResult:
    """
    Extracts a clean, actionable task description from an utterance.
    E.g. "Rahul will prepare the project presentation tomorrow." -> "Prepare the project presentation"
    """
    doc = sentence.doc
    action_verb = find_main_action_verb(doc)

    if action_verb is None:
        # Fallback: clean the sentence using preamble stripping
        clean_text = sentence.text
        for pat in PREAMBLE_PATTERNS:
            clean_text = re.sub(pat, "", clean_text, flags=re.IGNORECASE).strip()
        # Clean trailing temporal phrases
        clean_text = re.sub(r"\s+(by|before|on|within)\s+[A-Za-z0-9\s]+$", "", clean_text, flags=re.IGNORECASE).strip()
        return TaskDescriptionResult(
            description=clean_text.rstrip("."),
            action_verb="",
            confidence_penalty=0.15,
            needs_review=True,
        )

    # Collect tokens belonging to the action predicate
    collected_tokens: Set[Token] = {action_verb}

    # Include phrasal verb particles (e.g., "set up", "roll back")
    for child in action_verb.children:
        if child.dep_ == "prt":
            collected_tokens.add(child)

    # Include direct objects and their full noun phrase subtrees (det, amod, compound, etc.)
    for child in action_verb.children:
        if child.dep_ in ("dobj", "attr", "oprd"):
            for sub in child.subtree:
                if not is_temporal_token(sub):
                    collected_tokens.add(sub)

        # Include prepositional complements unless they are temporal (deadlines)
        elif child.dep_ == "prep":
            # Check if this prep is temporal (e.g., "by Friday", "before tomorrow")
            pobj_tokens = [c for c in child.subtree]
            if child.lower_ in TEMPORAL_PREPOSITIONS or any(is_temporal_token(t) for t in pobj_tokens):
                continue
            for sub in pobj_tokens:
                collected_tokens.add(sub)

        # Include conjunction verbs (e.g., "prepare and send")
        elif child.dep_ == "conj" and child.pos_ == "VERB":
            collected_tokens.add(child)
            for sub in child.children:
                if sub.dep_ in ("dobj", "prt"):
                    for s in sub.subtree:
                        if not is_temporal_token(s):
                            collected_tokens.add(s)

    # Sort collected tokens by original text order
    sorted_tokens = sorted(list(collected_tokens), key=lambda t: t.i)
    extracted_text = " ".join(t.text for t in sorted_tokens).strip()

    # Apply regex cleanups for any stray punctuation or preambles
    for pat in PREAMBLE_PATTERNS:
        extracted_text = re.sub(pat, "", extracted_text, flags=re.IGNORECASE).strip()

    # Remove trailing punctuation
    extracted_text = extracted_text.rstrip(".,;! ")

    # Quality check: if extraction is too short or empty, fallback to clause
    needs_review = False
    penalty = 0.0

    if len(extracted_text.split()) < 2:
        # Fall back to cleaned original sentence
        fallback = sentence.text
        for pat in PREAMBLE_PATTERNS:
            fallback = re.sub(pat, "", fallback, flags=re.IGNORECASE).strip()
        fallback = re.sub(r"\s+(by|before|on|within)\s+[A-Za-z0-9\s]+$", "", fallback, flags=re.IGNORECASE).strip()
        extracted_text = fallback.rstrip(".")
        needs_review = True
        penalty = 0.10

    # Capitalize the first letter for clean presentation
    if extracted_text:
        extracted_text = extracted_text[0].upper() + extracted_text[1:]

    return TaskDescriptionResult(
        description=extracted_text,
        action_verb=action_verb.lemma_,
        confidence_penalty=penalty,
        needs_review=needs_review,
    )
