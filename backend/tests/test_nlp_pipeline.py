import pytest
from datetime import date
from app.services.preprocessing import preprocess_transcript, parse_vtt_transcript, get_spacy_nlp, ProcessedSentence
from app.services.action_classifier import ActionClassifier
from app.schemas.task import TaskClassification
from app.services.task_extractor import extract_task_description
from app.services.assignee_extractor import extract_assignee
from app.services.deadline_extractor import extract_deadline
from app.services.pipeline import process_meeting_transcript


@pytest.fixture
def classifier():
    return ActionClassifier()


@pytest.fixture
def nlp():
    return get_spacy_nlp()


def create_sentence(text: str, speaker: str = None, nlp_model = None) -> ProcessedSentence:
    if nlp_model is None:
        nlp_model = get_spacy_nlp()
    doc = nlp_model(text)
    return ProcessedSentence(
        utterance_idx=0,
        sentence_idx=0,
        speaker=speaker,
        text=text,
        doc=doc,
        raw_utterance=f"{speaker}: {text}" if speaker else text,
    )


# 1. Actionable vs Non-Actionable Tests
def test_action_item_first_person_commitment(classifier, nlp):
    sent = create_sentence("I'll finish the backend API by Wednesday.", speaker="Rahul", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.ACTION_ITEM
    assert res.confidence >= 0.75


def test_action_item_directed_imperative(classifier, nlp):
    sent = create_sentence("Amit, please review the interface before Friday.", speaker="Priya", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.ACTION_ITEM


def test_action_item_modal_question(classifier, nlp):
    sent = create_sentence("Could Amit review the design before the next meeting?", speaker="Dev", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.ACTION_ITEM


def test_action_item_collective_obligation(classifier, nlp):
    sent = create_sentence("We need to submit the report next week.", speaker="Sneha", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.ACTION_ITEM


def test_non_actionable_general_discussion(classifier, nlp):
    sent = create_sentence("We discussed the possibility of improving the interface.", speaker="Amit", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.NON_ACTIONABLE


def test_non_actionable_observation(classifier, nlp):
    sent = create_sentence("The report looks good.", speaker="Rahul", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.NON_ACTIONABLE


# 2. Completed Actions and Suggestions Tests
def test_completed_action_past_passive(classifier, nlp):
    sent = create_sentence("The API was deployed yesterday.", speaker="Amit", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.COMPLETED_ACTION


def test_completed_action_past_active(classifier, nlp):
    sent = create_sentence("I finished the testing last week.", speaker="Rahul", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.COMPLETED_ACTION


def test_suggestion_hypothetical(classifier, nlp):
    sent = create_sentence("Maybe we could redesign the navigation bar next sprint.", speaker="Priya", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.SUGGESTION


def test_suggestion_what_if(classifier, nlp):
    sent = create_sentence("What if we try a dark theme for the analytics view?", speaker="Vikram", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.SUGGESTION


# 3. Negation Handling
def test_negation_rejection(classifier, nlp):
    sent = create_sentence("We don't need to update the legacy billing code anymore.", speaker="Rahul", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.NON_ACTIONABLE


def test_negation_wont_action(classifier, nlp):
    sent = create_sentence("I won't change the hero banner until marketing approves it.", speaker="Neha", nlp_model=nlp)
    res = classifier.classify(sent, mode="hybrid")
    assert res.classification == TaskClassification.NON_ACTIONABLE


# 4. Task Description Extraction Tests
def test_task_description_predicate_cleaning(nlp):
    sent = create_sentence("Rahul will prepare the project presentation tomorrow.", speaker="Dev", nlp_model=nlp)
    desc_res = extract_task_description(sent)
    assert "prepare" in desc_res.description.lower()
    assert "presentation" in desc_res.description.lower()
    # Preamble and temporal words should be stripped
    assert not desc_res.description.lower().startswith("rahul will")
    assert not desc_res.description.lower().endswith("tomorrow")


def test_task_description_imperative_cleaning(nlp):
    sent = create_sentence("Amit, please review the interface before Friday.", speaker="Priya", nlp_model=nlp)
    desc_res = extract_task_description(sent)
    assert "review the interface" in desc_res.description.lower()
    assert not desc_res.description.lower().startswith("amit")
    assert not desc_res.description.lower().startswith("please")


# 5. Assignee Extraction Tests
def test_assignee_first_person_resolution(nlp):
    sent = create_sentence("I'll complete the API integration.", speaker="Rahul", nlp_model=nlp)
    assignee_res = extract_assignee(sent)
    assert assignee_res.assignee == "Rahul"
    assert not assignee_res.needs_review


def test_assignee_vocative_direction(nlp):
    sent = create_sentence("Amit, please review the design.", speaker="Priya", nlp_model=nlp)
    assignee_res = extract_assignee(sent)
    assert assignee_res.assignee == "Amit"


def test_assignee_modal_request(nlp):
    sent = create_sentence("Could Amit review the design before the next meeting?", speaker="Dev", nlp_model=nlp)
    assignee_res = extract_assignee(sent)
    assert assignee_res.assignee == "Amit"


def test_assignee_indefinite_unknown(nlp):
    sent = create_sentence("Someone should investigate the issue.", speaker=None, nlp_model=nlp)
    assignee_res = extract_assignee(sent)
    assert assignee_res.assignee is None
    assert assignee_res.needs_review is True


# 6. Deadline Extraction Tests
def test_deadline_explicit_weekday(nlp):
    # Reference date: 2026-10-27 (Tuesday)
    ref_date = date(2026, 10, 27)
    sent = create_sentence("Rahul will prepare the presentation by Friday.", speaker="Dev", nlp_model=nlp)
    dl_res = extract_deadline(sent, ref_date)
    assert dl_res.deadline_text is not None
    assert "friday" in dl_res.deadline_text.lower()
    assert dl_res.deadline == "2026-10-30"


def test_deadline_relative_tomorrow(nlp):
    ref_date = date(2026, 10, 27)
    sent = create_sentence("I'll finish the backend API tomorrow.", speaker="Rahul", nlp_model=nlp)
    dl_res = extract_deadline(sent, ref_date)
    assert dl_res.deadline == "2026-10-28"


def test_deadline_ambiguous_next_meeting(nlp):
    ref_date = date(2026, 10, 27)
    sent = create_sentence("Review the design before the next meeting.", speaker="Dev", nlp_model=nlp)
    dl_res = extract_deadline(sent, ref_date)
    assert dl_res.is_ambiguous is True
    assert dl_res.deadline is None  # Cannot guess exact date
    assert dl_res.needs_review is True


def test_deadline_none_when_unmentioned(nlp):
    ref_date = date(2026, 10, 27)
    sent = create_sentence("I will write the test cases.", speaker="Arjun", nlp_model=nlp)
    dl_res = extract_deadline(sent, ref_date)
    assert dl_res.deadline is None
    assert dl_res.deadline_text is None


# 7. Deduplication and End-to-End Pipeline
def test_pipeline_deduplication():
    transcript = """
    Rahul: I'll complete the API integration by Wednesday.
    Rahul: By Wednesday, I will complete the API integration.
    """
    ref_date = date(2026, 10, 27)
    tasks, mode, total_sents, review_count = process_meeting_transcript(transcript, ref_date)
    assert len(tasks) == 1
    assert tasks[0].assignee == "Rahul"


# 8. WebVTT Transcript Format Parser
def test_vtt_parsing():
    vtt_content = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
<v Rahul>I'll finish the backend API by Wednesday.</v>

2
00:00:04.500 --> 00:00:08.000
<v Priya>Amit, please review the interface before Friday.</v>
"""
    cleaned = parse_vtt_transcript(vtt_content)
    assert "Rahul: I'll finish the backend API by Wednesday." in cleaned
    assert "Priya: Amit, please review the interface before Friday." in cleaned
    assert "00:00:01.000" not in cleaned
    assert "WEBVTT" not in cleaned
