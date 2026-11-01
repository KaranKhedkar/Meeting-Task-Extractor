import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.language import Language
from ..core.config import settings

_nlp_instance: Optional[Language] = None


def get_spacy_nlp() -> Language:
    """Singleton getter for the spaCy language model."""
    global _nlp_instance
    if _nlp_instance is None:
        try:
            _nlp_instance = spacy.load(settings.SPACY_MODEL)
        except Exception:
            # Fallback to minimal blank model if en_core_web_sm is not yet installed
            try:
                _nlp_instance = spacy.load("en_core_web_sm")
            except Exception:
                _nlp_instance = spacy.blank("en")
                _nlp_instance.add_pipe("sentencizer")
    return _nlp_instance


@dataclass
class ProcessedSentence:
    utterance_idx: int
    sentence_idx: int
    speaker: Optional[str]
    text: str
    doc: spacy.tokens.Doc
    raw_utterance: str


def parse_vtt_transcript(content: str) -> str:
    """
    Strips WebVTT headers, cue identifiers, and timestamp lines.
    Preserves speaker voice tags like <v Speaker>text</v> by converting them to Speaker: text.
    """
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines = []
    
    # Regex to match VTT timestamp lines (e.g. 00:00:01.000 --> 00:00:04.000)
    timestamp_pattern = re.compile(r"^\s*\d{2}:\d{2}(?::\d{2})?(?:\.\d{3})?\s*-->\s*\d{2}:\d{2}(?::\d{2})?(?:\.\d{3})?.*$")
    # Regex to match voice tag: <v Speaker>text</v>
    voice_tag_pattern = re.compile(r"<v\s+([^>]+)>(.*?)(?:</v>)?$", re.IGNORECASE)
    # Generic HTML/VTT tag pattern
    tag_cleaner = re.compile(r"<[^>]+>")

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        if line_strip.startswith("WEBVTT") or line_strip.startswith("NOTE") or line_strip.startswith("STYLE"):
            continue
        if timestamp_pattern.match(line_strip):
            continue
        if line_strip.isdigit():  # Cue index number
            continue
            
        # Check for <v Speaker>text</v>
        v_match = voice_tag_pattern.search(line_strip)
        if v_match:
            speaker_name = v_match.group(1).strip()
            rest_text = tag_cleaner.sub("", v_match.group(2)).strip()
            cleaned_lines.append(f"{speaker_name}: {rest_text}")
        else:
            # Clean residual HTML tags
            clean_text = tag_cleaner.sub("", line_strip).strip()
            if clean_text:
                cleaned_lines.append(clean_text)

    return "\n".join(cleaned_lines)


def extract_speaker_and_text(line: str) -> Tuple[Optional[str], str]:
    """
    Extracts speaker label from line if present.
    Recognizes:
      "Rahul: I'll complete..."
      "[Rahul]: I'll complete..."
      "(Rahul): I'll complete..."
      "Rahul - I'll complete..."
    Returns (speaker, text). If no speaker, returns (None, line).
    """
    clean_line = line.strip()
    
    # Pattern 1: [Speaker]: or (Speaker):
    bracket_match = re.match(r"^\[([A-Za-z0-9\s._'-]{1,50})\]\s*:\s*(.+)$", clean_line)
    if bracket_match:
        return bracket_match.group(1).strip(), bracket_match.group(2).strip()

    paren_match = re.match(r"^\(([A-Za-z0-9\s._'-]{1,50})\)\s*:\s*(.+)$", clean_line)
    if paren_match:
        return paren_match.group(1).strip(), paren_match.group(2).strip()

    # Pattern 2: Speaker: text (Avoid matching URLs or time like 10:30)
    colon_match = re.match(r"^([A-Z][A-Za-z0-9\s._'-]{0,40})\s*:\s*(.+)$", clean_line)
    if colon_match:
        cand = colon_match.group(1).strip()
        # Ensure it's not a protocol like http: or false match
        if not cand.lower().startswith(("http", "https", "note", "ftp")):
            return cand, colon_match.group(2).strip()

    # Pattern 3: Speaker - text
    dash_match = re.match(r"^([A-Z][A-Za-z0-9\s._'-]{1,40})\s+[-–—]\s+(.+)$", clean_line)
    if dash_match:
        return dash_match.group(1).strip(), dash_match.group(2).strip()

    return None, clean_line


def preprocess_transcript(transcript: str) -> List[ProcessedSentence]:
    """
    Main preprocessing entrypoint:
    1. Normalizes line endings and whitespace.
    2. Detects and parses VTT format if present.
    3. Segments utterances and extracts speaker tags.
    4. Segments utterances into individual grammatical sentences using spaCy.
    5. Returns a structured list of ProcessedSentence objects.
    """
    if not transcript or not transcript.strip():
        return []

    # Check if WebVTT
    is_vtt = "WEBVTT" in transcript or "-->" in transcript
    normalized_text = parse_vtt_transcript(transcript) if is_vtt else transcript.replace("\r\n", "\n").replace("\r", "\n")

    nlp = get_spacy_nlp()
    processed_sentences: List[ProcessedSentence] = []
    
    raw_lines = [l.strip() for l in normalized_text.split("\n") if l.strip()]
    sentence_counter = 0

    for u_idx, line in enumerate(raw_lines):
        speaker, utterance_text = extract_speaker_and_text(line)
        if not utterance_text:
            continue

        doc = nlp(utterance_text)
        
        # In case spaCy has no sentences component, fallback to doc.sents or whole doc
        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]

        for sent in sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Produce spaCy Doc for individual sentence for clean parsing
            sent_doc = sent.as_doc() if hasattr(sent, "as_doc") else nlp(sent_text)

            processed_sentences.append(
                ProcessedSentence(
                    utterance_idx=u_idx,
                    sentence_idx=sentence_counter,
                    speaker=speaker,
                    text=sent_text,
                    doc=sent_doc,
                    raw_utterance=line,
                )
            )
            sentence_counter += 1

    return processed_sentences
