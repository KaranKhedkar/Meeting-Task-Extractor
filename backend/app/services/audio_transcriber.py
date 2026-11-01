import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from faster_whisper import WhisperModel

# Global model cache to avoid reloading on each request
_WHISPER_MODEL: Optional[WhisperModel] = None


def get_whisper_model(model_size: str = "tiny.en") -> WhisperModel:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        # Load lightweight CPU-optimized model (39MB) with int8 quantization
        _WHISPER_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def format_timestamp(seconds: float) -> str:
    """Formats float seconds to MM:SS string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def transcribe_meeting_audio(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Transcribes uploaded meeting audio file using local Faster-Whisper.
    Returns:
      - formatted_transcript: Clean text formatted with timestamps and speaker placeholders
      - segments: Granular list of segments with start, end, and text
      - duration_seconds: Total length of the audio
    """
    model = get_whisper_model()

    # Determine extension
    ext = Path(filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
        temp_audio.write(file_bytes)
        temp_path = temp_audio.name

    try:
        segments_gen, info = model.transcribe(
            temp_path,
            beam_size=3,
            language="en",
            vad_filter=True, # Voice Activity Detection to skip silence
            vad_parameters=dict(min_silence_duration_ms=400),
        )

        segments_list = []
        transcript_lines = []

        # Alternating speaker estimation based on acoustic pauses
        current_speaker_idx = 1
        last_end = 0.0

        for s in segments_gen:
            text = s.text.strip()
            if not text:
                continue

            # If gap between utterances is > 1.2s, treat as turn transition
            if s.start - last_end > 1.2:
                current_speaker_idx = (current_speaker_idx % 4) + 1

            speaker_tag = f"Speaker {current_speaker_idx}"
            time_tag = format_timestamp(s.start)
            formatted_line = f"{speaker_tag} [{time_tag}]: {text}"

            transcript_lines.append(formatted_line)
            segments_list.append({
                "speaker": speaker_tag,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": text,
            })
            last_end = s.end

        formatted_transcript = "\n".join(transcript_lines)

        return {
            "filename": filename,
            "duration_seconds": round(info.duration, 2),
            "language": info.language,
            "segment_count": len(segments_list),
            "formatted_transcript": formatted_transcript,
            "segments": segments_list,
        }

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

