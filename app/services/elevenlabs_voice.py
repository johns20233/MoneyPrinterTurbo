"""
Professional narration with ElevenLabs (PDFLibero).

The whole script is spoken in ONE request (continuous intonation, natural
pauses chosen by the model) with the /with-timestamps endpoint, which returns
the start and end time of every character. Subtitle lines are matched to
those characters, so subtitles and scene timing stay exact.

Used only when an API key and a voice id for the narrator's gender are
configured. Any failure returns None and the caller uses the OpenAI voice.
"""

import base64
import re
from typing import List, Optional, Sequence, Tuple

import requests
from loguru import logger

from app.config import config

API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_MODEL = "eleven_multilingual_v2"
MAX_CHARS = 9500           # multilingual v2 accepts 10 000 characters
REQUEST_TIMEOUT = 120


def _cfg(name: str, default: str = "") -> str:
    value = config.elevenlabs.get(name, default)
    return str(value or default).strip()


def voice_id_for(gender: Optional[str]) -> str:
    key = "voice_female" if (gender or "").lower() == "female" else "voice_male"
    return _cfg(key)


def enabled_for(gender: Optional[str]) -> bool:
    return bool(_cfg("api_key")) and bool(voice_id_for(gender))


def _normalize(text: str) -> Tuple[str, List[int]]:
    """Lower-case letters/digits only, with the index of each kept character."""
    kept, index = [], []
    for i, ch in enumerate(text):
        if ch.isalnum():
            kept.append(ch.lower())
            index.append(i)
    return "".join(kept), index


def align_lines(
    lines: Sequence[str],
    characters: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
) -> Optional[List[Tuple[str, float, float]]]:
    """(line, start, end) for every subtitle line, or None if a line is not found."""
    if not lines or not characters or len(characters) != len(starts) or len(starts) != len(ends):
        return None
    spoken, spoken_index = _normalize("".join(characters))
    items, cursor = [], 0
    for line in lines:
        wanted, _ = _normalize(line)
        if not wanted:
            continue
        pos = spoken.find(wanted, cursor)
        if pos < 0:
            return None
        first = spoken_index[pos]
        last = spoken_index[pos + len(wanted) - 1]
        items.append((line, float(starts[first]), float(ends[last])))
        cursor = pos + len(wanted)
    if not items:
        return None
    # Each line stays until the next one starts (no flicker during pauses).
    joined = []
    for k, (line, start, end) in enumerate(items):
        if k + 1 < len(items):
            end = max(end, items[k + 1][1])
        joined.append((line, start, end))
    return joined


def synthesize(
    text: str,
    gender: Optional[str],
    voice_rate: float,
    voice_file: str,
    lines_of,
) -> Optional[List[Tuple[str, float, float]]]:
    """Write voice_file (MP3) and return timed subtitle lines, or None on any failure."""
    original = text or ""
    text = re.sub(r"\s+", " ", original).strip()
    voice_id = voice_id_for(gender)
    if not text or not voice_id or not _cfg("api_key") or len(text) > MAX_CHARS:
        return None
    try:
        speed = max(0.7, min(1.2, float(voice_rate or 1.0)))
    except (TypeError, ValueError):
        speed = 1.0
    payload = {
        "text": text,
        "model_id": _cfg("narration_model_id", DEFAULT_MODEL),
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": speed,
        },
    }
    try:
        response = requests.post(
            API_URL.format(voice_id=voice_id),
            params={"output_format": OUTPUT_FORMAT},
            headers={"xi-api-key": _cfg("api_key"), "Content-Type": "application/json"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            logger.warning(f"elevenlabs voice failed: HTTP {response.status_code}: {response.text[:200]}")
            return None
        data = response.json()
        audio = base64.b64decode(data.get("audio_base64") or "")
        alignment = data.get("alignment") or {}
        if len(audio) < 1000:
            logger.warning("elevenlabs voice failed: empty audio")
            return None
        with open(voice_file, "wb") as fp:
            fp.write(audio)
        items = align_lines(
            lines_of(original),
            alignment.get("characters") or [],
            alignment.get("character_start_times_seconds") or [],
            alignment.get("character_end_times_seconds") or [],
        )
        if not items:
            logger.warning("elevenlabs voice: subtitle alignment failed")
            return None
        logger.info(f"elevenlabs voice: {voice_id} ({payload['model_id']}), {len(items)} lines, "
                    f"{items[-1][2]:.2f}s")
        return items
    except Exception as exc:
        logger.warning(f"elevenlabs voice failed: {type(exc).__name__}: {str(exc)[:200]}")
        return None
