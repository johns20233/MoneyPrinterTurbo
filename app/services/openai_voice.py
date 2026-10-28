"""
Natural narration with OpenAI TTS (PDFLibero).

The standard Edge voices sound robotic. When enabled, the narration is
spoken by OpenAI's gpt-4o-mini-tts with narrator instructions, sentence by
sentence (parallel requests), trimmed and joined with short natural pauses.
Because every sentence is synthesised separately, its exact start and end
are known, so subtitles and scene timing stay precise without word
boundaries. Any failure returns None and the caller uses the standard voice.
"""

import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from app.config import config
from app.utils import utils

MODEL = "gpt-4o-mini-tts"
VOICES = {"Male": "ash", "Female": "coral"}
SENTENCE_GAP = 0.22          # pause between sentences (seconds)
WORKERS = 4
REQUEST_TIMEOUT = 40
MAX_CHARS = 3500             # OpenAI TTS input limit is 4096

ACCENTS = {
    "es-ES": "Spanish from Spain (Castilian accent)",
    "es-MX": "Mexican Spanish",
    "es-AR": "Rioplatense Spanish from Argentina",
    "es-US": "neutral Latin American Spanish",
    "it-IT": "Italian from Italy",
    "en-US": "American English",
    "en-GB": "British English",
    "fr-FR": "French from France",
    "de-DE": "German from Germany",
    "pt-BR": "Brazilian Portuguese",
    "pt-PT": "European Portuguese",
}

_SENTENCE_END = re.compile(r"(?<=[.!?…;:])\s+|\n+")
_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def instructions(locale: str) -> str:
    accent = ACCENTS.get(locale) or f"the language of the text, with a native accent for {locale}"
    return (
        f"You are a professional documentary and explainer-video narrator. Speak {accent}. "
        "Warm, clear and confident, with natural intonation and a fluid, steady pace, "
        "like a high-quality social media video. Never sound robotic, never rush, "
        "and do not add long pauses inside the sentence."
    )


def split_chunks(text: str) -> List[str]:
    """Sentences to synthesise (each keeps its final punctuation)."""
    chunks = [c.strip() for c in _SENTENCE_END.split(text or "") if c and c.strip()]
    return [c for c in chunks if re.search(r"\w", c)]


def assign_lines(
    chunks: Sequence[str],
    durations: Sequence[float],
    gap: float,
    lines_of,
) -> Optional[List[Tuple[str, float, float]]]:
    """
    Subtitle lines with their time span. lines_of(text) must split text into
    subtitle lines exactly like the subtitle builder does. Inside a sentence,
    time is shared in proportion to line length.
    """
    items = []
    start = 0.0
    for index, (chunk, duration) in enumerate(zip(chunks, durations)):
        lines = lines_of(chunk)
        if not lines:
            start += duration + (gap if index < len(chunks) - 1 else 0)
            continue
        weights = [max(1, len(line)) for line in lines]
        total = sum(weights)
        t = start
        for line, weight in zip(lines, weights):
            span = duration * weight / total
            items.append((line, t, t + span))
            t += span
        start += duration + (gap if index < len(chunks) - 1 else 0)
    return items or None


def _ffmpeg() -> str:
    return utils.get_ffmpeg_binary()


def _duration(path: str) -> float:
    result = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path], capture_output=True,
                            text=True, errors="replace", timeout=30, check=False)
    match = _DURATION.search(result.stderr or "")
    if not match:
        return 0.0
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def _speak(client, text: str, voice: str, locale: str, raw_path: str, clean_path: str) -> float:
    last_error = None
    for _ in range(2):
        try:
            response = client.audio.speech.create(
                model=config.app.get("openai_tts_model") or MODEL,
                voice=voice,
                input=text,
                instructions=instructions(locale),
                response_format="wav",
            )
            with open(raw_path, "wb") as fp:
                fp.write(response.read() if hasattr(response, "read") else response.content)
            break
        except Exception as exc:  # retried once
            last_error = exc
    else:
        raise RuntimeError(f"openai tts failed: {type(last_error).__name__}: {last_error}")
    # Trim leading/trailing silence so the pauses between sentences are even.
    trim = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.04,"
            "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.06,areverse")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", raw_path, "-af", trim,
                    "-ar", "24000", "-ac", "1", clean_path], check=True, timeout=60,
                   capture_output=True)
    duration = _duration(clean_path)
    if duration <= 0.1:
        raise RuntimeError("openai tts returned empty audio")
    return duration


def synthesize(text: str, voice_name: str, gender: Optional[str], voice_rate: float,
               voice_file: str, lines_of) -> Optional[List[Tuple[str, float, float]]]:
    """Write voice_file and return subtitle lines with times, or None on any failure."""
    api_key = config.app.get("openai_api_key", "")
    chunks = split_chunks(text)
    if not api_key or not chunks or sum(len(c) for c in chunks) > MAX_CHARS * 4:
        return None
    parts = voice_name.split("-")
    locale = "-".join(parts[:2]) if len(parts) >= 2 else "en-US"
    voice = VOICES.get(gender or "", VOICES["Female"])

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=config.app.get("openai_base_url") or None,
                    timeout=REQUEST_TIMEOUT, max_retries=0)
    work = tempfile.mkdtemp(prefix="voice-", dir=os.path.dirname(voice_file) or None)
    try:
        jobs = [(chunk, os.path.join(work, f"raw-{i:03d}.wav"), os.path.join(work, f"clean-{i:03d}.wav"))
                for i, chunk in enumerate(chunks)]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            durations = list(pool.map(lambda job: _speak(client, job[0], voice, locale, job[1], job[2]), jobs))

        # Join with even pauses; optional speed change for voice_rate.
        command = [_ffmpeg(), "-y", "-loglevel", "error"]
        for _, _, clean in jobs:
            command += ["-i", clean]
        graph = ""
        for i in range(len(jobs)):
            pad = f",apad=pad_dur={SENTENCE_GAP}" if i < len(jobs) - 1 else ""
            graph += f"[{i}:a]aformat=sample_rates=24000:channel_layouts=mono{pad}[a{i}];"
        graph += "".join(f"[a{i}]" for i in range(len(jobs))) + f"concat=n={len(jobs)}:v=0:a=1"
        rate = 1.0
        try:
            rate = max(0.5, min(2.0, float(voice_rate or 1.0)))
        except (TypeError, ValueError):
            rate = 1.0
        if abs(rate - 1.0) > 0.01:
            graph += f",atempo={rate:.3f}"
        graph += "[out]"
        command += ["-filter_complex", graph, "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "192k", voice_file]
        subprocess.run(command, check=True, timeout=120, capture_output=True)

        scale = 1.0 / rate
        items = assign_lines(chunks, [d * scale for d in durations], SENTENCE_GAP * scale, lines_of)
        total = _duration(voice_file)
        if items and total > 0:
            # Last subtitle ends exactly with the audio.
            line, start, _ = items[-1]
            items[-1] = (line, start, total)
        logger.info(f"openai voice: {voice} ({locale}), {len(chunks)} sentences, {total:.2f}s")
        return items
    except Exception as exc:
        logger.warning(f"openai voice failed: {type(exc).__name__}: {str(exc)[:300]}")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)
