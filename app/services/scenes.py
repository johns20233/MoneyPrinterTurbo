"""
Scene timing for "one scene per sentence" (PDFLibero).

PDFLibero sends one stock search term per script sentence, in order. This
module turns the real narration timing (from the generated SRT subtitles)
into one target duration per downloaded clip, so each image appears while
the voice is saying the sentence it illustrates.

Everything here is pure Python (no MoviePy), so it is unit-tested directly.
"""

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

MIN_SCENE_SECONDS = 1.5

_SENTENCE_SPLIT = re.compile(r"[.!?;:\n]+")
MAX_PART_CHARS = 120
_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _split_long(part: str) -> List[str]:
    """Split a part longer than MAX_PART_CHARS at the comma nearest its middle."""
    if len(part) <= MAX_PART_CHARS:
        return [part]
    commas = [i for i, ch in enumerate(part) if ch == ","]
    if not commas:
        return [part]
    middle = len(part) / 2
    cut = min(commas, key=lambda i: (abs(i - middle), i))
    left, right = part[:cut].strip(), part[cut + 1:].strip()
    if not left or not right:
        return [part]
    return _split_long(left) + _split_long(right)


def split_sentences(script: str) -> List[str]:
    """
    Narration parts, SAME rule as PDFLibero (stockNarrationScenes):
    cut on . ! ? ; : and line breaks, then split any part longer than
    120 characters at the comma nearest its middle (recursively).
    """
    parts = []
    for raw in _SENTENCE_SPLIT.split(script or ""):
        if raw.strip():
            parts.extend(_split_long(raw.strip()))
    return parts


def _letters(text: str) -> str:
    """Comparable text: lowercase letters and digits only, accents removed."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _seconds(stamp: str) -> float:
    h, m, s, ms = _SRT_TIME.match(stamp.strip()).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def parse_srt_entries(entries: Sequence[Tuple[int, str, str]]) -> List[Tuple[float, float, str]]:
    """Convert subtitle.file_to_subtitles() output to (start, end, text)."""
    parsed = []
    for _, times, text in entries or []:
        parts = times.split("-->")
        if len(parts) != 2:
            continue
        try:
            parsed.append((_seconds(parts[0]), _seconds(parts[1]), text))
        except (AttributeError, ValueError):
            continue
    return parsed


def sentence_end_times(
    sentences: List[str],
    srt: List[Tuple[float, float, str]],
    audio_duration: float,
) -> List[float]:
    """
    End time of every sentence.

    Subtitle lines and sentences do not split the text at the same places, so
    both are aligned by cumulative letter count: a sentence ends at the moment
    the subtitles have spoken as many letters as the script up to that
    sentence (interpolated inside the subtitle line). Without usable
    subtitles, time is shared in proportion to sentence length.
    """
    lengths = [max(1, len(_letters(s))) for s in sentences]
    total_letters = sum(lengths)
    cumulative, acc = [], 0
    for length in lengths:
        acc += length
        cumulative.append(acc)

    spoken = [(start, end, len(_letters(text))) for start, end, text in srt if _letters(text)]
    srt_letters = sum(n for _, _, n in spoken)
    if not spoken or srt_letters < total_letters * 0.6:
        return [audio_duration * c / total_letters for c in cumulative]

    scale = srt_letters / total_letters  # tolerate small text differences
    ends = []
    for target in cumulative:
        goal = target * scale
        done = 0
        moment = spoken[-1][1]
        for start, end, n in spoken:
            if done + n >= goal:
                fraction = (goal - done) / n if n else 1.0
                moment = start + (end - start) * fraction
                break
            done += n
        ends.append(min(moment, audio_duration))
    ends[-1] = audio_duration
    return ends


def scene_durations(
    script: str,
    term_count: int,
    term_indices: List[int],
    srt_entries: Sequence[Tuple[int, str, str]],
    audio_duration: float,
) -> Optional[List[float]]:
    """
    Target duration for every downloaded clip.

    term_indices[k] is the index (in the original term list) of the k-th
    downloaded clip. Terms without a clip give their time to the previous
    scene (or the next one at the start), so the timeline never drifts.
    Returns None when scene timing cannot be built; the caller then keeps
    the classic fixed-length montage.
    """
    sentences = split_sentences(script)
    if not sentences or term_count <= 0 or not term_indices or audio_duration <= 0:
        return None

    ends = sentence_end_times(sentences, parse_srt_entries(srt_entries), audio_duration)
    starts = [0.0] + ends[:-1]
    count = len(sentences)

    # Time span of every term: one sentence each when counts match; grouped
    # consecutive sentences when there are fewer terms; split sentences when
    # there are more terms than sentences.
    spans = []
    if count > term_count:
        # Same grouping as PDFLibero: group i = parts[floor(i*S/T) : floor((i+1)*S/T)].
        for i in range(term_count):
            first = (i * count) // term_count
            last = ((i + 1) * count) // term_count
            span_end = audio_duration if i == term_count - 1 else ends[last - 1]
            spans.append(max(0.0, span_end - starts[first]))
    for i in ([] if count > term_count else range(term_count)):
        first = i * count / term_count
        last = (i + 1) * count / term_count

        def at(position: float) -> float:
            index = min(int(position), count - 1)
            within = position - index
            return starts[index] + (ends[index] - starts[index]) * within

        span_start = at(first) if first < count else audio_duration
        span_end = audio_duration if i == term_count - 1 else at(last)
        spans.append(max(0.0, span_end - span_start))

    present = sorted(set(i for i in term_indices if 0 <= i < term_count))
    if not present:
        return None
    owned = {i: spans[i] for i in present}
    for i in range(term_count):
        if i in owned:
            continue
        previous = [p for p in present if p < i]
        owner = previous[-1] if previous else present[0]
        owned[owner] += spans[i]

    durations = [owned[i] for i in term_indices]
    # Very short scenes flash on screen; borrow time from the longest scene.
    for k, value in enumerate(durations):
        if value < MIN_SCENE_SECONDS and len(durations) > 1:
            donor = max(range(len(durations)), key=lambda j: durations[j])
            missing = MIN_SCENE_SECONDS - value
            if durations[donor] - missing >= MIN_SCENE_SECONDS:
                durations[donor] -= missing
                durations[k] = MIN_SCENE_SECONDS
    return durations


def narration_for_terms(script: str, term_count: int) -> List[str]:
    """
    Narration text each search term illustrates (same grouping as
    scene_durations / PDFLibero). Used to judge stock clips against what the
    narrator actually says, not only against the short search term.
    """
    parts = split_sentences(script)
    if not parts or term_count <= 0:
        return [""] * max(0, term_count)
    count = len(parts)
    texts = []
    for i in range(term_count):
        if count > term_count:
            first = (i * count) // term_count
            last = ((i + 1) * count) // term_count
            texts.append(" ".join(parts[first:last]))
        else:
            texts.append(parts[min(count - 1, (i * count) // term_count)])
    return texts
