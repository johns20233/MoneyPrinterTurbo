"""
Professional finishing for PDFLibero scene videos, done in one FFmpeg pass.

- Opening hook: the video title, large, over the first seconds.
- Social-style subtitles: 2-3 words at a time, big bold text with a thick
  outline, the word being spoken highlighted in colour.
- Background music that is clearly audible but ducks under the narrator
  (side-chain compression) and fades out at the end.

Subtitle and title overlays are drawn with Pillow into transparent PNG
"states" and joined into ONE overlay stream (concat demuxer with exact
durations), so no libass is needed and the filter graph stays small.
Anything unsupported (for example non-Latin scripts) or any failure raises,
and the caller falls back to the classic MoviePy finishing.
"""

import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from app.services import deadline
from app.utils import utils

FPS = 30
FONT_FILE = "BeVietnamPro-Bold.ttf"
HIGHLIGHT = (255, 214, 0, 255)      # spoken word
TEXT = (255, 255, 255, 255)
OUTLINE = (0, 0, 0, 255)
MAX_WORDS = 3
MAX_CHUNK_CHARS = 20
SUBTITLE_Y = 0.70                   # centre of the subtitle block (fraction of height)
TITLE_SECONDS = 2.4
MUSIC_LEVEL = 0.30                  # music level before ducking
VOICE_LEVEL = 1.0

_WORD = re.compile(r"\S+")
_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


# --------------------------------------------------------------------------
# Timing (pure Python, unit-tested)
# --------------------------------------------------------------------------

def word_times(lines: Sequence[Tuple[float, float, str]]) -> List[Tuple[float, float, str, int]]:
    """(start, end, word, line_index) with time shared by word length inside each line."""
    words = []
    for index, (start, end, text) in enumerate(lines):
        tokens = _WORD.findall(text or "")
        if not tokens or end <= start:
            continue
        weights = [len(t) + 1 for t in tokens]
        total = sum(weights)
        t = start
        for token, weight in zip(tokens, weights):
            span = (end - start) * weight / total
            words.append((t, t + span, token, index))
            t += span
    return words


def chunks(words: Sequence[Tuple[float, float, str, int]]) -> List[List[Tuple[float, float, str, int]]]:
    """Groups of at most MAX_WORDS words / MAX_CHUNK_CHARS chars, never across lines."""
    groups, current = [], []
    for word in words:
        chars = sum(len(w[2]) + 1 for w in current) + len(word[2])
        if current and (len(current) >= MAX_WORDS or chars > MAX_CHUNK_CHARS or current[-1][3] != word[3]):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def states(words, total: float):
    """
    Overlay timeline: [(start, end, chunk_index, highlighted_word_index)] and
    the chunks. A chunk stays on screen until the next one starts when the
    pause is shorter than 0.35 s; longer pauses show no subtitle.
    """
    groups = chunks(words)
    timeline = []
    for c, group in enumerate(groups):
        show_until = group[-1][1]
        if c + 1 < len(groups) and groups[c + 1][0][0] - show_until <= 0.35:
            show_until = groups[c + 1][0][0]
        for w, word in enumerate(group):
            start = word[0]
            end = group[w + 1][0] if w + 1 < len(group) else show_until
            if end - start > 1e-3:
                timeline.append((start, min(end, total), c, w))
    # Fill the gaps (no subtitle) so the timeline covers 0..total.
    filled, cursor = [], 0.0
    for start, end, c, w in timeline:
        if start > cursor + 1e-3:
            filled.append((cursor, start, None, None))
        start = max(start, cursor)
        if end > start:
            filled.append((start, end, c, w))
            cursor = end
    if cursor < total:
        filled.append((cursor, total, None, None))
    return filled, groups


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def font_path() -> str:
    return os.path.join(utils.font_dir(), FONT_FILE)


def supports_text(text: str, path: Optional[str] = None) -> bool:
    """True when the bold font has every glyph (Latin scripts); else fall back."""
    try:
        font = ImageFont.truetype(path or font_path(), 40)
    except Exception:
        return False
    for ch in set(text):
        if ch.isspace() or ch in "¿¡\"'«»“”‘’…–—-.,;:!?()%€$":
            continue
        if ord(ch) > 0x24F and not (0x1E00 <= ord(ch) <= 0x1EFF):
            return False  # non-Latin script (CJK, Arabic, Devanagari, Cyrillic, ...)
        try:
            if font.getmask(ch).getbbox() is None:
                return False
        except Exception:
            return False
    return True


def _fit_font(draw, text, max_width, size):
    while size > 30:
        font = ImageFont.truetype(font_path(), size)
        if draw.textlength(text, font=font) <= max_width:
            return font
        size -= 4
    return ImageFont.truetype(font_path(), size)


def _wrap(draw, text, font, max_width):
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and draw.textlength(trial, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def draw_state(width, height, group, highlight, title) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if title:
        size = int(width * 0.085)
        font = ImageFont.truetype(font_path(), size)
        lines = _wrap(draw, title, font, width * 0.84)[:4]
        line_h = int(size * 1.18)
        block_h = line_h * len(lines)
        top = int(height * 0.30 - block_h / 2)
        pad = int(size * 0.45)
        widest = max(draw.textlength(l, font=font) for l in lines)
        band = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(band).rounded_rectangle(
            [int((width - widest) / 2 - pad), top - pad, int((width + widest) / 2 + pad), top + block_h + pad],
            radius=int(size * 0.35), fill=(0, 0, 0, 150),
        )
        image = Image.alpha_composite(image, band)
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(lines):
            w = draw.textlength(line, font=font)
            draw.text(((width - w) / 2, top + i * line_h), line, font=font, fill=TEXT,
                      stroke_width=max(3, size // 14), stroke_fill=OUTLINE)
    if group:
        words = [w[2] for w in group]
        text = " ".join(words)
        font = _fit_font(draw, text, width * 0.88, int(width * 0.092))
        stroke = max(4, font.size // 10)
        space = draw.textlength(" ", font=font)
        total_w = draw.textlength(text, font=font)
        x = (width - total_w) / 2
        y = int(height * SUBTITLE_Y - font.size / 2)
        for i, word in enumerate(words):
            draw.text((x, y), word, font=font, fill=HIGHLIGHT if i == highlight else TEXT,
                      stroke_width=stroke, stroke_fill=OUTLINE)
            x += draw.textlength(word, font=font) + space
    return image


# --------------------------------------------------------------------------
# FFmpeg finishing
# --------------------------------------------------------------------------

def _duration(path: str) -> float:
    result = subprocess.run([utils.get_ffmpeg_binary(), "-hide_banner", "-i", path], capture_output=True,
                            text=True, errors="replace", timeout=30, check=False)
    match = _DURATION.search(result.stderr or "")
    if not match:
        return 0.0
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def _run(command: List[str], stage: str) -> None:
    deadline.check(stage)
    left = deadline.remaining()
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False,
                            timeout=None if left is None else max(10.0, left))
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed: {(result.stderr or '').strip()[-600:]}")


def finish(
    video_path: str,
    audio_path: str,
    subtitle_lines: Sequence[Tuple[float, float, str]],
    output_file: str,
    width: int,
    height: int,
    title: Optional[str] = None,
    bgm_file: Optional[str] = None,
    voice_volume: float = 1.0,
) -> str:
    all_text = " ".join(t for _, _, t in subtitle_lines) + " " + (title or "")
    if not os.path.exists(font_path()) or not supports_text(all_text):
        raise ValueError("text not supported by the professional finishing font")
    total = _duration(audio_path) or _duration(video_path)
    if total <= 0:
        raise ValueError("unknown duration")

    words = word_times(subtitle_lines)
    timeline, groups = states(words, total)
    work = tempfile.mkdtemp(prefix="finish-", dir=os.path.dirname(output_file) or None)
    try:
        # Overlay states (title over the first seconds, then karaoke chunks).
        cut_points = sorted({0.0, total, *(s for s, _, _, _ in timeline), *(e for _, e, _, _ in timeline)}
                            | ({min(TITLE_SECONDS, total)} if title else set()))
        cache = {}
        entries = []
        for a, b in zip(cut_points, cut_points[1:]):
            if b - a < 1e-3:
                continue
            middle = (a + b) / 2
            state = next((s for s in timeline if s[0] <= middle < s[1]), None)
            chunk_index, word_index = (state[2], state[3]) if state else (None, None)
            show_title = bool(title) and middle < TITLE_SECONDS
            key = (chunk_index, word_index, show_title)
            if key not in cache:
                path = os.path.join(work, f"s{len(cache):04d}.png")
                group = groups[chunk_index] if chunk_index is not None else None
                draw_state(width, height, group, word_index, title if show_title else None).save(path)
                cache[key] = path
            entries.append((cache[key], b - a))
        listing = os.path.join(work, "overlay.txt")
        with open(listing, "w", encoding="utf-8") as fp:
            for path, duration in entries:
                fp.write(f"file '{path}'\nduration {duration:.4f}\n")
            fp.write(f"file '{entries[-1][0]}'\n")  # concat demuxer needs the last file twice

        command = [utils.get_ffmpeg_binary(), "-y", "-loglevel", "error",
                   "-i", video_path, "-i", audio_path,
                   "-f", "concat", "-safe", "0", "-i", listing]
        music = bool(bgm_file and os.path.exists(bgm_file))
        if music:
            command += ["-stream_loop", "-1", "-i", bgm_file]
        fade_start = max(0.0, total - 1.5)
        graph = (
            f"[2:v]fps={FPS},format=rgba[ov];"
            f"[0:v][ov]overlay=0:0:format=auto:shortest=0,format=yuv420p[v];"
            f"[1:a]aresample=44100,volume={VOICE_LEVEL * float(voice_volume or 1.0):.3f},asplit=2[voice][side];"
        )
        if music:
            graph += (
                f"[3:a]aresample=44100,volume={MUSIC_LEVEL},atrim=0:{total:.3f},"
                f"afade=t=out:st={fade_start:.3f}:d=1.5[music];"
                f"[music][side]sidechaincompress=threshold=0.02:ratio=10:attack=40:release=450[ducked];"
                f"[voice][ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]"
            )
        else:
            graph += "[side]anullsink;[voice]anull[a]"
        cpu = max(1, min(os.cpu_count() or 2, 8))
        command += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                    "-t", f"{total:.3f}", "-r", str(FPS),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-threads", str(cpu), "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", output_file]
        _run(command, "final finishing")
        logger.success(
            f"professional finishing: {len(groups)} subtitle chunks, {len(cache)} overlay images, "
            f"title={'yes' if title else 'no'}, music={'yes' if music else 'no'}"
        )
        return output_file
    finally:
        shutil.rmtree(work, ignore_errors=True)
