"""
Visual relevance check for stock clips (PDFLibero).

Before a stock clip is downloaded, its preview image is shown to a vision
model together with the search term of that scene. Clips that do not show
the requested subject, or that are dark, blurry or mostly empty, are dropped;
the remaining ones are ordered best-first so the script-order downloader
uses the most relevant clip of every scene.

Design rules:
- Fail open: any error (no key, timeout, invalid answer) returns the
  candidates unchanged, so a vision outage never blocks a paid render.
  Safety is still enforced later by PDFLibero's mandatory final moderation.
- Bounded cost: one request per search term, at most MAX_CANDIDATES preview
  images at low detail.
- When the check runs, only approved clips are returned: a scene whose
  candidates are all rejected contributes no clip, and the script-order
  downloader fills the time with approved clips of the other scenes instead
  of showing something unrelated.
"""

import json
import re
from typing import List

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo

MAX_CANDIDATES = 8
REQUEST_TIMEOUT_SECONDS = 20
MIN_SCORE = 6

_PROMPT = (
    "You review stock video preview images for a narrated vertical video. "
    "Scene search term: {term!r}.{context} "
    "For each numbered image, score 0-10 how clearly it shows what this scene is about, "
    "as a viewer hearing the narration would understand it (an audio meter is NOT an electricity meter; "
    "a dam is NOT home insulation; food is NOT a light bulb; a car dashboard is NOT a home meter). "
    "Give 0-3 if the image is dark, blurry, mostly empty, abstract, or unrelated. "
    "Give 0-4 if the main element is a sign, label, logo, slogan, banner or large text, "
    "or a generic activity that does not show the specific action or object described. "
    "Answer JSON only: "
    '{{"scores": [s1, s2, ...]}} with exactly {count} integers in image order.'
)


def _enabled() -> bool:
    value = config.app.get("visual_relevance_check", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _thumbnail(item: MaterialInfo) -> str | None:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    url = source.get("thumbnail")
    return url if isinstance(url, str) and url.startswith("https://") else None


def _parse_scores(text: str, count: int) -> List[int] | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    scores = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(scores, list) or len(scores) != count:
        return None
    parsed = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        parsed.append(max(0, min(10, int(score))))
    return parsed


def _ask_vision(term: str, urls: List[str], context: str = "") -> List[int] | None:
    api_key = config.app.get("openai_api_key", "")
    if not api_key:
        return None
    from openai import OpenAI  # local import keeps tests and startup light

    client = OpenAI(
        api_key=api_key,
        base_url=config.app.get("openai_base_url") or None,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    narration = f" Narration of this scene: {context[:400]!r}." if context else ""
    content = [{"type": "text", "text": _PROMPT.format(term=term, context=narration, count=len(urls))}]
    for url in urls:
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})
    response = client.chat.completions.create(
        model=config.app.get("visual_relevance_model") or "gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=60,
    )
    text = response.choices[0].message.content if response and response.choices else ""
    return _parse_scores(text or "", len(urls))


def rank_by_relevance(search_term: str, items: List[MaterialInfo], context: str = "") -> List[MaterialInfo]:
    """Return only approved clips, best-first. Fail open (unchanged) on any error."""
    if not items or not _enabled():
        return items

    reviewed = [item for item in items[:MAX_CANDIDATES] if _thumbnail(item)]
    if not reviewed:
        return items

    try:
        scores = _ask_vision(search_term, [_thumbnail(item) for item in reviewed], context)
    except Exception as exc:
        logger.warning(
            "visual relevance check failed, keeping provider order: "
            f"term={search_term!r}, error={type(exc).__name__}"
        )
        return items
    if scores is None:
        logger.warning(f"visual relevance check unavailable, keeping provider order: term={search_term!r}")
        return items

    approved = sorted(
        (pair for pair in zip(scores, reviewed) if pair[0] >= MIN_SCORE),
        key=lambda pair: -pair[0],
    )
    logger.info(
        f"visual relevance: term={search_term!r}, scores={scores}, "
        f"approved={len(approved)}/{len(reviewed)}"
    )
    for score, item in approved:
        if isinstance(item.source_info, dict):
            item.source_info = dict(item.source_info)
            item.source_info["relevance_score"] = score
    return [item for _, item in approved]
