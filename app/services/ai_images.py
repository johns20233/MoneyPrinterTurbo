"""
AI-generated scene images for PDFLibero scene videos.

When the stock libraries have no good clip for a scene (every candidate was
rejected by the visual check), or a long scene would otherwise be one single
clip, an image is generated with OpenAI from the narration of that scene and
turned into a short still clip. The scene montage then animates it (zoom /
pan) like any other shot.

Design rules:
- OFF by default (config ai_scene_images); enabled only with the owner's OK.
- Hard cap per video (ai_scene_images_max, default 6) so the cost is bounded.
- Fail open: any error simply leaves the scene as it was.
- Photorealistic, no text, no logos, no recognisable real people.
"""

import base64
import os
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.utils import utils

DEFAULT_MODEL = "gpt-image-1"
DEFAULT_QUALITY = "medium"
DEFAULT_MAX = 6
LONG_SCENE_WORDS = 18        # about 6 s of narration
WEAK_SCORE = 7               # best stock clip below this: the AI image goes first
STILL_SECONDS = 12
WORKERS = 3
REQUEST_TIMEOUT = 90

_PROMPT = (
    "Photorealistic {orientation} photograph for a narrated explainer video. "
    "It must clearly show: {subject}. "
    "Narration of this scene: \"{narration}\". "
    "Natural light, cinematic composition, sharp focus, realistic everyday setting, "
    "high detail, professional stock photo look. "
    "Absolutely no text, letters, numbers, captions, signs, labels, logos, brands or watermarks. "
    "No recognisable real or famous people; people, if any, are anonymous and natural."
)


def _flag(name: str, default=False) -> bool:
    value = config.app.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def enabled() -> bool:
    return _flag("ai_scene_images", False) and bool(config.app.get("openai_api_key", ""))


def max_images() -> int:
    try:
        return max(0, min(12, int(config.app.get("ai_scene_images_max", DEFAULT_MAX))))
    except (TypeError, ValueError):
        return DEFAULT_MAX


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def weak_threshold() -> int:
    try:
        return int(config.app.get("ai_scene_images_weak_score", WEAK_SCORE))
    except (TypeError, ValueError):
        return WEAK_SCORE


def pick_scenes(
    term_count: int,
    clips_per_scene: Sequence[int],
    contexts: Optional[Sequence[str]],
    limit: int,
    best_scores: Optional[Sequence[Optional[int]]] = None,
    weak_below: int = WEAK_SCORE,
) -> List[int]:
    """
    Scenes that get an AI image, best first: scenes with no clip at all (in
    order), then scenes whose best stock clip only scored a weak pass
    (lowest first), then long scenes with a single clip (longest first).
    At most limit.
    """
    if limit <= 0:
        return []
    empty = [i for i in range(term_count) if clips_per_scene[i] == 0]
    weak = [
        i for i in range(term_count)
        if clips_per_scene[i] > 0 and best_scores and i < len(best_scores)
        and isinstance(best_scores[i], int) and best_scores[i] < weak_below
    ]
    weak.sort(key=lambda i: (best_scores[i], i))
    taken = set(empty) | set(weak)
    long_single = [
        i for i in range(term_count)
        if i not in taken and clips_per_scene[i] == 1 and contexts and i < len(contexts)
        and _words(contexts[i]) >= LONG_SCENE_WORDS
    ]
    long_single.sort(key=lambda i: -_words(contexts[i]))
    return (empty + weak + long_single)[:limit]


def is_weak(score: Optional[int]) -> bool:
    return isinstance(score, int) and score < weak_threshold()


def build_prompt(term: str, narration: str, aspect: VideoAspect) -> str:
    orientation = {
        VideoAspect.portrait: "vertical (portrait)",
        VideoAspect.landscape: "horizontal (landscape)",
    }.get(aspect, "square")
    narration = re.sub(r"\s+", " ", narration or "").strip()[:400] or term
    return _PROMPT.format(orientation=orientation, subject=term.strip(), narration=narration.replace('"', "'"))


def _size(aspect: VideoAspect) -> str:
    if aspect == VideoAspect.portrait:
        return "1024x1536"
    if aspect == VideoAspect.landscape:
        return "1536x1024"
    return "1024x1024"


def _target(aspect: VideoAspect) -> Tuple[int, int]:
    try:
        return aspect.to_resolution()
    except Exception:
        return (1080, 1920)


def _still_clip(image_path: str, out_path: str, width: int, height: int) -> None:
    command = [
        utils.get_ffmpeg_binary(), "-y", "-loglevel", "error",
        "-loop", "1", "-i", image_path, "-t", str(STILL_SECONDS),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=yuv420p",
        "-r", "30", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-tune", "stillimage",
        "-an", out_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=120, check=False)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"still clip failed: {(result.stderr or '').strip()[-300:]}")


# Models tried in order. gpt-image-1 / gpt-image-1-mini need a verified
# OpenAI organisation; dall-e-3 works without it. A model the account cannot
# use is remembered for the life of the process so it is not retried.
FALLBACK_MODELS = ["gpt-image-1", "gpt-image-1-mini", "dall-e-3"]
_unavailable: set = set()


def _models() -> List[str]:
    first = config.app.get("ai_scene_images_model", "") or DEFAULT_MODEL
    ordered = [first] + [m for m in FALLBACK_MODELS if m != first]
    return [m for m in ordered if m not in _unavailable]


def _model_unavailable(error: Exception) -> bool:
    text = str(error).lower()
    status = getattr(error, "status_code", None)
    return status in (403, 404) or "does not exist" in text or "must be verified" in text \
        or "not have access" in text or ("model" in text and "invalid_value" in text)


def _request(client, model: str, prompt: str, aspect: VideoAspect) -> str:
    if model.startswith("dall-e"):
        size = {"portrait": "1024x1792", "landscape": "1792x1024"}.get(aspect.name, "1024x1024")
        response = client.images.generate(model=model, prompt=prompt[:3900], size=size,
                                          quality="standard", response_format="b64_json", n=1)
    else:
        response = client.images.generate(model=model, prompt=prompt, size=_size(aspect),
                                          quality=config.app.get("ai_scene_images_quality", "") or DEFAULT_QUALITY, n=1)
    data = response.data[0].b64_json if response.data else None
    if not data:
        raise ValueError("empty image")
    return data


def generate_clip(term: str, narration: str, aspect: VideoAspect, directory: str) -> Optional[str]:
    """One AI image as a still MP4 clip, or None on any failure."""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config.app.get("openai_api_key", ""),
            base_url=config.app.get("openai_base_url") or None,
            timeout=REQUEST_TIMEOUT,
        )
        prompt = build_prompt(term, narration, aspect)
        data, used = None, None
        for model in _models():
            try:
                data, used = _request(client, model, prompt, aspect), model
                break
            except Exception as e:
                if not _model_unavailable(e):
                    raise
                _unavailable.add(model)
                logger.warning(f"AI image model {model} not available for this account, trying the next one")
        if not data:
            raise RuntimeError("no image model available for this OpenAI account")
        directory = directory or utils.storage_dir("cache_videos", create=True)
        os.makedirs(directory, exist_ok=True)
        name = f"ai-scene-{uuid.uuid4().hex[:12]}"
        image_path = os.path.join(directory, f"{name}.png")
        with open(image_path, "wb") as fp:
            fp.write(base64.b64decode(data))
        width, height = _target(aspect)
        clip_path = os.path.join(directory, f"{name}.mp4")
        _still_clip(image_path, clip_path, width, height)
        logger.info(f"AI scene image created with {used}: term={term!r}")
        return clip_path
    except Exception as e:
        logger.warning(f"AI scene image failed: term={term!r}, error={type(e).__name__}: {str(e)[:200]}")
        return None


def fill_scenes(
    search_terms: Sequence[str],
    clips_per_scene: Sequence[int],
    contexts: Optional[Sequence[str]],
    aspect: VideoAspect,
    directory: str,
    best_scores: Optional[Sequence[Optional[int]]] = None,
) -> List[Tuple[int, str]]:
    """(scene_index, clip_path) for every AI image generated (bounded, parallel)."""
    if not enabled():
        return []
    chosen = pick_scenes(len(search_terms), clips_per_scene, contexts, max_images(), best_scores, weak_threshold())
    if not chosen:
        return []
    logger.info(f"generating {len(chosen)} AI scene images for scenes {[i + 1 for i in chosen]}")

    def work(index: int):
        narration = contexts[index] if contexts and index < len(contexts) else ""
        return index, generate_clip(search_terms[index], narration, aspect, directory)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(work, chosen))
    made = [(i, p) for i, p in results if p]
    logger.success(f"AI scene images: {len(made)}/{len(chosen)} generated")
    return made
