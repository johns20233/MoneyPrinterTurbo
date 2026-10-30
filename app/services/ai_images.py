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


def pick_scenes(
    term_count: int,
    clips_per_scene: Sequence[int],
    contexts: Optional[Sequence[str]],
    limit: int,
) -> List[int]:
    """
    Scenes that get an AI image, best first: scenes with no clip at all (in
    order), then long scenes with a single clip (longest first). At most limit.
    """
    if limit <= 0:
        return []
    empty = [i for i in range(term_count) if clips_per_scene[i] == 0]
    long_single = [
        i for i in range(term_count)
        if clips_per_scene[i] == 1 and contexts and i < len(contexts) and _words(contexts[i]) >= LONG_SCENE_WORDS
    ]
    long_single.sort(key=lambda i: -_words(contexts[i]))
    return (empty + long_single)[:limit]


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


def generate_clip(term: str, narration: str, aspect: VideoAspect, directory: str) -> Optional[str]:
    """One AI image as a still MP4 clip, or None on any failure."""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config.app.get("openai_api_key", ""),
            base_url=config.app.get("openai_base_url") or None,
            timeout=REQUEST_TIMEOUT,
        )
        response = client.images.generate(
            model=config.app.get("ai_scene_images_model", "") or DEFAULT_MODEL,
            prompt=build_prompt(term, narration, aspect),
            size=_size(aspect),
            quality=config.app.get("ai_scene_images_quality", "") or DEFAULT_QUALITY,
            n=1,
        )
        data = response.data[0].b64_json if response.data else None
        if not data:
            raise ValueError("empty image")
        directory = directory or utils.storage_dir("cache_videos", create=True)
        os.makedirs(directory, exist_ok=True)
        name = f"ai-scene-{uuid.uuid4().hex[:12]}"
        image_path = os.path.join(directory, f"{name}.png")
        with open(image_path, "wb") as fp:
            fp.write(base64.b64decode(data))
        width, height = _target(aspect)
        clip_path = os.path.join(directory, f"{name}.mp4")
        _still_clip(image_path, clip_path, width, height)
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
) -> List[Tuple[int, str]]:
    """(scene_index, clip_path) for every AI image generated (bounded, parallel)."""
    if not enabled():
        return []
    chosen = pick_scenes(len(search_terms), clips_per_scene, contexts, max_images())
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
