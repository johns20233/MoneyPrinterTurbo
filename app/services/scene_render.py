"""
Professional scene montage for PDFLibero ("scene sync"), rendered with FFmpeg.

Every narration scene gets several DIFFERENT stock clips (one shot per clip,
never the same clip twice in a video). Each shot gets its own smooth camera
move (zoom in, zoom out or pan, never the same move twice in a row) and a
light colour grade. Scenes are joined with short varied transitions
(smooth slides, fades, circle reveal) timed so that every scene still starts
exactly when the narrator starts its sentence.

Everything runs in FFmpeg (no per-frame Python), so it is much faster than
the MoviePy path. Any failure raises; the caller falls back to the classic
montage, so a paid render never depends on this module.
"""

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

from loguru import logger

from app.services import deadline
from app.utils import utils

FPS = 30
SHOT_SECONDS = 2.6          # a new clip roughly every 2.6 s
MAX_SHOTS_PER_SCENE = 4
MIN_SPEED = 0.6             # slowest playback used to stretch a short clip
TRANSITION_SECONDS = 0.4
MIN_SCENE_FOR_TRANSITION = 1.6
ZOOM = 0.12                 # Ken Burns amplitude (12 %)
PRESCALE = 1.25             # render moves on a larger frame: smoother motion
# Shots and the joined montage are intermediate files (the final step
# re-encodes with subtitles), so they use fast presets with high quality.
SHOT_PRESET, SHOT_CRF = "ultrafast", "16"
X264_PRESET, COMPOSE_CRF = "superfast", "18"

# Only transitions available in FFmpeg 4.3 (Debian bullseye).
TRANSITIONS = [
    "smoothleft", "fade", "smoothup", "circleopen",
    "slideleft", "smoothright", "fadeblack", "smoothdown",
]
MOTIONS = ["in", "right", "out", "left", "up", "in", "down", "out"]


def _cpu() -> int:
    return max(1, min(os.cpu_count() or 2, 8))


# --------------------------------------------------------------------------
# Planning (pure Python, unit-tested)
# --------------------------------------------------------------------------

def plan(
    scene_durations: Sequence[float],
    scene_clips: Sequence[Sequence[str]],
    clip_lengths: Dict[str, float],
    fps: int = FPS,
) -> List[dict]:
    """
    Frame-exact plan. Scene i covers frames [round(B_i*fps), round(B_i+1*fps))
    of the narration. A scene followed by a transition renders T extra frames
    that the transition overlaps, so the timeline never drifts.
    """
    if not scene_durations or len(scene_durations) != len(scene_clips):
        raise ValueError("scene plan mismatch")

    bounds = [0.0]
    for duration in scene_durations:
        bounds.append(bounds[-1] + max(0.0, float(duration)))
    frames = [round(bounds[i + 1] * fps) - round(bounds[i] * fps) for i in range(len(scene_durations))]
    tf = round(TRANSITION_SECONDS * fps)
    min_frames_for_transition = round(MIN_SCENE_FOR_TRANSITION * fps)

    scenes = []
    motion_counter = 0
    transition_counter = 0
    for i, (scene_frames, clips) in enumerate(zip(frames, scene_clips)):
        clips = [c for c in clips if clip_lengths.get(c, 0) > 0]
        if scene_frames <= 0 or not clips:
            raise ValueError(f"scene {i + 1} has no usable clip")
        # Transition INTO the next scene, when both scenes are long enough.
        transition = None
        if i + 1 < len(frames) and scene_frames >= min_frames_for_transition and frames[i + 1] >= min_frames_for_transition:
            transition = TRANSITIONS[transition_counter % len(TRANSITIONS)]
            transition_counter += 1
        rendered = scene_frames + (tf if transition else 0)

        seconds = scene_frames / fps
        count = max(1, min(MAX_SHOTS_PER_SCENE, round(seconds / SHOT_SECONDS), len(clips)))
        base, extra = divmod(rendered, count)
        shots = []
        for k in range(count):
            shot_frames = base + (1 if k < extra else 0)
            path = clips[k]
            need = shot_frames / fps
            length = clip_lengths[path]
            if length >= need + 0.05:
                speed = 1.0
                start = min(1.0, (length - need) / 2)
            else:
                speed = max(MIN_SPEED, length / need)
                start = 0.0
            motion = MOTIONS[motion_counter % len(MOTIONS)]
            motion_counter += 1
            shots.append({
                "path": path,
                "start": round(start, 3),
                "speed": round(speed, 4),
                "frames": shot_frames,
                "motion": motion,
            })
        scenes.append({"frames": scene_frames, "rendered": rendered, "transition": transition, "shots": shots})
    return scenes


def _motion_filter(motion: str, frames: int, width: int, height: int, fps: int) -> str:
    n = max(1, frames - 1)
    p = f"min(on/{n},1)"
    e = f"(({p})*({p})*(3-2*({p})))"  # smoothstep easing
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    full = f"{1 + ZOOM:.3f}"
    if motion == "in":
        z, x, y = f"1+{ZOOM}*{e}", cx, cy
    elif motion == "out":
        z, x, y = f"{full}-{ZOOM}*{e}", cx, cy
    elif motion == "right":
        z, x, y = full, f"(iw-iw/zoom)*{e}", cy
    elif motion == "left":
        z, x, y = full, f"(iw-iw/zoom)*(1-{e})", cy
    elif motion == "up":
        z, x, y = full, cx, f"(ih-ih/zoom)*(1-{e})"
    else:  # down
        z, x, y = full, cx, f"(ih-ih/zoom)*{e}"
    return f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={width}x{height}:fps={fps}"


def shot_command(shot: dict, output: str, width: int, height: int, threads: int, fps: int = FPS) -> List[str]:
    need_source = shot["frames"] / fps * shot["speed"] + 0.5
    pw, ph = int(width * PRESCALE) // 2 * 2, int(height * PRESCALE) // 2 * 2
    filters = []
    if shot["speed"] != 1.0:
        filters.append(f"setpts=PTS/{shot['speed']}")
    filters += [
        f"fps={fps}",
        f"scale={pw}:{ph}:force_original_aspect_ratio=increase",
        f"crop={pw}:{ph}",
        "setsar=1",
        _motion_filter(shot["motion"], shot["frames"], width, height, fps),
        "eq=contrast=1.06:saturation=1.15",
        "tpad=stop_mode=clone:stop_duration=5",
        "format=yuv420p",
    ]
    return [
        utils.get_ffmpeg_binary(), "-y", "-loglevel", "error",
        "-ss", f"{shot['start']:.3f}", "-t", f"{need_source:.3f}", "-i", shot["path"],
        "-an", "-vf", ",".join(filters),
        "-frames:v", str(shot["frames"]), "-r", str(fps),
        "-c:v", "libx264", "-preset", SHOT_PRESET, "-crf", SHOT_CRF,
        "-pix_fmt", "yuv420p", "-threads", str(threads),
        output,
    ]


def compose_filter(scenes: List[dict], fps: int = FPS, transitions: bool = True) -> tuple[str, str]:
    """Filtergraph joining the rendered shot files (inputs in plan order)."""
    parts = []
    index = 0
    labels = []
    for s, scene in enumerate(scenes):
        inputs = ""
        for _ in scene["shots"]:
            inputs += f"[{index}:v]"
            index += 1
        frames = scene["rendered"] if transitions else scene["frames"]
        parts.append(
            f"{inputs}concat=n={len(scene['shots'])}:v=1:a=0,fps={fps},"
            f"trim=end_frame={frames},setpts=PTS-STARTPTS,format=yuv420p,settb=1/{fps}[s{s}]"
        )
        labels.append(f"[s{s}]")

    acc = labels[0]
    elapsed = scenes[0]["frames"]
    tf = round(TRANSITION_SECONDS * fps)
    for s in range(1, len(scenes)):
        out = f"[j{s}]"
        previous = scenes[s - 1]
        if transitions and previous["transition"]:
            parts.append(
                f"{acc}{labels[s]}xfade=transition={previous['transition']}:"
                f"duration={tf / fps:.4f}:offset={elapsed / fps:.4f},settb=1/{fps}{out}"
            )
        else:
            parts.append(f"{acc}{labels[s]}concat=n=2:v=1:a=0,fps={fps},settb=1/{fps}{out}")
        acc = out
        elapsed += scenes[s]["frames"]
    if len(scenes) == 1:
        return ";".join(parts), labels[0]
    return ";".join(parts), acc


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _run(command: List[str], stage: str) -> None:
    deadline.check(stage)
    left = deadline.remaining()
    timeout = None if left is None else max(5.0, left)
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed: {(result.stderr or '').strip()[-600:]}")


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_length(path: str) -> float:
    """Clip length read with ffmpeg itself (the server may have no ffprobe)."""
    result = subprocess.run(
        [utils.get_ffmpeg_binary(), "-hide_banner", "-i", path],
        capture_output=True, text=True, errors="replace", check=False, timeout=30,
    )
    match = _DURATION.search(result.stderr or "")
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def render(
    output_file: str,
    scene_clips: List[List[str]],
    scene_durations: List[float],
    width: int,
    height: int,
    total_duration: Optional[float] = None,
) -> str:
    output_dir = os.path.dirname(output_file)
    try:
        version = subprocess.run([utils.get_ffmpeg_binary(), "-version"], capture_output=True,
                                 text=True, errors="replace", timeout=15).stdout.splitlines()[0]
    except Exception:
        version = "unknown"
    logger.info(f"scene render using {utils.get_ffmpeg_binary()} ({version})")
    lengths = {}
    for clips in scene_clips:
        for path in clips:
            if path not in lengths:
                lengths[path] = probe_length(path)
    scenes = plan(scene_durations, scene_clips, lengths)
    shots = [shot for scene in scenes for shot in scene["shots"]]
    logger.info(
        f"scene render: {len(scenes)} scenes, {len(shots)} shots, "
        f"{len(set(s['path'] for s in shots))} different clips, "
        f"transitions={[s['transition'] for s in scenes]}"
    )

    cpu = _cpu()
    workers = max(1, min(4, cpu // 2))
    threads = max(1, cpu // workers)
    files = [os.path.join(output_dir, f"scene-shot-{k + 1:03d}.mp4") for k in range(len(shots))]
    budget = deadline.get()

    def work(k):
        deadline.adopt(budget)
        _run(shot_command(shots[k], files[k], width, height, threads), "shot rendering")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, range(len(shots))))

        def compose(transitions: bool):
            graph, label = compose_filter(scenes, transitions=transitions)
            command = [utils.get_ffmpeg_binary(), "-y", "-loglevel", "error"]
            for f in files:
                command += ["-i", f]
            command += ["-filter_complex", graph, "-map", label, "-an", "-r", str(FPS),
                        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", COMPOSE_CRF,
                        "-pix_fmt", "yuv420p", "-threads", str(cpu)]
            if total_duration:
                command += ["-t", f"{total_duration:.3f}"]
            command.append(output_file)
            _run(command, "scene composition")

        try:
            compose(transitions=True)
        except deadline.TaskDeadlineExceeded:
            raise
        except Exception as exc:
            logger.warning(f"transitions failed, joining scenes with hard cuts: {exc}")
            compose(transitions=False)
    finally:
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
    logger.success(f"scene render completed: {output_file}")
    return output_file
