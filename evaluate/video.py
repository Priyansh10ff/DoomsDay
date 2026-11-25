"""Write recorded frames (H, W, 3 uint8 RGB) as MP4 (OpenCV) and/or animated GIF (Pillow).

Only libraries already installed are used (cv2, PIL). Frames are upscaled by an integer
factor with nearest-neighbour (pixel-exact, no smoothing). If neither writer works,
numbered PNG frames are written instead (see `write_png_frames`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

MS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class VideoConfig:
    mp4_scale: int = 4                  # 160x120 -> 640x480
    gif_scale: int = 2                  # 160x120 -> 320x240 (GIF files grow fast)
    write_mp4: bool = True
    write_gif: bool = True
    mp4_fourcc: str = "mp4v"

    def __post_init__(self) -> None:
        if self.mp4_scale < 1 or self.gif_scale < 1:
            raise ValueError("scales must be >= 1")


def upscale(frame: np.ndarray, k: int) -> np.ndarray:
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8, \
        f"frame expected (H, W, 3) uint8, got {frame.shape} {frame.dtype}"
    return frame if k == 1 else np.repeat(np.repeat(frame, k, axis=0), k, axis=1)


def write_mp4(frames: list[np.ndarray], path: str, fps: float, scale: int, fourcc: str) -> bool:
    """True if a non-empty file was written."""
    try:
        import cv2
    except ImportError:
        return False
    h, w = frames[0].shape[0] * scale, frames[0].shape[1] * scale
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
    if not vw.isOpened():
        return False
    try:
        for f in frames:
            vw.write(cv2.cvtColor(upscale(f, scale), cv2.COLOR_RGB2BGR))
    finally:
        vw.release()
    return os.path.isfile(path) and os.path.getsize(path) > 0


def write_gif(frames: list[np.ndarray], path: str, fps: float, scale: int) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    imgs = [Image.fromarray(upscale(f, scale)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(round(MS_PER_SECOND / fps)),
                 loop=0, optimize=False)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def write_png_frames(frames: list[np.ndarray], folder: str) -> int:
    """Fallback. Assemble later with, for example:
    ffmpeg -framerate 8.75 -i frame_%05d.png -vf scale=640:480:flags=neighbor out.mp4"""
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(folder, f"frame_{i:05d}.png"))
    return len(frames)


def write_video(frames: list[np.ndarray], stem: str, fps: float, cfg: VideoConfig) -> dict[str, str]:
    """Writes `<stem>.mp4` / `<stem>.gif` as configured; PNG frames if both fail.
    Returns {kind: path} of what was written."""
    if not frames:
        raise ValueError("no frames to write")
    out: dict[str, str] = {}
    if cfg.write_mp4 and write_mp4(frames, stem + ".mp4", fps, cfg.mp4_scale, cfg.mp4_fourcc):
        out["mp4"] = stem + ".mp4"
    if cfg.write_gif and write_gif(frames, stem + ".gif", fps, cfg.gif_scale):
        out["gif"] = stem + ".gif"
    if not out:
        write_png_frames(frames, stem + "_frames")
        out["png_frames"] = stem + "_frames"
    return out
