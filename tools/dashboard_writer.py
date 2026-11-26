"""Streaming output for the demo dashboard: MP4 (OpenCV), GIF (Pillow), live window
(matplotlib TkAgg). Only installed libraries; display only.

Frames are written as they arrive (MP4) or kept downscaled, strided and palette-
quantised (GIF, capped at `gif_max_frames`), so a long episode never holds full-size
composites in memory.
"""
from __future__ import annotations

import os

import numpy as np

from evaluate.video import MS_PER_SECOND

from .dashboard_config import DashboardConfig


class EpisodeWriter:
    def __init__(self, stem: str, fps: float, cfg: DashboardConfig, mp4: bool = True, gif: bool = True) -> None:
        self.stem, self.fps, self.cfg = stem, float(fps), cfg
        self.want_mp4, self.want_gif = mp4, gif
        self._vw = None
        self._gif: list = []
        self._n = 0
        self.mp4_failed = False

    def _open_mp4(self, h: int, w: int) -> None:
        try:
            import cv2
        except ImportError:
            self.mp4_failed = True
            return
        vw = cv2.VideoWriter(self.stem + ".mp4", cv2.VideoWriter_fourcc(*self.cfg.mp4_fourcc), self.fps, (w, h))
        if vw.isOpened():
            self._vw = vw
        else:
            self.mp4_failed = True

    def add(self, img: np.ndarray) -> None:
        assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8, \
            f"composite expected (H, W, 3) uint8, got {img.shape} {img.dtype}"
        if self.want_mp4 and self._vw is None and not self.mp4_failed:
            self._open_mp4(img.shape[0], img.shape[1])
        if self._vw is not None:
            import cv2
            self._vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if (self.want_gif and self._n % self.cfg.gif_frame_stride == 0
                and len(self._gif) < self.cfg.gif_max_frames):
            self._gif_add(img)
        self._n += 1

    def _gif_add(self, img: np.ndarray) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.want_gif = False
            return
        k = self.cfg.gif_downscale
        small = Image.fromarray(img).resize((img.shape[1] // k, img.shape[0] // k), Image.BILINEAR)
        self._gif.append(small.quantize(colors=256, method=Image.Quantize.MEDIANCUT))

    def close(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self._vw is not None:
            self._vw.release()
            self._vw = None
            p = self.stem + ".mp4"
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                out["mp4"] = p
        if self._gif:
            p = self.stem + ".gif"
            dur = int(round(MS_PER_SECOND * self.cfg.gif_frame_stride / self.fps))
            self._gif[0].save(p, save_all=True, append_images=self._gif[1:], duration=dur, loop=0, optimize=False)
            self._gif = []
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                out["gif"] = p
        out["frames"] = str(self._n)
        return out


class LiveView:
    """A matplotlib window (TkAgg). OpenCV here is built without a GUI (cv2.imshow absent)."""

    def __init__(self, plt, fig, im) -> None:
        self.plt, self.fig, self.im = plt, fig, im

    @staticmethod
    def unavailable_reason() -> str:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt  # noqa: F401
            import tkinter  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on the machine
            return f"matplotlib TkAgg not usable ({type(e).__name__}: {e}); OpenCV has no GUI backend"
        return "no display could be opened"

    @classmethod
    def open(cls, cfg: DashboardConfig) -> "LiveView | None":
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            plt.ion()
            dpi = 100
            fig = plt.figure(figsize=(cfg.width / dpi, cfg.height / dpi), dpi=dpi)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.axis("off")
            im = ax.imshow(np.zeros((cfg.height, cfg.width, 3), np.uint8))
            fig.canvas.manager.set_window_title("DoomsDay agent dashboard")
            plt.show(block=False)
            return cls(plt, fig, im)
        except Exception:
            return None

    def show(self, img: np.ndarray) -> None:
        if not self.plt.fignum_exists(self.fig.number):
            return                                     # window closed by the viewer: keep recording
        self.im.set_data(img)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        try:
            self.plt.close(self.fig)
        except Exception:
            pass
