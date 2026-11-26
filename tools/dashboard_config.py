"""Layout, colours and limits of the demo dashboard (`python -m tools.dashboard`).

Display-only settings: nothing here reaches the model or the environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RGB = tuple[int, int, int]

# Sub-goal names shown when configs/goal_modes.py is absent or exposes no name tuple.
DEFAULT_GOAL_NAMES: tuple[str, ...] = (
    "exploring", "getting key", "to locked door", "to exit", "fighting", "opening door", "recovering",
)
# configs/goal_modes.py GOAL_MODES id name -> display name.
GOAL_DISPLAY: dict[str, str] = {
    "explore": "exploring", "get_key": "getting key", "go_locked_door": "to locked door", "go_exit": "to exit",
    "fight": "fighting", "open_door": "opening door", "recover": "recovering",
}
# Attribute names tried, in order, in configs/goal_modes.py for the goal-name tuple.
GOAL_NAME_ATTRS: tuple[str, ...] = ("GOAL_MODES", "GOAL_NAMES", "GOAL_MODE_NAMES", "MODES", "NAMES")


@dataclass(frozen=True)
class EgomapColours:
    """Top-down map colours; later entries are drawn over earlier ones."""
    unknown: RGB = (24, 24, 28)
    free: RGB = (70, 78, 92)
    visited: RGB = (60, 150, 230)
    wall: RGB = (225, 225, 225)
    door: RGB = (255, 150, 30)
    key: RGB = (250, 230, 40)
    locked: RGB = (235, 40, 60)
    agent: RGB = (40, 255, 120)


@dataclass(frozen=True)
class DashboardConfig:
    width: int = 1280
    height: int = 720
    margin: int = 16
    header_h: int = 36
    screen_scale: int = 5               # 160x120 -> 800x600
    door_strip_h: int = 22
    egomap_cell_px: int = 9             # 32x32 -> 288x288
    panel_title_h: int = 22
    value_panel_h: int = 118
    value_window: int = 200             # decisions shown in the rolling value line
    action_panel_h: int = 118
    goal_panel_h: int = 46
    background: RGB = (14, 15, 18)
    panel_bg: RGB = (28, 30, 36)
    text: RGB = (230, 230, 230)
    text_dim: RGB = (150, 155, 165)
    accent: RGB = (90, 200, 255)
    active: RGB = (60, 220, 110)
    fire: RGB = (240, 60, 50)
    attention: RGB = (255, 220, 40)
    door_lit: RGB = (255, 150, 30)
    door_unlit: RGB = (45, 45, 50)
    value_line: RGB = (90, 200, 255)
    attn_min_alpha: float = 0.15        # box blend at weight 0 (so every visible entity shows)
    attn_label_min_weight: float = 0.05  # weight label printed above this
    font_scale: float = 0.5
    title_font_scale: float = 0.55
    egomap: EgomapColours = field(default_factory=EgomapColours)
    # video output
    end_card_seconds: float = 1.5       # outcome banner held at the end of each episode
    gif_downscale: int = 2              # GIF frames are width/k x height/k
    gif_frame_stride: int = 2           # keep every k-th composite in the GIF
    gif_max_frames: int = 600           # GIF is cut after this many frames (memory cap)
    mp4_fourcc: str = "mp4v"

    def __post_init__(self) -> None:
        if min(self.screen_scale, self.egomap_cell_px, self.gif_downscale, self.gif_frame_stride) < 1:
            raise ValueError("scales and strides must be >= 1")
        if self.value_window < 2:
            raise ValueError("value_window must be >= 2")
