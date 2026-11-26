"""Pure drawing for the demo dashboard: one `FrameData` -> one (H, W, 3) uint8 RGB image.

Display only. Nothing here is called on the path from observation to action; it draws
what the policy saw (its own observation arrays) and what it produced (value, action,
pooling-attention weights, optional sub-goal) after the decision was made.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from env.egomap import CH_DOOR, CH_FREE, CH_KEY, CH_LOCKED, CH_VISITED, CH_WALL, EGOMAP_AGENT_CELL
from env.observation import ENTITY_TYPES, SCREEN_H, SCREEN_W

from .dashboard_config import DashboardConfig

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
U_COL, V_COL, W_COL, H_COL = 0, 1, 2, 3          # entity feature columns (env/observation.py)
MOVE_LABELS = ("-", "FWD", "BACK")                 # head 0 index -> label (env/actions.py)
STRAFE_LABELS = ("-", "LEFT", "RIGHT")             # head 1
ON_OFF = ("off", "ON")


@dataclass
class FrameData:
    """Everything one composite shows. Arrays are the policy's own observation for this
    decision (batch dim removed) and the network outputs for it."""
    screen: np.ndarray | None                      # (120, 160, 3) uint8 RGB
    door_cols: np.ndarray | None                   # (80,) v2 door channel, 1 = door face
    egomap: np.ndarray | None                      # (C, 32, 32) uint8 0/1
    entities: np.ndarray                           # (N, 8) float32
    entity_type: np.ndarray                        # (N,) int
    entity_mask: np.ndarray                        # (N,) bool
    attn: np.ndarray | None                        # (N,) pooling weight per entity row
    attn_empty: float | None                       # weight on the learned "no entity" token
    values: Sequence[float]                        # value_ext history, last = this decision
    action: np.ndarray                             # (6,) int head indices
    action_probs: np.ndarray | None                # (6,) probability of the chosen index per head
    turn_bins: tuple[float, ...]
    title: str = ""
    step: int = 0
    health: float | None = None
    keys_held: dict[str, bool] = field(default_factory=dict)
    goal: tuple[str, float] | None = None          # (name, probability) when the model has a goal head
    banner: str | None = None                      # outcome card text


# ---------------------------------------------------------------- primitives
def _text(img: np.ndarray, s: str, xy: tuple[int, int], col, scale: float, thick: int = 1) -> None:
    cv2.putText(img, s, xy, FONT, scale, col, thick, LINE)


def _label(img: np.ndarray, s: str, xy: tuple[int, int], col, scale: float, bg) -> None:
    """Text on a filled background box (readable over the game screen)."""
    (tw, th), base = cv2.getTextSize(s, FONT, scale, 1)
    x, y = xy
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), bg, -1)
    _text(img, s, xy, col, scale)


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, title: str, cfg: DashboardConfig) -> None:
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), cfg.panel_bg, -1)
    _text(img, title, (x + 6, y + 16), cfg.text_dim, cfg.font_scale * 0.9)


def _blend_rect(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], col, alpha: float) -> None:
    x0, y0 = max(p0[0], 0), max(p0[1], 0)
    x1, y1 = min(p1[0], img.shape[1]), min(p1[1], img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1.0 - alpha) + np.asarray(col, np.float32) * alpha).astype(np.uint8)


# ---------------------------------------------------------------- panels
def egomap_rgb(egomap: np.ndarray, cfg: DashboardConfig) -> np.ndarray:
    """(C, 32, 32) 0/1 -> (32*px, 32*px, 3) colour top-down map with the agent arrow."""
    assert egomap.ndim == 3 and egomap.shape[0] >= CH_KEY + 1, \
        f"egomap expected (C>=5, H, W), got {egomap.shape}"
    c = cfg.egomap
    _, hh, ww = egomap.shape
    rgb = np.empty((hh, ww, 3), np.uint8)
    rgb[:] = c.unknown
    layers = [(CH_FREE, c.free), (CH_VISITED, c.visited), (CH_WALL, c.wall), (CH_DOOR, c.door)]
    if egomap.shape[0] > CH_LOCKED:
        layers.append((CH_LOCKED, c.locked))
    layers.append((CH_KEY, c.key))
    for ch, col in layers:
        rgb[egomap[ch] > 0] = col
    k = cfg.egomap_cell_px
    big = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    r, col_ = EGOMAP_AGENT_CELL
    cx, cy = col_ * k + k // 2, r * k + k // 2
    tri = np.array([[cx, cy - k], [cx - int(0.7 * k), cy + int(0.7 * k)], [cx + int(0.7 * k), cy + int(0.7 * k)]],
                   np.int32)
    cv2.fillPoly(big, [tri], c.agent, LINE)
    return big


def _draw_egomap(img: np.ndarray, d: FrameData, x: int, y: int, w: int, cfg: DashboardConfig) -> int:
    k = cfg.egomap_cell_px
    side = 32 * k if d.egomap is None else d.egomap.shape[2] * k
    h = cfg.panel_title_h + side + 30
    _panel(img, x, y, w, h, "EGOMAP memory (agent-centred, facing up)", cfg)
    mx, my = x + (w - side) // 2, y + cfg.panel_title_h
    if d.egomap is None:
        _text(img, "no egomap in this observation version", (x + 12, my + side // 2), cfg.text_dim, cfg.font_scale)
    else:
        big = egomap_rgb(d.egomap, cfg)
        img[my:my + big.shape[0], mx:mx + big.shape[1]] = big
    c = cfg.egomap
    items = [("free", c.free), ("wall", c.wall), ("trail", c.visited), ("door", c.door), ("key", c.key)]
    if d.egomap is not None and d.egomap.shape[0] > CH_LOCKED:
        items.append(("locked", c.locked))
    lx, ly = x + 8, my + side + 20
    step = (w - 16) // len(items)
    for name, col in items:
        cv2.rectangle(img, (lx, ly - 10), (lx + 12, ly + 2), col, -1)
        _text(img, name, (lx + 16, ly), cfg.text, cfg.font_scale * 0.8)
        lx += step
    return h


def _draw_screen(img: np.ndarray, d: FrameData, x: int, y: int, cfg: DashboardConfig) -> None:
    s = cfg.screen_scale
    sw, sh = SCREEN_W * s, SCREEN_H * s
    if d.screen is None:
        cv2.rectangle(img, (x, y), (x + sw - 1, y + sh - 1), cfg.panel_bg, -1)
        _text(img, "no frame (terminal step)", (x + 20, y + sh // 2), cfg.text_dim, cfg.title_font_scale)
    else:
        assert d.screen.shape == (SCREEN_H, SCREEN_W, 3), \
            f"screen expected ({SCREEN_H}, {SCREEN_W}, 3), got {d.screen.shape}"
        img[y:y + sh, x:x + sw] = np.repeat(np.repeat(d.screen, s, axis=0), s, axis=1)
    _draw_attention(img, d, x, y, cfg)
    # door-cue strip: one cell per ray column
    sy = y + sh + 4
    if d.door_cols is not None:
        n = d.door_cols.shape[0]
        cw = sw / n
        for i in range(n):
            col = cfg.door_lit if d.door_cols[i] > 0.5 else cfg.door_unlit
            cv2.rectangle(img, (x + int(i * cw), sy), (x + int((i + 1) * cw) - 1, sy + cfg.door_strip_h), col, -1)
        _text(img, "door cue (v2 door channel, per ray column)", (x, sy + cfg.door_strip_h + 18),
              cfg.text_dim, cfg.font_scale * 0.9)
    else:
        _text(img, "no door channel in this observation version", (x, sy + 16), cfg.text_dim, cfg.font_scale)


def _draw_attention(img: np.ndarray, d: FrameData, x: int, y: int, cfg: DashboardConfig) -> None:
    s = cfg.screen_scale
    rows = np.flatnonzero(np.asarray(d.entity_mask, bool))
    for i in rows:
        u, v, w, h = (float(d.entities[i, c]) for c in (U_COL, V_COL, W_COL, H_COL))
        cx, cy = (u + 1.0) * 0.5 * SCREEN_W * s, (v + 1.0) * 0.5 * SCREEN_H * s
        bw, bh = max(w * SCREEN_W * s, 4.0), max(h * SCREEN_H * s, 4.0)
        p0 = (int(x + cx - bw / 2), int(y + cy - bh / 2))
        p1 = (int(x + cx + bw / 2), int(y + cy + bh / 2))
        wt = 0.0 if d.attn is None else float(d.attn[i])
        a = cfg.attn_min_alpha + (1.0 - cfg.attn_min_alpha) * min(max(wt, 0.0), 1.0)
        _blend_rect(img, p0, p1, cfg.attention, 0.45 * a)
        edge = tuple(int(c * (0.5 + 0.5 * a)) for c in cfg.attention)
        cv2.rectangle(img, p0, p1, edge, 1 + int(round(4 * a)))
        t = int(d.entity_type[i])
        name = ENTITY_TYPES[t] if 0 <= t < len(ENTITY_TYPES) else f"type{t}"
        label = name if d.attn is None or wt < cfg.attn_label_min_weight else f"{name} {wt:.2f}"
        _label(img, label, (p0[0], max(p0[1] - 5, y + 14)), edge, cfg.font_scale * 0.85, cfg.background)
    if d.attn_empty is not None:
        _label(img, f"entity attention: {len(rows)} visible, no-entity token {d.attn_empty:.2f}",
               (x + 8, y + 20), cfg.attention, cfg.font_scale * 0.9, cfg.background)


def _draw_value(img: np.ndarray, d: FrameData, x: int, y: int, w: int, cfg: DashboardConfig) -> None:
    h = cfg.value_panel_h
    vals = np.asarray(list(d.values)[-cfg.value_window:], np.float64)
    cur = f"{vals[-1]:+.3f}" if vals.size else "n/a"
    _panel(img, x, y, w, h, f"VALUE estimate (last {cfg.value_window} decisions)   now {cur}", cfg)
    px0, py0, pw, ph = x + 8, y + cfg.panel_title_h + 4, w - 16, h - cfg.panel_title_h - 12
    cv2.rectangle(img, (px0, py0), (px0 + pw, py0 + ph), cfg.background, -1)
    if vals.size < 2:
        return
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(1e-3, 0.1 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    if lo < 0.0 < hi:
        zy = int(py0 + ph * (hi - 0.0) / (hi - lo))
        cv2.line(img, (px0, zy), (px0 + pw, zy), cfg.text_dim, 1)
    xs = px0 + np.arange(vals.size) * (pw / (cfg.value_window - 1))
    ys = py0 + ph * (hi - vals) / (hi - lo)
    pts = np.stack([xs, ys], axis=1).round().astype(np.int32)
    cv2.polylines(img, [pts], False, cfg.value_line, 2, LINE)
    _text(img, f"{hi:+.2f}", (px0 + 2, py0 + 12), cfg.text_dim, cfg.font_scale * 0.7)
    _text(img, f"{lo:+.2f}", (px0 + 2, py0 + ph - 3), cfg.text_dim, cfg.font_scale * 0.7)


def _tile(img: np.ndarray, x: int, y: int, w: int, h: int, name: str, val: str, p: float | None,
          on: bool, on_col, cfg: DashboardConfig) -> None:
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), on_col if on else cfg.background, -1 if on else 1)
    if not on:
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), cfg.text_dim, 1)
    tc = cfg.background if on else cfg.text
    _text(img, name, (x + 6, y + 15), tc, cfg.font_scale * 0.8)
    _text(img, val, (x + 6, y + h - 9), tc, cfg.title_font_scale, 2)
    if p is not None:
        _text(img, f"p={p:.2f}", (x + w - 56, y + 15), tc, cfg.font_scale * 0.75)


def _draw_actions(img: np.ndarray, d: FrameData, x: int, y: int, w: int, cfg: DashboardConfig) -> None:
    h = cfg.action_panel_h
    _panel(img, x, y, w, h, "ACTION chosen (6 heads, one forward pass)", cfg)
    a = np.asarray(d.action, np.int64).reshape(-1)
    assert a.shape == (6,), f"action expected (6,), got {a.shape}"
    p = (None,) * 6 if d.action_probs is None else tuple(float(v) for v in d.action_probs)
    turn = float(d.turn_bins[a[2]]) if 0 <= a[2] < len(d.turn_bins) else 0.0
    turn_txt = "-" if turn == 0 else f"{'R' if turn > 0 else 'L'} {abs(turn):g} deg/tic"
    tiles = [
        ("move", MOVE_LABELS[a[0]] if a[0] < 3 else str(a[0]), p[0], a[0] != 0, cfg.active),
        ("strafe", STRAFE_LABELS[a[1]] if a[1] < 3 else str(a[1]), p[1], a[1] != 0, cfg.active),
        ("turn", turn_txt, p[2], turn != 0, cfg.active),
        ("attack", "FIRE" if a[3] else "off", p[3], bool(a[3]), cfg.fire),
        ("use", "USE" if a[4] else "off", p[4], bool(a[4]), cfg.door_lit),
        ("speed", "RUN" if a[5] else "walk", p[5], bool(a[5]), cfg.active),
    ]
    gap = 6
    tw = (w - 16 - 2 * gap) // 3
    th = (h - cfg.panel_title_h - 8 - gap) // 2
    for i, (name, val, pp, on, col) in enumerate(tiles):
        tx = x + 8 + (i % 3) * (tw + gap)
        ty = y + cfg.panel_title_h + 2 + (i // 3) * (th + gap)
        _tile(img, tx, ty, tw, th, name, val, pp, on, col, cfg)


def _draw_goal(img: np.ndarray, d: FrameData, x: int, y: int, w: int, cfg: DashboardConfig) -> None:
    assert d.goal is not None
    name, prob = d.goal
    _panel(img, x, y, w, cfg.goal_panel_h, "PREDICTED SUB-GOAL (goal head)", cfg)
    _text(img, f"{name}", (x + 10, y + cfg.goal_panel_h - 8), cfg.accent, cfg.title_font_scale * 1.2, 2)
    _text(img, f"p={prob:.2f}", (x + w - 80, y + cfg.goal_panel_h - 10), cfg.text, cfg.font_scale)


KEY_RGB = {"blue": (60, 110, 255), "yellow": (250, 220, 40), "red": (235, 50, 50)}


def _draw_header(img: np.ndarray, d: FrameData, cfg: DashboardConfig) -> None:
    cv2.rectangle(img, (0, 0), (cfg.width - 1, cfg.header_h - 1), cfg.panel_bg, -1)
    _text(img, d.title, (cfg.margin, 24), cfg.text, cfg.title_font_scale)
    x = cfg.width - 430
    _text(img, f"step {d.step}", (x, 24), cfg.text, cfg.title_font_scale)
    hp = "n/a" if d.health is None else f"{d.health:.0f}"
    _text(img, f"health {hp}", (x + 110, 24), cfg.text, cfg.title_font_scale)
    _text(img, "keys", (x + 250, 24), cfg.text, cfg.title_font_scale)
    kx = x + 300
    for colour, rgb in KEY_RGB.items():
        held = bool(d.keys_held.get(colour, False))
        cv2.rectangle(img, (kx, 10), (kx + 22, 28), rgb, -1 if held else 1)
        kx += 30


def _draw_banner(img: np.ndarray, text: str, cfg: DashboardConfig) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.6, 3)
    x, y = (cfg.width - tw) // 2, cfg.height // 2
    _blend_rect(img, (x - 30, y - th - 30), (x + tw + 30, y + 30), (0, 0, 0), 0.75)
    _text(img, text, (x, y), cfg.text, 1.6, 3)


def compose(d: FrameData, cfg: DashboardConfig) -> np.ndarray:
    """One dashboard frame, (cfg.height, cfg.width, 3) uint8 RGB."""
    img = np.empty((cfg.height, cfg.width, 3), np.uint8)
    img[:] = cfg.background
    _draw_header(img, d, cfg)
    top = cfg.header_h + 8
    _draw_screen(img, d, cfg.margin, top, cfg)
    rx = cfg.margin + SCREEN_W * cfg.screen_scale + cfg.margin
    rw = cfg.width - rx - cfg.margin
    y = top
    y += _draw_egomap(img, d, rx, y, rw, cfg) + 8
    _draw_value(img, d, rx, y, rw, cfg)
    y += cfg.value_panel_h + 8
    _draw_actions(img, d, rx, y, rw, cfg)
    y += cfg.action_panel_h + 8
    if d.goal is not None and y + cfg.goal_panel_h <= cfg.height:
        _draw_goal(img, d, rx, y, rw, cfg)
    if d.banner:
        _draw_banner(img, d.banner, cfg)
    assert img.shape == (cfg.height, cfg.width, 3), f"composite expected ({cfg.height}, {cfg.width}, 3)"
    return img
