"""DoomEnv: a gymnasium.Env written directly on vizdoom.DoomGame (ADR 0001, 0002).

Two data paths, kept separate on purpose:
  * policy path : _read_vars() -> ObsVars  -> _build_obs()   -> observation dict
  * privileged  : _read_vars() -> InfoVars -> _build_info()  -> info dict only
Nothing in InfoVars (absolute pose, counts, map name, seed, keys held) is ever
passed to _build_obs(). Reward is the engine reward, unshaped.

Observation v2 (cfg.obs_version == 2) adds two cues in _add_v2_cues(), after
_build_obs(), and only as finished arrays from env/door_cue.py:
  * rays gets a 4th row, the line-of-sight door channel. The pose (InfoVars) and the
    engine's sector list are consumed inside door_cue.door_channel / DoorTracker only,
    to put door faces in the player's frame and check them against the depth buffer;
    no coordinate, sector id or line id is output.
  * self gets 6 key bits: keys held (from vanilla pickup messages) and a one-decision
    "needs key" pulse (from the vanilla locked-door message). Only booleans; never the
    notification text.

Observation v3 (cfg.obs_version == 3) = v2 + `egomap` (5, 32, 32) uint8, added in
_add_v3_egomap() from env/egomap.py: an ego-centric occupancy crop (free / wall / visited
/ closed door / key seen) built from the middle depth band, the v2 door row, the
entity set (key rows) and the v2 keys-held bits. The pose (InfoVars)
is consumed inside EgoMap only, as odometry relative to the episode's spawn pose; no
coordinate leaves it (translation invariance tested).

Observation v4 (cfg.obs_version == 4) = v3 with `egomap` (6, 32, 32): channel 5 "locked
door known", marked from the v2 needs-key pulse bits (booleans only) at the door under the
view ray, cleared when that door is seen open or passed (env/egomap.py). Every other key
is byte-identical to v3.

Per-episode skill (training configs only): reset(options={"skill": k}), else a uniform
draw from cfg.skill_mix (env RNG, drawn after the map and episode seed), else cfg.skill.
Switched with the console command `skill k-1` before new_episode(), and only when it
changes: probe 2026-11-26 (docs/reference/engine-probes.md R6.1) found
DoomGame.set_doom_skill() after init() always yields the easy spawn set in vizdoom 1.3.3.
EVAL refuses the option (skill fixed by the harness). The skill is in info{} only.

Per-step `info` keys for the trainer (privileged; never an observation):
  final_obs_fresh (bool, every step and reset): True if `obs` was built from the engine
      state after this step's action. False only when the episode ended by death,
      level exit or the engine's backstop time limit: the engine then returns no
      state and `obs` is the previous step's observation, repeated. At the
      wrapper's own time limit (`truncated`) the observation is fresh (True).
  last_step_damage_taken (float, health points, end-of-episode steps only): damage
      taken during the final decision, read from DAMAGE_TAKEN even when the
      episode ended by death.
  dead, timeout, level_exit (bool, end-of-episode steps only).
  health, armor (float), ammo (tuple of 10 floats, AMMO0..AMMO9 per slot), every step
      and reset: reward inputs (ADR 0010). Stale (previous step's) on death / exit steps.
  damage_taken_total (float, every step and reset): DAMAGE_TAKEN since this episode's
      reset (0 at reset); refreshed on the ending step too.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from . import door_cue as D
from . import egomap as E
from . import observation as O
from .actions import BUTTONS, as_index_array, encode, head_sizes
from .config import EVAL_CONFIG, EnvConfig
from .wintimer import TimerRequest

GV = vzd.GameVariable
WEAPON_VARS = [getattr(GV, f"WEAPON{i}") for i in range(10)]
AMMO_VARS = [getattr(GV, f"AMMO{i}") for i in range(10)]  # per-slot ammo: reward input only
# One list, one order. Indices below split it into the two paths.
GAME_VARS = [
    GV.HEALTH, GV.ARMOR, GV.SELECTED_WEAPON, GV.SELECTED_WEAPON_AMMO, GV.DAMAGE_TAKEN,  # 0-4
    *WEAPON_VARS,                                                                    # 5-14
    GV.VELOCITY_X, GV.VELOCITY_Y, GV.VELOCITY_Z, GV.ANGLE,                           # 15-18 ego
    GV.POSITION_X, GV.POSITION_Y, GV.POSITION_Z,                                     # 19-21 info
    GV.KILLCOUNT, GV.ITEMCOUNT, GV.SECRETCOUNT, GV.DEAD,                             # 22-25 info
    *AMMO_VARS,                                                                      # 26-35 info
]


class ObsVars(NamedTuple):
    health: float
    armor: float
    selected_weapon: float
    selected_ammo: float
    damage_taken_total: float
    weapons: np.ndarray
    vel: tuple[float, float, float]
    heading: float  # consumed only by observation.ego_motion (rotation + delta)


class InfoVars(NamedTuple):
    pos_x: float
    pos_y: float
    pos_z: float
    angle: float
    killcount: float
    itemcount: float
    secretcount: float
    dead: float
    # Reward inputs (ADR 0010). health / armor / damage are also read by the policy
    # path through ObsVars; these are separate copies for info{} only.
    health: float
    armor: float
    damage_taken_raw: float          # engine counter; made episode-relative in the env
    ammo: tuple[float, ...]          # AMMO0..AMMO9, per slot (not SELECTED_WEAPON_AMMO)


def split_vars(gv: np.ndarray) -> tuple[ObsVars, InfoVars]:
    g = [float(x) for x in gv]
    obs = ObsVars(g[0], g[1], g[2], g[3], g[4], np.asarray(g[5:15], dtype=np.float32),
                  (g[15], g[16], g[17]), g[18])
    info = InfoVars(g[19], g[20], g[21], g[18], g[22], g[23], g[24], g[25],
                    g[0], g[1], g[4], tuple(g[26:36]))
    return obs, info


# Key PICKUP wording only. Verified on freedoom2 (probe `keys`): "Blue passcard
# secured!", "Red skeleton key secured!". Doom IWAD wording ("Picked up a blue
# keycard." / "Picked up a blue skull key.") is from the Doom string table, not
# probed (no Doom IWAD here). Locked-door/switch text ("Yellow key needed for this
# door.", "Blue key needed." -- probe `locked`) must NOT match.
_KEY_MSG = re.compile(
    r"\b(blue|yellow|red) (?:passcard|skeleton key) secured\b"
    r"|\bpicked up an? (blue|yellow|red) (?:keycard|skull key)\b",
    re.IGNORECASE,
)


def keys_picked_up(text: str) -> set[str]:
    """Colours whose key-pickup message appears in a notifications string."""
    return {(a or b).lower() for a, b in _KEY_MSG.findall(text or "")}


class DoomEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: EnvConfig = EVAL_CONFIG, render_mode: str | None = None):
        """The Windows timer request comes from config.win_timer_ms (on for TRAIN, off
        for EVAL, ADR 0006) or the $DOOMSDAY_WIN_TIMER_MS override. Without it,
        ViZDoom's Python-side wait rounds up to the 15.6 ms system tick (~76
        decisions/s). It is acquired only after the engine started, and released
        exactly once in close()."""
        self.game = None
        self._timer = None
        self._ini_dir = None
        config.validate()
        self.cfg = config
        # Audit E1: maps a reset may load (None under EVAL = any); read once here.
        self._allowed_maps = config.allowed_maps()
        self.render_mode = render_mode
        n = config.max_entities
        space = {
            "rays": spaces.Box(0.0, 1.0, (O.ray_channels(len(config.ray_bands), config.obs_version),
                                          O.N_RAY_COLS), np.float32),
            "entities": spaces.Box(-2.0, 2.0, (n, 8), np.float32),
            "entity_type": spaces.Box(0, len(O.ENTITY_TYPES) - 1, (n,), np.int64),
            "entity_mask": spaces.Box(0, 1, (n,), np.bool_),
            "self": spaces.Box(-1.0, 2.0, (O.self_dim(config.obs_version),), np.float32),
            "prev_action": spaces.Box(0, max(head_sizes(len(config.turn_bins))) - 1, (6,), np.int64),
        }
        if config.screen_enabled:
            space["screen"] = spaces.Box(0, 255, (3, 60, 80), np.uint8)
        space["modality"] = spaces.Box(0.0, 1.0, (3,), np.float32)
        if config.egomap:
            space["egomap"] = spaces.Box(0, 1, E.egomap_shape(config.obs_version), np.uint8)
        self.observation_space = spaces.Dict(space)
        self.action_space = spaces.MultiDiscrete(head_sizes(len(config.turn_bins)))
        self._tracker = O.EntityTracker()
        self._doors = D.DoorTracker()   # v2: per-episode door memory (private, never output)
        # v3 (v4: + locked-door channel): only crop() is output
        self._egomap = E.EgoMap(locked_channel=config.locked_door_channel) if config.egomap else None
        self._skill = int(config.skill)   # engine skill now in effect (set before init)
        self._keys_needed: set[str] = set()
        self._last_obs = None
        self._last_info: dict = {}
        self._map = None
        self._episode_seed = None
        self._timeout_tics = config.episode_timeout_tics
        timer_ms = config.effective_timer_ms()  # resolved (may raise) before the engine starts
        self.game = self._make_game()  # closes its own DoomGame if init() raises
        self._timer = TimerRequest(timer_ms)  # last step of __init__

    # ------------------------------------------------------------ engine setup
    def _make_game(self) -> vzd.DoomGame:
        c = self.cfg
        g = vzd.DoomGame()
        g.set_doom_game_path(c.game_path())  # stock IWAD, or the harness's iwad_file
        # TRAIN scenario PWAD (forbidden under EVAL) or the harness's map_wad, as given
        if c.pwad_path() is not None:
            g.set_doom_scenario_path(c.pwad_path())
        g.set_doom_map(c.maps[0])
        g.set_doom_skill(c.skill)
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_screen_format(vzd.ScreenFormat.CRCGCB)
        g.set_render_hud(c.render_hud)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_depth_buffer_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_objects_info_enabled(c.objects_info)
        # v1: only if the (TRAIN-only) sectors_info field asks. v2: on, for the
        # line-of-sight door channel only (EnvConfig.engine_sectors_enabled).
        g.set_sectors_info_enabled(c.engine_sectors_enabled)
        g.set_notifications_buffer_enabled(True)  # info path only (keys held)
        g.set_notifications_buffer_size(c.frame_skip)
        g.set_available_buttons(BUTTONS)
        g.set_available_game_variables(GAME_VARS)
        # The wrapper truncates at episode_timeout_tics itself, while the state is still
        # readable (probe `final_state`: get_state() is None once the engine ends the
        # episode). The engine limit is a backstop one decision later.
        g.set_episode_timeout(c.episode_timeout_tics + c.frame_skip + 1)
        g.set_mode(vzd.Mode.PLAYER)
        for k, v in c.engine_rewards:
            getattr(g, f"set_{k}")(float(v))
        for a in c.game_args:  # TRAIN only
            g.add_game_args(a)
        # Own engine config file per instance (probe `ini_path`): the default is
        # `_vizdoom.ini` in the working directory, read at start and rewritten at exit
        # by every instance. A fresh path starts from the engine's defaults; the file is
        # written on close and its directory removed in close().
        self._ini_dir = tempfile.mkdtemp(prefix="doomsday_vzd_ini_")
        g.set_doom_config_path(os.path.join(self._ini_dir, "vizdoom.ini"))
        try:
            g.init()
        except BaseException:
            g.close()
            self._remove_ini_dir()
            raise
        return g

    def _remove_ini_dir(self):
        d, self._ini_dir = getattr(self, "_ini_dir", None), None
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        opts = options or {}
        self._map = opts.get("map") or self.cfg.maps[int(self.np_random.integers(len(self.cfg.maps)))]
        # Audit E1: a training config may only load maps of its own PWAD (or its maps
        # tuple); any other name would fall through to the IWAD. EVAL: any map.
        self.cfg.check_reset_map(self._map, self._allowed_maps)
        self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._set_skill(self._episode_skill(opts))
        self.game.set_seed(self._episode_seed)
        self.game.set_doom_map(self._map)
        self.game.new_episode()
        self._tracker.reset()
        self._doors.reset()
        if self._egomap is not None:
            self._egomap.reset()
        self._keys_needed = set()
        self._prev_heading = None
        self._prev_damage = None
        self._prev_action = np.zeros(6, dtype=np.int64)
        self._total_reward = self.game.get_total_reward()
        self._keys_held = {"blue": False, "yellow": False, "red": False}
        self._episode_steps = 0
        # Info path: damage_taken_total is counted from this episode's start, whatever
        # the engine counter holds at reset.
        self._damage_base = float(self.game.get_game_variable(GV.DAMAGE_TAKEN))
        state = self.game.get_state()
        obs, info = self._observe(state)
        info["final_obs_fresh"] = True
        return obs, info

    def step(self, action):
        a = as_index_array(action)
        g = self.game
        g.set_action(encode(a, self.cfg.turn_bins))  # all buttons held for every tic
        g.advance_action(self.cfg.frame_skip, True)
        finished = g.is_episode_finished()
        state = None if finished else g.get_state()
        # Remaining gap (probe `final_state`): after death or level exit the engine
        # returns no state, so a key picked up in that final decision is not reported.
        # Time-limit endings are handled by the wrapper while the state is live.
        self._keys_needed = set()  # v2 "needs key" is a one-decision pulse
        if state is not None and isinstance(state.notifications_buffer, str):
            self._update_keys(state.notifications_buffer)  # buffer covers frame_skip tics
        total = g.get_total_reward()
        reward = float(total - self._total_reward)
        self._total_reward = total
        self._prev_action = a.copy()
        self._episode_steps += 1

        dead = g.is_player_dead()
        engine_timeout = finished and g.is_episode_timeout_reached()
        time_up = (not finished) and g.get_episode_time() >= self._timeout_tics
        terminated = bool(finished and (dead or not engine_timeout))           # death or exit
        truncated = bool(time_up or (finished and engine_timeout and not dead))  # time limit
        if finished:
            # Death or level exit (or the engine backstop limit): the engine returns no
            # state, so the previous observation is returned again -> final_obs_fresh False.
            prev_damage = self._prev_damage
            obs = self._last_obs
            info = dict(self._last_info)
            info.update(self._episode_end_info(dead, engine_timeout))
            total_damage = float(g.get_game_variable(GV.DAMAGE_TAKEN))
            info["last_step_damage_taken"] = (
                0.0 if prev_damage is None else max(total_damage - prev_damage, 0.0))
            # Refreshed (not stale) on the ending step; health / armor / ammo stay stale.
            info["damage_taken_total"] = (
                info.get("damage_taken_total", 0.0) + info["last_step_damage_taken"])
            info["final_obs_fresh"] = False
        else:
            prev_damage = self._prev_damage
            obs, info = self._observe(state)
            info["final_obs_fresh"] = True
            if time_up:
                info.update(self._episode_end_info(False, True))
                info["last_step_damage_taken"] = (
                    0.0 if prev_damage is None else max(self._prev_damage - prev_damage, 0.0))
        info["engine_reward"] = reward
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" and not self.game.is_episode_finished():
            s = self.game.get_state()
            return np.transpose(s.screen_buffer, (1, 2, 0)).copy()
        return None

    def close(self):
        """Idempotent. Releases the timer request even if closing the game raises."""
        g = getattr(self, "game", None)
        try:
            if g is not None:
                g.close()
        finally:
            self.game = None
            self._remove_ini_dir()
            t = getattr(self, "_timer", None)
            if t is not None:
                t.release()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------ internals
    def _episode_skill(self, opts: dict) -> int:
        """Skill for the next episode: options["skill"], else a draw from cfg.skill_mix
        (after the map / seed draws, so those sequences do not move), else cfg.skill."""
        if "skill" in opts and opts["skill"] is not None:
            if self.cfg.vanilla_eval:
                raise ValueError("reset option 'skill' refused under EVAL: skill is fixed by the config")
            k = int(opts["skill"])
            if not 1 <= k <= 5:
                raise ValueError(f"reset option skill={k} outside 1-5")
            return k
        if self.cfg.skill_mix:
            return int(self.cfg.skill_mix[int(self.np_random.integers(len(self.cfg.skill_mix)))])
        return int(self.cfg.skill)

    def _set_skill(self, k: int) -> None:
        if k != self._skill:
            self.game.send_game_command(f"skill {k - 1}")   # takes effect at new_episode()
            self._skill = k

    def _update_keys(self, text: str):
        """Keys held (info{} always; also the v2 key bits) from vanilla pickup messages,
        and this decision's locked-door colours (v2 pulse). Text never leaves here."""
        for colour in keys_picked_up(text):
            self._keys_held[colour] = True
        self._keys_needed = D.keys_needed(text)

    def _observe(self, state):
        obs_vars, info_vars = split_vars(state.game_variables)
        obs = self._build_obs(state, obs_vars)
        if self.cfg.door_channel:
            obs = self._add_v2_cues(obs, state, info_vars)
        if self._egomap is not None:
            obs = self._add_v3_egomap(obs, state, info_vars)
        info = self._build_info(state, info_vars)
        self._last_obs, self._last_info = obs, info
        return obs, info

    def _build_obs(self, state, v: ObsVars) -> dict:
        """POLICY PATH. Receives only ObsVars and sensor buffers."""
        c = self.cfg
        depth = state.depth_buffer
        labels_buf = state.labels_buffer
        telemetry_ok = depth is not None and labels_buf is not None and state.labels is not None
        n = c.max_entities
        if telemetry_ok:
            invalid = O.invalid_depth_mask(state.labels, labels_buf, depth)
            r = O.rays(depth, c.ray_bands, invalid)
            ent, et, em = O.entities(state.labels, labels_buf, depth, self._tracker, n)
            ego = O.ego_motion(*v.vel, v.heading, self._prev_heading)
            self._prev_heading = v.heading
        else:  # e.g. +viz_nocheat: buffers are None and ego variables read 0
            r = np.zeros((len(c.ray_bands), O.N_RAY_COLS), np.float32)
            ent = np.zeros((n, 8), np.float32)
            et = np.zeros((n,), np.int64)
            em = np.zeros((n,), bool)
            ego = np.zeros((4,), np.float32)
        dmg = 0.0 if self._prev_damage is None else v.damage_taken_total - self._prev_damage
        self._prev_damage = v.damage_taken_total
        selfv = O.self_vector(v.health, v.armor, v.selected_ammo, dmg, v.selected_weapon,
                              v.weapons, ego)
        obs = {
            "rays": r,
            "entities": ent,
            "entity_type": et,
            "entity_mask": em,
            "self": selfv,
            "prev_action": self._prev_action.copy(),
        }
        if c.screen_enabled:
            obs["screen"] = O.screen_small(state.screen_buffer)
        obs["modality"] = np.array([telemetry_ok, c.screen_enabled, telemetry_ok], dtype=np.float32)
        return obs

    def _add_v2_cues(self, obs: dict, state, v: InfoVars) -> dict:
        """Observation v2. The pose and sector list go into door_cue only; what is added
        to the observation is the (80,) door cue and the (6,) key bits, nothing else."""
        telemetry_ok = bool(obs["modality"][0])
        door = np.zeros((O.N_RAY_COLS,), np.float32)
        sectors = state.sectors if telemetry_ok else None
        if sectors:  # absent under +viz_nocheat (and with telemetry off): all zeros
            mid = np.rint(obs["rays"][1] * 255.0)  # middle band, raw levels; 0 = no reading
            faces, occ = self._doors.scene_now(sectors)
            door = D.door_channel(faces, v.pos_x, v.pos_y, v.angle, mid, occ, v.pos_z + D.EYE_HEIGHT)
        obs["rays"] = np.concatenate([obs["rays"], door[None, :]], axis=0)
        obs["self"] = np.concatenate([obs["self"], D.key_bits(self._keys_held, self._keys_needed)])
        return obs

    def _add_v3_egomap(self, obs: dict, state, v: InfoVars) -> dict:
        """Observation v3. The pose goes into env/egomap.EgoMap only (spawn-relative
        odometry, never output); what is added is the (5, 32, 32) ego-centric crop.
        Without depth/labels (e.g. +viz_nocheat) nothing is written: all zeros."""
        if bool(obs["modality"][0]):
            levels = E.world_column_levels(state.depth_buffer, state.labels_buffer,
                                           self.cfg.ray_bands[1])
            held = obs["self"][-O.N_KEY_BITS:-O.N_KEY_BITS + 3] > 0.5   # v2 keys-held bits
            # v4: the v2 needs-key pulse bits (booleans); ignored by a v3 EgoMap
            needs = obs["self"][-3:] > 0.5 if self.cfg.locked_door_channel else None
            self._egomap.update(v.pos_x, v.pos_y, v.angle, levels, obs["rays"][3],
                                obs["entities"], obs["entity_type"], obs["entity_mask"], held,
                                needs)
        obs["egomap"] = self._egomap.crop()
        return obs

    def _build_info(self, state, v: InfoVars) -> dict:
        """PRIVILEGED PATH. Never an input to the policy."""
        info = {
            "pos_x": v.pos_x, "pos_y": v.pos_y, "pos_z": v.pos_z, "angle": v.angle,
            "killcount": v.killcount, "itemcount": v.itemcount, "secretcount": v.secretcount,
            "map": self._map, "episode_seed": self._episode_seed, "config": self.cfg.name,
            "skill": self._skill,
            "keys_held": dict(self._keys_held), "tic": int(state.tic),
            "episode_steps": self._episode_steps,
            # reward inputs (ADR 0010): key names and shapes as rewards/terms.py reads them
            "health": v.health, "armor": v.armor,
            "ammo": v.ammo,  # tuple of 10 floats, AMMO0..AMMO9
            "damage_taken_total": v.damage_taken_raw - self._damage_base,
        }
        return info

    def diag_door_gaps(self) -> dict | None:
        """PRIVILEGED, TRAINING DIAGNOSTICS ONLY (train/episode_diag.py reads it in the
        worker for per-episode door-reversal logging). Not in the observation, not in info.
        {engine sector id: ceiling - floor} of the doors the v2 door channel remembers, as
        of the last observed state; None when no sector info was seen this episode."""
        return self._doors.gaps() if self.cfg.door_channel else None

    def _episode_end_info(self, dead, timeout) -> dict:
        return {"dead": bool(dead), "timeout": bool(timeout),
                "level_exit": bool(not dead and not timeout),
                "keys_held": dict(self._keys_held), "episode_steps": self._episode_steps}
