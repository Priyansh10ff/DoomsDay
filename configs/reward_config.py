"""Reward shaping and exploration-bonus settings (ADR 0010). Training only.

No torch and no vizdoom import: env worker processes import this module.

Every number here has its written reason in
docs/decisions/0010-reward-shaping-and-episodic-coverage-bonus.md. Do not add a
scale without adding its reason there.

Sign convention: every scale is a non-negative magnitude. The sign lives in the raw
component (damage, death and time are negative raw values), so a component's
contribution is always `scale * raw`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# Term names, in logging order. `rewards.terms` computes exactly these.
TERMS: tuple[str, ...] = ("level_exit", "keycard", "kill", "damage_taken", "health",
                          "ammo", "armor", "death", "time")
# Positive terms an agent could repeat; their episode sum is capped (`positive_shaping_cap`).
CAPPED_TERMS: tuple[str, ...] = ("kill", "health", "ammo", "armor")
N_KEY_COLOURS = 3

# Which of our terms each ViZDoom engine reward knob already pays. Used to refuse a
# setup that would pay the same event twice (ADR 0010, "No double counting").
KNOB_PAYS: dict[str, tuple[str, ...]] = {
    "living_reward": ("time",),
    "death_penalty": ("death",),
    "kill_reward": ("kill",),
    "frag_reward": ("kill",),
    "map_exit_reward": ("level_exit",),
    "health_reward": ("health",),
    "armor_reward": ("armor",),
    "damage_taken_penalty": ("damage_taken",),
    "hit_taken_penalty": ("damage_taken",),
    "item_reward": ("keycard", "health", "ammo", "armor"),
    "secret_reward": (),
    "hit_reward": (),
    "damage_made_reward": (),
}


@dataclass(frozen=True)
class RewardConfig:
    name: str = "full_level"
    # --- the scenario's own (engine / ACS) reward ---------------------------------
    # 0.0: the engine reward is ignored and only our terms pay.
    # 1.0: the scenario's reward passes through; every term it already pays must then
    #      have scale 0 here (checked by rewards.validate).
    engine_reward_scale: float = 0.0
    # Terms the scenario pays through its map script, which no engine knob reveals.
    scenario_script_pays: tuple[str, ...] = ()
    # Goal payout of the scenario when it (not we) pays the exit; used by the
    # exit-dominance check only.
    scenario_exit_value: float = 0.0
    # --- the nine extrinsic scales (ADR 0010 table) -------------------------------
    level_exit: float = 1.0        # per exit
    keycard: float = 0.1           # per key colour, once per colour per episode
    kill: float = 0.02             # per kill
    damage_taken: float = 0.003    # per health point lost (raw is negative)
    health: float = 0.002          # per health point GAINED (losses are damage_taken)
    ammo: float = 0.005            # per `ammo_unit` of ammo GAINED
    armor: float = 0.001           # per armor point GAINED
    death: float = 0.75            # per death (raw is negative)
    time: float = 0.0002           # per decision (raw is negative)
    # --- guards (part of the reward computation, never of the action path) --------
    # Episode cap on the summed positive pay of CAPPED_TERMS.
    positive_shaping_cap: float = 0.4
    ammo_unit: float = 10.0            # raw ammo units that count as one "unit"
    ammo_step_clip_units: float = 5.0  # most ammo units paid in one decision
    # Two-signal exit rule (0007-recipe ruling 1): our exit term pays only when the
    # final step's engine reward is at least this. None = `level_exit` flag alone.
    exit_confirm_engine_reward: float | None = 0.5
    # Refuse a setup in which not exiting could out-earn exiting.
    require_exit_dominance: bool = True
    # A non-zero scale whose `info` field is absent raises at reset. Never silent.
    # --- door progress (generated maps only; 2026-11-25 evening plan, coordinator) --
    # Paid once per door per episode, the first time the player gets through it
    # (rewards/doors.py). 0.0 = off, and then nothing about doors is computed or
    # logged, so presets without it behave exactly as before. Not one of TERMS: it
    # needs map geometry, not just `info`. Episode maximum = door x doors in the map.
    door: float = 0.0
    # Half-length, in map units, of the region in front of and behind a door slab in
    # which the player's side is remembered. Must exceed one decision's travel
    # (straferunning ~23.4 units/tic x frame_skip 4 ~ 94), so a step that skips the
    # 32-unit slab is still seen on both sides.
    door_near_margin: float = 128.0
    # --- USE / door discipline (preset full_level_doors_v2; rewards/use_doors.py) ---
    # All 0 / False = off: nothing is computed, the action is not read, nothing is
    # logged, so the older presets behave exactly as before. Magnitudes; the raw
    # values are negative counts. Definitions are those of train/episode_diag.py.
    # Penalty per USE press edge (0 -> 1) whose USE trace (64 units along the view
    # heading, pre-step pose) crosses no closed door face and no exit switch line.
    use_miss: float = 0.0
    use_miss_cap: float = 0.0          # most use_miss penalty of one episode
    # Penalty per door that starts closing again within ~1 s of a player press on it.
    door_reversal: float = 0.0
    door_reversal_cap: float = 0.0     # most door_reversal penalty of one episode
    # True: `door` pays only for doors on the start -> key -> exit route; others pay 0.
    door_on_path_only: bool = False
    # --- how the bonus reaches the trainer ----------------------------------------
    # True (v1): reward = extrinsic + coef * intrinsic, one value head.
    # False: reward = extrinsic only; the trainer reads info["shaping"]["intrinsic"]
    #        for the separate `value_int` stream (ADR 0005).
    sum_intrinsic_into_reward: bool = True

    def scale(self, term: str) -> float:
        return float(getattr(self, term))


@dataclass(frozen=True)
class CoverageConfig:
    """Episodic coverage bonus E1 (ADR 0005), from privileged x/y in `info`."""
    enabled: bool = True
    cell_size: float = 128.0     # map units per grid cell side
    # A new cell pays 1 / budget_cells, for at most budget_cells cells per episode,
    # so the raw intrinsic return of an episode is in [0, 1] by construction.
    budget_cells: int = 64
    # Coefficient schedule, in PER-ENV decisions (what a wrapper can count).
    coef_start: float = 0.25
    coef_end: float = 0.05
    hold_decisions: int = 0          # coefficient stays at coef_start this long
    anneal_decisions: int = 60_000   # then falls linearly to coef_end over this many

    @property
    def episode_max(self) -> float:
        """Largest raw intrinsic return of one episode."""
        return 1.0 if self.enabled else 0.0


# Full levels (generated maps, Freedoom Phase 1): our terms pay, engine reward ignored.
FULL_LEVEL_REWARD = RewardConfig()
FULL_LEVEL_COVERAGE = CoverageConfig()

# `full_level` plus the door-progress term (S1/S2 runs stalled at the first door: no
# term paid for opening one, and coverage cells behind a closed door are unreachable).
# Exit 1.0 stays the largest single payment (door 0.3 < 1.0, checked by
# rewards.validate). The summed exit-dominance bound does NOT include doors (see
# rewards.validate.check_door_setup): door pay is once per door and is never forgone
# by exiting after passing them, so it does not reward delaying the exit.
FULL_LEVEL_DOORS_REWARD = replace(FULL_LEVEL_REWARD, name="full_level_doors", door=0.3)

# `full_level_doors` plus USE / door discipline (requested 2026-11-26; ADR pending).
# Measured on reports/runs/20261126-night2-v2-bcppo (826 door episodes): USE press
# edges that hit nothing per episode: q10 2, q25 12, median 35, q75 70, q90 87; door
# reversals: median 0, q75 1, q90 2, q99 4, max 6; 374 episodes had an off-path door.
#  * use_miss 0.003, cap 0.15: the median spamming episode (35 misses) loses 0.105,
#    the cap binds at 50 misses (38 % of episodes), a clean episode (q10: 2) loses
#    0.006. The suggested 0.005 would hit the cap at 30 misses, i.e. in 55 % of
#    episodes, where one more useless press costs nothing: no gradient for the
#    majority. The cap (0.15) is half of one on-path door (0.3), so pressing to get
#    through a door is always worth it, even with every press before it wasted.
#  * door_reversal 0.05, cap 0.2: one reversal costs a sixth of the door it delays;
#    the cap covers 4 reversals (q99). Opening a door and reversing it once still
#    nets +0.25 when the player then gets through.
#  * door_on_path_only: side doors pay 0 (gap 6: a 5-door map could pay 1.5 > exit).
# Worst case of the two penalties together is 0.35: with the time penalty over a full
# episode (0.21) that stays below death (0.75) and below the exit (1.0); both are
# checked by rewards.validate.
FULL_LEVEL_DOORS_V2_REWARD = replace(FULL_LEVEL_DOORS_REWARD, name="full_level_doors_v2",
                                     use_miss=0.003, use_miss_cap=0.15,
                                     door_reversal=0.05, door_reversal_cap=0.2,
                                     door_on_path_only=True)

# Bundled `my_way_home`: the scenario pays the goal (+1 through ACS) and the time cost
# (living_reward), so both of ours are off. The other terms are off as well: the map
# has no monsters, keys or pickups other than the goal armour (an armor term would pay
# the goal a second time), and `info` does not carry health / armor / ammo yet.
MY_WAY_HOME_REWARD = RewardConfig(
    name="my_way_home", engine_reward_scale=1.0, scenario_script_pays=("level_exit",),
    scenario_exit_value=1.0, level_exit=0.0, keycard=0.0, kill=0.0, damage_taken=0.0,
    health=0.0, ammo=0.0, armor=0.0, death=0.0, time=0.0, exit_confirm_engine_reward=None)
MY_WAY_HOME_COVERAGE = CoverageConfig()

# Everything off: the wrapped env returns exactly the engine reward. For ablations
# ("no bonus" arm of ADR 0005's validation 2) and as the identity check in tests.
PASSTHROUGH_REWARD = RewardConfig(
    name="passthrough", engine_reward_scale=1.0, level_exit=0.0, keycard=0.0, kill=0.0,
    damage_taken=0.0, health=0.0, ammo=0.0, armor=0.0, death=0.0, time=0.0,
    exit_confirm_engine_reward=None, require_exit_dominance=False)
NO_COVERAGE = CoverageConfig(enabled=False)
