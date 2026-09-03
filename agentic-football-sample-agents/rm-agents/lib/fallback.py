"""RM_Agents — instinct layer + rule-based fallback.

Every tick is decided in this order (see agent_base.create_invoke_handler):

  0. instinct_command()   code, 0 ms. Everything ON the ball (shoot / carry / release),
                          everything while WE have the ball (support runs into the box,
                          defender holds shape), the goalkeeper at all times, and
                          "ball loose and I am the nearest teammate → INTERCEPT".
  1. LLM                  only the DEFENSIVE phase: opponent has the ball, or the ball is
                          loose and someone else is nearer. Pressing / marking decisions.
  2. fallback_commands()  rule-based defensive command when the LLM fails or is too slow.
  3. last resort          CLEAR_OVERRIDE — hand the player back to the engine's default AI.

Why: three practice matches (2026-09-03) produced 0 shots. The v2 "pressure rule" released
the ball every time an opponent came within 6 — on a 5v5 pitch that is every touch — so the
forwards never reached the shot gate. Shooting is now decided in code the moment a player is
in range, aimed at the corner away from the opposing keeper, and a pressed carrier inside
the attacking third shoots rather than turning the ball over.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from state import (
    get_goal_positions, dist, find_holder, is_holder,
    _player_idx, _is_my_team,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FallbackConfig:
    """Per-position doctrine. Everything the instinct layer needs lives here."""

    role: str = "FWD"
    """GK | DEF | MID | FWD."""

    # --- Shooting (distances are x-only, the way the engine's goal line works) ---
    shoot_threshold: float = 30.0
    """On the ball and within this x-distance of the opponent goal → SHOOT."""
    shoot_max_abs_y: float = 18.0
    """...and only when |y| is at most this (goalmouth is y -5..5)."""
    shoot_near: float = 15.0
    shoot_power_near: float = 0.85
    shoot_power_far: float = 1.0
    aim_flip: bool = False
    """Invert the corner mapping. Learned at match time by lib/match_memory.py (four scoreless shots → flip)."""

    # --- Pressure ---
    pressure_release_distance: float = 4.0
    """An opponent closer than this means I am about to be tackled."""
    pressure_shoot_distance: float = 38.0
    """Pressed inside this x-distance (and pressure_shoot_max_abs_y) → shoot anyway."""
    pressure_shoot_max_abs_y: float = 22.0
    outlet_clearance: float = 5.0
    """A teammate with no opponent inside this radius is an open outlet."""

    # --- Support run while a teammate has the ball ---
    support_x_ref: str = "opp_goal"
    """'opp_goal' → x = opp_goal_x pulled back by support_depth; 'my_goal' → my_goal_x * support_x_factor."""
    support_depth: float = 14.0
    support_x_factor: float = 0.45
    support_y: float | str = 0.0
    """Fixed y, or 'track_ball' (ball y * support_y_factor, clamped ±support_y_clamp)."""
    support_y_factor: float = 0.4
    support_y_clamp: float = 12.0
    support_sprint: bool = True

    # --- Carrying (on the ball, unpressed, out of range) ---
    carry_depth: float = 10.0
    """Carry toward opp_goal_x pulled back by this; y halves toward centre each tick."""
    prefer_pass_ahead: bool = False
    """DEF/MID: a forward clearly ahead and open beats carrying."""

    # --- Goalkeeper ---
    gk_line_depth: float = 3.0
    gk_track_factor: float = 0.4
    gk_max_abs_y: float = 4.0
    gk_intercept_radius: float = 10.0

    # --- Defensive fallback (LLM failed / too slow) ---
    press_distance: float = 20.0
    press_intensity: float = 0.8
    press_duration: int = 3
    mark_threshold: float = 0.0
    """DEF: mark the opponent nearest our goal when he is inside this distance of it."""
    mark_tightness: str = "TIGHT"
    default_x_ref: str = "opp_goal"
    """'opp_goal' | 'my_goal' | 'ball_x' — where to stand when nothing else applies."""
    default_x_factor: float = 0.45
    default_x_depth: float = 25.0
    default_y: float | str = 0.0

    pass_exclude_ids: list[int] = field(default_factory=lambda: [0])

    # --- Last resort (both LLM and fallback crashed) ---
    last_resort_command_type: str = "CLEAR_OVERRIDE"
    last_resort_params: dict = field(default_factory=dict)
    last_resort_duration: int = 0


GK_CONFIG = FallbackConfig(
    role="GK",
    shoot_threshold=0.0, pressure_release_distance=0.0,
)

DEF_CONFIG = FallbackConfig(
    role="DEF",
    shoot_threshold=22.0, shoot_max_abs_y=12.0,
    pressure_shoot_distance=28.0, pressure_shoot_max_abs_y=14.0,
    support_x_ref="my_goal", support_x_factor=0.45, support_y="track_ball",
    support_y_factor=0.3, support_y_clamp=10.0, support_sprint=False,
    prefer_pass_ahead=True, carry_depth=30.0,
    press_distance=18.0, press_intensity=0.9,
    mark_threshold=35.0,
    default_x_ref="my_goal", default_x_factor=0.55, default_y="track_ball",
)

MID_CONFIG = FallbackConfig(
    role="MID",
    shoot_threshold=28.0, shoot_max_abs_y=15.0,
    support_x_ref="opp_goal", support_depth=26.0, support_y="track_ball",
    support_y_factor=0.4, support_y_clamp=12.0, support_sprint=True,
    prefer_pass_ahead=True, carry_depth=14.0,
    press_distance=20.0, press_intensity=0.7,
    default_x_ref="ball_x", default_x_factor=0.5, default_y="track_ball",
)

FWD1_CONFIG = FallbackConfig(
    role="FWD",
    shoot_threshold=30.0, shoot_max_abs_y=18.0,
    support_x_ref="opp_goal", support_depth=14.0, support_y=-7.0, support_sprint=True,
    carry_depth=10.0,
    press_distance=18.0, press_intensity=0.8,
    default_x_ref="opp_goal", default_x_depth=25.0, default_y=-8.0,
)

FWD2_CONFIG = FallbackConfig(
    role="FWD",
    shoot_threshold=30.0, shoot_max_abs_y=18.0,
    support_x_ref="opp_goal", support_depth=14.0, support_y=7.0, support_sprint=True,
    carry_depth=10.0,
    press_distance=18.0, press_intensity=0.8,
    default_x_ref="opp_goal", default_x_depth=25.0, default_y=8.0,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cmd(cmd_type: str, pid: int, tid: int, params: dict, duration: int = 0) -> dict:
    return {"commandType": cmd_type, "playerId": pid, "teamId": tid,
            "parameters": params, "duration": duration}


def _dir(opp_goal_x: float) -> float:
    """+1 when we attack toward +x, -1 toward -x."""
    return 1.0 if opp_goal_x > 0 else -1.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _pos(p: dict) -> dict:
    return p.get("position") or {"x": 0, "y": 0}


def _split(players, team_id):
    mine = [p for p in players if _is_my_team(p, team_id)]
    opps = [p for p in players if not _is_my_team(p, team_id)]
    return mine, opps


def _nearest_opp_dist(opps, pos) -> float:
    return min((dist(_pos(o), pos) for o in opps), default=999.0)


def _in_shot_range(cfg, pos, opp_goal_x) -> bool:
    return (cfg.shoot_threshold > 0
            and abs(pos.get("x", 0) - opp_goal_x) <= cfg.shoot_threshold
            and abs(pos.get("y", 0)) <= cfg.shoot_max_abs_y)


GOAL_HALF_WIDTH = 5.0
AIM_INSET = 1.5           # aim this far inside the post
KEEPER_LEAD_S = 0.4       # seconds of keeper velocity to lead
KEEPER_REACH = 2.5
FAR_POST_BONUS = 1.5
BLOCK_RADIUS = 1.6


def _opp_keeper(players, team_id, opp_goal_x):
    _, opps = _split(players, team_id)
    if not opps:
        return None
    gk = next((p for p in opps if _player_idx(p) == 0), None)
    return gk or min(opps, key=lambda p: abs(_pos(p).get("x", 0) - opp_goal_x))


def opp_keeper_y(players, team_id, opp_goal_x) -> float:
    """Predicted y of the opposing keeper: position led by KEEPER_LEAD_S of velocity."""
    gk = _opp_keeper(players, team_id, opp_goal_x)
    if gk is None:
        return 0.0
    vel = gk.get("velocity") or {}
    return float(_pos(gk).get("y", 0)) + float(vel.get("y", 0) or 0) * KEEPER_LEAD_S


def _blockers_on_line(shooter_pos, target, players, team_id):
    """Opponents (keeper excluded) standing inside BLOCK_RADIUS of the shot line."""
    _, opps = _split(players, team_id)
    sx, sy = shooter_pos.get("x", 0), shooter_pos.get("y", 0)
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    length2 = dx * dx + dy * dy or 1.0
    n = 0
    for o in opps:
        if _player_idx(o) == 0:
            continue
        ox, oy = _pos(o).get("x", 0), _pos(o).get("y", 0)
        u = ((ox - sx) * dx + (oy - sy) * dy) / length2
        if 0.05 < u < 0.95:
            px, py = sx + u * dx, sy + u * dy
            if ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5 <= BLOCK_RADIUS:
                n += 1
    return n


def shot_plan(shooter_pos, players, team_id, opp_goal_x, flip: bool = False) -> dict:
    """Pick the side of the goal the keeper is least able to cover.

    Score per side (+y / -y): keeper's predicted distance to the aim point, minus blockers on
    the line, plus a far-post bonus from wide angles. Returns aim_location, side, blocked (both
    sides blocked), target (x, y).
    """
    gk_y = opp_keeper_y(players, team_id, opp_goal_x)
    sy = shooter_pos.get("y", 0)
    far_post = -1 if sy > 0 else 1
    plans = {}
    for side in (1, -1):
        target = (opp_goal_x, side * (GOAL_HALF_WIDTH - AIM_INSET))
        keeper_gap = abs(target[1] - gk_y)
        blockers = _blockers_on_line(shooter_pos, target, players, team_id)
        score = keeper_gap - blockers * 10.0
        if abs(sy) >= 4 and side == far_post:
            score += FAR_POST_BONUS
        plans[side] = {"side": side, "score": score, "blockers": blockers, "target": target}
    best = max(plans.values(), key=lambda p: (p["score"], p["side"] == far_post))
    side = best["side"]
    if flip:
        side = -side
    tb = "T" if side > 0 else "B"
    left_is_pos_y = opp_goal_x > 0
    lr = "L" if ((side > 0) == left_is_pos_y) else "R"
    return {"aim_location": tb + lr, "side": best["side"], "blocked": best["blockers"] > 0,
            "target": best["target"], "keeper_y": gk_y}


def aim_corner(shooter_pos, players, team_id, opp_goal_x, flip: bool = False) -> str:
    """Corner name the plan chose. Engine corner names are TL/TR/BL/BR; their axis mapping is
    undocumented, so both letters come from the same side decision (top = +y; left = shooter's
    left facing the goal) and lib/match_memory.py flips the mapping if shots keep missing."""
    return shot_plan(shooter_pos, players, team_id, opp_goal_x, flip)["aim_location"]


def _shoot(cfg, pid, tid, pos, players, opp_goal_x, force_power=None) -> list[dict]:
    d = abs(pos.get("x", 0) - opp_goal_x)
    power = force_power if force_power is not None else (
        cfg.shoot_power_near if d <= cfg.shoot_near else cfg.shoot_power_far)
    plan = shot_plan(pos, players, tid, opp_goal_x, cfg.aim_flip)
    return [_cmd("SHOOT", pid, tid, {"aim_location": plan["aim_location"], "power": power})]


def _sidestep_for_shot(cfg, pid, tid, pos, players, opp_goal_x):
    """Both shot lines blocked and nobody on me: move 4 units toward the freer side, still in range."""
    plan = shot_plan(pos, players, tid, opp_goal_x, cfg.aim_flip)
    d = _dir(opp_goal_x)
    ty = _clamp(pos.get("y", 0) + 4.0 * plan["side"], -cfg.shoot_max_abs_y, cfg.shoot_max_abs_y)
    tx = _clamp(pos.get("x", 0) + d * 3.0, -52, 52)
    return [_cmd("MOVE_TO", pid, tid, {"target_x": round(tx, 1), "target_y": round(ty, 1), "sprint": True})]


def _clearance(p, opps) -> float:
    return _nearest_opp_dist(opps, _pos(p))


def _best_outlet(cfg, mine, opps, my_pid, pos, opp_goal_x):
    """Most advanced open teammate; if nobody is open, the one with the most room. Never the GK."""
    exclude = set(cfg.pass_exclude_ids) | {my_pid}
    outlets = [p for p in mine if _player_idx(p) not in exclude]
    if not outlets:
        return None
    open_ = [p for p in outlets if _clearance(p, opps) >= cfg.outlet_clearance]
    if open_:
        return min(open_, key=lambda p: abs(_pos(p).get("x", 0) - opp_goal_x))
    return max(outlets, key=lambda p: _clearance(p, opps))


def _pass_to(cfg, target, my_pid, tid, pos, opp_goal_x) -> list[dict]:
    ahead = (abs(pos.get("x", 0) - opp_goal_x) - abs(_pos(target).get("x", 0) - opp_goal_x))
    kind = "THROUGH" if ahead > 5 else "GROUND"
    return [_cmd("PASS", my_pid, tid, {"target_player_id": _player_idx(target), "type": kind})]


def _forward_ahead(cfg, mine, opps, pos, opp_goal_x):
    """A forward (3/4) clearly ahead of me and open, else None."""
    my_d = abs(pos.get("x", 0) - opp_goal_x)
    cands = [p for p in mine if _player_idx(p) in (3, 4)
             and abs(_pos(p).get("x", 0) - opp_goal_x) < my_d - 5
             and _clearance(p, opps) >= cfg.outlet_clearance]
    if not cands:
        return None
    return min(cands, key=lambda p: abs(_pos(p).get("x", 0) - opp_goal_x))


# ---------------------------------------------------------------------------
# Instinct layer
# ---------------------------------------------------------------------------

def instinct_command(cfg: FallbackConfig, game_state: dict, team_id: int, my_pid: int):
    """Code-decided command for this tick, or None when the LLM should decide (defensive phase)."""
    players = game_state.get("players", [])
    ball = game_state.get("ball", {})
    ball_pos = ball.get("position") or {"x": 0, "y": 0}
    mine, opps = _split(players, team_id)
    me = next((p for p in mine if _player_idx(p) == my_pid), None)
    if not me:
        return None
    pos = _pos(me)
    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    holder = find_holder(ball, players)
    i_have_ball = is_holder(holder, team_id, my_pid)
    we_have_ball = holder is not None and _is_my_team(holder, team_id)

    if cfg.role == "GK":
        return _gk(cfg, my_pid, team_id, pos, ball, ball_pos, players, mine, opps,
                   my_goal_x, opp_goal_x, i_have_ball)

    if i_have_ball:
        return _on_ball(cfg, my_pid, team_id, pos, players, mine, opps, my_goal_x, opp_goal_x)

    if we_have_ball:
        return _support(cfg, my_pid, team_id, ball_pos, my_goal_x, opp_goal_x)

    if holder is None:                           # loose ball
        my_d = dist(pos, ball_pos)
        nearest = min(mine, key=lambda p: dist(_pos(p), ball_pos))
        if _player_idx(nearest) == my_pid or my_d < 4:
            return [_cmd("INTERCEPT", my_pid, team_id, {"aggressive": True}, duration=2)]

    return None                                  # opponent has it → LLM


def _on_ball(cfg, pid, tid, pos, players, mine, opps, my_goal_x, opp_goal_x):
    pressed = (cfg.pressure_release_distance > 0
               and _nearest_opp_dist(opps, pos) < cfg.pressure_release_distance)

    if _in_shot_range(cfg, pos, opp_goal_x):
        plan = shot_plan(pos, players, tid, opp_goal_x, cfg.aim_flip)
        close = abs(pos.get("x", 0) - opp_goal_x) <= 10
        if plan["blocked"] and not pressed and not close:
            return _sidestep_for_shot(cfg, pid, tid, pos, players, opp_goal_x)
        return _shoot(cfg, pid, tid, pos, players, opp_goal_x)
    if pressed:
        if (abs(pos.get("x", 0) - opp_goal_x) <= cfg.pressure_shoot_distance
                and abs(pos.get("y", 0)) <= cfg.pressure_shoot_max_abs_y):
            return _shoot(cfg, pid, tid, pos, players, opp_goal_x, force_power=cfg.shoot_power_far)
        target = _best_outlet(cfg, mine, opps, pid, pos, opp_goal_x)
        if target is not None:
            return _pass_to(cfg, target, pid, tid, pos, opp_goal_x)

    if cfg.prefer_pass_ahead:
        target = _forward_ahead(cfg, mine, opps, pos, opp_goal_x)
        if target is not None:
            return _pass_to(cfg, target, pid, tid, pos, opp_goal_x)

    d = _dir(opp_goal_x)
    tx = opp_goal_x - d * cfg.carry_depth
    # never carry backwards: at least 8 further forward than where I stand
    if (tx - pos.get("x", 0)) * d < 8:
        tx = pos.get("x", 0) + d * 8
    return [_cmd("MOVE_TO", pid, tid,
                 {"target_x": _clamp(tx, -52, 52), "target_y": round(pos.get("y", 0) * 0.5, 1),
                  "sprint": True})]


def _support(cfg, pid, tid, ball_pos, my_goal_x, opp_goal_x):
    d = _dir(opp_goal_x)
    if cfg.support_x_ref == "my_goal":
        tx = my_goal_x * cfg.support_x_factor
    else:
        tx = opp_goal_x - d * cfg.support_depth
    if cfg.support_y == "track_ball":
        ty = _clamp(ball_pos.get("y", 0) * cfg.support_y_factor, -cfg.support_y_clamp, cfg.support_y_clamp)
    else:
        ty = float(cfg.support_y)
    return [_cmd("MOVE_TO", pid, tid,
                 {"target_x": round(_clamp(tx, -52, 52), 1), "target_y": round(ty, 1),
                  "sprint": cfg.support_sprint})]


def _gk(cfg, pid, tid, pos, ball, ball_pos, players, mine, opps, my_goal_x, opp_goal_x, has_ball):
    d = _dir(opp_goal_x)
    if has_ball:
        forwards = [p for p in mine if _player_idx(p) in (3, 4)]
        open_fwd = [p for p in forwards if _clearance(p, opps) >= 8]
        if open_fwd:
            target = min(open_fwd, key=lambda p: abs(_pos(p).get("x", 0) - opp_goal_x))
            return [_cmd("GK_DISTRIBUTE", pid, tid,
                         {"target_player_id": _player_idx(target), "method": "KICK"})]
        short = [p for p in mine if _player_idx(p) in (1, 2)] or [p for p in mine if _player_idx(p) != pid]
        if short:
            target = max(short, key=lambda p: _clearance(p, opps))
            return [_cmd("GK_DISTRIBUTE", pid, tid,
                         {"target_player_id": _player_idx(target), "method": "THROW"})]
        return [_cmd("GK_DISTRIBUTE", pid, tid, {"target_player_id": 1, "method": "THROW"})]

    loose = find_holder(ball, players) is None
    in_my_third = abs(ball_pos.get("x", 0) - my_goal_x) < 25
    if loose and in_my_third and dist(pos, ball_pos) < cfg.gk_intercept_radius:
        return [_cmd("INTERCEPT", pid, tid, {"aggressive": True}, duration=2)]

    tx = my_goal_x + d * cfg.gk_line_depth
    ty = _clamp(ball_pos.get("y", 0) * cfg.gk_track_factor, -cfg.gk_max_abs_y, cfg.gk_max_abs_y)
    return [_cmd("MOVE_TO", pid, tid,
                 {"target_x": round(tx, 1), "target_y": round(ty, 1), "sprint": False})]


# ---------------------------------------------------------------------------
# Rule-based fallback (defensive phase; also covers instinct for completeness)
# ---------------------------------------------------------------------------

def build_last_resort(cfg: FallbackConfig, player_id: int) -> dict:
    return {
        "commandType": cfg.last_resort_command_type,
        "playerId": player_id,
        "parameters": dict(cfg.last_resort_params),
        "duration": cfg.last_resort_duration,
    }


def build_fallback(cfg: FallbackConfig) -> Callable[[dict, int, int], list[dict]]:
    """Return fallback_commands(game_state, team_id, my_player_id)."""

    def fallback_commands(game_state: dict, team_id: int, my_pid: int) -> list[dict]:
        instinct = instinct_command(cfg, game_state, team_id, my_pid)
        if instinct:
            return instinct

        players = game_state.get("players", [])
        ball = game_state.get("ball", {})
        ball_pos = ball.get("position") or {"x": 0, "y": 0}
        mine, opps = _split(players, team_id)
        me = next((p for p in mine if _player_idx(p) == my_pid), None)
        my_goal_x, opp_goal_x = get_goal_positions(team_id)
        if not me:
            return [_cmd("CLEAR_OVERRIDE", my_pid, team_id, {})]
        pos = _pos(me)

        # DEF: mark the most dangerous opponent
        if cfg.mark_threshold > 0 and opps:
            dangerous = min(opps, key=lambda p: abs(_pos(p).get("x", 0) - my_goal_x))
            if abs(_pos(dangerous).get("x", 0) - my_goal_x) < cfg.mark_threshold:
                return [_cmd("MARK", my_pid, team_id,
                             {"target_player_id": _player_idx(dangerous),
                              "tightness": cfg.mark_tightness}, duration=3)]

        # One presser: the teammate nearest the ball
        nearest = min(mine, key=lambda p: dist(_pos(p), ball_pos))
        if _player_idx(nearest) == my_pid and dist(pos, ball_pos) < cfg.press_distance:
            return [_cmd("PRESS_BALL", my_pid, team_id,
                         {"intensity": cfg.press_intensity}, duration=cfg.press_duration)]

        tx, ty = _default_pos(cfg, my_goal_x, opp_goal_x, ball_pos)
        return [_cmd("MOVE_TO", my_pid, team_id,
                     {"target_x": round(tx, 1), "target_y": round(ty, 1), "sprint": False})]

    return fallback_commands


def _default_pos(cfg, my_goal_x, opp_goal_x, ball_pos):
    d = _dir(opp_goal_x)
    if cfg.default_x_ref == "my_goal":
        tx = my_goal_x * cfg.default_x_factor
    elif cfg.default_x_ref == "opp_goal":
        tx = opp_goal_x - d * cfg.default_x_depth
    elif cfg.default_x_ref == "ball_x":
        tx = ball_pos.get("x", 0) * cfg.default_x_factor
    else:
        tx = 0.0
    if cfg.default_y == "track_ball":
        ty = _clamp(ball_pos.get("y", 0) * 0.3, -10, 10)
    else:
        ty = float(cfg.default_y)
    return _clamp(tx, -52, 52), ty
