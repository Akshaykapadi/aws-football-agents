"""Fast, deterministic 5v5 tactics used before considering an LLM call.

These calculations deliberately run in-process.  They replace the sample
Gateway's remote calculator tools on the match hot path: no MCP round trip, no
model tool-selection turn, and a valid command is always available immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

from state import (
    _is_my_team,
    _player_idx,
    dist,
    find_possession_holder,
    get_goal_positions,
)


@dataclass(frozen=True)
class TacticalDecision:
    commands: list[dict]
    confidence: float
    reason: str


def memory_adjustments(learned_context: str) -> dict[str, bool]:
    """Turn retrieved match lessons into conservative rule adjustments.

    Episodic records are natural-language/XML, so the rule engine only reacts
    to clear phrases. Unknown memory content cannot destabilize the base shape.
    """
    text = (learned_context or "").lower()
    return {
        "release_earlier": any(
            phrase in text
            for phrase in ("release the ball earlier", "turnovers lost", "lost possession under pressure")
        ),
        "attack_higher": any(
            phrase in text
            for phrase in ("support line higher", "failed to score", "create a final-third pass")
        ),
        "protect_transition": any(
            phrase in text
            for phrase in ("staggered during attacks", "conceded on transition", "defensive transition")
        ),
        "avoid_low_angle_shots": any(
            phrase in text
            for phrase in ("low-angle shots", "poor shooting angle", "forced shots")
        ),
        "conserve_stamina": any(
            phrase in text
            for phrase in ("late-match stamina", "excessive sprinting", "finished with low stamina")
        ),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _cmd(kind: str, pid: int, tid: int, parameters: dict, duration: int = 0) -> dict:
    return {
        "commandType": kind,
        "playerId": pid,
        "teamId": tid,
        "parameters": parameters,
        "duration": duration,
    }


def _attack_direction(team_id: int) -> int:
    return 1 if team_id == 0 else -1


def _attack_x(world_x: float, team_id: int) -> float:
    """Coordinate where both teams attack toward increasing x."""
    return world_x * _attack_direction(team_id)


def _world_x(attack_x: float, team_id: int) -> float:
    return attack_x * _attack_direction(team_id)


def _predicted_position(player: dict, seconds: float = 0.8) -> dict:
    pos = player.get("position", {})
    velocity = player.get("velocity", {})
    return {
        "x": float(pos.get("x", 0) or 0) + seconds * float(velocity.get("x", 0) or 0),
        "y": float(pos.get("y", 0) or 0) + seconds * float(velocity.get("y", 0) or 0),
    }


def _nearest_distance(position: dict, players: list[dict]) -> float:
    return min((dist(position, _predicted_position(p)) for p in players), default=99.0)


def _stamina_ratio(player: dict) -> float:
    """Normalise the engine's raw 0..1 or 0..100 stamina value."""
    try:
        value = float(player.get("stamina", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if value > 1.0:
        value /= 100.0
    return _clamp(value, 0.0, 1.0)


def _can_sprint(
    player: dict,
    *,
    decisive: bool = False,
    emergency: bool = False,
    conserve: bool = False,
) -> bool:
    """Spend stamina only on a scoring break or a genuine defensive emergency."""
    minimum = 0.18 if emergency else (0.55 if decisive else 0.72)
    if conserve and not emergency:
        minimum += 0.10
    return _stamina_ratio(player) >= minimum


def _lane_clearance(start: dict, end: dict, opponents: list[dict]) -> float:
    """Minimum predicted opponent distance from the useful part of a pass lane."""
    sx, sy = float(start.get("x", 0)), float(start.get("y", 0))
    ex, ey = float(end.get("x", 0)), float(end.get("y", 0))
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq < 0.01:
        return 0.0

    clearance = 30.0
    for opponent in opponents:
        point = _predicted_position(opponent)
        projection = ((point["x"] - sx) * dx + (point["y"] - sy) * dy) / length_sq
        if 0.08 <= projection <= 1.05:
            closest = {"x": sx + projection * dx, "y": sy + projection * dy}
            clearance = min(clearance, dist(point, closest))
    return clearance


def _pass_options(
    me: dict,
    teammates: list[dict],
    opponents: list[dict],
    team_id: int,
    exclude: set[int] | None = None,
) -> list[tuple[float, dict, float, float]]:
    """Rank receivers by progression, lane clearance, pressure, and distance."""
    exclude = exclude or set()
    origin = me.get("position", {})
    origin_attack_x = _attack_x(float(origin.get("x", 0)), team_id)
    ranked = []
    for teammate in teammates:
        pid = _player_idx(teammate)
        if pid in exclude or pid == _player_idx(me):
            continue
        target = _predicted_position(teammate, 0.6)
        pass_distance = dist(origin, target)
        if pass_distance < 2.0 or pass_distance > 62.0:
            continue
        progression = _attack_x(target["x"], team_id) - origin_attack_x
        clearance = _lane_clearance(origin, target, opponents)
        pressure = _nearest_distance(target, opponents)
        receiver_vx = float(teammate.get("velocity", {}).get("x", 0) or 0)
        forward_velocity = receiver_vx * _attack_direction(team_id)

        score = (
            0.035 * progression
            + 0.09 * min(clearance, 12.0)
            + 0.055 * min(pressure, 14.0)
            + 0.04 * forward_velocity
            - 0.012 * pass_distance
        )
        if clearance < 2.2:
            score -= 1.0
        if pressure < 3.0:
            score -= 0.8
        ranked.append((score, teammate, clearance, progression))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def _best_shot_lane(
    start: dict,
    opponent_goal_x: float,
    opponents: list[dict],
    player_id: int,
) -> tuple[str, float]:
    """Choose the less-covered corner and return its predicted lane clearance."""
    corners = [("TR", 4.2), ("BL", -4.2)]
    if player_id % 2 == 0:
        corners.reverse()
    ranked = [
        (
            _lane_clearance(start, {"x": opponent_goal_x, "y": target_y}, opponents),
            aim,
        )
        for aim, target_y in corners
    ]
    clearance, aim = max(ranked, key=lambda item: item[0])
    return aim, clearance


def _shoot_if_open(
    position_label: str,
    pos: dict,
    opponent_goal_x: float,
    opponents: list[dict],
    player_id: int,
    team_id: int,
    avoid_low_angle_shots: bool,
) -> TacticalDecision | None:
    """Shoot-first policy for any player with a viable sight of goal.

    The keeper almost never reaches this range, but is intentionally eligible:
    possession and an open goal should not be turned into an unnecessary pass.
    """
    goal_distance = dist(pos, {"x": opponent_goal_x, "y": 0.0})
    range_by_role = {"GK": 28.0, "DEF": 30.0, "MID": 33.0, "FWD1": 36.0, "FWD2": 36.0}
    long_range_by_role = {"GK": 42.0, "DEF": 42.0, "MID": 46.0, "FWD1": 48.0, "FWD2": 48.0}
    max_distance = range_by_role.get(position_label, 32.0)
    max_width = 20.0 if avoid_low_angle_shots else 25.0
    if abs(float(pos.get("y", 0) or 0)) > max_width:
        return None

    aim, lane_clearance = _best_shot_lane(pos, opponent_goal_x, opponents, player_id)
    close_finish = goal_distance <= 19.0 and lane_clearance >= 1.5
    clear_shot = goal_distance <= max_distance and lane_clearance >= 3.0
    very_open_shot = goal_distance <= max_distance + 4.0 and lane_clearance >= 5.5
    long_open_shot = (
        goal_distance <= long_range_by_role.get(position_label, 44.0)
        and abs(float(pos.get("y", 0) or 0)) <= 18.0
        and lane_clearance >= 7.0
    )
    if not (close_finish or clear_shot or very_open_shot or long_open_shot):
        return None

    power = _clamp(0.78 + goal_distance / 130.0, 0.82, 0.99)
    return TacticalDecision(
        [_cmd("SHOOT", player_id, team_id, {
            "aim_location": aim,
            "power": round(power, 2),
        })],
        0.99 if long_open_shot else (0.98 if clear_shot or very_open_shot else 0.95),
        (
            f"clearly visible long-range goal ({lane_clearance:.1f}m clearance)"
            if long_open_shot and goal_distance > max_distance
            else f"open shooting lane ({lane_clearance:.1f}m clearance)"
        ),
    )


def _pass_command(option, pid: int, tid: int) -> dict:
    _, receiver, clearance, progression = option
    pass_type = "THROUGH" if progression > 8.0 and clearance > 4.0 else "GROUND"
    return _cmd(
        "PASS",
        pid,
        tid,
        {"target_player_id": _player_idx(receiver), "type": pass_type},
    )


def _on_ball(
    game_state: dict,
    team_id: int,
    player_id: int,
    position_label: str,
    me: dict,
    teammates: list[dict],
    opponents: list[dict],
    adjustments: dict[str, bool],
) -> TacticalDecision:
    pos = me.get("position", {})
    _, opponent_goal_x = get_goal_positions(team_id)
    goal = {"x": opponent_goal_x, "y": 0.0}
    goal_distance = dist(pos, goal)
    pressure = _nearest_distance(pos, opponents)

    shot = _shoot_if_open(
        position_label,
        pos,
        opponent_goal_x,
        opponents,
        player_id,
        team_id,
        adjustments["avoid_low_angle_shots"],
    )
    if shot is not None:
        return shot

    if position_label == "GK":
        options = _pass_options(me, teammates, opponents, team_id)
        if options:
            receiver = options[0][1]
            method = "THROW" if dist(pos, receiver.get("position", {})) <= 32.0 else "KICK"
            return TacticalDecision(
                [_cmd("GK_DISTRIBUTE", player_id, team_id, {
                    "target_player_id": _player_idx(receiver), "method": method,
                })],
                0.98,
                "goalkeeper distribution ranked locally",
            )
        return TacticalDecision(
            [_cmd("GK_DISTRIBUTE", player_id, team_id, {"target_player_id": 1, "method": "THROW"})],
            0.95,
            "safe goalkeeper outlet",
        )

    options = _pass_options(
        me,
        teammates,
        opponents,
        team_id,
        exclude={0} if position_label == "DEF" else set(),
    )

    if position_label == "DEF":
        if options:
            confidence = 0.93 if options[0][0] > 0.0 else 0.76
            return TacticalDecision(
                [_pass_command(options[0], player_id, team_id)],
                confidence,
                "defender chose safest progressive lane",
            )
        return TacticalDecision(
            [_cmd("MOVE_TO", player_id, team_id, {
                "target_x": _clamp(float(pos.get("x", 0)) + 7 * _attack_direction(team_id), -52, 52),
                "target_y": _clamp(float(pos.get("y", 0)) * 0.7, -28, 28),
                "sprint": pressure < 5 and _can_sprint(
                    me, decisive=True, conserve=adjustments["conserve_stamina"]
                ),
            })],
            0.82,
            "defender carried away from immediate danger",
        )

    release_pressure = 7.5 if adjustments["release_earlier"] else 5.5
    if options and (pressure < release_pressure or options[0][0] > 0.75):
        return TacticalDecision(
            [_pass_command(options[0], player_id, team_id)],
            0.91,
            "pressure release or clearly superior passing lane",
        )

    # The shoot/pass/carry trade-off in this band is the only common situation
    # worth spending a model call on.  The local carry remains the immediate
    # command if the model is disabled, slow, or malformed.
    direction = _attack_direction(team_id)
    target_x = _clamp(float(pos.get("x", 0)) + 9 * direction, -51, 51)
    target_y = _clamp(float(pos.get("y", 0)) * 0.72, -27, 27)
    ambiguous = 20.0 < goal_distance < 38.0 and 5.0 <= pressure <= 13.0 and bool(options)
    return TacticalDecision(
        [_cmd("MOVE_TO", player_id, team_id, {
            "target_x": round(target_x, 1),
            "target_y": round(target_y, 1),
            "sprint": pressure < 8.0 and _can_sprint(
                me, decisive=True, conserve=adjustments["conserve_stamina"]
            ),
        })],
        0.58 if ambiguous else 0.86,
        "ambiguous final-third choice" if ambiguous else "controlled forward carry",
    )


def _off_ball(
    game_state: dict,
    team_id: int,
    player_id: int,
    position_label: str,
    me: dict,
    teammates: list[dict],
    opponents: list[dict],
    holder: dict | None,
    adjustments: dict[str, bool],
) -> TacticalDecision:
    ball = game_state.get("ball", {})
    ball_pos = ball.get("position", {})
    pos = me.get("position", {})
    direction = _attack_direction(team_id)
    ball_attack_x = _attack_x(float(ball_pos.get("x", 0)), team_id)
    distance_to_ball = dist(pos, ball_pos)
    our_possession = bool(holder and _is_my_team(holder, team_id))
    opponent_possession = bool(holder and not _is_my_team(holder, team_id))

    if position_label == "GK":
        target_x = _world_x(-49.0 if ball_attack_x < -8 else -47.0, team_id)
        target_y = _clamp(float(ball_pos.get("y", 0)) * 0.45, -11, 11)
        return TacticalDecision(
            [_cmd("MOVE_TO", player_id, team_id, {
                "target_x": target_x,
                "target_y": round(target_y, 1),
                "sprint": distance_to_ball < 10 and _can_sprint(me, emergency=True),
            })],
            0.99,
            "goalkeeper angle and line control",
        )

    if holder is None and distance_to_ball <= (17 if position_label == "MID" else 12):
        return TacticalDecision(
            [_cmd("INTERCEPT", player_id, team_id, {"aggressive": position_label != "DEF"}, duration=3)],
            0.96,
            "nearby loose ball",
        )

    if position_label == "DEF":
        dangerous = min(
            opponents,
            key=lambda p: _attack_x(float(p.get("position", {}).get("x", 0)), team_id),
            default=None,
        )
        if opponent_possession and distance_to_ball <= 10:
            return TacticalDecision(
                [_cmd("PRESS_BALL", player_id, team_id, {"intensity": 0.82}, duration=3)],
                0.95,
                "defender can pressure the carrier",
            )
        if dangerous and _attack_x(float(dangerous.get("position", {}).get("x", 0)), team_id) < -20:
            return TacticalDecision(
                [_cmd("MARK", player_id, team_id, {
                    "target_player_id": _player_idx(dangerous), "tightness": "TIGHT",
                }, duration=3)],
                0.94,
                "marking the deepest threat",
            )
        cover_depth = 16 if adjustments["protect_transition"] else 13
        target_attack_x = _clamp(ball_attack_x - cover_depth, -39, -10)
        return TacticalDecision(
            [_cmd("MOVE_TO", player_id, team_id, {
                "target_x": round(_world_x(target_attack_x, team_id), 1),
                "target_y": round(_clamp(float(ball_pos.get("y", 0)) * 0.45, -18, 18), 1),
                "sprint": False,
            })],
            0.96,
            "compact defensive cover",
        )

    if position_label == "MID":
        if opponent_possession and distance_to_ball <= 17:
            return TacticalDecision(
                [_cmd("PRESS_BALL", player_id, team_id, {"intensity": 0.7}, duration=3)],
                0.95,
                "midfield counter-pressure",
            )
        if our_possession:
            offset = 2 if adjustments["attack_higher"] else 5
        else:
            offset = 12 if adjustments["protect_transition"] else 10
        target_attack_x = _clamp(ball_attack_x - offset, -25, 25)
        return TacticalDecision(
            [_cmd("MOVE_TO", player_id, team_id, {
                "target_x": round(_world_x(target_attack_x, team_id), 1),
                "target_y": round(_clamp(float(ball_pos.get("y", 0)) * 0.65, -22, 22), 1),
                "sprint": (
                    opponent_possession
                    and 20 < distance_to_ball < 32
                    and _can_sprint(
                        me,
                        emergency=ball_attack_x < -12,
                        conserve=adjustments["conserve_stamina"],
                    )
                ),
            })],
            0.94,
            "midfield rest-defense and support position",
        )

    lane_y = -10.0 if position_label == "FWD1" else 10.0
    if opponent_possession and ball_attack_x > -5 and distance_to_ball <= 15:
        return TacticalDecision(
            [_cmd("PRESS_BALL", player_id, team_id, {"intensity": 0.72}, duration=3)],
            0.95,
            "forward press in useful territory",
        )
    forward_run = 14 if adjustments["attack_higher"] else 11
    target_attack_x = _clamp(ball_attack_x + (forward_run if our_possession else 5), 4, 42)
    # Move the two forwards on different vertical lanes and slightly toward the
    # opposite side of the ball to stretch the last defender.
    target_y = _clamp(lane_y - 0.18 * float(ball_pos.get("y", 0)), -25, 25)
    run_gain = target_attack_x - _attack_x(float(pos.get("x", 0)), team_id)
    decisive_break = our_possession and ball_attack_x > 4 and run_gain > 8
    return TacticalDecision(
        [_cmd("MOVE_TO", player_id, team_id, {
            "target_x": round(_world_x(target_attack_x, team_id), 1),
            "target_y": round(target_y, 1),
            "sprint": decisive_break and _can_sprint(
                me, decisive=True, conserve=adjustments["conserve_stamina"]
            ),
        })],
        0.95,
        "staggered forward support lane",
    )


def decide_locally(
    game_state: dict,
    team_id: int,
    player_id: int,
    position_label: str,
    learned_context: str = "",
) -> TacticalDecision | None:
    """Return one immediate tactical command, or ``None`` for malformed state."""
    players = game_state.get("players", [])
    me = next(
        (p for p in players if _player_idx(p) == player_id and _is_my_team(p, team_id)),
        None,
    )
    if me is None:
        return None

    teammates = [p for p in players if _is_my_team(p, team_id)]
    opponents = [p for p in players if not _is_my_team(p, team_id)]
    holder = find_possession_holder(game_state.get("ball", {}), players)
    adjustments = memory_adjustments(learned_context)
    has_ball = bool(holder and _is_my_team(holder, team_id) and _player_idx(holder) == player_id)

    if has_ball:
        return _on_ball(
            game_state, team_id, player_id, position_label, me, teammates, opponents, adjustments,
        )
    return _off_ball(
        game_state, team_id, player_id, position_label, me, teammates, opponents, holder, adjustments,
    )
