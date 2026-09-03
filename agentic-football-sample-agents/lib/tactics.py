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


def _shot_aim(opponents: list[dict], player_id: int) -> str:
    keeper = next((p for p in opponents if _player_idx(p) == 0), None)
    keeper_y = float(keeper.get("position", {}).get("y", 0)) if keeper else 0.0
    if keeper_y > 1.0:
        return "BL"
    if keeper_y < -1.0:
        return "TR"
    return "TR" if player_id % 2 else "BL"


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
                "sprint": pressure < 5,
            })],
            0.82,
            "defender carried away from immediate danger",
        )

    shot_distance = 27.0 if position_label.startswith("FWD") else 21.5
    shot_width = 20 if adjustments["avoid_low_angle_shots"] else 24
    if adjustments["avoid_low_angle_shots"]:
        shot_distance -= 2.0
    if goal_distance <= shot_distance and abs(float(pos.get("y", 0))) <= shot_width:
        power = _clamp(0.72 + goal_distance / 120.0, 0.78, 0.96)
        return TacticalDecision(
            [_cmd("SHOOT", player_id, team_id, {
                "aim_location": _shot_aim(opponents, player_id), "power": round(power, 2),
            })],
            0.95,
            "high-value shooting position",
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
            "target_x": round(target_x, 1), "target_y": round(target_y, 1), "sprint": pressure < 8.0,
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
                "target_x": target_x, "target_y": round(target_y, 1), "sprint": distance_to_ball < 10,
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
                "sprint": opponent_possession and distance_to_ball > 20,
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
    return TacticalDecision(
        [_cmd("MOVE_TO", player_id, team_id, {
            "target_x": round(_world_x(target_attack_x, team_id), 1),
            "target_y": round(target_y, 1),
            "sprint": our_possession and target_attack_x > _attack_x(float(pos.get("x", 0)), team_id) + 4,
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
