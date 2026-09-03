"""Game state summarization utilities for AI soccer agents."""

import math


# ---------------------------------------------------------------------------
# Format-agnostic helpers — handle both new (agentId/teamCode/possessionAgentId)
# and old (playerId/teamId/possessionPlayerId) game server formats.
# ---------------------------------------------------------------------------

def _player_idx(p: dict) -> int:
    """Numeric index (0-4) from a player dict — new agentId or old playerId."""
    if "agentId" in p:
        try:
            return int(p["agentId"].rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return 0
    return p.get("playerId", 0)


def _is_my_team(p: dict, team_id: int) -> bool:
    """True if player belongs to team_id — new teamCode or old teamId."""
    if "teamCode" in p:
        return p["teamCode"] == ("home" if team_id == 0 else "away")
    return p.get("teamId") == team_id


def _possession_idx(ball: dict):
    """Numeric possession player index from ball dict — new possessionAgentId or old possessionPlayerId.
    Returns int or None."""
    agent_id = ball.get("possessionAgentId")
    if agent_id is not None:
        try:
            return int(agent_id.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return None
    return ball.get("possessionPlayerId")


def find_possession_holder(ball: dict, players: list) -> dict | None:
    """Return the actual player holding the ball.

    Player numbers are duplicated across HOME and AWAY.  Older sample code used
    the first matching number, which can report the wrong team whenever the away
    player has possession.  Prefer an explicit possession team when the engine
    supplies it; otherwise the holder is the matching player nearest the ball.
    """
    possession_id = _possession_idx(ball)
    if possession_id is None:
        return None

    candidates = [p for p in players if _player_idx(p) == possession_id]
    if not candidates:
        return None

    possession_code = (
        ball.get("possessionTeamCode")
        or ball.get("teamCode")
        or ball.get("possessionTeam")
    )
    possession_team_id = ball.get("possessionTeamId")
    if possession_code is not None:
        expected = str(possession_code).lower()
        explicit = [p for p in candidates if str(p.get("teamCode", "")).lower() == expected]
        if explicit:
            return explicit[0]
    if possession_team_id is not None:
        explicit = [p for p in candidates if p.get("teamId") == possession_team_id]
        if explicit:
            return explicit[0]

    ball_pos = ball.get("position", {})
    return min(candidates, key=lambda p: dist(p.get("position", {}), ball_pos))


def get_goal_positions(team_id: int) -> tuple[float, float]:
    """Return (my_goal_x, opp_goal_x) based on team."""
    if team_id == 0:
        return -55.0, 55.0
    return 55.0, -55.0


def get_possession_info(ball: dict, players: list, team_id: int) -> tuple:
    """Return (possession_id, ball_status_str, we_have_ball)."""
    possession_id = _possession_idx(ball)
    if possession_id is not None:
        holder = find_possession_holder(ball, players)
        if holder:
            is_mine = _is_my_team(holder, team_id)
            side = "MY" if is_mine else "OPP"
            return possession_id, f"{side} player {possession_id}", is_mine
        return possession_id, "unknown", False
    return None, "free", False


def dist(pos1: dict, pos2: dict) -> float:
    """Euclidean distance between two position dicts with x,y keys."""
    return math.sqrt(
        (pos1.get("x", 0) - pos2.get("x", 0)) ** 2
        + (pos1.get("y", 0) - pos2.get("y", 0)) ** 2
    )


def _velocity(entity: dict) -> tuple[float, float]:
    velocity = entity.get("velocity", {})
    return float(velocity.get("x", 0) or 0), float(velocity.get("y", 0) or 0)


def _stamina_percent(value) -> int:
    """Normalise both 0..1 and 0..100 stamina formats to a useful percent."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 100
    return round(number * 100 if number <= 1.0 else number)


def summarize_state(
    game_state: dict,
    team_id: int,
    my_player_id: int,
    position_label: str,
) -> str:
    """Build a concise text summary of the game state for a single-player agent."""
    ball = game_state.get("ball", {})
    ball_pos = ball.get("position", {"x": 0, "y": 0})
    score = game_state.get("score", {})
    game_time = game_state.get("gameTime", 0)
    play_mode = game_state.get("playMode", 0)
    players = game_state.get("players", [])

    my_team = sorted(
        [p for p in players if _is_my_team(p, team_id)],
        key=lambda p: _player_idx(p),
    )
    opponents = sorted(
        [p for p in players if not _is_my_team(p, team_id)],
        key=lambda p: _player_idx(p),
    )

    me = next((p for p in my_team if _player_idx(p) == my_player_id), None)
    _, ball_status, _ = get_possession_info(ball, players, team_id)
    possession_holder = find_possession_holder(ball, players)

    my_goal_x, opp_goal_x = get_goal_positions(team_id)

    ball_vx, ball_vy = _velocity(ball)
    attack_direction = "+x" if team_id == 0 else "-x"
    lines = [
        f"Time: {game_time:.0f}s | Score: {score.get('home',0)}-{score.get('away',0)} | "
        f"Team: {team_id} ({'HOME' if team_id == 0 else 'AWAY'}) | PlayMode: {play_mode}",
        f"Attack direction: {attack_direction} | Your goal x={my_goal_x:.0f} | Opponent goal x={opp_goal_x:.0f}",
        f"Ball: ({ball_pos.get('x',0):.1f},{ball_pos.get('y',0):.1f}) "
        f"velocity=({ball_vx:.1f},{ball_vy:.1f}) held by {ball_status}",
        f"Ball projected in 2s: ({ball_pos.get('x',0) + 2 * ball_vx:.1f},"
        f"{ball_pos.get('y',0) + 2 * ball_vy:.1f})",
        "",
    ]

    # My player info
    if me:
        pos = me.get("position", {})
        stam = _stamina_percent(me.get("stamina", 100))
        dist_ball = dist(pos, ball_pos)
        has_ball = bool(
            possession_holder
            and _is_my_team(possession_holder, team_id)
            and _player_idx(possession_holder) == my_player_id
        )
        vx, vy = _velocity(me)
        nearest_pressure = min(
            (dist(pos, p.get("position", {})) for p in opponents),
            default=99.0,
        )
        extra = f" distOppGoal={abs(pos.get('x', 0) - opp_goal_x):.1f}" if position_label in ("MID", "FWD1", "FWD2") else ""
        lines.append(
            f">>> YOUR PLAYER ({position_label}, id={my_player_id}): "
            f"pos=({pos.get('x',0):.1f},{pos.get('y',0):.1f}) "
            f"velocity=({vx:.1f},{vy:.1f}) stamina={stam}% "
            f"distBall={dist_ball:.1f} nearestPressure={nearest_pressure:.1f}{extra} hasBall={has_ball}"
        )
    lines.append("")

    # Teammates
    lines.append("Teammates:")
    for p in my_team:
        if _player_idx(p) == my_player_id:
            continue
        pos = p.get("position", {})
        pid = _player_idx(p)
        role = "GK" if pid == 0 else f"P{pid}"
        vx, vy = _velocity(p)
        extra = ""
        if position_label == "MID":
            extra = f" distOppGoal={abs(pos.get('x', 0) - opp_goal_x):.1f}"
        lines.append(
            f"  {role}(id={pid}): ({pos.get('x',0):.1f},{pos.get('y',0):.1f}) "
            f"v=({vx:.1f},{vy:.1f}){extra}"
        )

    lines.append("")

    # Opponents
    opp_header = "Opponents (defenders to watch):" if position_label in ("FWD1", "FWD2") else "Opponents:"
    lines.append(opp_header)
    for p in opponents:
        pos = p.get("position", {})
        pid = _player_idx(p)
        vx, vy = _velocity(p)
        d_goal = abs(pos.get("x", 0) - my_goal_x)
        d_me = dist(pos, me.get("position", {})) if me else 0
        lines.append(
            f"  P{pid}: ({pos.get('x',0):.1f},{pos.get('y',0):.1f}) "
            f"v=({vx:.1f},{vy:.1f}) distToMyGoal={d_goal:.1f} distToMe={d_me:.1f}"
        )

    return "\n".join(lines)
