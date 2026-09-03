"""RM_Agents tools — open-position finder and shot-opportunity planner.

Both are plain functions the instinct layer calls every tick (0 ms) AND Strands `@tool`s so an
agent can be given them. They are deliberately NOT registered on the match agents: one tool call
costs two model turns (~1.1 s on Nova Lite) against a 500 ms budget. Instead the handler runs
them before asking the model and puts their results in the briefing ("OPEN POSITIONS", "SHOT"),
so the model chooses among computed options in a single turn.
"""

from __future__ import annotations

import json

try:
    from strands import tool
except Exception:  # local tests without strands
    def tool(fn=None, **_):
        return fn if fn else (lambda f: f)

from state import dist, _player_idx, _is_my_team, get_goal_positions, find_holder

CANDIDATE_DEPTHS = (9.0, 14.0, 20.0, 27.0, 34.0)
CANDIDATE_YS = (-18.0, -10.0, -4.0, 0.0, 4.0, 10.0, 18.0)
LANE_BLOCK_RADIUS = 2.5


def _pos(p):
    return p.get("position") or {"x": 0, "y": 0}


def _lane_clear(a: dict, b: dict, opps: list, radius: float = LANE_BLOCK_RADIUS) -> bool:
    ax, ay, bx, by = a.get("x", 0), a.get("y", 0), b.get("x", 0), b.get("y", 0)
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy or 1.0
    for o in opps:
        ox, oy = _pos(o).get("x", 0), _pos(o).get("y", 0)
        u = ((ox - ax) * dx + (oy - ay) * dy) / l2
        if 0.05 < u < 0.95:
            px, py = ax + u * dx, ay + u * dy
            if ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5 <= radius:
                return False
    return True


def open_positions(players: list, team_id: int, my_pid: int, ball_pos: dict, opp_goal_x: float,
                   side_y: float = 0.0, max_depth: float = 34.0, top: int = 3) -> list[dict]:
    """Rank candidate positions for an off-ball attacker.

    Score = space from opponents + progress toward goal + clear passing lane from the carrier
    + shot-gate bonus − crowding a teammate − running too far. side_y biases a wing (FWD1 -1, FWD2 +1).
    """
    mine = [p for p in players if _is_my_team(p, team_id)]
    opps = [p for p in players if not _is_my_team(p, team_id)]
    me = next((p for p in mine if _player_idx(p) == my_pid), None)
    my_pos = _pos(me) if me else ball_pos
    d = 1.0 if opp_goal_x > 0 else -1.0
    out = []
    for depth in CANDIDATE_DEPTHS:
        if depth > max_depth:
            continue
        x = opp_goal_x - d * depth
        # never behind the ball by more than 6 (we are attacking)
        if (x - ball_pos.get("x", 0)) * d < -6:
            continue
        for y in CANDIDATE_YS:
            cand = {"x": round(x, 1), "y": y}
            space = min((dist(_pos(o), cand) for o in opps), default=20.0)
            score = min(space, 12.0)
            score += max(0.0, 45.0 - depth) * 0.25
            score += 6.0 if _lane_clear(ball_pos, cand, opps) else -6.0
            # shootable: central and inside the gate beats open space in the corner
            score -= 0.35 * abs(y)
            if depth <= 25 and abs(y) <= 12:
                score += 6.0
            if depth < 15 and abs(y) > 8:
                score -= 6.0
            if any(dist(_pos(p), cand) < 6 and _player_idx(p) != my_pid for p in mine):
                score -= 4.0
            run = dist(my_pos, cand)
            if run > 30:
                score -= (run - 30) * 0.3
            if side_y and (y * side_y) < 0:
                score -= 3.0
            if abs(dist(cand, ball_pos)) < 5:
                score -= 5.0
            out.append({"x": cand["x"], "y": y, "score": round(score, 1), "space": round(space, 1)})
    out.sort(key=lambda c: -c["score"])
    return out[:top]


BANDS = ((0, 12), (12, 25), (25, 40), (40, 999))
CORNER_RANGE = 15.0
CORNER_KEEPER_OFFSET = 1.5


def shot_band(d: float) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= d < hi:
            return i
    return len(BANDS) - 1


GOAL_INSIDE = 4.5
AIMS = ("TL", "TR", "BL", "BR", "CENTER")


def pick_aim_from_landing(aim_map: dict, keeper_y: float, default: str, min_n: int = 2):
    """Corner whose KNOWN landing y is inside the posts and farthest from the keeper; None if nothing is known."""
    known = {a: v for a, v in (aim_map or {}).items() if v.get("n", 0) >= min_n}
    if not known:
        return None
    inside = {a: v["y"] for a, v in known.items() if abs(v["y"]) <= GOAL_INSIDE}
    if inside:
        return max(inside, key=lambda a: abs(inside[a] - keeper_y))
    if len(known) >= 4 and all(abs(v["y"]) > GOAL_INSIDE for v in known.values()) and "CENTER" not in known:
        return "CENTER"           # every corner is known to miss wide — the only untested aim
    return None


def shot_opportunity(players: list, team_id: int, pos: dict, opp_goal_x: float, flip: bool = False,
                     prefer_low: bool = False, aim_map: dict | None = None) -> dict:
    """Aim + power for a shot from here. Power drops close in (accuracy), climbs with distance."""
    from fallback import shot_plan  # late import: fallback imports tools
    plan = shot_plan(pos, players, team_id, opp_goal_x, flip)
    d = abs(pos.get("x", 0) - opp_goal_x)
    keeper_gap = abs(plan["target"][1] - plan["keeper_y"])
    gap = (not plan["blocked"]) and keeper_gap >= 2.0
    band = shot_band(d)
    power = (0.8, 0.9, 1.0, 1.0)[band]
    # The goal is at (opp_goal_x, 0) and the corner labels have no published geometry (AGENT_PROTOCOL §4):
    # CENTER is the only aim that is on target by construction. Corners only from close range when the
    # keeper has clearly committed to one side; a learned landing map (below) can override either way.
    keeper_y = plan["keeper_y"]
    if d <= CORNER_RANGE and abs(keeper_y) >= CORNER_KEEPER_OFFSET:
        aim = plan["aim_location"]
        if prefer_low:
            aim = "B" + aim[1]
    else:
        aim = "CENTER"
    learned = pick_aim_from_landing(aim_map, plan["keeper_y"], aim)
    if learned:
        aim = learned
    return {"gap": gap, "learned_aim": bool(learned), "aim_location": aim, "power": power, "dist": round(d, 1), "band": band,
            "keeper_gap": round(keeper_gap, 1), "blocked": plan["blocked"], "keeper_y": round(plan["keeper_y"], 1),
            "side": -plan["side"] if flip else plan["side"]}


def shot_line(game_state: dict, team_id: int, my_pid: int, flip: bool = False, prefer_low: bool = False,
              banned_bands=(), aim_map: dict | None = None) -> str:
    """SHOT TOOL briefing line for the carrier."""
    players = game_state.get("players", [])
    me = next((p for p in players if _is_my_team(p, team_id) and _player_idx(p) == my_pid), None)
    if me is None:
        return ""
    _, opp_goal_x = get_goal_positions(team_id)
    sh = shot_opportunity(players, team_id, _pos(me), opp_goal_x, flip, prefer_low, aim_map)
    verdict = "SHOOT NOW" if sh["band"] not in banned_bands else "this distance has not produced shots on target — carry closer first"
    return (f"\nSHOT TOOL (plan_shot): dist={sh['dist']} keeper_y={sh['keeper_y']} open_side={'+y' if sh['side'] > 0 else '-y'} "
            f"aim={sh['aim_location']}{' (learned: lands inside the posts)' if sh['learned_aim'] else ''} power={sh['power']} "
            f"lane={'clear' if not sh['blocked'] else 'BLOCKED'} → {verdict}")


def briefing_lines(game_state: dict, team_id: int, my_pid: int, side_y: float = 0.0) -> str:
    """Tool results for the LLM briefing: open positions (possession) or nothing (defence)."""
    players = game_state.get("players", [])
    ball = game_state.get("ball", {})
    holder = find_holder(ball, players)
    if holder is None or not _is_my_team(holder, team_id) or _player_idx(holder) == my_pid:
        return ""
    _, opp_goal_x = get_goal_positions(team_id)
    cands = open_positions(players, team_id, my_pid, ball.get("position") or {}, opp_goal_x, side_y)
    if not cands:
        return ""
    opts = "  ".join(f"{i+1}) x={c['x']} y={c['y']} score={c['score']} space={c['space']}" for i, c in enumerate(cands))
    return f"\nOPEN POSITIONS (tool find_open_position, best first): {opts}"


# --- Strands tools (available to any agent that wants to call them explicitly) ---

@tool
def find_open_position(game_state_json: str, team_id: int, player_id: int, side_y: float = 0.0) -> str:
    """Rank the best open positions for an off-ball attacker from the raw game state JSON.
    Returns a JSON list of {x, y, score, space}, best first."""
    gs = json.loads(game_state_json)
    _, opp_goal_x = get_goal_positions(team_id)
    return json.dumps(open_positions(gs.get("players", []), team_id, player_id,
                                     (gs.get("ball") or {}).get("position") or {}, opp_goal_x, side_y))


@tool
def plan_shot(game_state_json: str, team_id: int, player_id: int) -> str:
    """Shot plan for the player: whether there is a gap, the corner to aim at (TL/TR/BL/BR) and the power.
    Returns JSON {gap, aim_location, power, dist, keeper_gap, blocked}."""
    gs = json.loads(game_state_json)
    players = gs.get("players", [])
    me = next((p for p in players if _is_my_team(p, team_id) and _player_idx(p) == player_id), None)
    _, opp_goal_x = get_goal_positions(team_id)
    return json.dumps(shot_opportunity(players, team_id, _pos(me) if me else {}, opp_goal_x))
