"""Response parsing utilities for AI soccer agents."""

import re

from json_tolerant import parse_json_tolerant

VALID_COMMANDS = {
    "MOVE_TO", "PASS", "SHOOT", "SLIDE_TACKLE", "PRESS_BALL", "INTERCEPT", "MARK",
    "FOLLOW_PLAYER", "GK_DISTRIBUTE", "SET_STANCE", "CLEAR_OVERRIDE", "RESET",
}


def parse_commands(text: str, team_id: int, my_player_id: int, on_recovered=None) -> list[dict]:
    """Extract commands from LLM response, forcing the given player ID on all commands.

    A model that writes Python-flavoured JSON (``True`` instead of ``true``, a trailing
    comma, a markdown fence) still has its commands used — see lib/json_tolerant.py for
    what is recovered and what is deliberately left alone. Pass ``on_recovered(raw)`` if
    you want to log how often that happens; the commands themselves are used exactly as
    if the model had emitted valid JSON.
    """
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        commands = _loads(match.group(), on_recovered)
        if isinstance(commands, list):
            return _tag_commands(commands, team_id, my_player_id)

    parsed = _loads(text, on_recovered)
    if isinstance(parsed, list):
        return _tag_commands(parsed, team_id, my_player_id)
    if isinstance(parsed, dict) and "commandType" in parsed:
        parsed["teamId"] = team_id
        parsed["playerId"] = my_player_id
        return [parsed]

    return []


def _loads(candidate: str, on_recovered):
    """Parse one candidate payload; None when it does not parse even after normalisation."""
    result = parse_json_tolerant(candidate)
    if result is None:
        return None
    value, recovered = result
    if recovered and on_recovered is not None:
        on_recovered(candidate)
    return value


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _tag_commands(commands: list, team_id: int, my_player_id: int) -> list[dict]:
    """Add teamId and playerId to each command, filtering to valid ones."""
    result = []
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        cmd["teamId"] = team_id
        cmd["playerId"] = my_player_id
        if "commandType" in cmd:
            # Skip unrecognized command types
            if cmd["commandType"] not in VALID_COMMANDS:
                continue
            params = cmd.get("parameters", {})
            if not isinstance(params, dict):
                continue
            cmd["parameters"] = params

            # Clamp values the match engine otherwise accepts unchecked.
            if cmd["commandType"] == "MOVE_TO":
                if not isinstance(params.get("target_x"), (int, float)) or not isinstance(
                    params.get("target_y"), (int, float)
                ):
                    continue
                if isinstance(params.get("target_x"), (int, float)):
                    params["target_x"] = _clamp(params["target_x"], -55, 55)
                if isinstance(params.get("target_y"), (int, float)):
                    params["target_y"] = _clamp(params["target_y"], -35, 35)
                params["sprint"] = bool(params.get("sprint", False))
            elif cmd["commandType"] == "SHOOT":
                if params.get("aim_location") not in {"TL", "TR", "BL", "BR", "CENTER"}:
                    params["aim_location"] = "CENTER"
                power = params.get("power", 0.8)
                params["power"] = _clamp(power, 0.0, 1.0) if isinstance(power, (int, float)) else 0.8
            elif cmd["commandType"] == "PRESS_BALL":
                intensity = params.get("intensity", 0.6)
                params["intensity"] = (
                    _clamp(intensity, 0.0, 1.0)
                    if isinstance(intensity, (int, float))
                    else 0.6
                )
            elif cmd["commandType"] == "SET_STANCE":
                stance = params.get("stance", 0)
                params["stance"] = int(_clamp(stance, 0, 2)) if isinstance(stance, (int, float)) else 0
            # Ensure PASS/MARK/FOLLOW_PLAYER/GK_DISTRIBUTE/SLIDE_TACKLE have target_player_id
            if cmd["commandType"] in ("PASS", "MARK", "FOLLOW_PLAYER", "GK_DISTRIBUTE", "SLIDE_TACKLE"):
                if params.get("target_player_id") is None:
                    # Default: pass to a forward, mark nearest opponent, tackle ball carrier
                    if cmd["commandType"] in ("PASS", "GK_DISTRIBUTE"):
                        params["target_player_id"] = 3 if my_player_id != 3 else 4
                    elif cmd["commandType"] == "SLIDE_TACKLE":
                        params["target_player_id"] = -1  # target ball carrier
                    else:
                        params["target_player_id"] = 0  # mark/follow opponent 0
                    cmd["parameters"] = params
                target = params.get("target_player_id")
                if not isinstance(target, int) or not 0 <= target <= 4:
                    continue
                if cmd["commandType"] == "PASS":
                    if target == my_player_id:
                        continue
                    if params.get("type") not in {"GROUND", "AERIAL", "THROUGH"}:
                        params["type"] = "GROUND"
                elif cmd["commandType"] == "GK_DISTRIBUTE":
                    if params.get("method") not in {"THROW", "KICK"}:
                        params["method"] = "THROW"

            duration = cmd.get("duration", 0)
            cmd["duration"] = int(_clamp(duration, 0, 5)) if isinstance(duration, (int, float)) else 0
            result.append(cmd)
    return result
