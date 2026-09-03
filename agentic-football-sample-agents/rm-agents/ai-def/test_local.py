"""Local test for one RM_Agents position — instinct layer, fallback, parsing, and (with --llm) the real model.

    python3 test_local.py          # no AWS needed
    python3 test_local.py --llm    # calls the model: prints latency against the 500 ms budget
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from test_helpers import mock_agentcore, GAME_STATE, TEAM_ID
mock_agentcore()

from state import summarize_state
from parsing import parse_commands
from fallback import instinct_command
from main import fallback_commands, FALLBACK_CONFIG, MY_PLAYER_ID, POSITION_LABEL, SYSTEM_PROMPT


def _state(**changes):
    s = json.loads(json.dumps(GAME_STATE))
    s.update(changes)
    return s


def _show(cmds):
    for c in cmds:
        print(f"  P{c['playerId']} T{c['teamId']}: {c['commandType']} {c.get('parameters', {})} d={c.get('duration', 0)}")


def test_summarize():
    print(f"=== BRIEFING ({POSITION_LABEL}, player {MY_PLAYER_ID}) ===")
    print(summarize_state(GAME_STATE, TEAM_ID, MY_PLAYER_ID, POSITION_LABEL))
    print()


def test_instinct_we_have_ball():
    """Teammate 3 has the ball in the sample state → every position gets a code-decided command."""
    print(f"=== INSTINCT: teammate has ball ({POSITION_LABEL}) ===")
    cmds = instinct_command(FALLBACK_CONFIG, _state(), TEAM_ID, MY_PLAYER_ID)
    assert cmds, "instinct should decide the possession phase"
    _show(cmds)
    assert all(c["playerId"] == MY_PLAYER_ID and c["teamId"] == TEAM_ID for c in cmds)
    print()


def test_instinct_on_ball_in_range():
    print(f"=== INSTINCT: on ball, in range ({POSITION_LABEL}) ===")
    s = _state()
    s["ball"]["possessionAgentId"] = f"agentId_{MY_PLAYER_ID}"
    s["players"][MY_PLAYER_ID]["position"] = {"x": 40, "y": -3}
    s["ball"]["position"] = {"x": 40, "y": -3, "z": 0}
    cmds = instinct_command(FALLBACK_CONFIG, s, TEAM_ID, MY_PLAYER_ID)
    _show(cmds)
    expected = "GK_DISTRIBUTE" if POSITION_LABEL == "GK" else "SHOOT"
    assert cmds[0]["commandType"] == expected, cmds
    print()


def test_defensive_phase():
    print(f"=== DEFENSIVE PHASE: opponent has ball ({POSITION_LABEL}) ===")
    s = _state()
    s["ball"]["possessionAgentId"] = "agentId_2"
    s["players"][7]["position"] = {"x": 0, "y": 5}   # away P2 carries at midfield
    s["ball"]["position"] = {"x": 0, "y": 5, "z": 0}
    inst = instinct_command(FALLBACK_CONFIG, s, TEAM_ID, MY_PLAYER_ID)
    if POSITION_LABEL == "GK":
        assert inst, "GK is always code-decided"
        print("  GK instinct:"); _show(inst)
    else:
        assert inst is None, "outfield defensive phase must go to the LLM"
        print("  → LLM would be asked. Rule-based fallback if it is slow:")
        _show(fallback_commands(s, TEAM_ID, MY_PLAYER_ID))
    print()


def test_parse():
    print("=== PARSE ===")
    cases = [
        ('[{"commandType":"PRESS_BALL","playerId":9,"parameters":{"intensity":0.8},"duration":3}]', 1),
        ('[{"commandType":"MARK","target_player_id":3,"tightness":"TIGHT"}]', 1),
        ("no json here", 0),
    ]
    for text, n in cases:
        cmds = parse_commands(text, TEAM_ID, MY_PLAYER_ID)
        ok = len(cmds) == n and all(c["playerId"] == MY_PLAYER_ID for c in cmds)
        print(f"  [{'PASS' if ok else 'FAIL'}] {text[:50]}... -> {len(cmds)}")
        assert ok
    print()


def test_llm():
    print(f"=== LLM ({POSITION_LABEL}) — defensive-phase briefing, 3 calls ===")
    from main import agent, MODEL_ID, invoke
    s = _state()
    s["ball"]["possessionAgentId"] = "agentId_2"
    s["players"][7]["position"] = {"x": 0, "y": 5}
    s["ball"]["position"] = {"x": 0, "y": 5, "z": 0}
    summary = summarize_state(s, TEAM_ID, MY_PLAYER_ID, POSITION_LABEL)
    print(f"{MODEL_ID}, prompt {len(SYSTEM_PROMPT)} chars, briefing {len(summary)} chars")
    for i in range(3):
        t0 = time.perf_counter()
        text = str(agent(summary))
        ms = (time.perf_counter() - t0) * 1000
        bare = text.strip().startswith("[") and text.strip().endswith("]")
        cmds = parse_commands(text, TEAM_ID, MY_PLAYER_ID)
        gate = "OK" if ms < 450 else ("WARN" if ms < 600 else "OVER BUDGET")
        print(f"  call {i+1}: {ms:.0f}ms [{gate}] bare_json={bare} -> "
              f"{[c['commandType'] for c in cmds] or text[:120]}")

    import asyncio
    payload = {"prompt": json.dumps({"gameState": s, "teamId": TEAM_ID, "myPlayers": [MY_PLAYER_ID]})}

    async def run():
        t0 = time.perf_counter()
        out = [json.loads(chunk) async for chunk in invoke(payload, None)]
        return out, (time.perf_counter() - t0) * 1000

    out, ms = asyncio.run(run())
    print(f"  handler end-to-end: {ms:.0f}ms -> {out[0][0]['commandType']} {out[0][0].get('parameters', {})}")


if __name__ == "__main__":
    test_summarize()
    test_instinct_we_have_ball()
    test_instinct_on_ball_in_range()
    test_defensive_phase()
    test_parse()
    if "--llm" in sys.argv:
        test_llm()
    else:
        print("Skipping LLM test. Run with --llm to call the model.")
