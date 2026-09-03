"""Offline tests for local tactics, selective calls, and match LTM.

Run from ``agentic-football-sample-agents``:
    python3 lib/test_learned_team.py

No AWS credentials, model invocation, AgentCore package, or network is needed.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from fallback import FWD1_CONFIG, build_fallback
from learned_handler import create_learned_invoke_handler
from match_memory import AgentCoreMatchMemory, MatchTracker
from state import find_possession_holder, summarize_state, _is_my_team
from tactics import decide_locally, memory_adjustments
from test_helpers import GAME_STATE


def test_all_roles_return_one_local_command():
    for pid, label in enumerate(("GK", "DEF", "MID", "FWD1", "FWD2")):
        decision = decide_locally(copy.deepcopy(GAME_STATE), 0, pid, label)
        assert decision is not None
        assert len(decision.commands) == 1
        command = decision.commands[0]
        assert command["playerId"] == pid
        assert command["teamId"] == 0
        assert 0.0 <= decision.confidence <= 1.0


def _shooting_state(
    player_id: int,
    stamina: float = 0.8,
    shooter_x: float = 30.0,
    keeper_y: float = -9.0,
) -> dict:
    state = copy.deepcopy(GAME_STATE)
    holder = next(
        p for p in state["players"]
        if _is_my_team(p, 0) and p["agentId"] == f"agentId_{player_id}"
    )
    holder["position"] = {"x": shooter_x, "y": 0.0}
    holder["velocity"] = {"x": 0.0, "y": 0.0}
    holder["stamina"] = stamina
    state["ball"].update({
        "position": {"x": shooter_x, "y": 0.0, "z": 0.0},
        "possessionAgentId": f"agentId_{player_id}",
        "possessionTeamCode": "home",
        "isFree": False,
    })
    for opponent in (p for p in state["players"] if not _is_my_team(p, 0)):
        opponent["position"] = {
            "x": 50.0 if opponent["agentId"] == "agentId_0" else 20.0,
            "y": keeper_y if opponent["agentId"] == "agentId_0" else 28.0,
        }
        opponent["velocity"] = {"x": 0.0, "y": 0.0}
    return state


def test_every_player_shoots_an_open_goal_lane_in_range():
    for pid, label in enumerate(("GK", "DEF", "MID", "FWD1", "FWD2")):
        decision = decide_locally(_shooting_state(pid), 0, pid, label)
        assert decision.commands[0]["commandType"] == "SHOOT", (label, decision)
        assert decision.confidence >= 0.95


def test_every_player_takes_a_clearly_open_long_shot():
    for pid, label in enumerate(("GK", "DEF", "MID", "FWD1", "FWD2")):
        state = _shooting_state(pid, shooter_x=13.0, keeper_y=-13.0)
        decision = decide_locally(state, 0, pid, label)
        assert decision.commands[0]["commandType"] == "SHOOT", (label, decision)
        assert "long-range" in decision.reason


def test_low_stamina_disables_routine_forward_sprint():
    state = copy.deepcopy(GAME_STATE)
    midfielder = next(
        p for p in state["players"] if _is_my_team(p, 0) and p["agentId"] == "agentId_2"
    )
    forward = next(
        p for p in state["players"] if _is_my_team(p, 0) and p["agentId"] == "agentId_4"
    )
    midfielder["position"] = {"x": 10.0, "y": 0.0}
    forward["position"] = {"x": 0.0, "y": 12.0}
    forward["stamina"] = 0.30
    state["ball"].update({
        "position": {"x": 10.0, "y": 0.0, "z": 0.0},
        "possessionAgentId": "agentId_2",
        "possessionTeamCode": "home",
        "isFree": False,
    })

    tired = decide_locally(state, 0, 4, "FWD2")
    assert tired.commands[0]["commandType"] == "MOVE_TO"
    assert tired.commands[0]["parameters"]["sprint"] is False

    forward["stamina"] = 0.90
    fresh = decide_locally(state, 0, 4, "FWD2")
    assert fresh.commands[0]["parameters"]["sprint"] is True


def test_duplicate_player_numbers_find_correct_possession_team():
    state = copy.deepcopy(GAME_STATE)
    away_forward = next(
        p for p in state["players"]
        if not _is_my_team(p, 0) and p["agentId"] == "agentId_3"
    )
    state["ball"]["position"] = copy.deepcopy(away_forward["position"])
    state["ball"]["possessionAgentId"] = "agentId_3"
    holder = find_possession_holder(state["ball"], state["players"])
    assert holder is away_forward


def test_summary_exposes_motion_and_pressure():
    summary = summarize_state(GAME_STATE, 0, 2, "MID")
    assert "Ball projected in 2s" in summary
    assert "velocity=" in summary
    assert "nearestPressure=" in summary
    assert "Attack direction: +x" in summary


def test_retrieved_lessons_adjust_the_local_policy():
    adjustments = memory_adjustments(
        "We lost possession under pressure; release the ball earlier. "
        "Keep the defender and midfielder staggered during attacks."
    )
    assert adjustments["release_earlier"]
    assert adjustments["protect_transition"]
    assert not adjustments["attack_higher"]
    assert not adjustments["conserve_stamina"]

    stamina_adjustment = memory_adjustments(
        "We finished with low energy; reduce excessive sprinting to preserve late-match stamina."
    )
    assert stamina_adjustment["conserve_stamina"]


def test_away_targets_are_mirrored():
    home = copy.deepcopy(GAME_STATE)
    home_decision = decide_locally(home, 0, 0, "GK")

    away = copy.deepcopy(GAME_STATE)
    # The away goalkeeper is already represented in the shared snapshot.
    away["ball"]["position"] = {"x": -15.3, "y": -5.2, "z": 0}
    away["ball"]["possessionAgentId"] = None
    away["ball"]["isFree"] = True
    away_decision = decide_locally(away, 1, 0, "GK")

    assert home_decision.commands[0]["parameters"]["target_x"] < 0
    assert away_decision.commands[0]["parameters"]["target_x"] > 0


def test_tracker_finishes_practice_match_and_starts_fresh_match():
    tracker = MatchTracker(player_id=2)
    first = copy.deepcopy(GAME_STATE)
    first["tick"] = 1
    first["gameTime"] = 1
    transition = tracker.observe({"isPractice": True}, first, 0)
    assert transition.started and transition.completed_summary is None
    tracker.record_command({"commandType": "PASS"})

    final = copy.deepcopy(first)
    final["tick"] = 100
    final["gameTime"] = 90
    final["playMode"] = "FULL_TIME"
    final["score"] = {"home": 2, "away": 0}
    transition = tracker.observe({"isPractice": True}, final, 0)
    episode = json.loads(transition.completed_summary)
    assert episode["match_type"] == "practice"
    assert episode["outcome"] == "WIN"
    assert episode["command_mix"]["PASS"] == 1

    next_match = copy.deepcopy(first)
    next_match["tick"] = 1
    next_match["gameTime"] = 1
    transition = tracker.observe({}, next_match, 0)
    assert transition.started
    assert transition.match_key != episode["match_id"]


def test_match_summary_records_team_stamina_and_learns_conservation():
    tracker = MatchTracker(player_id=2)
    state = copy.deepcopy(GAME_STATE)
    state["tick"] = 1
    state["gameTime"] = 1
    for player in (p for p in state["players"] if _is_my_team(p, 0)):
        player["stamina"] = 0.35
        player["isSprinting"] = True
    tracker.observe({"isPractice": True}, state, 0)

    state["tick"] = 90
    state["gameTime"] = 90
    state["playMode"] = "FULL_TIME"
    transition = tracker.observe({"isPractice": True}, state, 0)
    episode = json.loads(transition.completed_summary)
    assert episode["final_team_stamina_percent"] == 35
    assert episode["sprinting_player_tick_percent"] == 100
    assert any("late-match stamina" in lesson for lesson in episode["lessons"])


class _FakeMemoryClient:
    def __init__(self):
        self.retrieve_args = None
        self.event_args = None

    def retrieve_memory_records(self, **kwargs):
        self.retrieve_args = kwargs
        return {
            "memoryRecordSummaries": [
                {"content": {"text": "Earlier pressing trap worked on the right."}}
            ]
        }

    def create_event(self, **kwargs):
        self.event_args = kwargs
        return {"event": {"eventId": "event-1"}}


def test_agentcore_adapter_reads_and_writes_once_per_operation():
    fake = _FakeMemoryClient()
    memory = AgentCoreMatchMemory("memory-123", "test-team")
    memory._runtime_client = fake

    lessons = memory.retrieve(team_id=0)
    assert "pressing trap" in lessons
    assert fake.retrieve_args["namespace"] == "/episodes/football-test-team"

    assert memory.store_episode(0, "practice-7", '{"outcome":"WIN"}')
    assert fake.event_args["sessionId"].startswith("football-match-")
    assert len(fake.event_args["payload"]) == 2


class _Logger:
    def info(self, message):
        pass

    def warn(self, message):
        pass

    def error(self, message):
        pass


class _App:
    logger = _Logger()

    def entrypoint(self, fn):
        return fn


class _Agent:
    def __init__(self, delay=0):
        self.calls = 0
        self.messages = []
        self.delay = delay

    async def invoke_async(self, prompt):
        self.calls += 1
        self.messages.append(prompt)
        await asyncio.sleep(self.delay)
        return '[{"commandType":"SHOOT","parameters":{"aim_location":"TR","power":0.8},"duration":0}]'


async def _invoke_handler(mode: str, delay: float = 0, ticks: int = 1):
    old_mode = os.environ.get("FOOTBALL_LLM_MODE")
    old_memory = os.environ.pop("MEMORY_TEAM_MEMORY_ID", None)
    old_legacy_memory = os.environ.pop("MEMORY_ID", None)
    os.environ["FOOTBALL_LLM_MODE"] = mode
    try:
        agent = _Agent(delay)
        handler = create_learned_invoke_handler(
            _App(),
            agent,
            3,
            "FWD1",
            build_fallback(FWD1_CONFIG),
            FWD1_CONFIG,
        )
        outputs = []
        latencies = []
        for offset in range(ticks):
            state = copy.deepcopy(GAME_STATE)
            state["tick"] += offset
            payload = {
                "prompt": json.dumps({"gameState": state, "teamId": 0, "myPlayers": [3]})
            }
            started = time.perf_counter()
            async for output in handler(payload, None):
                outputs.append(json.loads(output))
            latencies.append(time.perf_counter() - started)
            # Let a background task run without ever awaiting it in the handler.
            await asyncio.sleep(delay + 0.01 if offset + 1 < ticks else 0)
        return agent, outputs, latencies
    finally:
        if old_mode is None:
            os.environ.pop("FOOTBALL_LLM_MODE", None)
        else:
            os.environ["FOOTBALL_LLM_MODE"] = old_mode
        if old_memory is not None:
            os.environ["MEMORY_TEAM_MEMORY_ID"] = old_memory
        if old_legacy_memory is not None:
            os.environ["MEMORY_ID"] = old_legacy_memory


def test_llm_off_makes_zero_model_calls():
    agent, outputs, _ = asyncio.run(_invoke_handler("off"))
    assert agent.calls == 0
    assert len(outputs) == 1 and len(outputs[0]) == 1


def test_slow_llm_never_blocks_current_tick():
    agent, outputs, latencies = asyncio.run(_invoke_handler("always", delay=1.5))
    assert agent.calls == 1
    assert len(outputs) == 1
    assert latencies[0] < 0.1
    assert agent.messages == []


def test_fresh_background_result_can_be_used_on_next_tick():
    agent, outputs, latencies = asyncio.run(_invoke_handler("always", ticks=2))
    assert agent.calls == 1
    assert outputs[1][0]["commandType"] == "SHOOT"
    assert max(latencies) < 0.1
    assert agent.messages == []


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} learned-team tests passed (offline)")
