"""RM_Agents doctrine tests — shoot-first instinct layer, keeper-opposite aim, mirror drill,
LLM-only-in-defence, hard timeout, latency caps, prompts.

Run:  python3 test_rm_agents.py
"""

import asyncio
import copy
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

from test_helpers import mock_agentcore, GAME_STATE, TEAM_ID, _FakeApp  # noqa: E402
mock_agentcore()

from dataclasses import replace  # noqa: E402
from state import summarize_state  # noqa: E402
from fallback import (  # noqa: E402
    build_fallback, instinct_command, aim_corner,
    GK_CONFIG, DEF_CONFIG, MID_CONFIG, FWD1_CONFIG, FWD2_CONFIG,
)
from parsing import parse_commands  # noqa: E402
from fallback import shot_plan  # noqa: E402
from match_memory import MatchTracker, MemoryStore, priors_from_events, parse_event_payloads  # noqa: E402

AWAY = 1
HOME_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
AWAY_IDX = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9}


def _state(**changes):
    s = copy.deepcopy(GAME_STATE)
    s.update(changes)
    return s


def _on_ball(pid, x, y, opp_moves=None):
    """Player pid (HOME) has the ball at (x, y)."""
    s = _state()
    s["ball"]["possessionAgentId"] = f"agentId_{pid}"
    s["players"][HOME_IDX[pid]]["position"] = {"x": x, "y": y}
    s["ball"]["position"] = {"x": x, "y": y, "z": 0}
    for idx, pos in (opp_moves or {}).items():
        s["players"][AWAY_IDX[idx]]["position"] = pos
    return s


def _far_opps():
    """Push every away outfielder away from the HOME attackers (keeper stays)."""
    return {1: {"x": -30, "y": 25}, 2: {"x": -35, "y": -25}, 3: {"x": -40, "y": 20}, 4: {"x": -45, "y": -20}}


def _mirrored(state):
    s = copy.deepcopy(state)
    for p in s["players"]:
        p["teamCode"] = "away" if p["teamCode"] == "home" else "home"
        p["position"]["x"] = -p["position"]["x"]
        p["velocity"]["x"] = -p["velocity"]["x"]
    s["ball"]["position"]["x"] = -s["ball"]["position"]["x"]
    return s


# ---------------------------------------------------------------------------
# 1. Shooting — in range → SHOOT, corner away from the keeper
# ---------------------------------------------------------------------------

def test_close_range_shoots_away_from_a_committed_keeper_else_center():
    s = _on_ball(3, 45, -4, {0: {"x": 52, "y": 2}})            # 10 out, keeper on +y side
    cmds = instinct_command(FWD1_CONFIG, s, TEAM_ID, 3)
    assert cmds[0]["commandType"] == "SHOOT", cmds
    assert cmds[0]["parameters"]["aim_location"] == "BR", cmds  # -y side: Bottom, shooter's Right
    s = _on_ball(3, 45, -4, {0: {"x": 52, "y": -2}})           # keeper on -y side
    assert instinct_command(FWD1_CONFIG, s, TEAM_ID, 3)[0]["parameters"]["aim_location"] == "TL"
    s = _on_ball(3, 45, -4, {0: {"x": 52, "y": 0}})            # keeper central → CENTER is the only known-on-target aim
    assert instinct_command(FWD1_CONFIG, s, TEAM_ID, 3)[0]["parameters"]["aim_location"] == "CENTER"
    s = _on_ball(3, 30, -4, {0: {"x": 52, "y": 3}})            # 25 out → CENTER regardless of the keeper
    assert instinct_command(FWD1_CONFIG, s, TEAM_ID, 3)[0]["parameters"]["aim_location"] == "CENTER"


def test_keeper_central_means_far_post():
    s = _on_ball(4, 38, 6, {0: {"x": 52, "y": 0}})
    assert aim_corner({"x": 38, "y": 6}, s["players"], TEAM_ID, 55) == "BR"
    assert aim_corner({"x": 38, "y": -6}, s["players"], TEAM_ID, 55) == "TL"





def test_on_ball_means_shoot_from_anywhere_for_every_position():
    for cfg, pid in ((DEF_CONFIG, 1), (MID_CONFIG, 2), (FWD1_CONFIG, 3), (FWD2_CONFIG, 4)):
        for x, y in ((-30, 10), (0, -20), (25, 0), (48, 3)):
            cmds = instinct_command(cfg, _on_ball(pid, x, y), TEAM_ID, pid)
            assert cmds[0]["commandType"] == "SHOOT", (pid, x, y, cmds)
            d = abs(x - 55)
            assert cmds[0]["parameters"]["power"] == (0.8 if d < 12 else (0.9 if d < 25 else 1.0)), cmds
    away = instinct_command(FWD1_CONFIG, _mirrored(_on_ball(3, 0, -5)), AWAY, 3)
    assert away[0]["commandType"] == "SHOOT"


def test_every_outfield_position_shoots_in_range():
    for cfg, pid in ((DEF_CONFIG, 1), (MID_CONFIG, 2), (FWD1_CONFIG, 3), (FWD2_CONFIG, 4)):
        cmds = instinct_command(cfg, _on_ball(pid, 40, -3, _far_opps()), TEAM_ID, pid)
        assert cmds[0]["commandType"] == "SHOOT", (pid, cmds)





def test_every_position_shoots_from_range_with_a_gap():
    for cfg, pid, x in ((DEF_CONFIG, 1, 22), (MID_CONFIG, 2, 18), (FWD1_CONFIG, 3, 12), (FWD2_CONFIG, 4, 12)):
        cmds = instinct_command(cfg, _on_ball(pid, x, 0, _far_opps()), TEAM_ID, pid)
        assert cmds[0]["commandType"] == "SHOOT" and cmds[0]["parameters"]["power"] == 1.0, (pid, cmds)





# ---------------------------------------------------------------------------
# 1b. Shot accuracy — keeper velocity, blockers, far post, sidestep
# ---------------------------------------------------------------------------

def test_shot_leads_a_moving_keeper():
    s = _on_ball(3, 38, 0, {0: {"x": 52, "y": 0}})
    s["players"][AWAY_IDX[0]]["velocity"] = {"x": 0, "y": 4.0}       # keeper diving toward +y
    plan = shot_plan({"x": 38, "y": 0}, s["players"], TEAM_ID, 55)
    assert plan["side"] == -1 and plan["keeper_y"] > 1.0, plan


def test_shot_avoids_a_defender_on_the_line():
    # keeper slightly toward -y (so +y would normally win), but a defender blocks the +y line
    s = _on_ball(3, 35, 0, {0: {"x": 52, "y": -1.5}, 1: {"x": 45, "y": 1.8}})
    plan = shot_plan({"x": 35, "y": 0}, s["players"], TEAM_ID, 55)
    assert plan["side"] == -1 and not plan["blocked"], plan





def test_aim_flip_inverts_the_corner():
    s = _on_ball(3, 45, -4, {0: {"x": 52, "y": 2}})
    assert aim_corner({"x": 45, "y": -4}, s["players"], TEAM_ID, 55) == "BR"
    assert aim_corner({"x": 45, "y": -4}, s["players"], TEAM_ID, 55, flip=True) == "TL"
    cfg = replace(FWD1_CONFIG, aim_flip=True)
    assert instinct_command(cfg, s, TEAM_ID, 3)[0]["parameters"]["aim_location"] == "TL"


# ---------------------------------------------------------------------------
# 1c. Memory — STM dynamic doctrine, aim learning, LTM priors
# ---------------------------------------------------------------------------

class _FakeStore:
    enabled = True

    def __init__(self, events=None, lessons=None):
        self.events, self.lessons, self.recorded = events or [], lessons or [], []

    def record(self, event):
        self.recorded.append(event)

    def load_priors_async(self, on_ready):
        on_ready(priors_from_events(self.events), self.lessons)


def _tracker(store=None):
    return MatchTracker("FWD1", 3, store=store)


def _tick(tr, t, home, away, team=TEAM_ID, session="m1"):
    tr.observe({"gameTime": t, "score": {"home": home, "away": away}}, team, session)


def test_losing_widens_the_shot_gate_and_late_losing_widens_more():
    tr = _tracker(); _tick(tr, 100, 0, 1)
    cfg = tr.adjust(FWD1_CONFIG)
    assert cfg.shoot_threshold == 50.0 and cfg.pressure_shoot_distance == 55.0 and cfg.support_depth == 12.0, cfg
    _tick(tr, 250, 0, 1)
    assert tr.adjust(FWD1_CONFIG).shoot_threshold == 58.0


def test_two_up_tightens_and_drops_the_defender():
    tr = MatchTracker("DEF", 1); _tick(tr, 100, 2, 0)
    cfg = tr.adjust(DEF_CONFIG)
    assert cfg.shoot_threshold == 31.0 and cfg.support_x_factor == 0.6, cfg


def test_early_goal_against_turns_on_the_transition_guard():
    tr = MatchTracker("MID", 2); _tick(tr, 10, 0, 0); _tick(tr, 40, 0, 1)
    assert tr.conceded_early
    assert tr.adjust(MID_CONFIG).support_depth == 32.0
    assert "scored early" in tr.briefing_extra() and "LOSING" in tr.briefing_extra()


def test_a_goal_from_a_shot_locks_the_mapping():
    store = _FakeStore(); tr = _tracker(store); _tick(tr, 10, 0, 0)
    shot = {"commandType": "SHOOT", "parameters": {"aim_location": "BR", "power": 0.9}}
    tr.note_shot(shot, {"x": 40, "y": 0}, 55)
    _tick(tr, 14, 1, 0)                                              # goal within 8 s of the shot
    assert tr.aim_locked and any(e["type"] == "goal_from_shot" for e in store.recorded)
    for _ in range(3):                                               # contradicting observations no longer flip
        _shot_then_ball(tr, "TL", {"x": 35, "y": 0}, {"x": 45, "y": -1.0, "z": 0.5}, {"x": 20, "y": -2.0, "z": 0}, t=20)
    assert tr.aim_flip is False


def test_priors_from_past_matches_seed_the_next_one():
    events = [
        {"type": "match_start", "session": "a", "ts": 1}, {"type": "match_start", "session": "b", "ts": 2},
        {"type": "goal_against", "session": "b", "early": True, "ts": 3},
        {"type": "goal_from_shot", "dist": 26, "flip": True, "ts": 4},
        {"type": "snapshot", "shots": 7, "ts": 5},
    ]
    priors = priors_from_events(events)
    assert priors == {"aim_flip": True, "aim_locked": True, "shoot_bonus": 4.0,
                      "guard_transitions": True, "past_matches": 2, "past_shots": 7}, priors
    store = _FakeStore(events, ["Long shots from 26 beat their keeper"])
    tr = MatchTracker("DEF", 1, store=store); _tick(tr, 5, 0, 0, session="c")
    assert tr.aim_flip and tr.aim_locked
    cfg = tr.adjust(DEF_CONFIG)
    assert cfg.shoot_threshold == 39.0 and cfg.support_x_factor == 0.6, cfg
    assert "LESSONS FROM PAST MATCHES" in tr.briefing_extra() and "26 beat" in tr.briefing_extra()
    assert store.recorded[0]["type"] == "match_start"


def test_event_payload_parsing_and_store_is_noop_without_memory_id():
    raw = [{"payload": [{"conversational": {"role": "ASSISTANT", "content": {"text": '{"type":"snapshot","shots":3}'}}},
                        {"conversational": {"role": "USER", "content": {"text": "not json"}}}]}]
    assert parse_event_payloads(raw) == [{"type": "snapshot", "shots": 3}]
    os.environ.pop("MEMORY_RM_MEMORY_ID", None)
    store = MemoryStore("FWD1", memory_id=None)
    assert not store.enabled
    store.record({"type": "x"}); store.load_priors_async(lambda *a: None)   # must not raise


def test_handler_tracks_shots_and_briefs_the_llm_with_the_situation():
    from agent_base import create_invoke_handler
    seen = []
    invoke = create_invoke_handler(_FakeApp(), lambda s: seen.append(s) or "[]", 3, "FWD1",
                                   build_fallback(FWD1_CONFIG), FWD1_CONFIG, llm_timeout_s=0.5)

    async def run(state, t, home, away):
        state = copy.deepcopy(state); state["gameTime"] = t; state["score"] = {"home": home, "away": away}
        payload = {"prompt": json.dumps({"gameState": state, "teamId": TEAM_ID, "myPlayers": [3]})}
        return [json.loads(c) async for c in invoke(payload, None)]

    out = asyncio.run(run(_on_ball(3, 40, -3), 20, 0, 1))
    assert out[0][0]["commandType"] == "SHOOT" and len(invoke.tracker.shots) == 1
    asyncio.run(run(_defensive_state(), 30, 0, 1))
    assert seen and "SITUATION: we are LOSING" in seen[-1], seen


# ---------------------------------------------------------------------------
# 2. Pressure — attacking third: shoot anyway; deep: release forward
# ---------------------------------------------------------------------------

def test_pressed_in_attacking_third_shoots_rather_than_turning_over():
    s = _on_ball(3, 20, 6, {1: {"x": 22, "y": 7}})               # opponent 2.2 away, 35 from goal
    cmds = instinct_command(FWD1_CONFIG, s, TEAM_ID, 3)
    assert cmds[0]["commandType"] == "SHOOT" and cmds[0]["parameters"]["power"] == 1.0, cmds











# ---------------------------------------------------------------------------
# 3. Possession without the ball — forwards into the box, defender holds, keeper on his line
# ---------------------------------------------------------------------------

def test_forwards_run_to_open_positions_when_a_teammate_has_the_ball():
    from tools import open_positions
    s = _state()                                                  # P3 has the ball at (14,-5)
    f2 = instinct_command(FWD2_CONFIG, s, TEAM_ID, 4)[0]
    assert f2["commandType"] == "MOVE_TO" and f2["parameters"]["sprint"], f2
    assert f2["parameters"]["target_x"] > 20 and f2["parameters"]["target_y"] >= 0, f2        # ahead, on his wing
    best = open_positions(s["players"], TEAM_ID, 4, s["ball"]["position"], 55, side_y=1.0)[0]
    assert (f2["parameters"]["target_x"], f2["parameters"]["target_y"]) == (best["x"], best["y"])
    # the LLM gets to choose off the ball: instinct yields None with allow_llm_positions
    assert instinct_command(FWD2_CONFIG, s, TEAM_ID, 4, allow_llm_positions=True) is None
    # a defender parked on the #1 spot pushes the run elsewhere
    s2 = copy.deepcopy(s); s2["players"][AWAY_IDX[2]]["position"] = {"x": best["x"], "y": best["y"]}
    f2b = instinct_command(FWD2_CONFIG, s2, TEAM_ID, 4)[0]["parameters"]
    assert (f2b["target_x"], f2b["target_y"]) != (best["x"], best["y"]), f2b


def test_defender_holds_shape_while_we_attack():
    d = instinct_command(DEF_CONFIG, _state(), TEAM_ID, 1)[0]
    assert d["commandType"] == "MOVE_TO" and d["parameters"]["target_x"] == -24.8 and not d["parameters"]["sprint"], d


def test_keeper_holds_line_tracks_ball_and_distributes():
    gk = instinct_command(GK_CONFIG, _state(), TEAM_ID, 0)[0]
    assert gk["commandType"] == "MOVE_TO" and gk["parameters"]["target_x"] == -52.0, gk
    assert abs(gk["parameters"]["target_y"]) <= 4
    s = _state(); s["ball"]["possessionAgentId"] = "agentId_0"
    s["ball"]["position"] = {"x": -49, "y": 0, "z": 0}            # ids repeat across teams: nearest keeper holds it
    gk = instinct_command(GK_CONFIG, s, TEAM_ID, 0)[0]
    assert gk["commandType"] == "GK_DISTRIBUTE" and gk["parameters"]["target_player_id"] in (1, 2, 3, 4), gk


def test_loose_ball_nearest_teammate_intercepts_others_ask_llm():
    s = _state(); s["ball"]["possessionAgentId"] = None
    s["ball"]["position"] = {"x": 12, "y": -4, "z": 0}
    assert instinct_command(FWD1_CONFIG, s, TEAM_ID, 3)[0]["commandType"] == "INTERCEPT"
    assert instinct_command(FWD2_CONFIG, s, TEAM_ID, 4) is None


def test_opponent_possession_goes_to_llm_for_outfield_not_keeper():
    s = _state(); s["ball"]["possessionAgentId"] = "agentId_1"     # away P1 (index 6) has it
    s["ball"]["position"] = {"x": 10, "y": -3, "z": 0}
    for cfg, pid in ((DEF_CONFIG, 1), (MID_CONFIG, 2), (FWD1_CONFIG, 3), (FWD2_CONFIG, 4)):
        assert instinct_command(cfg, s, TEAM_ID, pid) is None, pid
    assert instinct_command(GK_CONFIG, s, TEAM_ID, 0)[0]["commandType"] == "MOVE_TO"


# ---------------------------------------------------------------------------
# 4. Mirror drill — drawn AWAY, everything points at -x and the aim mirrors
# ---------------------------------------------------------------------------

def test_away_forward_shoots_toward_minus_x_and_mirrors_the_corner():
    s = _mirrored(_on_ball(3, 45, -4, {0: {"x": 52, "y": 2}}))     # now at (-45,-4); keeper at (-52, 2)
    cmds = instinct_command(FWD1_CONFIG, s, AWAY, 3)
    assert cmds[0]["commandType"] == "SHOOT", cmds
    assert cmds[0]["parameters"]["aim_location"] == "BL", cmds   # -y side; facing -x, left is -y
    support = instinct_command(FWD2_CONFIG, _mirrored(_state()), AWAY, 4)[0]
    assert support["parameters"]["target_x"] < -20, support
    gk = instinct_command(GK_CONFIG, _mirrored(_state()), AWAY, 0)[0]
    assert gk["parameters"]["target_x"] == 52.0, gk


def test_away_briefing_names_the_mirrored_goals():
    assert "Your goal at x=55 | Opponent goal at x=-55" in summarize_state(_mirrored(_state()), AWAY, 3, "FWD1")


# ---------------------------------------------------------------------------
# 5. Handler — instinct skips the LLM; slow LLM falls back inside the budget
# ---------------------------------------------------------------------------

def _run_invoke(state, pid, label, cfg, agent_fn, timeout=0.45):
    from agent_base import create_invoke_handler
    calls = []

    def fake_agent(summary):
        calls.append(summary)
        return agent_fn(summary)

    invoke = create_invoke_handler(_FakeApp(), fake_agent, pid, label, build_fallback(cfg), cfg, llm_timeout_s=timeout)
    payload = {"prompt": json.dumps({"gameState": state, "teamId": TEAM_ID, "myPlayers": [pid]})}

    async def collect():
        # time to the first reply chunk — asyncio.run's shutdown waits for background threads,
        # which the long-lived AgentCore server never does
        t0 = time.perf_counter()
        out = []
        async for chunk in invoke(payload, None):
            out.append(json.loads(chunk))
        return out, time.perf_counter() - t0

    out, took = asyncio.run(collect())
    return out, calls, took


def _defensive_state():
    s = _state(); s["ball"]["possessionAgentId"] = "agentId_2"
    s["players"][7]["position"] = {"x": 0, "y": 5}
    s["ball"]["position"] = {"x": 0, "y": 5, "z": 0}
    return s


def test_handler_asks_llm_to_shoot_with_the_tool_and_shoots_anyway_when_late():
    reply = '[{"commandType":"SHOOT","playerId":3,"parameters":{"aim_location":"BL","power":0.9},"duration":0}]'
    out, calls, _ = _run_invoke(_on_ball(3, 40, -3), 3, "FWD1", FWD1_CONFIG, lambda s: reply)
    assert len(calls) == 1 and "SHOT TOOL" in calls[0] and "SHOOT NOW" in calls[0], calls
    assert out[0][0]["commandType"] == "SHOOT" and out[0][0]["parameters"]["aim_location"] == "BL", out
    def slow(s):
        time.sleep(0.8); return reply
    out, _, took = _run_invoke(_on_ball(3, 40, -3), 3, "FWD1", FWD1_CONFIG, slow, timeout=0.2)
    assert took < 0.6 and out[0][0]["commandType"] == "SHOOT" and out[0][0]["parameters"]["power"] == 0.9, out
    time.sleep(0.9)
    # code path (no LLM) still shoots from anywhere
    assert instinct_command(FWD1_CONFIG, _on_ball(3, -20, 0), TEAM_ID, 3)[0]["commandType"] == "SHOOT"


def _shot_then_ball(tr, aim, from_pos, ball_pos, ball_vel, t=10):
    tr.note_shot({"commandType": "SHOOT", "parameters": {"aim_location": aim, "power": 1.0}}, from_pos, 55)
    gs = {"gameTime": t + 2, "score": {"home": 0, "away": 0}, "ball": {"position": ball_pos, "velocity": ball_vel}}
    tr.observe(gs, TEAM_ID, "m1")


def test_shot_judged_from_ball_trajectory_on_target_and_bands():
    tr = _tracker(); _tick(tr, 10, 0, 0)
    _shot_then_ball(tr, "TL", {"x": 35, "y": 0}, {"x": 45, "y": 1.0, "z": 0.5}, {"x": 20, "y": 2.0, "z": 0})   # y at goal = 2 → on target, +y side
    assert tr.band_obs == {1: [1, 1]} and tr.side_obs == [(1, 1)], (tr.band_obs, tr.side_obs)
    _shot_then_ball(tr, "TL", {"x": 10, "y": 0}, {"x": 30, "y": 4.0, "z": 1}, {"x": 20, "y": 6.0, "z": 0})     # y at goal = 11.5 → off
    assert tr.band_obs[3] == [0, 1]
    for _ in range(2):
        _shot_then_ball(tr, "TL", {"x": 10, "y": 0}, {"x": 30, "y": 4.0, "z": 1}, {"x": 20, "y": 6.0, "z": 0})
    assert tr.banned_bands() == (3,)
    cfg = tr.adjust(FWD1_CONFIG)
    assert cfg.banned_bands == (3,)
    cmds = instinct_command(cfg, _on_ball(3, 10, 0), TEAM_ID, 3)          # dist 45 → banned → carry closer
    assert cmds[0]["commandType"] == "MOVE_TO" and cmds[0]["parameters"]["target_x"] == 22.0, cmds
    assert instinct_command(cfg, _on_ball(3, 25, 0), TEAM_ID, 3)[0]["commandType"] == "SHOOT"


def test_shots_landing_on_the_keepers_side_flip_the_mapping():
    tr = _tracker(); _tick(tr, 10, 0, 0)
    for _ in range(3):                                                    # intended +y (T), observed -y
        _shot_then_ball(tr, "TL", {"x": 35, "y": 0}, {"x": 45, "y": -1.0, "z": 0.5}, {"x": 20, "y": -2.0, "z": 0})
    assert tr.aim_flip is True and tr.adjust(FWD1_CONFIG).aim_flip is True


def test_high_top_shots_switch_to_low():
    tr = _tracker(); _tick(tr, 10, 0, 0)
    for _ in range(3):
        _shot_then_ball(tr, "TR", {"x": 35, "y": 0}, {"x": 45, "y": 0, "z": 2.5}, {"x": 20, "y": 0, "z": 3.0})
    assert tr.prefer_low is True
    from tools import shot_opportunity
    s = _on_ball(3, 45, -4, {0: {"x": 52, "y": 2}})
    assert shot_opportunity(s["players"], TEAM_ID, {"x": 45, "y": -4}, 55, prefer_low=True)["aim_location"] == "BR"


def test_landing_map_picks_a_corner_known_to_land_inside():
    from tools import pick_aim_from_landing, shot_opportunity
    amap = {"TL": {"y": 8.0, "n": 3}, "BR": {"y": -3.0, "n": 2}, "TR": {"y": 2.5, "n": 2}}
    assert pick_aim_from_landing(amap, keeper_y=2.0, default="TL") == "BR"      # inside and away from the keeper
    assert pick_aim_from_landing(amap, keeper_y=-2.0, default="TL") == "TR"
    assert pick_aim_from_landing({"TL": {"y": 8.0, "n": 1}}, 0.0, "TL") is None   # too few observations
    allwide = {a: {"y": 7.0, "n": 2} for a in ("TL", "TR", "BL", "BR")}
    assert pick_aim_from_landing(allwide, 0.0, "TL") == "CENTER"
    s = _on_ball(3, 35, -4, {0: {"x": 52, "y": 2}})
    sh = shot_opportunity(s["players"], TEAM_ID, {"x": 35, "y": -4}, 55, aim_map=amap)   # learned map overrides CENTER
    assert sh["aim_location"] == "BR" and sh["learned_aim"]
    # learned this match: three TL shots landing at y=8 → tracker's aim_map feeds the config
    tr = _tracker(); _tick(tr, 10, 0, 0)
    for _ in range(3):
        _shot_then_ball(tr, "TL", {"x": 35, "y": 0}, {"x": 45, "y": 4.0, "z": 0.5}, {"x": 20, "y": 8.0, "z": 0})   # y at goal = 8
    for _ in range(2):
        _shot_then_ball(tr, "BR", {"x": 35, "y": 0}, {"x": 45, "y": -1.0, "z": 0.5}, {"x": 20, "y": -2.0, "z": 0})  # y at goal = -2
    cfg = tr.adjust(FWD1_CONFIG)
    assert cfg.aim_map["TL"]["y"] == 8.0 and cfg.aim_map["BR"] == {"y": -2.0, "n": 2}, cfg.aim_map
    assert instinct_command(cfg, _on_ball(3, 35, -4, {0: {"x": 52, "y": 2}}), TEAM_ID, 3)[0]["parameters"]["aim_location"] == "BR"


def test_priors_learn_band_stats_and_mapping_from_shot_results():
    events = [{"type": "shot_result", "band": 3, "on_target": False, "side_match": False, "tb": "T", "high": False, "flip": False, "ts": i} for i in range(3)]
    events += [{"type": "shot_result", "band": 1, "on_target": True, "side_match": False, "tb": "T", "high": False, "flip": False, "ts": 9, "aim": "TR", "y_goal": 3.0}]
    pr = priors_from_events(events)
    assert pr["aim_flip"] is True and pr["band_stats"] == {"3": [0, 3], "1": [1, 1]} and pr["on_target_rate"] == 0.25, pr
    assert pr["aim_landing"] == {"TR": [3.0, 1]}, pr
    tr = MatchTracker("FWD1", 3, store=_FakeStore(events)); _tick(tr, 5, 0, 0, session="c")
    assert tr.aim_flip is True and tr.banned_bands() == (3,)


def test_handler_asks_llm_in_defensive_phase():
    reply = '[{"commandType":"PRESS_BALL","playerId":3,"parameters":{"intensity":0.8},"duration":3}]'
    out, calls, _ = _run_invoke(_defensive_state(), 3, "FWD1", FWD1_CONFIG, lambda s: reply)
    assert len(calls) == 1 and out[0][0]["commandType"] == "PRESS_BALL", (calls, out)
    assert "Opponent goal at x=55" in calls[0]


def test_slow_llm_falls_back_within_budget():
    def slow(summary):
        time.sleep(1.0)
        return "[]"
    out, calls, took = _run_invoke(_defensive_state(), 3, "FWD1", FWD1_CONFIG, slow, timeout=0.2)
    assert took < 0.6, took
    assert out[0][0]["commandType"] in ("PRESS_BALL", "MOVE_TO", "MARK"), out
    time.sleep(1.0)   # let the background call finish and release the lock


def test_last_resort_is_clear_override():
    from fallback import build_last_resort
    assert build_last_resort(FWD1_CONFIG, 3)["commandType"] == "CLEAR_OVERRIDE"


# ---------------------------------------------------------------------------
# 6. Latency caps, prompts, parser
# ---------------------------------------------------------------------------

def _load_main(name):
    sys.path.insert(0, os.path.join(HERE, f"ai-{name}", "src"))
    sys.modules.pop("main", None)
    try:
        return importlib.import_module("main")
    finally:
        sys.path.pop(0)


def test_hybrid_positions_use_llm_answer_or_tool_best_when_late():
    from tools import open_positions
    s = _state()                                                  # P3 has the ball; FWD2 off the ball
    best = open_positions(s["players"], TEAM_ID, 4, s["ball"]["position"], 55, side_y=1.0)[0]
    reply = '[{"commandType":"MOVE_TO","playerId":4,"parameters":{"target_x":30,"target_y":16,"sprint":true},"duration":0}]'
    out, calls, _ = _run_invoke(s, 4, "FWD2", FWD2_CONFIG, lambda x: reply)
    assert len(calls) == 1 and "OPEN POSITIONS" in calls[0] and out[0][0]["parameters"]["target_y"] == 16, (calls, out)
    def slow(x):
        time.sleep(0.8); return reply
    out, _, took = _run_invoke(s, 4, "FWD2", FWD2_CONFIG, slow, timeout=0.2)
    assert took < 0.6 and (out[0][0]["parameters"]["target_x"], out[0][0]["parameters"]["target_y"]) == (best["x"], best["y"]), out
    time.sleep(0.9)


def test_aggressive_defensive_fallback_press_lane_mark():
    s = _defensive_state()                                        # away P2 carries at (0,5)
    fb = {pid: build_fallback(cfg)(s, TEAM_ID, pid)[0] for cfg, pid in ((DEF_CONFIG, 1), (MID_CONFIG, 2), (FWD1_CONFIG, 3), (FWD2_CONFIG, 4))}
    kinds = {pid: c["commandType"] for pid, c in fb.items()}
    assert "PRESS_BALL" in kinds.values() and "INTERCEPT" in kinds.values(), kinds
    assert kinds[1] in ("MARK", "PRESS_BALL", "INTERCEPT"), kinds
    press = [c for c in fb.values() if c["commandType"] == "PRESS_BALL"]
    assert all(c["parameters"]["intensity"] == 1.0 for c in press)


def test_agent_is_latency_capped():
    from agent_base import create_agent
    agent = create_agent("test prompt")
    assert agent.conversation_manager.window_size <= 4, agent.conversation_manager.window_size
    cfg = agent.model.get_config()
    assert cfg["model_id"] == "us.amazon.nova-micro-v1:0", cfg
    assert cfg.get("max_tokens", 10**9) <= 80 and cfg.get("temperature", 1.0) <= 0.1, cfg


def test_every_agent_is_lite_short_prompted_and_shoot_wired():
    for name, pid, role in (("gk", 0, "GK"), ("def", 1, "DEF"), ("mid", 2, "MID"), ("fwd1", 3, "FWD"), ("fwd2", 4, "FWD")):
        m = _load_main(name)
        assert m.MY_PLAYER_ID == pid and m.FALLBACK_CONFIG.role == role, name
        assert m.agent.model.get_config()["model_id"] == "us.amazon.nova-lite-v1:0", name
        sp = m.SYSTEM_PROMPT
        assert len(sp) < 3300, (name, len(sp))
        for must in ("Your goal at x=A", "Bare JSON only", "no code fences", f'"playerId":{pid}'):
            assert must in sp, (name, must)
        assert "SET_STANCE" not in sp and "<ID>" not in sp, name
        if role != "GK":
            assert "SHOT TOOL" in sp and "- SHOOT:" in sp, name
            assert m.FALLBACK_CONFIG.shoot_threshold >= 35, name
            assert "OPEN POSITIONS" in sp and "PRESS_BALL 1.0" in sp, name


def test_parser_lifts_flattened_parameters_and_fills_shot_defaults():
    flat = '[{"commandType":"MARK","playerId":1,"target_player_id":3,"tightness":"TIGHT"}]'
    cmd = parse_commands(flat, TEAM_ID, 1)[0]
    assert cmd["parameters"] == {"target_player_id": 3, "tightness": "TIGHT"} and cmd["duration"] == 3, cmd
    shot = parse_commands('[{"commandType":"SHOOT","parameters":{}}]', TEAM_ID, 3)[0]
    assert shot["parameters"] == {"aim_location": "BR", "power": 0.9}, shot


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}\n{type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
