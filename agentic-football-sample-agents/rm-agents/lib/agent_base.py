"""Base agent factory + invoke handler for RM_Agents.

Latency is the ruling constraint: the match server invokes every ~2 s and expects a reply
within 500 ms. So:

  * instinct_command() decides every on-ball / possession / keeper / loose-ball tick in code (0 ms)
  * the LLM is only consulted in the defensive phase, with a hard timeout — if Nova Micro has
    not answered in LLM_TIMEOUT_S the rule-based fallback replies instead and the slow call is
    left to finish in the background (the agent is locked until it does, so the conversation
    never sees two calls at once)
  * the Strands conversation window, max_tokens and temperature are capped small
  * lib/match_memory.py tracks the match (STM) and AgentCore Memory (LTM) in background threads:
    the doctrine is re-tuned to the score/time/priors each tick, the LLM briefing gains a
    SITUATION line and LESSONS from past matches, and nothing waits on memory I/O
"""

import asyncio
import json
import os
import threading
import time
from typing import Callable

from strands import Agent
from strands.models import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager

from parsing import parse_commands
from state import summarize_state
from fallback import FallbackConfig, build_last_resort, instinct_command
from match_memory import MatchTracker, MemoryStore
from tools import briefing_lines, shot_line
from state import get_goal_positions, _player_idx, _is_my_team


LLM_TIMEOUT_S = float(os.environ.get("RM_LLM_TIMEOUT_S", "0.55"))


def create_agent(
    system_prompt: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
    window_size: int = 2,
    max_tokens: int = 64,
    temperature: float = 0.1,
) -> Agent:
    """Strands Agent tuned for a sub-500 ms reply.

    window_size 2: the briefing is ~250 tokens; every remembered turn is prefill the model
    has to read again before it answers. max_tokens 64: one bare-JSON command is ~35 tokens
    and output decodes sequentially, so this cap IS the worst-case generation time.
    """
    model = BedrockModel(model_id=model_id, max_tokens=max_tokens, temperature=temperature)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        conversation_manager=SlidingWindowConversationManager(window_size=window_size),
        callback_handler=None,
    )


def create_invoke_handler(
    app,
    agent,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
    llm_timeout_s: float = LLM_TIMEOUT_S,
):
    """Create and register the @app.entrypoint invoke handler."""
    log = app.logger
    last_resort = build_last_resort(fallback_cfg, my_player_id)
    llm_lock = threading.Lock()
    store = MemoryStore(position_label, log)
    tracker = MatchTracker(position_label, my_player_id, store=store, log=log)
    log.info(f"{position_label} memory {'ON' if store.enabled else 'off (no MEMORY_*_ID env)'}")

    def _call_llm(summary: str) -> str:
        t0 = time.perf_counter()
        try:
            return str(agent(summary))
        finally:
            llm_lock.release()
            ms = (time.perf_counter() - t0) * 1000
            if ms > llm_timeout_s * 1000:
                log.info(f"{position_label} LLM late call finished in {ms:.0f}ms")

    async def _ask_llm(summary: str):
        """LLM reply text, or None when busy / timed out / errored.

        asyncio.wait (not wait_for): cancelling an executor future whose thread is already
        running blocks until the thread finishes, which would defeat the timeout. We simply
        stop waiting; the thread completes in the background and releases the lock.
        """
        if not llm_lock.acquire(blocking=False):
            log.warn(f"{position_label} LLM still busy from a previous tick — using fallback")
            return None
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, _call_llm, summary)
        fut.add_done_callback(lambda f: None if f.cancelled() else f.exception())  # retrieve, never raise later
        done, _ = await asyncio.wait({fut}, timeout=llm_timeout_s)
        if fut not in done:
            log.warn(f"{position_label} LLM exceeded {llm_timeout_s:.2f}s — using fallback")
            return None
        try:
            return fut.result()
        except Exception as e:
            log.error(f"{position_label} LLM error: {e}")
            return None

    @app.entrypoint
    async def invoke(payload, context):
        try:
            prompt = payload.get("prompt", "{}")
            prompt_data = json.loads(prompt) if isinstance(prompt, str) else prompt
            game_state = prompt_data.get("gameState", {})
            team_id = prompt_data.get("teamId", 0)
            my_players = prompt_data.get("myPlayers", [my_player_id])
            pid = my_players[0] if my_players else my_player_id

            tracker.observe(game_state, team_id, getattr(context, "session_id", None))
            cfg = tracker.adjust(fallback_cfg)

            instinct = instinct_command(cfg, game_state, team_id, pid, allow_llm_positions=True)
            if instinct:
                c = instinct[0]
                tracker.note_tick("instinct")
                if c["commandType"] == "SHOOT":
                    me = next((p for p in game_state.get("players", [])
                               if _is_my_team(p, team_id) and _player_idx(p) == pid), None)
                    tracker.note_shot(c, (me or {}).get("position") or {}, get_goal_positions(team_id)[1])
                log.info(f"{position_label} instinct {c['commandType']} {c.get('parameters', {})}")
                yield json.dumps(instinct)
                return

            summary = (summarize_state(game_state, team_id, pid, position_label)
                       + shot_line(game_state, team_id, pid, cfg.aim_flip, cfg.prefer_low, cfg.banned_bands, cfg.aim_map)
                       + briefing_lines(game_state, team_id, pid, fallback_cfg.side_y)
                       + tracker.briefing_extra()
                       + f"\n\nReply now with the JSON array containing exactly one command for player {pid} (never an empty array):")
            t0 = time.perf_counter()
            response_text = await _ask_llm(summary)
            ms = (time.perf_counter() - t0) * 1000

            commands = []
            if response_text is not None:
                def on_recovered(raw: str) -> None:
                    log.warn(f"{position_label} recovered malformed JSON: {raw[:200]}")
                commands = parse_commands(response_text, team_id, pid, on_recovered)

            if commands:
                tracker.note_tick("llm")
                if commands[0]["commandType"] == "SHOOT":
                    me = next((p for p in game_state.get("players", [])
                               if _is_my_team(p, team_id) and _player_idx(p) == pid), None)
                    tracker.note_shot(commands[0], (me or {}).get("position") or {}, get_goal_positions(team_id)[1])
                log.info(f"{position_label} LLM {ms:.0f}ms returned "
                         f"{[c.get('commandType') for c in commands]}")
                yield json.dumps(commands)
            else:
                if response_text is not None:
                    log.warn(f"{position_label} LLM parse failed after {ms:.0f}ms: {response_text[:200]}")
                tracker.note_tick("fallback")
                commands = instinct_command(cfg, game_state, team_id, pid) or fallback_fn(game_state, team_id, pid)
                if commands[0]["commandType"] == "SHOOT":
                    me = next((p for p in game_state.get("players", [])
                               if _is_my_team(p, team_id) and _player_idx(p) == pid), None)
                    tracker.note_shot(commands[0], (me or {}).get("position") or {}, get_goal_positions(team_id)[1])
                log.info(f"{position_label} fallback {commands[0]['commandType']} "
                         f"{commands[0].get('parameters', {})}")
                yield json.dumps(commands)

        except Exception as e:
            log.error(f"{position_label} agent error: {e}")
            try:
                prompt_data = json.loads(payload.get("prompt", "{}"))
                team_id = prompt_data.get("teamId", 0)
                my_players = prompt_data.get("myPlayers", [my_player_id])
                pid = my_players[0] if my_players else my_player_id
                yield json.dumps(fallback_fn(prompt_data.get("gameState", {}), team_id, pid))
            except Exception:
                cmd = dict(last_resort)
                cmd["teamId"] = 0
                yield json.dumps([cmd])

    invoke.tracker = tracker   # exposed for tests and local probes
    return invoke
