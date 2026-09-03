"""Low-latency invoke handler for the learned football team."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Callable

from fallback import FallbackConfig, build_last_resort
from match_memory import AgentCoreMatchMemory, MatchTracker
from parsing import parse_commands
from state import _is_my_team, _player_idx, find_possession_holder, summarize_state
from tactics import decide_locally


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


async def _invoke_model(agent, prompt: str, timeout_seconds: float):
    """Invoke without retaining an ever-growing model conversation."""
    messages = getattr(agent, "messages", None)
    if isinstance(messages, list):
        messages.clear()
    try:
        invoke_async = getattr(agent, "invoke_async", None)
        if invoke_async is not None:
            return await asyncio.wait_for(invoke_async(prompt), timeout=timeout_seconds)
        return await asyncio.wait_for(asyncio.to_thread(agent, prompt), timeout=timeout_seconds)
    finally:
        messages = getattr(agent, "messages", None)
        if isinstance(messages, list):
            messages.clear()


def _state_signature(game_state: dict, team_id: int) -> tuple:
    """Coarse tactical signature used to reject stale background decisions."""
    ball = game_state.get("ball", {})
    players = game_state.get("players", [])
    holder = find_possession_holder(ball, players)
    if holder is None:
        holder_key = "free"
    else:
        holder_key = (
            "ours" if _is_my_team(holder, team_id) else "theirs",
            _player_idx(holder),
        )
    direction = 1 if team_id == 0 else -1
    attack_x = direction * float(ball.get("position", {}).get("x", 0) or 0)
    zone = max(0, min(9, int((attack_x + 55) // 11)))
    score = game_state.get("score", {})
    return (
        holder_key,
        zone,
        int(score.get("home", 0) or 0),
        int(score.get("away", 0) or 0),
        str(game_state.get("playMode", "")),
    )


def create_learned_invoke_handler(
    app,
    agent,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
):
    """Register a rule-first, selective-model, match-LTM entrypoint.

    Default call budget:
      * deterministic local calculation on every tick;
      * zero model calls for clear decisions;
      * an ambiguous decision may schedule Nova in the background, but the
        current tick never waits for it;
      * a completed model command is used only on a later matching state;
      * one asynchronous LTM read at match start;
      * one asynchronous LTM event at match end, written by MID only.
    """
    log = app.logger
    last_resort = build_last_resort(fallback_cfg, my_player_id)
    tracker = MatchTracker(my_player_id)
    memory = AgentCoreMatchMemory()
    model_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()
    retrieval_task: asyncio.Task | None = None
    retrieval_match_key = ""
    learned_context = ""
    model_task: asyncio.Task | None = None
    model_task_match_key = ""
    model_task_tick = -10_000.0
    model_task_signature: tuple | None = None
    last_model_tick = -10_000.0

    model_mode = os.environ.get("FOOTBALL_LLM_MODE", "selective").strip().lower()
    if model_mode not in {"off", "selective", "always"}:
        model_mode = "selective"
    timeout_seconds = _env_float("FOOTBALL_LLM_TIMEOUT_MS", 2200, 500, 4000) / 1000
    min_tick_gap = _env_int("FOOTBALL_LLM_MIN_TICK_GAP", 6, 1, 120)
    max_model_age = _env_int("FOOTBALL_LLM_MAX_AGE_TICKS", 3, 1, 10)
    confidence_threshold = _env_float("FOOTBALL_LOCAL_CONFIDENCE", 0.72, 0.5, 0.98)
    writer_position = os.environ.get("FOOTBALL_MEMORY_WRITER", "MID").strip().upper()

    def _keep_task(task: asyncio.Task) -> None:
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    def _log_memory_failure(prefix: str) -> None:
        if memory.last_error:
            log.warn(f"{position_label} {prefix}: {memory.last_error}")

    @app.entrypoint
    async def invoke(payload, context):
        nonlocal retrieval_task, retrieval_match_key, learned_context
        nonlocal model_task, model_task_match_key, model_task_tick, model_task_signature
        nonlocal last_model_tick
        reaction_started = time.perf_counter()
        try:
            prompt = payload.get("prompt", "{}")
            prompt_data = json.loads(prompt) if isinstance(prompt, str) else prompt
            if not isinstance(prompt_data, dict):
                raise ValueError("prompt must decode to an object")

            # The match service may keep its identifier in AgentCore runtime
            # context rather than the football prompt. Use it when available so
            # a new match cannot inherit the previous match's counters.
            if not any(
                prompt_data.get(key)
                for key in ("matchId", "match_id", "gameId", "game_id", "sessionId")
            ):
                runtime_session = None
                if isinstance(context, dict):
                    runtime_session = context.get("sessionId") or context.get("session_id")
                elif context is not None:
                    runtime_session = getattr(context, "session_id", None) or getattr(
                        context, "sessionId", None
                    )
                if runtime_session:
                    prompt_data = dict(prompt_data)
                    prompt_data["sessionId"] = str(runtime_session)

            game_state = prompt_data.get("gameState", {})
            team_id = int(prompt_data.get("teamId", 0))
            my_players = prompt_data.get("myPlayers", [my_player_id])
            effective_pid = int(my_players[0]) if my_players else my_player_id
            tick = float(game_state.get("tick", game_state.get("gameTime", 0)) or 0)

            transition = tracker.observe(prompt_data, game_state, team_id)
            if transition.started:
                # If the engine signals a new match by resetting its clock (and
                # never emitted FULL_TIME), reuse the just-completed summary
                # immediately while AgentCore extracts it in the background.
                learned_context = transition.completed_summary or ""
                last_model_tick = -10_000.0
                if model_task is not None and not model_task.done():
                    model_task.cancel()
                model_task = None
                model_task_match_key = ""
                model_task_signature = None
                retrieval_match_key = transition.match_key
                if memory.enabled:
                    retrieval_task = asyncio.create_task(
                        asyncio.to_thread(memory.retrieve, team_id, 3)
                    )
                    _keep_task(retrieval_task)

            # Pick up the asynchronous result on a later tick; never hold the
            # first match command waiting for a memory network request.
            if (
                retrieval_task is not None
                and retrieval_match_key == transition.match_key
                and retrieval_task.done()
            ):
                try:
                    retrieved_context = retrieval_task.result()
                    if retrieved_context:
                        learned_context = "\n".join(
                            part for part in (learned_context, retrieved_context) if part
                        )[-3600:]
                        log.info(f"{position_label} loaded past-match lessons")
                    else:
                        _log_memory_failure("LTM retrieval skipped")
                except Exception as exc:
                    log.warn(f"{position_label} LTM retrieval failed: {exc}")
                retrieval_task = None

            # Only one player writes the shared team episode, avoiding five
            # duplicate extraction jobs for the same match snapshot.
            if (
                transition.completed_summary
                and memory.enabled
                and position_label.upper() == writer_position
            ):
                task = asyncio.create_task(
                    asyncio.to_thread(
                        memory.store_episode,
                        team_id,
                        transition.completed_match_key or transition.match_key,
                        transition.completed_summary,
                    )
                )
                _keep_task(task)

            local = decide_locally(
                game_state,
                team_id,
                effective_pid,
                position_label,
                learned_context=learned_context,
            )
            if local is None or not local.commands:
                commands = fallback_fn(game_state, team_id, effective_pid)
                local_reason = "fallback for incomplete state"
                local_confidence = 1.0
            else:
                commands = local.commands
                local_reason = local.reason
                local_confidence = local.confidence

            used_model = False
            current_signature = _state_signature(game_state, team_id)

            # Polling task.done()/result() is non-blocking. A model decision is
            # accepted only while the tactical situation that produced it is
            # still materially the same; otherwise the stale command is dropped.
            if model_task is not None and model_task.done():
                try:
                    response = model_task.result()

                    def on_recovered(raw: str) -> None:
                        log.warn(f"{position_label} recovered malformed model JSON: {raw[:160]}")

                    model_commands = parse_commands(
                        str(response), team_id, effective_pid, on_recovered
                    )
                    fresh = (
                        model_task_match_key == transition.match_key
                        and 0 <= tick - model_task_tick <= max_model_age
                        and model_task_signature == current_signature
                        and (model_mode == "always" or local_confidence < confidence_threshold)
                    )
                    if fresh and model_commands:
                        commands = model_commands[:1]
                        used_model = True
                        log.info(
                            f"{position_label} used fresh background Nova decision: "
                            f"{commands[0].get('commandType')}"
                        )
                    else:
                        log.info(f"{position_label} discarded stale background Nova decision")
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log.warn(f"{position_label} background Nova failed: {exc}")
                finally:
                    model_task = None
                    model_task_match_key = ""
                    model_task_signature = None

            should_schedule_model = (
                model_task is None
                and tick - last_model_tick >= min_tick_gap
                and (
                    model_mode == "always"
                    or (
                        model_mode == "selective"
                        and local_confidence < confidence_threshold
                    )
                )
            )

            if should_schedule_model:
                state_summary = summarize_state(
                    game_state, team_id, effective_pid, position_label
                )
                context_parts = [tracker.current_context()]
                if learned_context:
                    context_parts.append("Relevant lessons from completed past matches:\n" + learned_context)
                model_prompt = (
                    state_summary
                    + "\n\n"
                    + "\n".join(part for part in context_parts if part)
                    + f"\n\nLocal engine is uncertain because: {local_reason}. "
                    "Return one immediately executable command only."
                )
                last_model_tick = tick

                async def run_model_in_background():
                    async with model_lock:
                        return await _invoke_model(agent, model_prompt, timeout_seconds)

                model_task = asyncio.create_task(run_model_in_background())
                model_task_match_key = transition.match_key
                model_task_tick = tick
                model_task_signature = current_signature
                _keep_task(model_task)
                log.info(
                    f"{position_label} scheduled background Nova; current tick stays local"
                )

            if not commands:
                commands = fallback_fn(game_state, team_id, effective_pid)
            if not commands:
                emergency = dict(last_resort)
                emergency["teamId"] = team_id
                emergency["playerId"] = effective_pid
                commands = [emergency]

            tracker.record_command(commands[0])
            reaction_ms = (time.perf_counter() - reaction_started) * 1000
            log.info(
                f"{position_label} tick={tick:.0f} command={commands[0].get('commandType')} "
                f"source={'model-cache' if used_model else 'local'} "
                f"reactionMs={reaction_ms:.1f} localReason={local_reason}"
            )
            yield json.dumps(commands[:1])

        except Exception as exc:
            log.error(f"{position_label} learned handler error: {exc}")
            try:
                raw = payload.get("prompt", "{}")
                prompt_data = json.loads(raw) if isinstance(raw, str) else raw
                team_id = int(prompt_data.get("teamId", 0))
                my_players = prompt_data.get("myPlayers", [my_player_id])
                effective_pid = int(my_players[0]) if my_players else my_player_id
                commands = fallback_fn(prompt_data.get("gameState", {}), team_id, effective_pid)
                yield json.dumps(commands[:1])
            except Exception:
                command = dict(last_resort)
                command["teamId"] = 0
                yield json.dumps([command])

    return invoke
