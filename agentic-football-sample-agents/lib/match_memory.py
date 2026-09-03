"""Bounded match memory and AgentCore long-term-memory integration.

The hot path keeps only small numeric counters in process.  At most one role
(MID by default) writes one deterministic summary after a match.  AgentCore's
episodic strategy performs its extraction asynchronously; the next match reads
the most relevant episodes once, never once per tick.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from state import _is_my_team, _player_idx, dist, find_possession_holder


TERMINAL_MODES = {
    "ENDED",
    "FINISHED",
    "FULL_TIME",
    "GAME_OVER",
    "MATCH_END",
    "MATCH_ENDED",
    "MATCH_FINISHED",
}


def _safe_number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score(game_state: dict) -> tuple[int, int]:
    score = game_state.get("score", {})
    return int(_safe_number(score.get("home", 0))), int(_safe_number(score.get("away", 0)))


def _explicit_match_id(prompt_data: dict, game_state: dict) -> str | None:
    for source in (prompt_data, game_state):
        for key in ("matchId", "match_id", "gameId", "game_id", "sessionId"):
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _is_terminal(prompt_data: dict, game_state: dict) -> bool:
    values = (
        game_state.get("playMode"),
        game_state.get("status"),
        prompt_data.get("matchStatus"),
        prompt_data.get("status"),
    )
    return any(str(value).upper().replace(" ", "_") in TERMINAL_MODES for value in values)


@dataclass
class MatchStats:
    match_key: str
    team_id: int
    match_type: str
    start_home: int
    start_away: int
    end_home: int
    end_away: int
    ticks: int = 0
    possession_ticks: int = 0
    opponent_possession_ticks: int = 0
    free_ball_ticks: int = 0
    territory_sum: float = 0.0
    pressure_sum: float = 0.0
    pressure_samples: int = 0
    turnovers_won: int = 0
    turnovers_lost: int = 0
    commands: Counter = field(default_factory=Counter)
    last_possession_side: str | None = None

    def observe(self, game_state: dict, player_id: int) -> None:
        self.ticks += 1
        self.end_home, self.end_away = _score(game_state)
        players = game_state.get("players", [])
        ball = game_state.get("ball", {})
        holder = find_possession_holder(ball, players)

        if holder is None:
            side = "free"
            self.free_ball_ticks += 1
        elif _is_my_team(holder, self.team_id):
            side = "ours"
            self.possession_ticks += 1
        else:
            side = "theirs"
            self.opponent_possession_ticks += 1

        if self.last_possession_side == "theirs" and side == "ours":
            self.turnovers_won += 1
        elif self.last_possession_side == "ours" and side == "theirs":
            self.turnovers_lost += 1
        if side != "free":
            self.last_possession_side = side

        direction = 1 if self.team_id == 0 else -1
        ball_pos = ball.get("position", {})
        self.territory_sum += direction * _safe_number(ball_pos.get("x", 0))

        me = next(
            (p for p in players if _player_idx(p) == player_id and _is_my_team(p, self.team_id)),
            None,
        )
        opponents = [p for p in players if not _is_my_team(p, self.team_id)]
        if me and opponents:
            pressure = min(dist(me.get("position", {}), p.get("position", {})) for p in opponents)
            self.pressure_sum += pressure
            self.pressure_samples += 1

    def record_command(self, command: dict) -> None:
        command_type = command.get("commandType")
        if command_type:
            self.commands[str(command_type)] += 1

    def summary(self) -> str:
        our_start, their_start = (
            (self.start_home, self.start_away)
            if self.team_id == 0
            else (self.start_away, self.start_home)
        )
        our_end, their_end = (
            (self.end_home, self.end_away)
            if self.team_id == 0
            else (self.end_away, self.end_home)
        )
        goals_observed_for = our_end - our_start
        goals_observed_against = their_end - their_start
        if our_end > their_end:
            outcome = "WIN"
        elif our_end < their_end:
            outcome = "LOSS"
        else:
            outcome = "DRAW"

        contested = self.possession_ticks + self.opponent_possession_ticks
        possession_pct = round(100 * self.possession_ticks / contested) if contested else 0
        avg_territory = self.territory_sum / max(self.ticks, 1)
        avg_pressure = self.pressure_sum / max(self.pressure_samples, 1)
        command_mix = dict(self.commands.most_common())

        lessons = []
        if possession_pct < 45:
            lessons.append("use safer outlets and compact counter-pressure to retain possession")
        if self.turnovers_lost > self.turnovers_won:
            lessons.append("release the ball earlier when pressure is close")
        if avg_territory < 0:
            lessons.append("move the midfield support line higher after regains")
        if our_end == 0:
            lessons.append("create a final-third pass before forcing low-angle shots")
        if their_end > 0:
            lessons.append("keep defender and midfielder staggered during attacks")
        if not lessons:
            lessons.append("preserve the successful shape while varying the final action")

        data = {
            "kind": "football_match_episode",
            "match_id": self.match_key,
            "match_type": self.match_type,
            "team_side": "HOME" if self.team_id == 0 else "AWAY",
            "outcome": outcome,
            "goals_for": our_end,
            "goals_against": their_end,
            "goals_observed_after_tracker_start": goals_observed_for,
            "goals_conceded_after_tracker_start": goals_observed_against,
            "possession_percent": possession_pct,
            "average_attack_axis_ball_x": round(avg_territory, 1),
            "average_nearest_pressure": round(avg_pressure, 1),
            "turnovers_won": self.turnovers_won,
            "turnovers_lost": self.turnovers_lost,
            "command_mix": command_mix,
            "lessons": lessons,
        }
        return json.dumps(data, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class MatchTransition:
    started: bool = False
    completed_summary: str | None = None
    match_key: str = ""
    completed_match_key: str | None = None


class MatchTracker:
    """Small per-runtime tracker; memory use stays bounded across a match."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.current: MatchStats | None = None
        self.last_tick: float | None = None
        self.last_game_time: float | None = None
        self.finalized = False

    def observe(self, prompt_data: dict, game_state: dict, team_id: int) -> MatchTransition:
        tick = _safe_number(game_state.get("tick", 0))
        game_time = _safe_number(game_state.get("gameTime", tick))
        explicit_id = _explicit_match_id(prompt_data, game_state)
        reset_clock = (
            self.current is not None
            and (
                (self.last_tick is not None and tick + 2 < self.last_tick)
                or (self.last_game_time is not None and game_time + 2 < self.last_game_time)
            )
        )
        changed_id = bool(
            self.current is not None
            and explicit_id
            and explicit_id != self.current.match_key
        )

        completed = None
        completed_match_key = None
        started = False
        if self.current is None or reset_clock or changed_id:
            if self.current is not None and not self.finalized:
                completed_match_key = self.current.match_key
                completed = self.current.summary()
            match_key = explicit_id or f"local-{uuid.uuid4().hex}"
            home, away = _score(game_state)
            match_type = str(
                prompt_data.get("matchType")
                or ("practice" if prompt_data.get("isPractice") else "competitive")
            )
            self.current = MatchStats(
                match_key=match_key,
                team_id=team_id,
                match_type=match_type,
                start_home=home,
                start_away=away,
                end_home=home,
                end_away=away,
            )
            self.finalized = False
            started = True

        self.current.observe(game_state, self.player_id)
        self.last_tick = tick
        self.last_game_time = game_time

        if _is_terminal(prompt_data, game_state) and not self.finalized:
            completed_match_key = self.current.match_key
            completed = self.current.summary()
            self.finalized = True

        return MatchTransition(
            started=started,
            completed_summary=completed,
            match_key=self.current.match_key,
            completed_match_key=completed_match_key,
        )

    def record_command(self, command: dict) -> None:
        if self.current is not None:
            self.current.record_command(command)

    def current_context(self) -> str:
        """Compact same-match context without persisting every model turn."""
        if self.current is None:
            return ""
        contested = self.current.possession_ticks + self.current.opponent_possession_ticks
        possession = round(100 * self.current.possession_ticks / contested) if contested else 0
        return (
            f"Current match: possession={possession}% "
            f"turnoversWon={self.current.turnovers_won} "
            f"turnoversLost={self.current.turnovers_lost} "
            f"recentCommandMix={dict(self.current.commands.most_common(4))}"
        )


class AgentCoreMatchMemory:
    """One-read/one-write match-level adapter around AgentCore Memory."""

    def __init__(self, memory_id: str | None = None, deployment_team: str | None = None):
        self.memory_id = memory_id or os.environ.get("MEMORY_TEAM_MEMORY_ID") or os.environ.get("MEMORY_ID")
        self.deployment_team = deployment_team or os.environ.get("TEAM_ID", "default-team")
        self.region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self._runtime_client = None
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.memory_id)

    def actor_id(self, team_id: int) -> str:
        # HOME/AWAY changes between fixtures; use one deployment-level actor so
        # lessons learned on either side are available in every future match.
        raw = f"football-{self.deployment_team}"
        return re.sub(r"[^A-Za-z0-9_-]", "-", raw)[:100]

    def _client(self):
        if self._runtime_client is None:
            import boto3

            kwargs = {"region_name": self.region} if self.region else {}
            self._runtime_client = boto3.client("bedrock-agentcore", **kwargs)
        return self._runtime_client

    def retrieve(self, team_id: int, top_k: int = 3) -> str:
        """Retrieve relevant episodes once at match start; failures are non-fatal."""
        if not self.enabled:
            return ""
        actor = self.actor_id(team_id)
        try:
            client = self._client()
            request = dict(
                memoryId=self.memory_id,
                searchCriteria={
                    "searchQuery": (
                        "5v5 football match tactics, opponent pressing, passing lanes, "
                        "turnovers, scoring chances, defensive shape, and lessons"
                    ),
                    "topK": max(1, min(int(top_k), 10)),
                },
                maxResults=max(1, min(int(top_k), 10)),
            )
            # Newer SDKs expose namespacePath for all session children. Older
            # SDKs only expose namespace; that exact actor namespace still
            # retrieves the cross-match reflection records.
            try:
                members = client.meta.service_model.operation_model(
                    "RetrieveMemoryRecords"
                ).input_shape.members
            except (AttributeError, KeyError):
                members = {}
            namespace_key = "namespacePath" if "namespacePath" in members else "namespace"
            request[namespace_key] = f"/episodes/{actor}"
            response = client.retrieve_memory_records(**request)
            memories = []
            for record in response.get("memoryRecordSummaries", []):
                text = record.get("content", {}).get("text")
                if text:
                    memories.append(text.strip())
            return "\n".join(f"- {item}" for item in memories)[:2400]
        except Exception as exc:  # memory must never cost us a match command
            self.last_error = f"retrieve failed: {exc}"
            return ""

    def store_episode(self, team_id: int, match_key: str, summary: str) -> bool:
        """Write one deterministic completed-match event for episodic extraction."""
        if not self.enabled or not summary:
            return False
        actor = self.actor_id(team_id)
        digest = hashlib.sha256(f"{actor}:{match_key}".encode("utf-8")).hexdigest()
        session_id = f"football-match-{digest[:40]}"
        try:
            self._client().create_event(
                memoryId=self.memory_id,
                actorId=actor,
                sessionId=session_id,
                eventTimestamp=datetime.now(timezone.utc),
                clientToken=digest[:64],
                payload=[
                    {
                        "conversational": {
                            "content": {"text": "Completed match telemetry: " + summary},
                            "role": "USER",
                        }
                    },
                    {
                        "conversational": {
                            "content": {
                                "text": (
                                    "Match episode complete. Extract the outcome, tactical choices, "
                                    "failures, successes, and reusable lessons for future matches."
                                )
                            },
                            "role": "ASSISTANT",
                        }
                    },
                ],
            )
            return True
        except Exception as exc:  # the rule engine remains fully operational
            self.last_error = f"store failed: {exc}"
            return False
