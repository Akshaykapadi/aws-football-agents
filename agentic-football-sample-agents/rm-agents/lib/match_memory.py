"""RM_Agents memory — short-term match tracking and long-term learning, never on the tick path.

STM  MatchTracker — in-process, per runtime session. Watches score, time and possession every
     tick, counts the shots the instinct layer takes and the goals for/against, and turns that
     into DYNAMIC DOCTRINE: `adjust()` returns a FallbackConfig tuned to the match situation
     (losing → wider shot gate; two up → tighter and deeper; conceded early → transition guard;
     last minute → volume). It also LEARNS shooting from the ball itself: the tick after a shot,
     the ball's position + velocity give the y where it crosses the goal line and its height → whether
     the corner letters point the right way (flip), whether T shots sail high (prefer_low), and which
     distance bands never hit the target (banned → carry closer). Zero latency — pure arithmetic.

LTM  MemoryStore — Amazon Bedrock AgentCore Memory. Every notable moment (match start, goal for,
     goal against, aim flip, 60-second snapshots) is written as an event to the actor's "career"
     session in a background thread. At the start of each match the previous matches' events are
     read back (also in a background thread) and turned into PRIORS: the aim mapping that scored,
     whether long shots scored (shot-gate bonus), whether we tend to concede early (start with the
     transition guard on). Extracted long-term records (SEMANTIC strategy on the memory resource)
     are retrieved as one or two LESSONS lines for the LLM briefing.

The memory id arrives as MEMORY_RM_MEMORY_ID (injected by the CDK stack for the "rm_memory"
resource in agentcore/agentcore.json). Without it — local tests — the store is a no-op.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace

MIN_SHOTS_TO_JUDGE = 3
GOAL_HALF_WIDTH = 5.0
HIGH_Z = 3.0
EARLY_CONCEDE_SECONDS = 90.0
LATE_MATCH_SECONDS = 240.0
SNAPSHOT_EVERY_SECONDS = 60.0
CAREER_SESSION = "career"


# ---------------------------------------------------------------------------
# LTM — AgentCore Memory client (background threads; every call is best-effort)
# ---------------------------------------------------------------------------

class MemoryStore:
    """Thin wrapper over bedrock_agentcore.memory.MemoryClient. All I/O happens off-thread."""

    def __init__(self, position_label: str, log=None, memory_id: str | None = None, client=None):
        self.actor_id = f"rm-{position_label}"
        self.log = log
        self.memory_id = memory_id or _memory_id_from_env()
        self._client = client
        self.enabled = bool(self.memory_id)

    # -- client -------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            from bedrock_agentcore.memory import MemoryClient
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            self._client = MemoryClient(region_name=region)
        return self._client

    def _warn(self, msg):
        if self.log:
            self.log.warn(f"memory: {msg}")

    # -- writes -------------------------------------------------------------
    def record(self, event: dict) -> None:
        """Append one event to the actor's career session (fire-and-forget)."""
        if not self.enabled:
            return
        threading.Thread(target=self._record, args=(event,), daemon=True).start()

    def _record(self, event: dict) -> None:
        try:
            self._get_client().create_event(
                memory_id=self.memory_id, actor_id=self.actor_id, session_id=CAREER_SESSION,
                messages=[(json.dumps(event, separators=(",", ":")), "ASSISTANT")],
            )
        except Exception as e:
            self._warn(f"create_event failed: {e}")

    # -- reads --------------------------------------------------------------
    def load_priors_async(self, on_ready) -> None:
        """Read past events + extracted records in a thread; call on_ready(priors, lessons)."""
        if not self.enabled:
            return
        threading.Thread(target=self._load, args=(on_ready,), daemon=True).start()

    def _load(self, on_ready) -> None:
        events, lessons = [], []
        try:
            events = self.fetch_events()
        except Exception as e:
            self._warn(f"list_events failed: {e}")
        try:
            lessons = self.fetch_lessons()
        except Exception as e:
            self._warn(f"retrieve_memories failed: {e}")
        try:
            on_ready(priors_from_events(events), lessons)
        except Exception as e:
            self._warn(f"applying priors failed: {e}")

    def fetch_events(self) -> list[dict]:
        raw = self._get_client().list_events(
            memory_id=self.memory_id, actor_id=self.actor_id, session_id=CAREER_SESSION,
            max_results=100, include_payload=True,
        )
        return parse_event_payloads(raw)

    def fetch_lessons(self) -> list[str]:
        records = self._get_client().retrieve_memories(
            memory_id=self.memory_id, namespace=f"/rm/lessons/{self.actor_id}",
            query="what shooting, pressing or defending decisions scored or conceded goals", top_k=2,
        )
        out = []
        for r in records or []:
            text = (r.get("content") or {}).get("text") if isinstance(r, dict) else None
            if text:
                out.append(re.sub(r"\s+", " ", text.strip())[:140])
        return out


def _memory_id_from_env() -> str | None:
    if os.environ.get("MEMORY_RM_MEMORY_ID"):
        return os.environ["MEMORY_RM_MEMORY_ID"]
    for k, v in os.environ.items():
        if k.startswith("MEMORY_") and k.endswith("_ID") and v:
            return v
    return os.environ.get("MEMORY_ID")


def parse_event_payloads(raw_events: list) -> list[dict]:
    """Our events are one JSON object per conversational message. Ignore anything else."""
    out = []
    for ev in raw_events or []:
        for item in ev.get("payload", []) if isinstance(ev, dict) else []:
            text = ((item.get("conversational") or {}).get("content") or {}).get("text")
            if not text:
                continue
            try:
                obj = json.loads(text)
            except ValueError:
                continue
            if isinstance(obj, dict) and "type" in obj:
                out.append(obj)
    return out


def priors_from_events(events: list[dict]) -> dict:
    """Turn past matches' events into starting settings for this one."""
    priors: dict = {}
    goals = [e for e in events if e.get("type") == "goal_from_shot"]
    if goals:
        last = max(goals, key=lambda e: e.get("ts", 0))
        priors["aim_flip"] = bool(last.get("flip", False))
        priors["aim_locked"] = True
        if any(e.get("dist", 0) >= 24 for e in goals):
            priors["shoot_bonus"] = 4.0
    starts = sorted((e for e in events if e.get("type") == "match_start"), key=lambda e: e.get("ts", 0))
    recent = {e.get("session") for e in starts[-3:]}
    early = {e.get("session") for e in events if e.get("type") == "goal_against" and e.get("early")}
    if recent and len(recent & early) >= 1:
        priors["guard_transitions"] = True
    results = [e for e in events if e.get("type") == "shot_result"]
    if results:
        recent = results[-12:]
        matches = [e for e in recent if "side_match" in e]
        if len(matches) >= MIN_SHOTS_TO_JUDGE:
            rate = sum(1 for e in matches if e["side_match"]) / len(matches)
            last_flip = bool(recent[-1].get("flip", False))
            if rate <= 0.34:
                priors["aim_flip"] = not last_flip
            elif rate >= 0.66:
                priors["aim_flip"] = last_flip
                priors["aim_locked"] = True
        tops = [e for e in recent if e.get("tb") == "T"]
        if len(tops) >= MIN_SHOTS_TO_JUDGE and sum(1 for e in tops if e.get("high")) / len(tops) >= 0.67:
            priors["prefer_low"] = True
        stats: dict = {}
        for e in results:
            b = str(e.get("band", 3)); st = stats.setdefault(b, [0, 0])
            st[0] += int(bool(e.get("on_target"))); st[1] += 1
        priors["band_stats"] = stats
        landing: dict = {}
        for e in results:
            if e.get("aim") and "y_goal" in e:
                l = landing.setdefault(e["aim"], [0.0, 0]); l[0] += float(e["y_goal"]); l[1] += 1
        priors["aim_landing"] = landing
        priors["on_target_rate"] = round(sum(1 for e in results if e.get("on_target")) / len(results), 2)
    snaps = [e for e in events if e.get("type") == "snapshot"]
    if snaps:
        priors["past_matches"] = len(recent)
        priors["past_shots"] = max(e.get("shots", 0) for e in snaps)
    return priors


# ---------------------------------------------------------------------------
# STM — match tracker + dynamic doctrine
# ---------------------------------------------------------------------------

@dataclass
class MatchTracker:
    position_label: str
    my_player_id: int
    store: MemoryStore | None = None
    log: object = None

    session_id: str | None = None
    team_id: int = 0
    game_time: float = 0.0
    score_for: int = 0
    score_against: int = 0
    shots: list = field(default_factory=list)
    shots_since_goal: int = 0
    aim_flip: bool = False
    aim_locked: bool = False
    prefer_low: bool = False
    pending_shot: dict | None = None
    side_obs: list = field(default_factory=list)      # (intended_side, observed_sign)
    high_obs: list = field(default_factory=list)      # (tb_letter, went_high)
    band_obs: dict = field(default_factory=dict)      # band -> [on_target, total]
    aim_landing: dict = field(default_factory=dict)   # aim letters -> [sum_y_at_goal, count]  (this match)
    conceded_early: bool = False
    llm_ticks: int = 0
    fallback_ticks: int = 0
    instinct_ticks: int = 0
    priors: dict = field(default_factory=dict)
    lessons: list = field(default_factory=list)
    _last_snapshot: float = -1e9
    _seen_score: bool = False
    _cache_key: tuple = ()
    _cache_cfg: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- per tick -----------------------------------------------------------
    def observe(self, game_state: dict, team_id: int, session_id: str | None) -> None:
        t = float(game_state.get("gameTime", 0) or 0)
        new_match = (session_id and session_id != self.session_id) or t + 30 < self.game_time
        if new_match or not self._seen_score:
            if new_match and self._seen_score:
                self._snapshot(force=True)
            self._start_match(session_id, t, team_id)
        self.team_id = team_id
        self.game_time = t

        score = game_state.get("score") or {}
        mine = int(score.get("home", 0) if team_id == 0 else score.get("away", 0))
        theirs = int(score.get("away", 0) if team_id == 0 else score.get("home", 0))
        if mine > self.score_for:
            self._goal_for(t)
        if theirs > self.score_against:
            self._goal_against(t)
        self.score_for, self.score_against = mine, theirs
        self._seen_score = True

        if self.pending_shot is not None:
            self._judge_shot(game_state, team_id)

        if t - self._last_snapshot >= SNAPSHOT_EVERY_SECONDS:
            self._snapshot()

    def _start_match(self, session_id, t, team_id):
        self.session_id = session_id or f"local-{int(time.time())}"
        self.game_time = t
        self.score_for = self.score_against = 0
        self.shots = []
        self.shots_since_goal = 0
        self.conceded_early = False
        self.llm_ticks = self.fallback_ticks = self.instinct_ticks = 0
        self._last_snapshot = t
        self._cache_key = ()
        self.aim_flip = bool(self.priors.get("aim_flip", self.aim_flip))
        self.aim_locked = bool(self.priors.get("aim_locked", self.aim_locked))
        self.prefer_low = bool(self.priors.get("prefer_low", self.prefer_low))
        self.pending_shot = None
        self.side_obs, self.high_obs, self.band_obs, self.aim_landing = [], [], {}, {}
        self._emit({"type": "match_start", "session": self.session_id, "team": team_id})
        if self.store and not self.priors:
            self.store.load_priors_async(self._apply_priors)

    def _apply_priors(self, priors: dict, lessons: list) -> None:
        with self._lock:
            self.priors = dict(priors or {})
            self.lessons = list(lessons or [])
            if "aim_flip" in self.priors and not self.shots:
                self.aim_flip = bool(self.priors["aim_flip"])
                self.aim_locked = bool(self.priors.get("aim_locked", False))
            self.prefer_low = bool(self.priors.get("prefer_low", self.prefer_low))
            self._cache_key = ()
        self._info(f"priors applied {self.priors} lessons={len(self.lessons)}")

    def _goal_for(self, t):
        last = self.shots[-1] if self.shots else None
        if last and t - last["t"] <= 8:
            self.aim_locked = True
            self._emit({"type": "goal_from_shot", "t": t, "dist": last["dist"], "aim": last["aim"],
                        "flip": self.aim_flip, "score": [self.score_for + 1, self.score_against]})
        else:
            self._emit({"type": "goal_for", "t": t, "score": [self.score_for + 1, self.score_against]})
        self.shots_since_goal = 0

    def _goal_against(self, t):
        early = t <= EARLY_CONCEDE_SECONDS
        if early:
            self.conceded_early = True
        self._emit({"type": "goal_against", "t": t, "early": early, "session": self.session_id,
                    "score": [self.score_for, self.score_against + 1]})
        self._cache_key = ()

    def note_shot(self, cmd: dict, pos: dict, opp_goal_x: float) -> None:
        params = cmd.get("parameters", {})
        aim = params.get("aim_location") or "BR"
        d = abs(pos.get("x", 0) - opp_goal_x)
        left_is_pos_y = opp_goal_x > 0
        side = 1 if (aim[0] == "T") else -1          # intended y side under the T=+y convention
        if aim[0] == "B" and self.prefer_low:          # prefer_low makes the side live in L/R
            side = 1 if ((aim[1] == "L") == left_is_pos_y) else -1
        if self.aim_flip:
            side = -side
        self.shots.append({"t": self.game_time, "dist": round(d, 1), "aim": aim, "power": params.get("power"),
                           "flip": self.aim_flip, "x": pos.get("x", 0), "y": pos.get("y", 0)})
        self.shots_since_goal += 1
        self.pending_shot = {"t": self.game_time, "side": side, "tb": aim[0], "aim": aim, "dist": d,
                             "opp_goal_x": opp_goal_x, "from": dict(pos)}

    def _judge_shot(self, game_state: dict, team_id: int) -> None:
        """Next tick after a shot: where is the ball crossing the goal line, and how high?"""
        shot = self.pending_shot
        ball = game_state.get("ball") or {}
        bp = ball.get("position") or {}
        bv = ball.get("velocity") or {}
        gx = shot["opp_goal_x"]
        d = 1.0 if gx > 0 else -1.0
        vx = float(bv.get("x", 0) or 0)
        result = None
        if vx * d > 1.0:                                   # still travelling at goal → extrapolate
            tt = (gx - bp.get("x", 0)) / vx
            if tt >= 0:
                y_goal = bp.get("y", 0) + float(bv.get("y", 0) or 0) * tt
                z_goal = float(bp.get("z", 0) or 0) + float(bv.get("z", 0) or 0) * tt
                result = {"y": y_goal, "high": z_goal > HIGH_Z}
        elif (bp.get("x", 0) - gx) * d >= -6.0:            # already at/over the line (or dead near it)
            result = {"y": bp.get("y", 0), "high": float(bp.get("z", 0) or 0) > HIGH_Z}
        self.pending_shot = None
        if result is None:
            self._info(f"shot not judged: ball at ({bp.get('x', 0):.1f},{bp.get('y', 0):.1f},{bp.get('z', 0) or 0:.1f}) "
                       f"vel=({vx:.1f},{float(bv.get('y', 0) or 0):.1f}) holder={find_holder_idx(game_state)}")
            return                                         # blocked / tackled / reset — no lesson
        on_target = abs(result["y"]) <= GOAL_HALF_WIDTH and not result["high"]
        band = self._band(shot["dist"])
        stat = self.band_obs.setdefault(band, [0, 0])
        stat[0] += int(on_target); stat[1] += 1
        observed_sign = 1 if result["y"] > 0 else -1
        self.side_obs.append((shot["side"], observed_sign))
        land = self.aim_landing.setdefault(shot["aim"], [0.0, 0])
        land[0] += float(result["y"]); land[1] += 1
        self.high_obs.append((shot["tb"], bool(result["high"])))
        self._learn()
        self._emit({"type": "shot_result", "t": self.game_time, "dist": round(shot["dist"], 1), "band": band, "aim": shot["aim"],
                    "on_target": on_target, "y_goal": round(result["y"], 1), "high": result["high"],
                    "side_match": shot["side"] == observed_sign, "tb": shot["tb"], "flip": self.aim_flip})
        self._info(f"shot judged: dist={shot['dist']:.0f} y_at_goal={result['y']:.1f} high={result['high']} on_target={on_target}")

    @staticmethod
    def _band(d):
        return 0 if d < 12 else (1 if d < 25 else (2 if d < 40 else 3))

    def _learn(self) -> None:
        """Turn observations into settings: corner mapping, low preference, banned distance bands."""
        if len(self.side_obs) >= MIN_SHOTS_TO_JUDGE and not self.aim_locked:
            recent = self.side_obs[-6:]
            mismatch = sum(1 for a, b in recent if a != b) / len(recent)
            if mismatch >= 0.67:
                self.aim_flip = not self.aim_flip
                self.side_obs = []
                self._emit({"type": "aim_flip", "t": self.game_time, "flip": self.aim_flip})
                self._info(f"shots landing on the keeper's side — flipping corner mapping to flip={self.aim_flip}")
            elif mismatch <= 0.34:
                self.aim_locked = True
        tops = [h for tb, h in self.high_obs[-6:] if tb == "T"]
        if len(tops) >= MIN_SHOTS_TO_JUDGE and sum(tops) / len(tops) >= 0.67 and not self.prefer_low:
            self.prefer_low = True
            self._info("T shots sail high — keeping every shot low (B)")
        self._cache_key = ()

    def aim_map(self) -> dict:
        """aim letters -> mean observed y at the goal line (this match merged with LTM priors)."""
        out = {}
        prior = self.priors.get("aim_landing", {})
        for aim in set(prior) | set(self.aim_landing):
            ps, pn = prior.get(aim, [0.0, 0])
            ls, ln = self.aim_landing.get(aim, [0.0, 0])
            if pn + ln:
                out[aim] = {"y": round((ps + ls) / (pn + ln), 1), "n": pn + ln}
        return out

    def banned_bands(self) -> tuple:
        out = []
        for band, (hit, total) in self.band_obs.items():
            prior = self.priors.get("band_stats", {}).get(str(band), [0, 0])
            h, n = hit + prior[0], total + prior[1]
            if n >= MIN_SHOTS_TO_JUDGE and h == 0 and band > 0:
                out.append(band)
        for band, (h, n) in self.priors.get("band_stats", {}).items():
            if int(band) not in self.band_obs and n >= MIN_SHOTS_TO_JUDGE and h == 0 and int(band) > 0:
                out.append(int(band))
        return tuple(sorted(set(out)))

    def note_tick(self, kind: str) -> None:
        if kind == "llm":
            self.llm_ticks += 1
        elif kind == "fallback":
            self.fallback_ticks += 1
        else:
            self.instinct_ticks += 1

    # -- dynamic doctrine ---------------------------------------------------
    def adjust(self, cfg):
        """FallbackConfig for this tick, derived from the match situation (memoised per situation)."""
        diff = self.score_for - self.score_against
        late = self.game_time >= LATE_MATCH_SECONDS
        guard = self.conceded_early or bool(self.priors.get("guard_transitions"))
        banned = self.banned_bands()
        amap = self.aim_map()
        key = (id(cfg), diff if -1 <= diff <= 2 else (2 if diff > 2 else -1), late, guard,
               self.aim_flip, self.prefer_low, banned, self.priors.get("shoot_bonus", 0),
               tuple(sorted((k, v["y"], v["n"]) for k, v in amap.items())))
        if key == self._cache_key and self._cache_cfg is not None:
            return self._cache_cfg

        changes: dict = {"aim_flip": self.aim_flip, "prefer_low": self.prefer_low, "banned_bands": banned,
                         "aim_map": amap}
        bonus = float(self.priors.get("shoot_bonus", 0))
        if diff < 0:
            bonus += 5.0
            changes["pressure_shoot_distance"] = cfg.pressure_shoot_distance + 5
            if cfg.role == "FWD":
                changes["support_depth"] = max(10.0, cfg.support_depth - 2)
            if cfg.role == "DEF":
                changes["support_x_factor"] = min(cfg.support_x_factor, 0.35)
        elif diff >= 2:
            bonus -= 4.0
            if cfg.role == "DEF":
                changes["support_x_factor"] = max(cfg.support_x_factor, 0.6)
            if cfg.role == "MID":
                changes["support_depth"] = cfg.support_depth + 6
        if guard:
            if cfg.role == "DEF":
                changes["support_x_factor"] = max(changes.get("support_x_factor", cfg.support_x_factor), 0.6)
            if cfg.role == "MID":
                changes["support_depth"] = max(changes.get("support_depth", cfg.support_depth), 32.0)
        if late and diff <= 0:
            bonus += 8.0
            changes["pressure_shoot_distance"] = changes.get("pressure_shoot_distance", cfg.pressure_shoot_distance) + 6
        if cfg.shoot_threshold > 0 and bonus:
            changes["shoot_threshold"] = max(18.0, cfg.shoot_threshold + bonus)

        self._cache_cfg = replace(cfg, **changes)
        self._cache_key = key
        return self._cache_cfg

    def briefing_extra(self) -> str:
        """One or two lines for the LLM: match situation + lessons from past matches."""
        diff = self.score_for - self.score_against
        lines = []
        if diff < 0:
            lines.append("SITUATION: we are LOSING — press higher, win the ball in their half.")
        elif diff >= 2:
            lines.append("SITUATION: protecting a lead — hold shape, no risky tackles.")
        if self.conceded_early:
            lines.append("SITUATION: they scored early on a transition — stay goal-side, mark the runner.")
        if self.lessons:
            lines.append("LESSONS FROM PAST MATCHES:")
            lines.extend(f"  - {l}" for l in self.lessons[:2])
        return ("\n" + "\n".join(lines)) if lines else ""

    # -- memory writes ------------------------------------------------------
    def _snapshot(self, force: bool = False) -> None:
        self._last_snapshot = self.game_time
        self._emit({"type": "snapshot", "t": self.game_time, "session": self.session_id,
                    "score": [self.score_for, self.score_against], "shots": len(self.shots),
                    "flip": self.aim_flip, "locked": self.aim_locked, "low": self.prefer_low,
                    "bands": self.band_obs, "landing": self.aim_landing, "early": self.conceded_early,
                    "llm": self.llm_ticks, "fallback": self.fallback_ticks, "instinct": self.instinct_ticks,
                    "final": force})

    def _emit(self, event: dict) -> None:
        event = dict(event, pos=self.position_label, ts=time.time())
        if self.store:
            self.store.record(event)

    def _info(self, msg: str) -> None:
        if self.log:
            self.log.info(f"{self.position_label} memory: {msg}")


def find_holder_idx(game_state: dict):
    from state import find_holder, _player_idx
    h = find_holder(game_state.get("ball") or {}, game_state.get("players", []))
    return None if h is None else _player_idx(h)
