# RM_Agents — change log

## v6 — 2026-09-03 — aim CENTER (iteration 5)

AGENT_PROTOCOL.md: goals at (±55, 0), mouth y −5..5, "corner labels have no published mapping to goalmouth
geometry". CENTER is the only aim on target by construction, so it is the default from anywhere; a corner
(away from the keeper) only inside 15 when the keeper is ≥ 1.5 off centre. Power 0.8 / 0.9 / 1.0 by distance.
The landing map (aim → observed y at the goal line, STM + LTM) overrides once a corner is proven to land inside;
long-range CENTER rule and all-corners-wide → CENTER kept. Shot frequency and shoot-from-anywhere unchanged.


## v5 — 2026-09-03 — shot accuracy learned from the ball (iteration 4)

Input: match with 35 SHOOT commands → 8 shots, 0 on target.

- On the ball the LLM is asked with a SHOT TOOL line (keeper x,y + velocity lead, open side, lane, aim, power by
  distance band 0.7 / 0.85 / 1.0 / 1.0); if it is late the tool's shot is taken. SHOOT back in the command list.
- lib/match_memory.py judges every shot on the next tick from the ball's position + velocity: y where it crosses the
  goal line (on target if |y| ≤ 5 and not high). Learns: corner mapping flip when ≥ 2/3 of shots land on the intended
  side's opposite; prefer_low when T shots sail high; distance bands with 0 on-target from ≥ 3 shots are banned →
  carry 12 closer instead. All of it persisted as `shot_result` events; priors rebuild band stats / mapping next match.
- Shooting still from anywhere (no positional optimisation), except banned bands.


## v4 — 2026-09-03 — all-out attack (iteration 3)

Input: first RM_Agents match logs — 8 shots, forwards parked 276 ticks on a static fallback spot, 5 backward passes
to the DEF, Nova Lite p50 548 ms in-region so the 0.40 s timeout cut 671/700 model calls. Audit of the sample
"extremely aggressive" team (Total Attack's blueprint): shoot from ~40 at power 1.0, DEF joins attacks, press at 1.0.

- lib/tools.py (new): `open_positions()` ranks candidate spots by space, lane from the carrier, progress, shot gate,
  crowding; `shot_opportunity()` = gap (no blocker, keeper gap ≥ 2) + power by distance (0.8 / 0.95 / 1.0).
  Both also exposed as Strands @tools (`find_open_position`, `plan_shot`) — not registered on the match agents
  because a tool call is two model turns (~1.1 s) against a 500 ms budget; their results go into the briefing.
- lib/fallback.py: gates FWD 45/22, MID 40/18, DEF 35/15 — shoot the moment there is a gap (close range or pressed:
  regardless); pressed inside 50 → shoot; outlets must be ahead of the carrier (backward only from our own third);
  support runs = tool's best open position (dynamic, wing-biased); defensive fallback = nearest presses at 1.0,
  second cuts the lane, others man-mark, DEF always marks the deepest opponent.
- lib/agent_base.py: hybrid positioning — off the ball while we attack the LLM is asked with OPEN POSITIONS in the
  briefing; if late, the tool's best spot is used. Timeout 0.55 s (Lite p50) — replies over the server's 500 ms
  are the trade-off for keeping Nova Lite.
- Prompts: two moments (A teammate has ball → pick an open position; B opponent has ball → press 1.0 / lane / mark).
- test_rm_agents.py: 38 tests.


## v3 — 2026-09-03 — shoot-first, accurate, remembering (iteration 2)

Input: three practice matches (Fort Knox Athletic 0-1, Total Attack United 0-4, The Benchmark FC 1-0),
coach's-corner feedback, portal latency panel, and the runtimes' CloudWatch logs (~600 ticks/agent).

Findings
- 0 shots in all three matches. Logs: forwards returned 0 SHOOT; pressure rule fired 228 times.
- LLM ticks 660-880 ms vs the server's 500 ms budget (opponents ran 430-490 ms).
- No parse failures, no errors — the model obeyed; the doctrine was wrong.

Changes
- lib/fallback.py: instinct layer now decides every on-ball tick (SHOOT in range, pressed-in-attacking-third
  SHOOT, else release/carry), every possession tick (forwards into the box, DEF holds, MID second wave),
  the keeper always, and loose-ball intercepts. `aim_corner()` aims away from the opposing keeper.
  Pressure distance 6 → 4. Shot gates widened (FWD 30/18, MID 28/15, DEF 22/12). Last resort = CLEAR_OVERRIDE.
- lib/agent_base.py: LLM only in the defensive phase; hard timeout (RM_LLM_TIMEOUT_S, 0.40 s) with
  rule-based fallback and a lock so a slow call never overlaps the next tick; window 2, max_tokens 64,
  temperature 0.1; per-tick one-line logs (instinct / LLM ms / fallback).
- lib/fallback.py `shot_plan()`: shot accuracy from game state — keeper position led by velocity, defenders on the
  shot line, far-post bonus; sidestep before shooting into a block (unless pressed or inside 10).
- lib/match_memory.py (new): STM MatchTracker re-tunes the doctrine per tick (score, time, early concession) and
  learns the corner mapping (flip after 4 scoreless shots, lock on a goal); LTM MemoryStore writes events to
  AgentCore Memory off-thread and seeds priors + LESSONS lines from past matches at the next match start.
- agentcore/agentcore.json: `rm_memory` memory resource (30-day events, SEMANTIC + SUMMARIZATION strategies).
- lib/parsing.py: fills aim_location/power for SHOOT, type for PASS, duration for maintained commands.
- ai-*/src/main.py: Nova Lite everywhere (per request; the hard timeout keeps replies in budget). Lite answered `[]`
  to the v3 prompt until the prompt said an empty array is never valid and the briefing ended with a one-line nudge; prompts cut to the defensive phase (~2.3k chars),
  SET_STANCE removed (engine docs: 1=DEFENSIVE, 2=ATTACKING — the stock prompts had it backwards).
- test_rm_agents.py: 34 tests; ai-*/test_local.py: instinct scenarios + `--llm` latency probe.
- Self-contained team directory; AgentCore project `RM`, runtimes `Agents_*` → deployed as `RM_Agents_*`.

## v2 — 2026-09-03 — counter-press (iteration 1)

Side-aware prompts, pressure rule (release at 6), shot geometry (|y| ≤ 15), instinct release layer,
parser lift for flattened parameters, latency caps (window 12, max_tokens 150). Deployed as
TeamStrandsBalanced. Result: 0 shots in three matches — superseded by v3.
