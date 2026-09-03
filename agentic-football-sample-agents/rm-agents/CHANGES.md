# RM_Agents — change log

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
