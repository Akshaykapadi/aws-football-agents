# RM_Agents — Shoot-First Playbook v3

Five single-player agents for the 5v5 Agentic Football match server, built with the
[Strands Agents SDK](https://github.com/strands-agents/sdk-python) and deployed to
[Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/) as AgentCore project `RM`
(runtimes `RM_Agents_gk`, `RM_Agents_def`, `RM_Agents_mid`, `RM_Agents_fwd1`, `RM_Agents_fwd2`;
the CLI requires an alphanumeric project name, so the `RM_Agents` prefix is project + runtime name).

This directory is self-contained: `lib/` lives here, so nothing in the sibling teams changes.

## What the practice matches taught us (2026-09-03)

| Opponent | Result | Shots for/against | Lesson |
|---|---|---|---|
| Fort Knox Athletic (defensive) | 0-1 | 0 / 5 | 58% possession, zero shots. Passive marking and pressing; the attack never fired. |
| Total Attack United (attacking) | 0-4 | 0 / 12 | Conceded 4 inside two minutes on transitions; our defender chased play, keeper wandered. |
| The Benchmark FC (balanced) | 1-0 | 0 / 4 | Won on a scramble. Still no shots. |

CloudWatch logs from the runtimes told the real story: across ~600 ticks per agent the forwards
returned **0 SHOOT** commands (FWD1: 372 INTERCEPT, 156 MOVE_TO; FWD2: 258 INTERCEPT, 225 MOVE_TO)
and the "pressure rule" fired 228 times — every touch on a 5v5 pitch has an opponent within 6, so
each touch became a give-away. The LLM ticks ran 660-880 ms against a 500 ms server budget.

## The v3 doctrine

| Edge | Where | What it does |
|---|---|---|
| Shoot in code | `lib/fallback.py` `instinct_command` | On the ball inside the shot gate (FWD 30 / MID 28 / DEF 22 x-units, \|y\| ≤ 18/15/12) → `SHOOT` immediately, 0 ms, no model. Power 0.85 close, 1.0 far. |
| Accurate placement | `shot_plan()` | Scores both sides of the goal from the game state: the keeper's position led by his velocity, defenders standing on the shot line, and a far-post bonus from wide angles. Picks the side the keeper cannot cover; if both lines are blocked and nobody is on the shooter, sidestep 4 units first. Mirrors for AWAY. |
| Learns the corner names | `lib/match_memory.py` | The engine documents TL/TR/BL/BR without axes. Four scoreless shots → the mapping is flipped; a goal locks the mapping that scored and stores it in AgentCore Memory for the next match. |
| Pressed → shoot, not turnover | `pressure_shoot_distance` | Pressed (opponent < 4) inside 38 of goal → shoot at full power. Deeper → THROUGH/GROUND to the most advanced open teammate. Never the keeper. |
| Forwards live in the box | `_support()` | Whenever a teammate has the ball both forwards sprint to 14 units from goal, y ∓7 — they are always in the shot gate when the ball arrives. |
| Defender holds shape | `DEF_CONFIG.support_*` | While we attack the defender sits at 0.45 of our half, tracking ball y. Fixes the Total Attack transitions. |
| Keeper on his line | `_gk()` | Positioning and distribution are code every tick: 3 units off the line, tracks ball y (±4), intercepts loose balls in his third, KICKs to an open forward or THROWs to the freer of DEF/MID. |
| Loose ball | `instinct_command` | Nearest teammate to a free ball → `INTERCEPT`, no model call. |
| LLM only in defence | `lib/agent_base.py` | The model is asked only when the opponent has the ball (press / mark / intercept decisions). Prompts are ~2.3k chars and describe only that phase. |
| Hard timeout | `RM_LLM_TIMEOUT_S` (default 0.40) | If Nova Micro has not answered in time, the rule-based fallback replies and the slow call finishes in the background (agent locked meanwhile). Every tick answers inside the budget. |
| Latency caps | `create_agent` | Nova Lite everywhere, `SlidingWindowConversationManager(2)`, `max_tokens=64`, `temperature=0.1`. Lite answers in ~450-650 ms, so a good share of defensive ticks land on the rule-based fallback — the log line says which. Set `MODEL_ID` to `us.amazon.nova-micro-v1:0` for more model-decided ticks. |
| Short-term memory (STM) | `MatchTracker` | In-process, per session: score, time, shots, goals for/against. Re-tunes the doctrine every tick — losing → shot gate +5 and forwards deeper; two up → tighter gate, defender drops; conceded inside 90 s → transition guard (DEF at 0.6, MID at 32); last minute behind → gate +8. Adds a SITUATION line to the LLM briefing. Zero latency. |
| Long-term memory (LTM) | `MemoryStore` + `agentcore/agentcore.json` `rm_memory` | AgentCore Memory resource (30-day events, SEMANTIC `rm_lessons` + SUMMARIZATION `rm_summaries` strategies). Match start, goals, aim flips and 60 s snapshots are written as events off-thread; at the next match start the actor's career events become priors (aim mapping that scored, long-shot bonus, transition guard) and extracted records become LESSONS lines for the LLM. Nothing on the tick path waits for memory. |

Decision order each tick: instinct (code) → LLM → rule-based fallback → `CLEAR_OVERRIDE`.

## Layout

```
rm-agents/
├── lib/                 # shared code for the five agents (copied into each agent at deploy time)
│   ├── fallback.py      # doctrine: FallbackConfig per position, instinct layer, shot_plan, rule-based fallback
│   ├── match_memory.py  # STM MatchTracker (dynamic doctrine, aim learning) + LTM MemoryStore (AgentCore Memory)
│   ├── agent_base.py    # Strands agent factory + invoke handler (timeout, lock, logging)
│   ├── state.py         # game state → text briefing for the LLM
│   ├── parsing.py       # LLM text → validated commands
│   └── ...
├── ai-gk/ ai-def/ ai-mid/ ai-fwd1/ ai-fwd2/
│   ├── src/main.py      # prompt + doctrine overrides + wiring
│   └── test_local.py    # local tests; --llm calls Nova Micro and prints latency
├── agentcore/           # AgentCore CLI project (CDK); project RM, runtimes Agents_*, memory rm_memory
├── deploy_all.py        # cdk bootstrap → inject lib/ → agentcore deploy
└── test_rm_agents.py    # 34 doctrine tests (shooting, accuracy, pressure, support, keeper, mirror, timeout, memory)
```

## Run

```bash
uv venv --python 3.10 .venv --seed && source .venv/bin/activate
uv pip install strands-agents bedrock-agentcore boto3

python3 test_rm_agents.py             # doctrine — no AWS needed
python3 ai-fwd1/test_local.py         # one position, no AWS
python3 ai-fwd1/test_local.py --llm   # calls Nova Lite: latency vs the 400 ms gate
```

## Deploy

```bash
npm install -g @aws/agentcore aws-cdk       # once
PATH="$HOME/.npm-global/bin:$PATH" AWS_DEFAULT_REGION=us-east-1 python3 deploy_all.py
aws bedrock-agentcore-control list-agent-runtimes --region us-east-1 \
  --query "agentRuntimes[?starts_with(agentRuntimeName,'RM_Agents')].[agentRuntimeName,status,agentRuntimeArn]" --output text
```

`deploy_all.py` copies `lib/` into each agent directory, runs one `agentcore deploy` (five runtimes plus
the `rm_memory` AgentCore Memory resource, whose id reaches every runtime as `MEMORY_RM_MEMORY_ID`),
and removes the copies afterwards. Same command redeploys in place (ARNs unchanged).

## Reading a match afterwards

Every tick logs one line to CloudWatch (`/aws/bedrock-agentcore/runtimes/RM_Agents_*-DEFAULT`):

```
FWD1 memory ON
FWD1 memory: priors applied {'aim_flip': True, 'aim_locked': True} lessons=1
FWD1 instinct SHOOT {'aim_location': 'TL', 'power': 0.85}
FWD1 memory: 4 scoreless shots — flipping aim mapping to flip=True
FWD1 LLM 392ms returned ['PRESS_BALL']
FWD1 LLM exceeded 0.40s — using fallback
FWD1 fallback PRESS_BALL {'intensity': 0.8}
```

Count them to see the shot volume, the share of ticks the model actually decided, and how often
the timeout fired. Tune `shoot_threshold` / `pressure_shoot_distance` in `src/main.py`, rerun the
tests, redeploy the one agent that changed.

## Corner naming caveat

The engine documents `aim_location` as TL/TR/BL/BR without saying which axis is which.
`shot_plan()` derives both letters from the same "away from the keeper" side (top = +y; left
= the shooter's left facing the goal). `MatchTracker` flips that mapping after four scoreless
shots and locks in whichever mapping scores; the locked value is persisted to memory as a
`goal_from_shot` event and applied as a prior in the next match.
