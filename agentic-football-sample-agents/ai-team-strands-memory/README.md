# Learned 5v5 Team — Local Tactics + Selective Nova + Match LTM

Five single-player agents that respond inside the match deadline without sending
every tick through a model. Routine decisions are deterministic and local. Amazon
Nova is called only for an uncertain final-third decision, and AgentCore Memory
stores one completed-match episode so future matches can reuse learned lessons.

## Hot-path design

| Work | Frequency | Network/model cost |
|---|---:|---|
| Velocity, pressure, pass lanes, shape, and role action | Every tick | Local Python only |
| Nova decision | Only when local confidence is low | Background call; the current tick never waits, and only a fresh later result can be used |
| Past-match retrieval | Match start | One asynchronous AgentCore read per player |
| Completed-match write | Match end | One asynchronous event from MID for the whole team |

This keeps the live reaction path local and normally in a few milliseconds; a
remote model response can never push the current command beyond the one-second
limit. There is no AgentCore Gateway or MCP server. The former Gateway calculators are
normal in-process functions, so there is no tool-selection turn or remote tool
round trip. The stateless Strands agent is also cleared after each selective
call; a full match transcript is never replayed to the model.

## Football strategy

- Both HOME and AWAY are normalized to one attack axis before decisions, then
  converted back to world coordinates.
- Ball and player velocity project receivers and blockers before ranking passes.
- Defenders protect the center and mark the deepest threat; midfield holds a
  staggered rest-defense position; forwards occupy separate lanes.
- Under pressure, pass selection scores progression, lane clearance, receiver
  pressure, travel distance, and receiver motion.
- Shooting uses true distance and chooses the side away from the goalkeeper.

## Match memory

`agentcore/agentcore.json` creates `team_memory` with an **EPISODIC** strategy.
During a match, `MatchTracker` keeps only bounded counters such as possession,
territory, turnovers, pressure, score, and command mix. On full time—or when a
new match clock is detected—the midfielder writes one deterministic JSON summary.
Practice matches are marked as practice when the input includes
`isPractice=true` or `matchType=practice`.

AgentCore extracts the episode and cross-match reflections asynchronously. At
the next match start, each role retrieves up to three relevant records in the
background. They are added only to a selective model prompt; memory never blocks
the first command. Clear retrieved lessons also tune conservative local thresholds
(earlier release under pressure, higher support, transition cover, and shot angle),
so learning affects routine ticks without another model call. If memory is
unavailable, the base tactical engine continues.

AgentCore extraction can take a minute or more. An immediate rematch may use
older lessons; the just-completed episode becomes available once extraction
finishes, without delaying live play.

## Architecture

```
agents/
├── lib/                          # Shared library (same as other teams)
└── ai-team-strands-memory/
    ├── ai-gk/                    # Goalkeeper  (player 0) — Nova Micro when needed
    ├── ai-def/                   # Defender    (player 1) — Nova Lite when needed
    ├── ai-mid/                   # Midfielder  (player 2) — Nova Lite when needed
    ├── ai-fwd1/                  # Forward 1   (player 3) — Nova Lite when needed
    ├── ai-fwd2/                  # Forward 2   (player 4) — Nova Lite when needed
    ├── agentcore/                # AgentCore project config (agents + memory)
    │   ├── agentcore.json        # Runtimes + team_memory declaration
    │   └── cdk/                  # CDK app used by `agentcore deploy`
    ├── memory_agent_base_cdk.py  # Stateless/capped model factory (CDK flow)
    ├── deploy_all.py             # Cross-platform deploy script
    ├── destroy_all.py            # Cross-platform teardown script
    └── README.md
```

The five configured runtime names end in `_ak` (`ai_gk_memory_agent_ak`,
`ai_def_memory_agent_ak`, `ai_mid_memory_agent_ak`,
`ai_fwd1_memory_agent_ak`, and `ai_fwd2_memory_agent_ak`) so future deployments
are identifiable as Akshay's agents.

## Prerequisites

- Python 3.10+
- Node.js 20 or 22 and npm (the locally installed AgentCore CLI currently
  throws a JavaScript syntax error under Node 24; `nvm use 20` works)
- AWS CLI configured with valid credentials
- AgentCore CLI and CDK: `npm install -g @aws/agentcore aws-cdk`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — used by the AgentCore CLI to package Python dependencies
- AWS account with Bedrock model access (Nova Micro and Nova Lite by default)
- CDK bootstrap (one-time per account/region): `cdk bootstrap aws://<account-id>/<region>`

Works on macOS, Linux, and Windows (PowerShell) — no WSL required.
No Python packages needed to deploy — `deploy_all.py` uses only the standard library.

## Quick Start

### 1. Deploy (memory + all 5 agents)

```bash
# macOS/Linux
nvm use 20
AWS_DEFAULT_REGION=us-east-1 python deploy_all.py
```

```powershell
# Windows PowerShell
$env:AWS_DEFAULT_REGION = "us-east-1"
python deploy_all.py
```

The script checks prerequisites, writes the deploy target, bootstraps CDK if
needed, stages the shared `lib/` and `memory_agent_base_cdk.py` into each agent
directory, and runs `agentcore deploy --yes`. That single deploy creates the
`team_memory` resource, grants the runtimes access to it, and injects
`MEMORY_TEAM_MEMORY_ID` into every agent — nothing to export or configure.

### Runtime controls

| Variable | Default | Purpose |
|---|---|---|
| `FOOTBALL_LLM_MODE` | `selective` | `off`, `selective`, or `always` |
| `FOOTBALL_LLM_TIMEOUT_MS` | `2200` | Background model timeout; it never delays the current tick |
| `FOOTBALL_LLM_MIN_TICK_GAP` | `6` | Per-player model-call rate limit |
| `FOOTBALL_LLM_MAX_AGE_TICKS` | `3` | Discard a completed decision after this many ticks |
| `FOOTBALL_LOCAL_CONFIDENCE` | `0.72` | Model is considered only below this local confidence |
| `FOOTBALL_MEMORY_WRITER` | `MID` | The only role that writes the shared match episode |
| `FOOTBALL_MODEL_ID` | role default | Override the model for every role |
| `FOOTBALL_MODEL_ID_GK`, `..._MID`, `..._FWD1` | unset | Override one role |

Nova Lite is the balanced default for the outfield decision points. For the
lowest possible latency/cost, set `FOOTBALL_MODEL_ID` to
`us.amazon.nova-micro-v1:0`. If Nova 2 Lite is enabled in the deployment Region,
set it to `us.amazon.nova-2-lite-v1:0`; keep reasoning disabled on this hot path.
Set `FOOTBALL_LLM_MODE=off` for a zero-model-call team.

### 2. Local test

```bash
python3 ../lib/test_learned_team.py
python3 ai-gk/test_local.py
python3 ai-gk/test_local.py --llm  # needs AWS credentials
```

### 3. Teardown

```bash
python destroy_all.py            # remove all 5 agents
python destroy_all.py ai-gk      # remove one agent
```

The memory resource is kept; remove its entry from `agentcore/agentcore.json`
and redeploy (or delete the CloudFormation stack) to delete it.

## Legacy scripts

`deploy-all.sh` / `deploy-all-windows.ps1` (with `create_memory.py` and
`memory_agent_base.py`) are the previous deployment path and still work.
Note the two paths manage **separate resources**: the legacy flow creates a
memory named `AITeamMatchMemory` outside CloudFormation, while this flow
creates `team_memory` inside the stack. Running both against one account
leaves two memory resources and two sets of agent runtimes.
