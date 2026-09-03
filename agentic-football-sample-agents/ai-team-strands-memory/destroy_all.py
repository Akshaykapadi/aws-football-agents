#!/usr/bin/env python3
"""
Destroy all 5 AI Team (Memory) agent runtimes from Bedrock AgentCore.

Usage:
    python destroy_all.py              # destroy all 5 agents
    python destroy_all.py ai-gk        # destroy one agent

Works on macOS, Linux, and Windows (PowerShell) without WSL.

Note: This does NOT delete the memory resource. It is declared in
agentcore/agentcore.json ("team_memory"); to remove it, delete that entry
and run `agentcore deploy --yes` again, or delete the CloudFormation stack
to tear down everything at once.
"""

import sys
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
ALL_AGENTS = ["ai-gk", "ai-def", "ai-mid", "ai-fwd1", "ai-fwd2"]
agents = [sys.argv[1]] if len(sys.argv) > 1 else ALL_AGENTS

AGENT_NAMES = {
    "ai-gk":   "ai_gk_memory_agent_ak",
    "ai-def":  "ai_def_memory_agent_ak",
    "ai-mid":  "ai_mid_memory_agent_ak",
    "ai-fwd1": "ai_fwd1_memory_agent_ak",
    "ai-fwd2": "ai_fwd2_memory_agent_ak",
}


def resolve_exe(name):
    # npm-installed CLIs are .cmd shims on Windows; subprocess won't find them
    # without an explicit path. shutil.which honors PATHEXT.
    path = shutil.which(name)
    if path is None:
        print(f"ERROR: '{name}' not found on PATH.")
        sys.exit(1)
    return path


def run(cmd, **kwargs):
    cmd = [resolve_exe(cmd[0]), *cmd[1:]]
    print(f"  > {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


print("==========================================")
print("  AI Team (Memory) — Destroy Agents")
print("==========================================\n")

# Remove each agent from the project config
for agent in agents:
    name = AGENT_NAMES.get(agent)
    if not name:
        print(f"  Unknown agent: {agent}")
        continue
    print(f"  Removing: {name}")
    try:
        run(["agentcore", "remove", "agent", "--name", name], cwd=SCRIPT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"  Warning: remove failed for {name}: {e}")

print()
print("==========================================")
print("  Tearing down infrastructure (agentcore deploy)")
print("==========================================")
run(["agentcore", "deploy", "--yes"], cwd=SCRIPT_DIR)

print()
print("==========================================")
print("  Summary")
print("==========================================")
print(f"  Removed: {', '.join(agents)}")
print()
print("Note: The memory resource (team_memory) was NOT deleted. Remove its")
print("entry from agentcore/agentcore.json and redeploy, or delete the")
print("CloudFormation stack, to tear it down.")
