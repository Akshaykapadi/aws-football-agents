#!/usr/bin/env python3
"""
Destroy all 5 AI Team (Gateway) agent runtimes from Bedrock AgentCore.

Usage:
    python destroy_all.py              # destroy all 5 agents
    python destroy_all.py ai-gk        # destroy one agent

Works on macOS, Linux, and Windows (PowerShell) without WSL.

Note: This does NOT delete the gateway or its Lambda tools. They are
declared in agentcore/agentcore.json ("tactical-tools"); to remove them,
delete that entry and run `agentcore deploy --yes` again, or delete the
CloudFormation stack to tear down everything at once.
"""

import sys
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
ALL_AGENTS = ["ai-gk", "ai-def", "ai-mid", "ai-fwd1", "ai-fwd2"]
agents = [sys.argv[1]] if len(sys.argv) > 1 else ALL_AGENTS

AGENT_NAMES = {
    "ai-gk":   "ai_gk_gateway_agent",
    "ai-def":  "ai_def_gateway_agent",
    "ai-mid":  "ai_mid_gateway_agent",
    "ai-fwd1": "ai_fwd1_gateway_agent",
    "ai-fwd2": "ai_fwd2_gateway_agent",
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
print("  AI Team (Gateway) — Destroy Agents")
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
print("Note: The gateway (tactical-tools) and its Lambda tools were NOT")
print("deleted. Remove the agentCoreGateways entry from agentcore/agentcore.json")
print("and redeploy, or delete the CloudFormation stack, to tear them down.")
