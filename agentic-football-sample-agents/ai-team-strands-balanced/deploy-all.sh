#!/usr/bin/env bash
set -euo pipefail

# Deploy the five _ak runtimes and their shared episodic memory as one
# AgentCore/CDK project. Keeping these resources in one stack lets AgentCore
# inject MEMORY_TEAM_MEMORY_ID into every runtime and grant the required IAM
# permissions automatically.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$#" -ne 0 ]; then
  echo "ERROR: This team uses one shared memory, so deploy all five agents together."
  echo "Usage: ./deploy-all.sh"
  exit 2
fi

# Some macOS NVM installations expose an AgentCore CLI installed beside Node
# 24 even though this CLI release currently runs on Node 20. Prefer a local
# Node 20 binary when one is available; the agentcore shim remains on PATH.
if command -v agentcore >/dev/null 2>&1 && ! agentcore --version >/dev/null 2>&1; then
  for node20_dir in "$HOME"/.nvm/versions/node/v20*/bin; do
    if [ -x "$node20_dir/node" ]; then
      export PATH="$node20_dir:$PATH"
    fi
  done
fi

exec python3 "$SCRIPT_DIR/deploy_all.py"
