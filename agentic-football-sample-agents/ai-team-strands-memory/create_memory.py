"""One-time script to create episodic AgentCore Memory for the learned team.

The normal ``python deploy_all.py`` path declares this resource in
``agentcore/agentcore.json``. This script is retained for the legacy shell and
PowerShell deployment paths.

Usage:
    AWS_DEFAULT_REGION=us-east-1 python3 create_memory.py

Prints the MEMORY_ID to stdout. Export it before deploying agents:
    export MEMORY_ID=<printed-id>
"""

import os
from bedrock_agentcore.memory import MemoryClient

region = os.environ.get("AWS_DEFAULT_REGION")
if not region:
    raise RuntimeError("AWS_DEFAULT_REGION environment variable is required")

client = MemoryClient(region_name=region)

# Check if memory already exists
existing = client.list_memories()
for mem in existing:
    mem_id = mem.get("id") or mem.get("memoryId", "")
    mem_name = mem.get("name", "")
    if mem_name == "AITeamMatchMemory" or mem_id.startswith("AITeamMatchMemory"):
        memory_id = mem_id
        print(f"Memory resource ready: {memory_id}")
        print("Note: an existing resource is reused; verify that it has an EPISODIC strategy.")
        print(f"Export it:  export MEMORY_ID={memory_id}")
        exit(0)

# Create new memory (handle "already exists" gracefully)
try:
    memory = client.create_memory(
        name="AITeamMatchMemory",
        description="Completed football match episodes and cross-match tactical reflections",
        event_expiry_days=30,
        strategies=[
            {
                "episodicMemoryStrategy": {
                    "name": "FootballMatchEpisodes",
                    "namespaceTemplates": ["/episodes/{actorId}/{sessionId}"],
                    "reflection": {
                        "namespaceTemplates": ["/episodes/{actorId}"]
                    },
                }
            }
        ],
    )
    memory_id = memory.get("id") or memory.get("memoryId")
except Exception as e:
    if "already exists" in str(e):
        # Memory exists but list didn't find it — re-list and search
        existing = client.list_memories()
        for mem in existing:
            name = mem.get("name", "")
            if name == "AITeamMatchMemory":
                memory_id = mem.get("id") or mem.get("memoryId")
                break
        else:
            # Last resort: parse the error or use name as ID
            raise RuntimeError(f"Memory 'AITeamMatchMemory' exists but could not retrieve ID: {e}")
    else:
        raise

print(f"Memory resource ready: {memory_id}")
print(f"Export it:  export MEMORY_ID={memory_id}")
