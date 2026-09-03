"""Stateless model factory for the learned soccer team (CDK/npm-CLI flow).

Sibling of memory_agent_base.py with the same public API, for deployments
made with the npm AgentCore CLI (`python deploy_all.py`). Match-level memory is
handled by ``lib/learned_handler.py``: one retrieval per match and one completed
episode written by the midfielder. The model itself is intentionally stateless.

deploy_all.py stages this file into each agent directory AS
memory_agent_base.py, so src/main.py imports it unchanged. The original
memory_agent_base.py remains the module staged by the legacy deploy-all.sh
flow.
"""

import os

from strands import Agent
from strands.models import BedrockModel


def create_memory_agent(
    system_prompt: str,
    player_id: int,
    position_label: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
) -> Agent:
    """Create a reusable model client with a small response budget."""
    resolved_model_id = os.environ.get(
        f"FOOTBALL_MODEL_ID_{position_label}",
        os.environ.get("FOOTBALL_MODEL_ID", model_id),
    )
    model = BedrockModel(model_id=resolved_model_id, temperature=0.0, max_tokens=220)
    return Agent(model=model, system_prompt=system_prompt)
