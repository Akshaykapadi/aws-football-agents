"""Stateless model factory for the learned soccer team (legacy deploy flow).

AgentCore LTM is read and written by ``lib/learned_handler.py``. Keeping the
model stateless avoids replaying a growing transcript on every game tick.
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
