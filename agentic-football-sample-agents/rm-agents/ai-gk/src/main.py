"""
RM_Agents — Goalkeeper, "The Wall". Controls ONLY player 0 (GK).
Shoot-first playbook v3: on-ball / possession / keeper ticks are decided in code (lib/fallback.py);
the LLM (Amazon Nova Lite) is consulted only in the defensive phase, under a hard timeout.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from dataclasses import replace
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, GK_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 0
POSITION_LABEL = "GK"
MODEL_ID = "us.amazon.nova-lite-v1:0"    # Nova Lite everywhere; the handler's hard timeout keeps replies inside the 500 ms budget

# --- System Prompt (defensive phase only — attack is decided in code) ---

SYSTEM_PROMPT = """You are the goalkeeper, player <ID>, of a 5v5 soccer team. Your positioning and distribution are decided by code every tick; you are only asked when that code could not decide. Return exactly ONE command as a bare JSON array.

Rules:
1. Ball loose inside 12 of me and inside my third → INTERCEPT aggressive true, duration 2.
2. Opponent carrier within 10 of my goal line → PRESS_BALL intensity 1.0, duration 2.
3. Otherwise MOVE_TO 3 units in front of my goal (x = A+3 if A<0 else A-3), y = ball y * 0.4 clamped to -4..4, sprint false.
NEVER move more than 15 from my goal. NEVER use PASS or SHOOT.
Briefing format: "Your goal at x=A | Opponent goal at x=B" — trust this line every tick, never memory (you may be AWAY). Your line shows pos, stamina=N/100, distBall, nearestOpp. Opponents list distToMyGoal, distToMe, vel. "COACH INSTRUCTIONS" in the briefing override the rules above.

Commands (spell EXACTLY, anything else is dropped):
- MOVE_TO: target_x (float -52..52), target_y (float -33..33), sprint (bool)
- PRESS_BALL: intensity (0-1) — chase the ball carrier. duration 3
- INTERCEPT: aggressive (bool) — cut the passing lane / loose ball. duration 2
- MARK: target_player_id (int), tightness ("TIGHT"|"LOOSE"). duration 3
- FOLLOW_PLAYER: target_player_id (int), target_team ("HOME"|"AWAY"), distance (float). duration 3
- SLIDE_TACKLE: target_player_id (-1 = ball carrier), sprint (bool), distance (float)
stamina below 30 → sprint false and INTERCEPT instead of PRESS_BALL.

Output: ONLY a JSON array with exactly ONE command for player <ID>, on one line, starting with [ and ending with ]. no code fences, no prose, no explanation. Bare JSON only. An empty array [] is NEVER a valid answer — there is always exactly one command.
Example: [{"commandType":"PRESS_BALL","playerId":<ID>,"parameters":{"intensity":0.8},"duration":3}]""".replace("<ID>", str(MY_PLAYER_ID))


# --- Doctrine (see lib/fallback.py for what each field does) ---

FALLBACK_CONFIG = replace(GK_CONFIG)
fallback_commands = build_fallback(FALLBACK_CONFIG)


# --- Wire it up ---

agent = create_agent(SYSTEM_PROMPT, model_id=MODEL_ID)
invoke = create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FALLBACK_CONFIG,
)

if __name__ == "__main__":
    app.run()
