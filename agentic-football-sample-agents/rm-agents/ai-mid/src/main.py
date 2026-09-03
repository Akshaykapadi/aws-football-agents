"""
RM_Agents — Midfielder, "The Brain". Controls ONLY player 2 (MID).
Shoot-first playbook v3: on-ball / possession / keeper ticks are decided in code (lib/fallback.py);
the LLM (Amazon Nova Lite) is consulted only in the defensive phase, under a hard timeout.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from dataclasses import replace
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, MID_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 2
POSITION_LABEL = "MID"
MODEL_ID = "us.amazon.nova-lite-v1:0"    # Nova Lite everywhere; the handler's hard timeout keeps replies inside the 500 ms budget

# --- System Prompt (defensive phase only — attack is decided in code) ---

SYSTEM_PROMPT = """You are the midfielder, player <ID>, of a 5v5 soccer team. You are ONLY asked when the OPPONENT has the ball or the ball is loose far from you — passing, carrying and shooting are decided by code. Return exactly ONE command as a bare JSON array.

Rules (first match wins):
1. Their carrier within 10 of me → PRESS_BALL intensity 0.7, duration 3.
2. Their carrier within 20 → INTERCEPT aggressive true, duration 3 — cut the central lane, do not double-press a teammate's man.
3. Otherwise screen the middle between the ball and my goal A: MOVE_TO x = ball x moved 10 toward A, y = ball y * 0.5, sprint false.
NEVER be the second presser on one carrier.
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

FALLBACK_CONFIG = replace(MID_CONFIG, shoot_threshold=28.0, shoot_max_abs_y=15.0)
fallback_commands = build_fallback(FALLBACK_CONFIG)


# --- Wire it up ---

agent = create_agent(SYSTEM_PROMPT, model_id=MODEL_ID)
invoke = create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FALLBACK_CONFIG,
)

if __name__ == "__main__":
    app.run()
