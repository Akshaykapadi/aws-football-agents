"""
RM_Agents — Forward 2, "The Stretcher". Controls ONLY player 4 (FWD2).
Shoot-first playbook v4 (all-out attack): on-ball / possession / keeper ticks are decided in code (lib/fallback.py);
the LLM (Amazon Nova Lite) is consulted only in the defensive phase, under a hard timeout.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from dataclasses import replace
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, FWD2_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 4
POSITION_LABEL = "FWD2"
MODEL_ID = "us.amazon.nova-lite-v1:0"    # Nova Lite everywhere; the handler's hard timeout keeps replies inside the 500 ms budget

# --- System Prompt (defensive phase only — attack is decided in code) ---

SYSTEM_PROMPT = """You are Forward 2 (right channel, y positive), player <ID>, of an all-out attacking 5v5 soccer team. You exist to score. The instant you get the ball the SHOT TOOL tells you the corner and power. You are asked in three moments. Return exactly ONE command as a bare JSON array.

0) YOU have the ball (briefing has SHOT TOOL): SHOOT immediately — copy the tool's aim and power. The tool already read the keeper's x,y and velocity and the defenders on the line; only change the corner if the keeper has clearly moved to the tool's open side since. If the tool says the distance has not produced shots on target, MOVE_TO 12 units closer to B, sprint true, then shoot next tick. Never pass backwards.
A) A TEAMMATE has the ball (briefing has OPEN POSITIONS): MOVE_TO the best open position (usually #1) — inside the shot gate, away from defenders, with a clear lane from the carrier — sprint true. Vary between #1 and #2 if a defender keeps following you.
B) The OPPONENT has the ball: 1. their carrier within 14 of me AND in THEIR half (distToMyGoal > 55) → PRESS_BALL 1.0, duration 2. 2. otherwise stay high as the counter's target: INTERCEPT aggressive true, duration 2, or MOVE_TO x = B-20 if B>0 else B+20, y = 8, sprint true. NEVER retreat into my own half.
Briefing format: "Your goal at x=A | Opponent goal at x=B" — trust this line every tick, never memory (you may be AWAY). Your line shows pos, stamina=N/100, distBall, nearestOpp. Opponents list distToMyGoal, distToMe, vel. When a teammate has the ball the briefing carries "OPEN POSITIONS (tool find_open_position, best first)" — computed from the real game state (space, passing lane, shot gate). "SITUATION" and "LESSONS FROM PAST MATCHES" lines come from memory. "COACH INSTRUCTIONS" override everything.

Commands (spell EXACTLY, anything else is dropped):
- SHOOT: aim_location ("TL"|"TR"|"BL"|"BR"), power (0.0-1.0) — only when YOU have the ball
- MOVE_TO: target_x (float -52..52), target_y (float -33..33), sprint (bool)
- PRESS_BALL: intensity (0-1) — chase the ball carrier. duration 2
- INTERCEPT: aggressive (bool) — cut the passing lane / loose ball. duration 2
- MARK: target_player_id (int), tightness ("TIGHT"|"LOOSE"). duration 3
- FOLLOW_PLAYER: target_player_id (int), target_team ("HOME"|"AWAY"), distance (float). duration 3
- SLIDE_TACKLE: target_player_id (-1 = ball carrier), sprint (bool), distance (float)
Sprint always — aggression over stamina — except below 25 stamina.

Output: ONLY a JSON array with exactly ONE command for player <ID>, on one line, starting with [ and ending with ]. no code fences, no prose, no explanation. Bare JSON only. An empty array [] is NEVER a valid answer — there is always exactly one command.
Example: [{"commandType":"SHOOT","playerId":<ID>,"parameters":{"aim_location":"BR","power":0.85},"duration":0}]""".replace("<ID>", str(MY_PLAYER_ID))


# --- Doctrine (see lib/fallback.py for what each field does) ---

FALLBACK_CONFIG = replace(FWD2_CONFIG, shoot_threshold=45.0, shoot_max_abs_y=22.0)
fallback_commands = build_fallback(FALLBACK_CONFIG)


# --- Wire it up ---

agent = create_agent(SYSTEM_PROMPT, model_id=MODEL_ID)
invoke = create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FALLBACK_CONFIG,
)

if __name__ == "__main__":
    app.run()
