import anthropic
import asyncio
import json
import os
import re
import sys
import time
from typing import Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PROMPT_TEMPLATE = """You are the tactical brain of an AI combat bot in a 2D top-down shooter game.
You control the bot's directional movement and shooting reflexes.

CRITICAL PHYSICS & LATENCY RULES:
1. Coordinate System: [0, 0] is the top-left corner. +X is Right, +Y is Down. Map size is {MAP_WIDTH}x{MAP_HEIGHT}.
2. Network Latency: Your commands have a delay of approximately {LATENCY_MS}ms.
3. Dead Reckoning: Because of the delay, DO NOT aim at the enemy's current position. The input JSON provides `enemy.predicted_pos`—this is exactly where the human will be when your command executes. You must aim and maneuver based on this future position.
4. Continuous Movement: You do not output coordinates. You output a directional vector [dx, dy] between -1.0 and 1.0. The bot's reflex layer will continuously drive in that direction until your next command.

YOUR OBJECTIVES:
1. SURVIVE: Look at the `threats` array (incoming bullets). If a bullet is close, immediately set your [dx, dy] to move perpendicular to the bullet's velocity to dodge it.
2. AVOID TRAPS: Look at the `walls` array [North, East, South, West]. If a distance is less than {MIN_WALL_DISTANCE}, do not move in that direction.
3. DESTROY: If you are not dodging, maneuver to get a clear line of sight to `enemy.predicted_pos`. If you have a clear shot, set "shoot" to true.

INPUT FORMAT:
You will receive a JSON object representing the game state:
{{
  "bot": {{"pos": [x,y], "vel": [dx,dy], "hp": int, "ready": bool}},
  "enemy": {{"pos": [x,y], "vel": [dx,dy], "predicted_pos": [future_x, future_y]}},
  "threats": [{{"p": [x,y], "v": [dx,dy]}}],
  "walls": [dist_N, dist_E, dist_S, dist_W],
  "Style": "Short String"
}}

OUTPUT FORMAT:
First, write a brief <thinking> block analyzing the opponent's style based off of the past style input and this current step. This will be used in future steps.
Then, output ONLY a valid JSON object with your commands. Do not output any text after the JSON.

Example Output:
<thinking>
The opponent is extremely agressive, my goal is to outmaneuver them and find a weak spot.
</thinking>
{{
  "dx": 0.0,
  "dy": -1.0,
  "shoot": true
}}

Current game state:
{GAME_STATE_JSON}"""


def _build_prompt(game_state: dict, map_width: int, map_height: int,
                  latency_ms: int, min_wall_distance: int) -> str:
    return PROMPT_TEMPLATE.format(
        MAP_WIDTH=map_width,
        MAP_HEIGHT=map_height,
        LATENCY_MS=latency_ms,
        MIN_WALL_DISTANCE=min_wall_distance,
        GAME_STATE_JSON=json.dumps(game_state, indent=2),
    )


def _parse_response(text: str) -> tuple[dict | None, str | None]:
    """Return (decision_dict, thinking_text) parsed from a raw response."""
    # Extract thinking text
    thinking = None
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Strip thinking block, then parse JSON
    json_text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
    try:
        match = re.search(r"\{.*\}", json_text, re.DOTALL)
        if match:
            return json.loads(match.group()), thinking
    except json.JSONDecodeError:
        pass
    return None, thinking


async def run_pipeline(
    get_game_state: Callable[[], dict],
    on_ai_decision: Callable[[dict], None],
    map_width: int = 800,
    map_height: int = 600,
    latency_ms: int = 300,
    min_wall_distance: int = 50,
    num_instances: int = 4,
    fire_interval: float = 0.25,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 300,
) -> None:
    """
    Continuously fire parallel Claude instances with live game state.

    Args:
        get_game_state:    Called each cycle; returns the current game state dict.
        on_ai_decision:    Called with each parsed {dx, dy, shoot} decision.
        map_width/height:  Map dimensions for the prompt.
        latency_ms:        Estimated network latency injected into the prompt.
        min_wall_distance: Wall-avoidance threshold injected into the prompt.
        num_instances:     Number of rotating parallel instance slots.
        fire_interval:     Seconds between firing each new instance (0.25 = 4/sec).
        model:             Claude model to use.
        max_tokens:        Max tokens per response.
    """

    style_memory: list[str] = []  # grows with each thinking block; shared across instances

    async def call_claude(instance_id: int, tick: int) -> None:
        game_state = get_game_state()
        # Inject accumulated style analysis into the Style field
        if style_memory:
            game_state = {**game_state, "Style": style_memory[-1]}
        prompt = _build_prompt(game_state, map_width, map_height, latency_ms, min_wall_distance)

        client = anthropic.AsyncAnthropic(api_key=API_KEY)
        t_start = time.perf_counter()
        print(f"[Tick {tick:03d}] Instance {instance_id} FIRED")

        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        elapsed = time.perf_counter() - t_start
        raw = message.content[0].text
        decision, thinking = _parse_response(raw)

        if thinking:
            style_memory.append(thinking)

        if decision:
            style_preview = f" | style: {thinking[:60]}..." if thinking else ""
            print(f"[Tick {tick:03d}] Instance {instance_id} DONE ({elapsed:.2f}s) -> {decision}{style_preview}")
            on_ai_decision(decision)
        else:
            print(f"[Tick {tick:03d}] Instance {instance_id} DONE ({elapsed:.2f}s) -> PARSE FAILED: {raw[:80]}")

    tick = 0
    tasks: list[asyncio.Task] = []

    print(f"Pipeline started: {num_instances} instances, {fire_interval}s interval")

    try:
        while True:
            instance_id = (tick % num_instances) + 1
            task = asyncio.create_task(call_claude(instance_id, tick))
            tasks.append(task)
            tasks = [t for t in tasks if not t.done()]
            tick += 1
            await asyncio.sleep(fire_interval)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutting down pipeline...")
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        print("Done.")


# ---------------------------------------------------------------------------
# Quick smoke test — replace with real game state + handler when integrating
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    DUMMY_STATES = [
        {   # Scenario A: bot cornered north-west, enemy charging in with bullet incoming
            "bot":     {"pos": [60, 55], "vel": [0, 0], "hp": 80, "ready": True},
            "enemy":   {"pos": [400, 300], "vel": [2, 1], "predicted_pos": [460, 340]},
            "threats": [{"p": [200, 100], "v": [3, -1]}],
            "walls":   [55, 740, 545, 60],
            "Style":   "unknown",
        },
        {   # Scenario B: enemy retreating, bot healthy, no threats
            "bot":     {"pos": [400, 300], "vel": [1, 0], "hp": 100, "ready": True},
            "enemy":   {"pos": [650, 150], "vel": [-3, -2], "predicted_pos": [560, 90]},
            "threats": [],
            "walls":   [300, 150, 300, 400],
            "Style":   "unknown",
        },
        {   # Scenario C: bot low HP, two bullets incoming, enemy flanking
            "bot":     {"pos": [300, 500], "vel": [-1, 0], "hp": 25, "ready": False},
            "enemy":   {"pos": [500, 200], "vel": [0, 3], "predicted_pos": [500, 290]},
            "threats": [{"p": [350, 420], "v": [-1, 2]}, {"p": [280, 460], "v": [2, 1]}],
            "walls":   [500, 500, 100, 300],
            "Style":   "unknown",
        },
        {   # Scenario D: even fight, enemy strafing, bot in open centre
            "bot":     {"pos": [390, 295], "vel": [0, 1], "hp": 60, "ready": True},
            "enemy":   {"pos": [390, 100], "vel": [3, 0], "predicted_pos": [480, 100]},
            "threats": [{"p": [390, 200], "v": [0, 3]}],
            "walls":   [295, 410, 305, 390],
            "Style":   "unknown",
        },
    ]

    _tick_counter = 0

    def get_dummy_state() -> dict:
        """Rotate through scenarios so each instance sees different data."""
        global _tick_counter
        state = DUMMY_STATES[_tick_counter % len(DUMMY_STATES)]
        _tick_counter += 1
        return state

    def handle_decision(decision: dict) -> None:
        print(f"  => dx={decision.get('dx'):+.1f}  dy={decision.get('dy'):+.1f}  shoot={decision.get('shoot')}")

    asyncio.run(run_pipeline(
        get_game_state=get_dummy_state,
        on_ai_decision=handle_decision,
    ))
