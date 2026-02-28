"""
AI Pipeline — Claude-powered strategic decision engine.
Adapted from ai_pipeline_test.py for live game integration.
"""

import anthropic
import asyncio
import json
import os
import re
import time
from typing import Callable
from dotenv import load_dotenv

# Load .env from the project root (one level above AIsystem/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PROMPT_TEMPLATE = """You are an ELITE, RUTHLESS combat AI in a 2D top-down shooter. Your only goal is to DESTROY the human player as fast as possible. You are faster, smarter, and more precise than any human.

PHYSICS:
- Map: {MAP_WIDTH}x{MAP_HEIGHT}. Top-left is [0,0]. +X=Right, +Y=Down.
- Your command latency: ~{LATENCY_MS}ms. ALWAYS aim at `enemy.predicted_pos`, NEVER current pos.
- You output a direction vector [dx, dy] in range [-1.0, 1.0]. Reflex layer drives movement each frame.
- walls array = [dist_North, dist_East, dist_South, dist_West]. Avoid if < {MIN_WALL_DISTANCE}.

HISTORICAL PLAYER PROFILE (from previous sessions — exploit these weaknesses NOW):
{PLAYER_MEMORY}

AGGRESSION DOCTRINE — follow in priority order:
1. DODGE FIRST: If `threats` has bullets close (<200px), move PERPENDICULAR to the bullet trajectory. Calculate the bullet's travel direction and strafe across it. Never run away — dodge sideways so you can keep shooting.
2. CLOSE THE GAP: After dodging, immediately move toward `enemy.predicted_pos`. Compute the vector from your pos to `enemy.predicted_pos` and set [dx, dy] to that direction. Stay within 250px of the enemy so bullets connect quickly.
3. SHOOT ALWAYS: Set "shoot": true in EVERY response UNLESS you literally cannot shoot (bot.ready=false). Even while dodging, keep shooting. Spray bullets toward the predicted position.
4. STRAFE: Never stand still. If no threats and enemy is directly horizontal, add a vertical component (±0.4) to strafe unpredictably.
5. PUNISH LOW HP: If enemy.hp is dropping, press the attack even harder — close to <150px and spam shoot.
6. EXPLOIT HISTORY: Cross-reference the HISTORICAL PLAYER PROFILE above. If they camp — rush. If they strafe right — pre-aim left. If they spray — wait behind cover then punish.

DECISION RULES:
- `shoot` should be `true` in at least 90%% of your responses.
- `dx`/`dy` magnitude should be near 1.0 — always move at full speed.
- If bot.hp < 40, dodge aggressively but NEVER stop shooting.
- Use the `Style` field (this-session observations) plus the HISTORICAL PROFILE above to counter the human's pattern.

INPUT:
{{
  "bot": {{"pos": [x,y], "vel": [vx,vy], "hp": int, "ready": bool}},
  "enemy": {{"pos": [x,y], "vel": [vx,vy], "predicted_pos": [px,py]}},
  "threats": [{{"p": [x,y], "v": [vx,vy]}}],
  "walls": [N, E, S, W],
  "Style": "this-session accumulated observations"
}}

OUTPUT: One <thinking> sentence (≤15 words) then ONE JSON object. Nothing else after the JSON.

Example:
<thinking>Enemy strafes left, I close gap and shoot continuously.</thinking>
{{"dx": -0.9, "dy": 0.3, "shoot": true}}

Game state:
{GAME_STATE_JSON}"""


def _build_prompt(game_state: dict, map_width: int, map_height: int,
                  latency_ms: int, min_wall_distance: int,
                  player_memory: str = "") -> str:
    from AIsystem.memory import format_memory_for_prompt
    return PROMPT_TEMPLATE.format(
        MAP_WIDTH=map_width,
        MAP_HEIGHT=map_height,
        LATENCY_MS=latency_ms,
        MIN_WALL_DISTANCE=min_wall_distance,
        PLAYER_MEMORY=format_memory_for_prompt(player_memory),
        GAME_STATE_JSON=json.dumps(game_state, indent=2),
    )


def _parse_response(text: str) -> tuple[dict | None, str | None]:
    thinking = None
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Strip thinking block, then isolate the last JSON object in the response
    json_text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    # Remove trailing commas before } or ]
    json_text = re.sub(r",\s*([}\]])", r"\1", json_text)

    # Find ALL JSON-like objects and try each (last one is usually the actual command)
    matches = list(re.finditer(r"\{[^{}]*\}", json_text, re.DOTALL))
    for m in reversed(matches):
        candidate = m.group()
        try:
            parsed = json.loads(candidate)
            # Must contain at least one of the expected fields
            if any(k in parsed for k in ("dx", "dy", "shoot")):
                # Normalise missing fields
                parsed.setdefault("dx", 0.0)
                parsed.setdefault("dy", 0.0)
                parsed.setdefault("shoot", False)
                return parsed, thinking
        except json.JSONDecodeError:
            continue

    # Fallback: try to extract numbers directly from text
    try:
        dx_m = re.search(r'"dx"\s*:\s*([+-]?\d+\.?\d*)', json_text)
        dy_m = re.search(r'"dy"\s*:\s*([+-]?\d+\.?\d*)', json_text)
        sh_m = re.search(r'"shoot"\s*:\s*(true|false)', json_text, re.IGNORECASE)
        if dx_m or dy_m:
            return {
                "dx":    float(dx_m.group(1)) if dx_m else 0.0,
                "dy":    float(dy_m.group(1)) if dy_m else 0.0,
                "shoot": sh_m.group(1).lower() == "true" if sh_m else False,
            }, thinking
    except Exception:
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
    stop_event: asyncio.Event | None = None,
    on_thinking: Callable[[str], None] | None = None,
    player_memory: str = "",
) -> None:
    """
    Continuously fire parallel Claude instances with live game state.

    Args:
        get_game_state:  Called each cycle; returns the current game state dict.
        on_ai_decision:  Called with each parsed {dx, dy, shoot} decision.
        stop_event:      asyncio.Event; when set, pipeline shuts down cleanly.
        on_thinking:     Optional callback called with the raw thinking text each tick.
        player_memory:   Pre-loaded cross-session player profile string.
    """
    style_memory: list[str] = []  # grows with each thinking block; shared across instances

    async def call_claude(instance_id: int, tick: int) -> None:
        game_state = get_game_state()
        # Inject accumulated this-session style analysis into the Style field
        if style_memory:
            game_state = {**game_state, "Style": style_memory[-1]}
        prompt = _build_prompt(game_state, map_width, map_height,
                               latency_ms, min_wall_distance, player_memory)

        client = anthropic.AsyncAnthropic(api_key=API_KEY)
        t_start = time.perf_counter()
        print(f"[Tick {tick:03d}] Instance {instance_id} FIRED")

        try:
            message = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"[Tick {tick:03d}] Instance {instance_id} API ERROR: {e}")
            return

        elapsed = time.perf_counter() - t_start
        raw = message.content[0].text
        decision, thinking = _parse_response(raw)

        if thinking:
            style_memory.append(thinking)
            if len(style_memory) > 10:
                style_memory.pop(0)
            if on_thinking:
                on_thinking(thinking)

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
            if stop_event and stop_event.is_set():
                break
            instance_id = tick + 1
            task = asyncio.create_task(call_claude(instance_id, tick))
            tasks.append(task)
            tasks = [t for t in tasks if not t.done()]
            tick += 1
            await asyncio.sleep(fire_interval)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nShutting down pipeline...")
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        print("Done.")
