"""
Player Memory — Persistent cross-session profiling.

At the end of every game, the raw thinking observations collected by the
AI pipeline are sent to Claude for summarisation into a compact 5-point
behavioural profile.  That profile is appended to `player_memory.txt`
alongside a timestamp and the match result.

On the next launch, `load_player_memory()` reads the file and returns a
single condensed string that is injected into the pipeline prompt so the
AI enters every session with full knowledge of how this human plays.
"""

import anthropic
import asyncio
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# ── File location ──────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(_THIS_DIR, "player_memory.txt")

# Keep the last N sessions in the file so the prompt doesn't grow forever
MAX_SESSIONS_KEPT = 8

# Load .env from the project root (one level above AIsystem/)
load_dotenv(os.path.join(_THIS_DIR, "..", ".env"))
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SUMMARISE_PROMPT = """\
You are a combat analyst reviewing an AI's observations of a human player in a 2D top-down shooter.

Below are the AI's raw per-tick observations from a single match (each line is one Claude decision \
summary describing what the enemy was doing):

---
{OBSERVATIONS}
---

Match result: {RESULT}

Summarise the human player's combat style into EXACTLY 5 short bullet points (one per line, start \
each with "- ").  Focus on:
1. Movement patterns (do they strafe, rush, camp, circle?)
2. Shooting habits (spray, burst, patient sniper?)
3. Dodging behaviour (reactive, predictive, or none?)
4. Positioning preference (centre, edges, behind walls?)
5. Exploitable weakness observed this session

Be specific and tactical.  Max 15 words per bullet.  No extra text outside the 5 bullets."""


def load_player_memory() -> str:
    """
    Read all stored session summaries from disk.
    Returns a single string ready to paste into a prompt, or an empty string
    if no memory exists yet.
    """
    if not os.path.exists(MEMORY_FILE):
        return ""
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return ""
        # Return the full file — pipeline will truncate via max_tokens anyway
        return content
    except Exception as e:
        print(f"[Memory] Could not read memory file: {e}")
        return ""


def _trim_memory_file():
    """Keep only the last MAX_SESSIONS_KEPT sessions in the file."""
    if not os.path.exists(MEMORY_FILE):
        return
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        # Sessions are separated by a line of "═" characters
        separator = "═" * 60
        sessions = text.split(separator)
        sessions = [s.strip() for s in sessions if s.strip()]
        if len(sessions) > MAX_SESSIONS_KEPT:
            sessions = sessions[-MAX_SESSIONS_KEPT:]
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write(("\n" + separator + "\n").join(sessions) + "\n")
    except Exception as e:
        print(f"[Memory] Trim failed: {e}")


async def _summarise_async(observations: list[str], result: str) -> str:
    """Call Claude to summarise the session observations."""
    obs_text = "\n".join(observations[-80:])  # cap at 80 lines to stay within tokens
    prompt   = SUMMARISE_PROMPT.format(OBSERVATIONS=obs_text, RESULT=result)
    client   = anthropic.AsyncAnthropic(api_key=API_KEY)
    try:
        message = await client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 300,
            messages   = [{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"[Memory] Summarisation API error: {e}")
        return ""


def save_session(observations: list[str], winner: str) -> None:
    """
    Synchronous entry point called from the game loop at game-over.
    Runs the async summarise call in a temporary event loop so it doesn't
    block the main thread for long (Haiku responds in ~1-2 s).
    """
    if not observations:
        print("[Memory] No observations to save.")
        return

    result = "AI won" if winner == "bot" else "Human won"
    print(f"[Memory] Summarising session ({len(observations)} observations, {result})…")

    summary = asyncio.run(_summarise_async(observations, result))
    if not summary:
        print("[Memory] No summary returned — skipping save.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    separator = "═" * 60
    entry = (
        f"{separator}\n"
        f"SESSION: {timestamp}  |  Result: {result}\n"
        f"{summary}\n"
    )

    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        _trim_memory_file()
        print(f"[Memory] Saved to {MEMORY_FILE}")
    except Exception as e:
        print(f"[Memory] Write failed: {e}")


def format_memory_for_prompt(memory: str) -> str:
    """
    Wrap raw memory text in a clear block for injection into the AI prompt.
    Returns empty string if memory is empty.
    """
    if not memory.strip():
        return "No previous sessions recorded."
    # Truncate to last ~1500 chars so it doesn't bloat the prompt
    if len(memory) > 1500:
        memory = "…(older sessions omitted)…\n" + memory[-1500:]
    return memory
