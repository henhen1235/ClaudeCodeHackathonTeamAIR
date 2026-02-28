"""
Reflex Layer — Per-frame sanitizer that converts LLM strategic vectors
into safe, collision-aware movement commands.

Dodge algorithm (physics-based):
  For each incoming bullet, compute the time of closest approach (TCA)
  using relative velocity. If the bullet will pass within BOT_RADIUS at
  TCA, the bot is on a collision course. The dodge direction is the
  perpendicular to the bullet's travel vector that leads AWAY from the
  bullet's path, biased toward open space (away from walls).
  Dodge strength is inversely proportional to TCA — imminent bullets
  get maximum priority.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── Tuning constants ──────────────────────────────────────────────────────────
BOT_RADIUS             = 16    # px — must match PLAYER_R / BOT_R in game.py
BULLET_RADIUS          = 5     # px
COLLISION_MARGIN       = BOT_RADIUS + BULLET_RADIUS + 8   # px, generous buffer
DANGER_TIME_WINDOW     = 1.2   # seconds — only react to bullets arriving within this window
WALL_BUFFER            = 38    # px — soft repulsion zone near walls/perimeter
WALL_REPULSE_STRENGTH  = 1.8   # how hard walls push back
DODGE_BASE_STRENGTH    = 3.5   # multiplier applied to each threatening bullet's dodge vector
INTENT_BLEND           = 0.30  # how much of the LLM intent survives during active dodge (0=pure dodge, 1=pure intent)
ARENA_W                = 800
ARENA_H                = 600
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BotState:
    x: float = 400.0
    y: float = 300.0
    vx: float = 0.0
    vy: float = 0.0
    hp: int = 100
    # LLM strategic intent (updated by pipeline)
    intent_dx: float = 0.0
    intent_dy: float = 0.0
    shoot_intent: bool = False
    # Actual frame velocity after reflex processing
    frame_vx: float = 0.0
    frame_vy: float = 0.0


@dataclass
class BulletInfo:
    x: float
    y: float
    vx: float
    vy: float
    owner: str  # "player" or "bot"


@dataclass
class WallRect:
    x: float
    y: float
    w: float
    h: float


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return 0.0, 0.0
    return dx / mag, dy / mag


def compute_wall_distances(bot: BotState, walls: list[WallRect], arena_w: int, arena_h: int) -> list[float]:
    """Return [dist_N, dist_E, dist_S, dist_W] to nearest wall surface."""
    dist_n = bot.y
    dist_e = arena_w - bot.x
    dist_s = arena_h - bot.y
    dist_w = bot.x

    for wall in walls:
        wx1, wy1, wx2, wy2 = wall.x, wall.y, wall.x + wall.w, wall.y + wall.h
        if wy2 <= bot.y and wx1 <= bot.x <= wx2:
            dist_n = min(dist_n, bot.y - wy2)
        if wy1 >= bot.y and wx1 <= bot.x <= wx2:
            dist_s = min(dist_s, wy1 - bot.y)
        if wx2 <= bot.x and wy1 <= bot.y <= wy2:
            dist_w = min(dist_w, bot.x - wx2)
        if wx1 >= bot.x and wy1 <= bot.y <= wy2:
            dist_e = min(dist_e, wx1 - bot.x)

    return [dist_n, dist_e, dist_s, dist_w]


def _closest_approach(
    bx: float, by: float, bvx: float, bvy: float,
    tx: float, ty: float,
) -> tuple[float, float]:
    """
    Return (tca, miss_distance):
      tca           — time of closest approach (seconds). Negative means bullet
                      already passed.
      miss_distance — minimum distance between bullet and target at TCA.
    Bullet travels at (bvx, bvy) px/s starting from (bx, by).
    Target is stationary at (tx, ty).
    """
    rx = bx - tx
    ry = by - ty
    rel_speed_sq = bvx * bvx + bvy * bvy
    if rel_speed_sq < 1e-6:
        return 0.0, math.hypot(rx, ry)

    tca = -(rx * bvx + ry * bvy) / rel_speed_sq
    cx = bx + bvx * tca - tx
    cy = by + bvy * tca - ty
    miss_dist = math.hypot(cx, cy)
    return tca, miss_dist


def _wall_repulsion(
    bot_x: float, bot_y: float,
    walls: list[WallRect],
    arena_w: int, arena_h: int,
) -> tuple[float, float]:
    """
    Compute a soft repulsion vector that pushes the bot away from walls
    and arena edges when within WALL_BUFFER distance.
    """
    rx, ry = 0.0, 0.0

    # Arena perimeter repulsion
    for dist, push_dx, push_dy in (
        (bot_x,              1.0,  0.0),   # left edge
        (arena_w - bot_x,   -1.0,  0.0),   # right edge
        (bot_y,              0.0,  1.0),   # top edge
        (arena_h - bot_y,    0.0, -1.0),   # bottom edge
    ):
        if dist < WALL_BUFFER:
            strength = (1.0 - dist / WALL_BUFFER) * WALL_REPULSE_STRENGTH
            rx += push_dx * strength
            ry += push_dy * strength

    # Internal wall repulsion
    for wall in walls:
        cx = wall.x + wall.w / 2
        cy = wall.y + wall.h / 2
        # Closest point on wall AABB to the bot
        closest_x = max(wall.x, min(wall.x + wall.w, bot_x))
        closest_y = max(wall.y, min(wall.y + wall.h, bot_y))
        dist = math.hypot(bot_x - closest_x, bot_y - closest_y)
        if dist < WALL_BUFFER and dist > 1e-3:
            strength = (1.0 - dist / WALL_BUFFER) * WALL_REPULSE_STRENGTH
            nx, ny = _normalize(bot_x - closest_x, bot_y - closest_y)
            rx += nx * strength
            ry += ny * strength

    return rx, ry


def process_reflex(
    bot: BotState,
    player_bullets: list[BulletInfo],
    walls: list[WallRect],
    bot_speed: float,
    arena_w: int = ARENA_W,
    arena_h: int = ARENA_H,
    dt: float = 1 / 60,
) -> tuple[float, float, bool]:
    """
    Per-frame decision:
      1. For every incoming bullet compute TCA and miss distance.
         Bullets that will hit (miss_dist < COLLISION_MARGIN) within
         DANGER_TIME_WINDOW seconds are genuine threats.
      2. For each threat, compute a dodge vector perpendicular to the
         bullet's travel, choosing the side that leads away from the
         bullet's trajectory line and biased away from nearby walls.
         Weight by urgency: 1 / max(tca, 0.01).
      3. Sum all dodge vectors. If total dodge magnitude > 0, blend
         with LLM intent at INTENT_BLEND ratio (LLM keeps only 30%).
      4. Add wall repulsion so the bot never backs into a corner.
      5. Normalize and scale to bot_speed.
    """

    # ── 1. Threat assessment ─────────────────────────────────────────────────
    dodge_x, dodge_y = 0.0, 0.0
    threat_count = 0
    max_urgency  = 0.0

    for b in player_bullets:
        tca, miss_dist = _closest_approach(b.x, b.y, b.vx, b.vy, bot.x, bot.y)

        # Ignore bullets that already passed or are too far in the future
        if tca < -0.05 or tca > DANGER_TIME_WINDOW:
            continue
        # Ignore bullets whose closest approach misses by a safe margin
        if miss_dist >= COLLISION_MARGIN * 2.5:
            continue

        # ── 2. Dodge direction ───────────────────────────────────────────────
        # The bullet's travel unit vector
        travel_x, travel_y = _normalize(b.vx, b.vy)

        # Two perpendicular options (left/right of bullet path)
        perp_ax, perp_ay =  travel_y, -travel_x   # rotate +90°
        perp_bx, perp_by = -travel_y,  travel_x   # rotate -90°

        # The bot's position relative to the bullet at the moment of closest approach
        bullet_at_tca_x = b.x + b.vx * max(tca, 0.0)
        bullet_at_tca_y = b.y + b.vy * max(tca, 0.0)
        to_bot_x = bot.x - bullet_at_tca_x
        to_bot_y = bot.y - bullet_at_tca_y

        # Pick the perpendicular that points toward the bot's side of the path
        # (so we dodge away from the bullet's line, not toward it)
        dot_a = perp_ax * to_bot_x + perp_ay * to_bot_y
        if dot_a >= 0:
            chosen_px, chosen_py = perp_ax, perp_ay
        else:
            chosen_px, chosen_py = perp_bx, perp_by

        # Bias the dodge away from arena walls/internal walls using repulsion
        wall_rx, wall_ry = _wall_repulsion(
            bot.x + chosen_px * 30,   # probe 30px ahead in dodge direction
            bot.y + chosen_py * 30,
            walls, arena_w, arena_h,
        )
        # If the probe point is strongly repelled, flip to the other perpendicular
        probe_wall_dot = wall_rx * chosen_px + wall_ry * chosen_py
        if probe_wall_dot < -0.5:
            chosen_px = -chosen_px
            chosen_py = -chosen_py

        # Urgency = 1 / time-to-closest-approach (closer = more urgent)
        # Also scale by how "on target" the bullet is (miss_dist near 0 = maximum urgency)
        urgency = DODGE_BASE_STRENGTH / max(tca, 0.03)
        hit_factor = max(0.0, 1.0 - miss_dist / COLLISION_MARGIN)   # 1 when perfect hit
        urgency *= (0.4 + 0.6 * hit_factor)

        dodge_x += chosen_px * urgency
        dodge_y += chosen_py * urgency
        threat_count += 1
        max_urgency = max(max_urgency, urgency)

    # ── 3. Blend dodge with LLM intent ───────────────────────────────────────
    intent_x = bot.intent_dx
    intent_y = bot.intent_dy

    if threat_count > 0:
        # Normalize dodge vector first so urgency doesn't just explode magnitude
        norm_dodge_x, norm_dodge_y = _normalize(dodge_x, dodge_y)
        # Scale by urgency capped at 1.0 for blending purposes
        urgency_scale = min(max_urgency / DODGE_BASE_STRENGTH, 1.0)
        # The more urgent the threat, the less LLM intent survives
        blend = INTENT_BLEND * (1.0 - urgency_scale * 0.7)
        dx = norm_dodge_x * (1.0 - blend) + intent_x * blend
        dy = norm_dodge_y * (1.0 - blend) + intent_y * blend
    else:
        dx = intent_x
        dy = intent_y

    # ── 4. Wall repulsion (always applied) ───────────────────────────────────
    wr_x, wr_y = _wall_repulsion(bot.x, bot.y, walls, arena_w, arena_h)
    dx += wr_x * 0.5
    dy += wr_y * 0.5

    # ── 5. Hard clamp at perimeter ────────────────────────────────────────────
    if bot.x < WALL_BUFFER and dx < 0:
        dx = 0.0
    if bot.x > arena_w - WALL_BUFFER and dx > 0:
        dx = 0.0
    if bot.y < WALL_BUFFER and dy < 0:
        dy = 0.0
    if bot.y > arena_h - WALL_BUFFER and dy > 0:
        dy = 0.0

    # ── 6. Normalize + scale ─────────────────────────────────────────────────
    dx, dy = _normalize(dx, dy)
    vx = dx * bot_speed
    vy = dy * bot_speed

    # Always shoot if the LLM wants to — never suppress shoot intent in reflex
    should_shoot = bot.shoot_intent

    return vx, vy, should_shoot

    x: float = 400.0
    y: float = 300.0
    vx: float = 0.0
    vy: float = 0.0
    hp: int = 100
    # LLM strategic intent (updated by pipeline)
    intent_dx: float = 0.0
    intent_dy: float = 0.0
    shoot_intent: bool = False
    # Actual frame velocity after reflex processing
    frame_vx: float = 0.0
    frame_vy: float = 0.0


@dataclass
class BulletInfo:
    x: float
    y: float
    vx: float
    vy: float
    owner: str  # "player" or "bot"


@dataclass
class WallRect:
    x: float
    y: float
    w: float
    h: float


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return 0.0, 0.0
    return dx / mag, dy / mag


def compute_wall_distances(bot: BotState, walls: list[WallRect], arena_w: int, arena_h: int) -> list[float]:
    """Return [dist_N, dist_E, dist_S, dist_W] to nearest wall surface."""
    dist_n = bot.y
    dist_e = arena_w - bot.x
    dist_s = arena_h - bot.y
    dist_w = bot.x

    for wall in walls:
        # Only check walls that are roughly in each cardinal direction
        wx1, wy1, wx2, wy2 = wall.x, wall.y, wall.x + wall.w, wall.y + wall.h
        # North: wall is above bot and horizontally overlapping
        if wy2 <= bot.y and wx1 <= bot.x <= wx2:
            dist_n = min(dist_n, bot.y - wy2)
        # South: wall is below
        if wy1 >= bot.y and wx1 <= bot.x <= wx2:
            dist_s = min(dist_s, wy1 - bot.y)
        # West: wall is left
        if wx2 <= bot.x and wy1 <= bot.y <= wy2:
            dist_w = min(dist_w, bot.x - wx2)
        # East: wall is right
        if wx1 >= bot.x and wy1 <= bot.y <= wy2:
            dist_e = min(dist_e, wx1 - bot.x)

    return [dist_n, dist_e, dist_s, dist_w]

