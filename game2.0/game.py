"""
AI-Controlled Retro Shooter
============================
Human (WASD + Mouse aim + LMB/Space to shoot) vs LLM-driven AI Bot.

Run:
    python game2.0/game.py

Requires:
    pip install pygame anthropic
"""

import sys
import os
import math
import asyncio
import threading
import random
import json
from dataclasses import dataclass
from typing import Optional
import pygame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Path setup so AIsystem and reflex imports work ───────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reflex.reflex import (
    BotState, BulletInfo, WallRect,
    compute_wall_distances, process_reflex,
)
from AIsystem.ai_pipeline import run_pipeline

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 800, 600
FPS          = 60
TITLE        = "RETRO SHOOTER — Human vs AI"

PLAYER_SPEED = 200   # px/s
BOT_SPEED    = 230   # px/s  — AI is faster than the player
BULLET_SPEED = 420   # px/s
BULLET_R     = 5     # radius px
PLAYER_R     = 16    # collision radius px
BOT_R        = 16
SHOOT_COOLDOWN = 0.22  # seconds between shots (player)
BOT_SHOOT_COOLDOWN = 0.14  # seconds between shots (bot — fires faster)
DAMAGE       = 10
MAX_HP       = 100
PREDICT_T    = 0.35   # seconds ahead for bot's enemy prediction (longer look-ahead)

# ── Color palette (retro) ────────────────────────────────────────────────────
C_BG         = (10,  10,  20)
C_WALL       = (40,  40,  80)
C_WALL_EDGE  = (70,  70, 130)
C_PLAYER     = (50, 200, 255)
C_PLAYER_DARK= (20, 100, 140)
C_BOT        = (255,  60,  60)
C_BOT_DARK   = (140,  20,  20)
C_PBULLET    = (100, 220, 255)
C_BBULLET    = (255, 100,  50)
C_HP_GREEN   = (50,  220,  80)
C_HP_RED     = (220,  50,  50)
C_HP_BG      = (30,   30,  30)
C_TEXT       = (200, 200, 200)
C_GOLD       = (255, 200,  50)
C_FLASH_HIT  = (255,  80,  80)
C_THINKING   = (180, 255, 180)

# ── Internal wall layout (x, y, w, h) ────────────────────────────────────────
WALL_DEFS = [
    (160,  80, 80, 140),
    (350, 120, 100,  50),
    (560,  80,  80, 140),
    (120, 300, 140,  50),
    (540, 300, 140,  50),
    (300, 250, 200,  50),
    (160, 400,  80, 140),
    (560, 400,  80, 140),
    (340, 430, 120,  50),
]

# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    owner: str      # "player" or "bot"
    age: float = 0.0
    max_age: float = 2.5

    @property
    def alive(self):
        return self.age < self.max_age


@dataclass
class Entity:
    x: float
    y: float
    hp: int = MAX_HP
    vx: float = 0.0
    vy: float = 0.0
    shoot_timer: float = 0.0

    @property
    def alive(self):
        return self.hp > 0

    def can_shoot(self):
        return self.shoot_timer <= 0.0

    def fire(self, tx: float, ty: float, owner: str,
             cooldown: float = SHOOT_COOLDOWN) -> Optional[Bullet]:
        if not self.can_shoot():
            return None
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1:
            return None
        bvx = (dx / dist) * BULLET_SPEED
        bvy = (dy / dist) * BULLET_SPEED
        self.shoot_timer = cooldown
        return Bullet(self.x, self.y, bvx, bvy, owner)

    def rect_collide(self, wall: WallRect) -> bool:
        return (wall.x < self.x + PLAYER_R and
                self.x - PLAYER_R < wall.x + wall.w and
                wall.y < self.y + PLAYER_R and
                self.y - PLAYER_R < wall.y + wall.h)


# ═════════════════════════════════════════════════════════════════════════════
# EFFECTS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple

    @property
    def alive(self):
        return self.life > 0


def spawn_hit_particles(x, y, color, n=8) -> list[Particle]:
    parts = []
    for _ in range(n):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(60, 180)
        life  = random.uniform(0.3, 0.7)
        parts.append(Particle(x, y, math.cos(angle)*speed, math.sin(angle)*speed, life, life, color))
    return parts


def spawn_muzzle_flash(x, y, bvx, bvy, color) -> list[Particle]:
    parts = []
    angle = math.atan2(bvy, bvx)
    for _ in range(5):
        a = angle + random.uniform(-0.4, 0.4)
        speed = random.uniform(40, 120)
        life  = random.uniform(0.05, 0.12)
        parts.append(Particle(x, y, math.cos(a)*speed, math.sin(a)*speed, life, life, color))
    return parts


# ═════════════════════════════════════════════════════════════════════════════
# GAME STATE  (shared between game loop and AI pipeline — single asyncio thread)
# ═════════════════════════════════════════════════════════════════════════════

class GameState:
    """
    Shared state bridge between the synchronous game loop (main thread)
    and the async Claude pipeline (background daemon thread).
    Thread-safe via two lightweight locks.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._data: dict = {}
        self._ai_lock = threading.Lock()
        self.ai_dx: float = 0.0
        self.ai_dy: float = 0.0
        self.ai_shoot: bool = False
        self.latest_thinking: str = ""

    def update(self, data: dict):
        with self._lock:
            self._data = data

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def set_ai_decision(self, decision: dict):
        with self._ai_lock:
            self.ai_dx    = float(decision.get("dx", 0.0))
            self.ai_dy    = float(decision.get("dy", 0.0))
            self.ai_shoot = bool(decision.get("shoot", False))

    def get_ai_decision(self) -> tuple[float, float, bool]:
        with self._ai_lock:
            return self.ai_dx, self.ai_dy, self.ai_shoot


# ═════════════════════════════════════════════════════════════════════════════
# WALL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def make_walls() -> list[WallRect]:
    return [WallRect(x, y, w, h) for x, y, w, h in WALL_DEFS]


def bullet_hits_wall(b: Bullet, walls: list[WallRect]) -> bool:
    for w in walls:
        if w.x <= b.x <= w.x + w.w and w.y <= b.y <= w.y + w.h:
            return True
    # Perimeter
    if b.x < 0 or b.x > SCREEN_W or b.y < 0 or b.y > SCREEN_H:
        return True
    return False


def resolve_entity_walls(e: Entity, walls: list[WallRect]):
    """Push entity out of any wall it overlaps, zero velocity component."""
    for w in walls:
        overlap_x = (e.x + PLAYER_R) - w.x if e.vx > 0 else w.x + w.w - (e.x - PLAYER_R)
        overlap_y = (e.y + PLAYER_R) - w.y if e.vy > 0 else w.y + w.h - (e.y - PLAYER_R)

        if not e.rect_collide(w):
            continue
        # Resolve on minimum overlap axis
        if abs(overlap_x) < abs(overlap_y):
            if e.vx > 0:
                e.x = w.x - PLAYER_R
            else:
                e.x = w.x + w.w + PLAYER_R
            e.vx = 0.0
        else:
            if e.vy > 0:
                e.y = w.y - PLAYER_R
            else:
                e.y = w.y + w.h + PLAYER_R
            e.vy = 0.0


def clamp_to_arena(e: Entity):
    e.x = max(PLAYER_R, min(SCREEN_W - PLAYER_R, e.x))
    e.y = max(PLAYER_R, min(SCREEN_H - PLAYER_R, e.y))


# ═════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

# Module-level font + surface caches — populated once after pygame.init()
_FONT_LABEL: pygame.font.Font | None = None
_FONT_HUD:   pygame.font.Font | None = None
_FONT_SMALL: pygame.font.Font | None = None
_GRID_SURF:  pygame.Surface   | None = None   # pre-rendered grid overlay

def _init_draw_caches():
    """Call once after pygame.init() to populate all caches."""
    global _FONT_LABEL, _FONT_HUD, _FONT_SMALL, _GRID_SURF
    _FONT_LABEL = pygame.font.SysFont("monospace", 11, bold=True)
    _FONT_HUD   = pygame.font.SysFont("monospace", 12, bold=True)
    _FONT_SMALL = pygame.font.SysFont("monospace", 11)
    # Pre-render the static grid so we blit one surface per frame instead of 55 line calls
    _GRID_SURF = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    for gx in range(0, SCREEN_W, 40):
        pygame.draw.line(_GRID_SURF, (18, 18, 30, 255), (gx, 0), (gx, SCREEN_H))
    for gy in range(0, SCREEN_H, 40):
        pygame.draw.line(_GRID_SURF, (18, 18, 30, 255), (0, gy), (SCREEN_W, gy))
    _init_bullet_surfaces()

def draw_walls(surf: pygame.Surface, walls: list[WallRect]):
    for w in walls:
        pygame.draw.rect(surf, C_WALL, (w.x, w.y, w.w, w.h))
        pygame.draw.rect(surf, C_WALL_EDGE, (w.x, w.y, w.w, w.h), 2)


def draw_entity(surf: pygame.Surface, e: Entity, color: tuple, dark: tuple, label: str):
    if not e.alive:
        return
    pygame.draw.circle(surf, dark,  (int(e.x), int(e.y)), PLAYER_R + 3)
    pygame.draw.circle(surf, color, (int(e.x), int(e.y)), PLAYER_R)
    pygame.draw.circle(surf, (255, 255, 255), (int(e.x), int(e.y)), 4)
    txt = _FONT_LABEL.render(label, True, color)
    surf.blit(txt, (int(e.x) - txt.get_width()//2, int(e.y) - PLAYER_R - 16))


# Pre-baked bullet glow surfaces (created once after pygame.init)
_BULLET_GLOW_P: pygame.Surface | None = None
_BULLET_GLOW_B: pygame.Surface | None = None

def _init_bullet_surfaces():
    global _BULLET_GLOW_P, _BULLET_GLOW_B
    sz = BULLET_R * 4
    _BULLET_GLOW_P = pygame.Surface((sz, sz), pygame.SRCALPHA)
    pygame.draw.circle(_BULLET_GLOW_P, (*C_PBULLET, 60), (sz//2, sz//2), BULLET_R*2)
    _BULLET_GLOW_B = pygame.Surface((sz, sz), pygame.SRCALPHA)
    pygame.draw.circle(_BULLET_GLOW_B, (*C_BBULLET, 60), (sz//2, sz//2), BULLET_R*2)

def draw_bullet(surf: pygame.Surface, b: Bullet):
    color = C_PBULLET if b.owner == "player" else C_BBULLET
    glow  = _BULLET_GLOW_P if b.owner == "player" else _BULLET_GLOW_B
    if glow:
        surf.blit(glow, (int(b.x) - BULLET_R*2, int(b.y) - BULLET_R*2))
    pygame.draw.circle(surf, color, (int(b.x), int(b.y)), BULLET_R)
    pygame.draw.circle(surf, (255, 255, 255), (int(b.x), int(b.y)), max(1, BULLET_R - 2))


def draw_hp_bar(surf: pygame.Surface, x: int, y: int, hp: int, w: int, label: str, color: tuple):
    pct = max(0, hp) / MAX_HP
    bar_color = C_HP_GREEN if pct > 0.5 else (C_GOLD if pct > 0.25 else C_HP_RED)
    pygame.draw.rect(surf, C_HP_BG, (x, y, w, 14))
    pygame.draw.rect(surf, bar_color, (x, y, int(w * pct), 14))
    pygame.draw.rect(surf, (80, 80, 80), (x, y, w, 14), 1)
    txt = _FONT_HUD.render(f"{label} {hp:3d}HP", True, color)
    surf.blit(txt, (x, y - 17))


def draw_particles(surf: pygame.Surface, parts: list[Particle]):
    for p in parts:
        r = max(1, int(3 * (p.life / p.max_life)))
        # Fade colour by alpha-blending toward background manually (no Surface alloc)
        t = p.life / p.max_life
        c = (int(p.color[0]*t), int(p.color[1]*t), int(p.color[2]*t))
        pygame.draw.circle(surf, c, (int(p.x), int(p.y)), r)


def draw_aim_line(surf: pygame.Surface, ex: float, ey: float, mx: int, my: int):
    dx, dy = mx - ex, my - ey
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    nx, ny = dx/dist, dy/dist
    for i in range(0, int(dist), 14):
        pygame.draw.circle(surf, (50, 120, 200), (int(ex + nx*i), int(ey + ny*i)), 1)


# Cached screen-flash overlay (avoids allocating a new Surface every frame)
_FLASH_SURF: pygame.Surface | None = None

def draw_hud(surf: pygame.Surface, player: Entity, bot: Entity,
             ai_dx: float, ai_dy: float, ai_shoot: bool,
             thinking_lines: list[str], fps_val: float,
             screen_flash: float):
    global _FLASH_SURF
    if screen_flash > 0:
        if _FLASH_SURF is None:
            _FLASH_SURF = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        alpha = int(min(180, screen_flash * 400))
        _FLASH_SURF.fill((255, 40, 40, alpha))
        surf.blit(_FLASH_SURF, (0, 0))

    draw_hp_bar(surf,  20, 30, player.hp, 180, "YOU",    C_PLAYER)
    draw_hp_bar(surf, 600, 30, bot.hp,    180, "AI BOT", C_BOT)

    panel_x, panel_y = 10, SCREEN_H - 130
    pygame.draw.rect(surf, (15, 15, 30), (panel_x, panel_y, 220, 120), border_radius=4)
    pygame.draw.rect(surf, C_BOT, (panel_x, panel_y, 220, 120), 1, border_radius=4)
    surf.blit(_FONT_HUD.render("▶ AI VECTOR", True, C_BOT), (panel_x+6, panel_y+5))
    surf.blit(_FONT_HUD.render(f"  dx: {ai_dx:+.2f}  dy: {ai_dy:+.2f}", True, C_TEXT), (panel_x+6, panel_y+22))
    shoot_col = C_BBULLET if ai_shoot else (80, 80, 80)
    surf.blit(_FONT_HUD.render(f"  shoot: {'FIRE!' if ai_shoot else 'hold'}", True, shoot_col), (panel_x+6, panel_y+38))
    surf.blit(_FONT_HUD.render("▶ AI THINKING", True, C_THINKING), (panel_x+6, panel_y+56))
    for i, line in enumerate(thinking_lines[-3:]):
        surf.blit(_FONT_SMALL.render(line[:28], True, (120, 200, 120)), (panel_x+6, panel_y+72 + i*14))

    surf.blit(_FONT_HUD.render(f"FPS {fps_val:5.1f}", True, (100, 100, 100)), (SCREEN_W - 90, SCREEN_H - 20))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN GAME
# ═════════════════════════════════════════════════════════════════════════════

def game_over_screen(surf: pygame.Surface, winner: str):
    font_big = pygame.font.SysFont("monospace", 60, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 22)
    col = C_PLAYER if winner == "player" else C_BOT
    msg = "YOU WIN!" if winner == "player" else "AI WINS!"
    surf.fill(C_BG)
    txt = font_big.render(msg, True, col)
    surf.blit(txt, txt.get_rect(center=(SCREEN_W//2, SCREEN_H//2 - 40)))
    sub = font_sm.render("Press R to restart  |  ESC to quit", True, C_TEXT)
    surf.blit(sub, sub.get_rect(center=(SCREEN_W//2, SCREEN_H//2 + 40)))
    pygame.display.flip()
def build_ai_game_state(player: Entity, bot: Entity,
                        bullets: list[Bullet],
                        walls: list[WallRect]) -> dict:
    """Serialise the dynamic game situation for the Claude pipeline."""
    # Predict player pos PREDICT_T seconds ahead
    pred_px = player.x + player.vx * PREDICT_T
    pred_py = player.y + player.vy * PREDICT_T
    pred_px = max(0, min(SCREEN_W,  pred_px))
    pred_py = max(0, min(SCREEN_H, pred_py))

    # Player-owned bullets near the bot
    threats = []
    for b in bullets:
        if b.owner != "player":
            continue
        dist = math.hypot(b.x - bot.x, b.y - bot.y)
        if dist < 300:
            threats.append({"p": [round(b.x, 1), round(b.y, 1)],
                            "v": [round(b.vx, 1), round(b.vy, 1)]})

    wall_dists = compute_wall_distances(
        BotState(x=bot.x, y=bot.y), walls, SCREEN_W, SCREEN_H
    )

    return {
        "bot":    {"pos": [round(bot.x, 1), round(bot.y, 1)],
                   "vel": [round(bot.vx, 1), round(bot.vy, 1)],
                   "hp": bot.hp, "ready": bot.can_shoot()},
        "enemy":  {"pos": [round(player.x, 1), round(player.y, 1)],
                   "vel": [round(player.vx, 1), round(player.vy, 1)],
                   "predicted_pos": [round(pred_px, 1), round(pred_py, 1)]},
        "threats": threats,
        "walls":  [round(d, 1) for d in wall_dists],
        "Style":  "unknown",
    }


def run_game(shared_state: GameState, stop_event: threading.Event):
    """
    Synchronous game loop running in the main thread at a solid 60 FPS.
    The Claude pipeline runs in a separate daemon thread — completely decoupled.
    """
    pygame.init()
    pygame.font.init()
    _init_draw_caches()   # build font + grid + bullet surface caches

    surf  = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    walls = make_walls()

    thinking_lines: list[str] = []

    # ── mutable game objects ──────────────────────────────────────────────────
    player         = Entity(x=150, y=300)
    bot            = Entity(x=650, y=300)
    bullets        = []
    particles      = []
    screen_flash   = 0.0
    telemetry_timer = 0.0
    bot_reflex     = BotState(x=bot.x, y=bot.y)
    game_over      = False
    winner         = ""
    fps_val        = 60.0

    def reset():
        nonlocal player, bot, bullets, particles, screen_flash, telemetry_timer, bot_reflex, game_over, winner
        player          = Entity(x=150, y=300)
        bot             = Entity(x=650, y=300)
        bullets         = []
        particles       = []
        screen_flash    = 0.0
        telemetry_timer = 0.0
        bot_reflex      = BotState(x=bot.x, y=bot.y)
        game_over       = False
        winner          = ""

    print("[Game] Loop started — pipeline is live.")

    while not stop_event.is_set():
        # ── Tick at 60 FPS — blocks only this thread, not the AI pipeline ──────
        dt = clock.tick(FPS) / 1000.0

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    stop_event.set()
                    break
                if game_over and event.key == pygame.K_r:
                    reset()

        if stop_event.is_set():
            break

        if game_over:
            game_over_screen(surf, winner)
            continue

        # ── Player input ──────────────────────────────────────────────────────
        keys = pygame.key.get_pressed()
        pdx, pdy = 0.0, 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:    pdy -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  pdy += 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  pdx -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: pdx += 1

        mag = math.hypot(pdx, pdy)
        if mag > 0:
            pdx, pdy = pdx / mag, pdy / mag
        player.vx = pdx * PLAYER_SPEED
        player.vy = pdy * PLAYER_SPEED

        mx, my = pygame.mouse.get_pos()

        fire_player = (pygame.mouse.get_pressed()[0] or keys[pygame.K_SPACE])
        if fire_player and player.can_shoot():
            b = player.fire(mx, my, "player")
            if b:
                bullets.append(b)
                particles += spawn_muzzle_flash(player.x, player.y, b.vx, b.vy, C_PBULLET)

        # ── AI decision (written by pipeline coroutine, read here) ────────────
        ai_dx, ai_dy, ai_shoot = shared_state.get_ai_decision()

        # Feed into reflex layer
        bot_reflex.x           = bot.x
        bot_reflex.y           = bot.y
        bot_reflex.vx          = bot.vx
        bot_reflex.vy          = bot.vy
        bot_reflex.intent_dx   = ai_dx
        bot_reflex.intent_dy   = ai_dy
        bot_reflex.shoot_intent = ai_shoot

        player_bullet_infos = [
            BulletInfo(b.x, b.y, b.vx, b.vy, b.owner)
            for b in bullets if b.owner == "player"
        ]

        rvx, rvy, rshoot = process_reflex(
            bot_reflex, player_bullet_infos, walls, BOT_SPEED,
            SCREEN_W, SCREEN_H, dt
        )
        bot.vx = rvx
        bot.vy = rvy

        if rshoot and bot.can_shoot():
            pred_px = player.x + player.vx * PREDICT_T
            pred_py = player.y + player.vy * PREDICT_T
            b = bot.fire(pred_px, pred_py, "bot", cooldown=BOT_SHOOT_COOLDOWN)
            if b:
                bullets.append(b)
                particles += spawn_muzzle_flash(bot.x, bot.y, b.vx, b.vy, C_BBULLET)

        # ── Physics ───────────────────────────────────────────────────────────
        player.x += player.vx * dt;  player.y += player.vy * dt
        bot.x    += bot.vx    * dt;  bot.y    += bot.vy    * dt

        resolve_entity_walls(player, walls);  resolve_entity_walls(bot, walls)
        clamp_to_arena(player);               clamp_to_arena(bot)

        player.shoot_timer -= dt
        bot.shoot_timer    -= dt

        for b in bullets:
            b.x += b.vx * dt;  b.y += b.vy * dt;  b.age += dt

        # ── Bullet collisions ─────────────────────────────────────────────────
        remaining = []
        for b in bullets:
            if not b.alive:
                continue
            if bullet_hits_wall(b, walls):
                particles += spawn_hit_particles(b.x, b.y, (120, 120, 160), 4)
                continue
            hit = False
            if b.owner == "player" and bot.alive:
                if math.hypot(b.x - bot.x, b.y - bot.y) < BOT_R + BULLET_R:
                    bot.hp -= DAMAGE
                    particles += spawn_hit_particles(b.x, b.y, C_BOT, 10)
                    hit = True
            if b.owner == "bot" and player.alive:
                if math.hypot(b.x - player.x, b.y - player.y) < PLAYER_R + BULLET_R:
                    player.hp -= DAMAGE
                    screen_flash = 0.35
                    particles += spawn_hit_particles(b.x, b.y, C_PLAYER, 10)
                    hit = True
            if not hit:
                remaining.append(b)
        bullets = remaining

        for p in particles:
            p.x += p.vx * dt;  p.y += p.vy * dt;  p.life -= dt
        particles = [p for p in particles if p.alive]
        screen_flash = max(0.0, screen_flash - dt)

        # ── Game over ─────────────────────────────────────────────────────────
        if not player.alive:
            winner = "bot";   game_over = True
        elif not bot.alive:
            winner = "player"; game_over = True

        # ── Telemetry → pipeline (every 250 ms) ───────────────────────────────
        telemetry_timer += dt
        if telemetry_timer >= 0.25:
            telemetry_timer = 0.0
            shared_state.update(build_ai_game_state(player, bot, bullets, walls))

        # ── Render ────────────────────────────────────────────────────────────
        surf.fill(C_BG)
        surf.blit(_GRID_SURF, (0, 0))   # pre-rendered grid — single blit

        draw_walls(surf, walls)
        draw_particles(surf, particles)
        for b in bullets:
            draw_bullet(surf, b)
        draw_entity(surf, player, C_PLAYER, C_PLAYER_DARK, "YOU")
        draw_entity(surf, bot,    C_BOT,    C_BOT_DARK,    "AI")
        if player.alive:
            draw_aim_line(surf, player.x, player.y, mx, my)
        pygame.draw.rect(surf, C_WALL_EDGE, (0, 0, SCREEN_W, SCREEN_H), 3)

        fps_val = clock.get_fps()

        # Pull latest thinking text from the shared state to display
        t = shared_state.latest_thinking
        if t and t not in thinking_lines:
            for line in t.split("\n"):
                line = line.strip()
                if line:
                    thinking_lines.append(line)
            thinking_lines = thinking_lines[-12:]

        draw_hud(surf, player, bot, ai_dx, ai_dy, ai_shoot,
                 thinking_lines, fps_val, screen_flash)
        pygame.display.flip()

    pygame.quit()
    print("[Game] Exited cleanly.")


# ═════════════════════════════════════════════════════════════════════════════
# AI PIPELINE BACKGROUND THREAD
# ═════════════════════════════════════════════════════════════════════════════

class AIThread(threading.Thread):
    """
    Daemon thread that owns its own asyncio event loop and runs the Claude
    pipeline continuously.  The game loop never waits on it — they share
    data only via GameState's thread-safe locks.
    """
    def __init__(self, shared_state: GameState, stop_event: threading.Event,
                 on_thinking):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event   = stop_event
        self.on_thinking  = on_thinking

    def run(self):
        async def _pipeline():
            # Bridge threading.Event → asyncio.Event via periodic polling
            async_stop = asyncio.Event()
            loop = asyncio.get_event_loop()
            def _poll():
                if self.stop_event.is_set():
                    async_stop.set()
                else:
                    loop.call_later(0.05, _poll)
            loop.call_soon(_poll)

            await run_pipeline(
                get_game_state    = self.shared_state.snapshot,
                on_ai_decision    = self.shared_state.set_ai_decision,
                on_thinking       = self.on_thinking,
                map_width         = SCREEN_W,
                map_height        = SCREEN_H,
                latency_ms        = 300,
                min_wall_distance = 50,
                num_instances     = 4,
                fire_interval     = 0.25,
                model             = "claude-haiku-4-5-20251001",
                max_tokens        = 300,
                stop_event        = async_stop,
            )

        asyncio.run(_pipeline())


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    shared_state = GameState()
    stop_event   = threading.Event()

    # Seed initial state so the pipeline fires immediately with valid data
    shared_state.update({
        "bot":    {"pos": [650, 300], "vel": [0, 0], "hp": 100, "ready": True},
        "enemy":  {"pos": [150, 300], "vel": [0, 0], "predicted_pos": [150, 300]},
        "threats": [],
        "walls":  [300, 150, 300, 650],
        "Style":  "unknown",
    })

    def on_thinking(text: str):
        shared_state.latest_thinking = text

    ai_thread = AIThread(shared_state, stop_event, on_thinking)
    ai_thread.start()
    print("[Main] AI pipeline thread started.")

    # Game loop blocks here at 60 FPS — Claude runs freely in the background
    run_game(shared_state, stop_event)

    stop_event.set()
    ai_thread.join(timeout=3.0)
    print("[Main] Done.")


if __name__ == "__main__":
    main()
