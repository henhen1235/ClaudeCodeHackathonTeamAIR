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
from AIsystem.memory import load_player_memory, save_session

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 800, 600
FPS          = 60
TITLE        = "RETRO SHOOTER — Human vs AI"

PLAYER_SPEED = 200   # px/s
BULLET_SPEED = 420   # px/s
BULLET_R     = 5     # radius px
PLAYER_R     = 16    # collision radius px
BOT_R        = 16
SHOOT_COOLDOWN = 0.22  # seconds between shots (player)
DAMAGE       = 10
MAX_HP       = 100

# ── Difficulty Levels ────────────────────────────────────────────────────────
@dataclass
class DifficultyConfig:
    name: str
    bot_speed: float
    bot_cooldown: float
    predict_time: float
    color: tuple

DIFFICULTY_EASY = DifficultyConfig(
    name="EASY",
    bot_speed=180.0,
    bot_cooldown=0.25,
    predict_time=0.20,
    color=(100, 200, 100)
)

DIFFICULTY_NORMAL = DifficultyConfig(
    name="NORMAL",
    bot_speed=230.0,
    bot_cooldown=0.14,
    predict_time=0.35,
    color=(100, 150, 255)
)

DIFFICULTY_HARD = DifficultyConfig(
    name="HARD",
    bot_speed=280.0,
    bot_cooldown=0.10,
    predict_time=0.50,
    color=(255, 100, 100)
)

DIFFICULTY_EXPERT = DifficultyConfig(
    name="EXPERT",
    bot_speed=320.0,
    bot_cooldown=0.08,
    predict_time=0.65,
    color=(255, 50, 255)
)

DIFFICULTIES = [DIFFICULTY_EASY, DIFFICULTY_NORMAL, DIFFICULTY_HARD, DIFFICULTY_EXPERT]

# ── Color palette (neon cyberpunk theme) ─────────────────────────────────────
C_BG         = (8, 8, 18)
C_WALL       = (25, 45, 75)
C_WALL_EDGE  = (100, 180, 255)
C_PLAYER     = (0, 255, 150)
C_PLAYER_DARK= (0, 180, 100)
C_BOT        = (255, 50, 100)
C_BOT_DARK   = (180, 20, 60)
C_PBULLET    = (0, 255, 200)
C_BBULLET    = (255, 80, 150)
C_HP_GREEN   = (0, 255, 100)
C_HP_RED     = (255, 50, 80)
C_HP_BG      = (15, 15, 35)
C_TEXT       = (200, 220, 240)
C_GOLD       = (255, 200, 80)
C_FLASH_HIT  = (255, 100, 150)
C_THINKING   = (150, 255, 200)

# ═════════════════════════════════════════════════════════════════════════════
# MAP SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MapConfig:
    """A complete map configuration with layout, colors, and spawn points."""
    name: str
    walls: list[tuple[int, int, int, int]]  # (x, y, w, h)
    player_spawn: tuple[int, int]
    bot_spawn: tuple[int, int]
    bg_color: tuple[int, int, int]
    wall_color: tuple[int, int, int]
    wall_edge_color: tuple[int, int, int]

# ── Map 1: Symmetrical Arena (balanced, modern) ──────────────────────────────
MAP_SYMMETRICAL = MapConfig(
    name="SYMMETRICAL ARENA",
    walls=[
        (160,  80, 80, 140),
        (350, 120, 100,  50),
        (560,  80,  80, 140),
        (120, 300, 140,  50),
        (540, 300, 140,  50),
        (300, 250, 200,  50),
        (160, 400,  80, 140),
        (560, 400,  80, 140),
        (340, 430, 120,  50),
    ],
    player_spawn=(150, 300),
    bot_spawn=(650, 300),
    bg_color=(5, 8, 15),
    wall_color=(30, 60, 120),
    wall_edge_color=(100, 180, 255),
)

# ── Map 2: Open Field (fast-paced, minimal walls) ─────────────────────────────
MAP_OPEN_FIELD = MapConfig(
    name="OPEN FIELD",
    walls=[
        (350, 200, 100, 200),
        (150, 150, 50, 80),
        (600, 150, 50, 80),
        (200, 450, 400, 40),
    ],
    player_spawn=(100, 100),
    bot_spawn=(700, 100),
    bg_color=(5, 12, 8),
    wall_color=(60, 140, 80),
    wall_edge_color=(100, 255, 150),
)

# ── Map 3: Maze (complex, tactical) ───────────────────────────────────────────
MAP_MAZE = MapConfig(
    name="MAZE",
    walls=[
        (80, 50, 300, 40),
        (420, 50, 300, 40),
        (80, 150, 40, 300),
        (680, 150, 40, 300),
        (150, 200, 100, 50),
        (550, 200, 100, 50),
        (250, 300, 300, 50),
        (150, 400, 100, 50),
        (550, 400, 100, 50),
        (300, 150, 50, 100),
        (450, 200, 50, 150),
        (200, 450, 400, 50),
    ],
    player_spawn=(120, 100),
    bot_spawn=(680, 500),
    bg_color=(12, 8, 18),
    wall_color=(120, 40, 140),
    wall_edge_color=(200, 100, 255),
)

# Available maps for rotation
MAPS = [MAP_SYMMETRICAL, MAP_OPEN_FIELD, MAP_MAZE]

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
        # All thinking observations from this session (for end-of-game memory save)
        self._obs_lock: threading.Lock = threading.Lock()
        self.observations: list[str] = []

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

    def add_observation(self, text: str):
        """Thread-safe: append a thinking line to this session's observations."""
        with self._obs_lock:
            self.observations.append(text)
            # Cap to last 200 so memory doesn't grow unbounded mid-session
            if len(self.observations) > 200:
                self.observations = self.observations[-200:]

    def get_observations(self) -> list[str]:
        with self._obs_lock:
            return list(self.observations)


# ═════════════════════════════════════════════════════════════════════════════
# WALL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def make_walls(map_config: MapConfig | None = None) -> list[WallRect]:
    """Create walls from a map config, or use default symmetrical map."""
    config = map_config or MAP_SYMMETRICAL
    return [WallRect(x, y, w, h) for x, y, w, h in config.walls]


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

def draw_walls(surf: pygame.Surface, walls: list[WallRect], map_config: MapConfig | None = None):
    config = map_config or MAP_SYMMETRICAL
    for w in walls:
        pygame.draw.rect(surf, config.wall_color, (w.x, w.y, w.w, w.h))
        pygame.draw.rect(surf, config.wall_edge_color, (w.x, w.y, w.w, w.h), 2)


def draw_entity(surf: pygame.Surface, e: Entity, color: tuple, dark: tuple, label: str):
    if not e.alive:
        return
    # Draw enhanced glow around entity
    glow_surf = pygame.Surface((PLAYER_R*6, PLAYER_R*6), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*color, 80), (PLAYER_R*3, PLAYER_R*3), PLAYER_R*2.5)
    pygame.draw.circle(glow_surf, (*color, 40), (PLAYER_R*3, PLAYER_R*3), PLAYER_R*3.2)
    pygame.draw.circle(glow_surf, (*color, 15), (PLAYER_R*3, PLAYER_R*3), PLAYER_R*3.8)
    surf.blit(glow_surf, (int(e.x - PLAYER_R*3), int(e.y - PLAYER_R*3)))
    # Draw shadow
    pygame.draw.circle(surf, (0, 0, 0, 100),  (int(e.x), int(e.y) + 3), PLAYER_R + 3)
    # Draw main body with enhanced outline
    pygame.draw.circle(surf, dark,  (int(e.x), int(e.y)), PLAYER_R + 2)
    pygame.draw.circle(surf, color, (int(e.x), int(e.y)), PLAYER_R)
    pygame.draw.circle(surf, (255, 255, 200), (int(e.x), int(e.y)), 5)
    pygame.draw.circle(surf, color, (int(e.x), int(e.y)), PLAYER_R, 2)
    txt = _FONT_LABEL.render(label, True, color)
    text_shadow = pygame.font.SysFont("monospace", 11, bold=True).render(label, True, (0, 0, 0))
    surf.blit(text_shadow, (int(e.x) - text_shadow.get_width()//2 + 1, int(e.y) - PLAYER_R - 21))
    surf.blit(txt, (int(e.x) - txt.get_width()//2, int(e.y) - PLAYER_R - 22))


# Pre-baked bullet glow surfaces (created once after pygame.init)
_BULLET_GLOW_P: pygame.Surface | None = None
_BULLET_GLOW_B: pygame.Surface | None = None

def _init_bullet_surfaces():
    global _BULLET_GLOW_P, _BULLET_GLOW_B
    sz = BULLET_R * 8
    _BULLET_GLOW_P = pygame.Surface((sz, sz), pygame.SRCALPHA)
    pygame.draw.circle(_BULLET_GLOW_P, (*C_PBULLET, 120), (sz//2, sz//2), BULLET_R*2.5)
    pygame.draw.circle(_BULLET_GLOW_P, (*C_PBULLET, 60), (sz//2, sz//2), BULLET_R*3.5)
    pygame.draw.circle(_BULLET_GLOW_P, (*C_PBULLET, 20), (sz//2, sz//2), BULLET_R*4.5)
    _BULLET_GLOW_B = pygame.Surface((sz, sz), pygame.SRCALPHA)
    pygame.draw.circle(_BULLET_GLOW_B, (*C_BBULLET, 120), (sz//2, sz//2), BULLET_R*2.5)
    pygame.draw.circle(_BULLET_GLOW_B, (*C_BBULLET, 60), (sz//2, sz//2), BULLET_R*3.5)
    pygame.draw.circle(_BULLET_GLOW_B, (*C_BBULLET, 20), (sz//2, sz//2), BULLET_R*4.5)

def draw_bullet(surf: pygame.Surface, b: Bullet):
    color = C_PBULLET if b.owner == "player" else C_BBULLET
    glow  = _BULLET_GLOW_P if b.owner == "player" else _BULLET_GLOW_B
    if glow:
        surf.blit(glow, (int(b.x) - BULLET_R*4, int(b.y) - BULLET_R*4))
    pygame.draw.circle(surf, color, (int(b.x), int(b.y)), BULLET_R)
    pygame.draw.circle(surf, (255, 255, 220), (int(b.x), int(b.y)), max(2, BULLET_R - 2))


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


def get_ui_alpha(entity_x, entity_y, ui_x, ui_y, threshold=150):
    dist = math.hypot(entity_x - ui_x, entity_y - ui_y)
    if dist < threshold:
        return max(30, int(100 - (threshold - dist) * 0.4))
    return 100

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
             screen_flash: float, map_config: MapConfig, walls: list[WallRect]):
    global _FLASH_SURF
    if screen_flash > 0:
        if _FLASH_SURF is None:
            _FLASH_SURF = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        alpha = int(min(180, screen_flash * 400))
        _FLASH_SURF.fill((255, 40, 40, alpha))
        surf.blit(_FLASH_SURF, (0, 0))

    # Calculate transparency based on entity proximity to HP bars
    left_hp_alpha = get_ui_alpha(player.x, player.y, 90, 55) if player.alive else 100
    right_hp_alpha = get_ui_alpha(bot.x, bot.y, 715, 55) if bot.alive else 100
    
    # HP bar backgrounds fade when entities approach
    temp_bg_left = pygame.Surface((160, 50), pygame.SRCALPHA)
    pygame.draw.rect(temp_bg_left, (0, 0, 0, int(100 * left_hp_alpha / 100)), (0, 0, 160, 50), border_radius=6)
    pygame.draw.rect(temp_bg_left, (50, 50, 100, int(200 * left_hp_alpha / 100)), (0, 0, 160, 50), 2, border_radius=6)
    surf.blit(temp_bg_left, (10, 20))
    
    temp_bg_right = pygame.Surface((160, 50), pygame.SRCALPHA)
    pygame.draw.rect(temp_bg_right, (0, 0, 0, int(100 * right_hp_alpha / 100)), (0, 0, 160, 50), border_radius=6)
    pygame.draw.rect(temp_bg_right, (50, 50, 100, int(200 * right_hp_alpha / 100)), (0, 0, 160, 50), 2, border_radius=6)
    surf.blit(temp_bg_right, (635, 20))
    
    draw_hp_bar(surf,  20, 30, player.hp, 130, "YOU",    C_PLAYER)
    draw_hp_bar(surf, 650, 30, bot.hp,    130, "AI BOT", C_BOT)

    # ── Map name display (top center) ─────────────────────────────────────────
    map_txt = _FONT_HUD.render(map_config.name, True, C_GOLD)
    surf.blit(map_txt, (SCREEN_W//2 - map_txt.get_width()//2, 12))

    panel_x, panel_y = 10, SCREEN_H - 130
    # AI panel proximity transparency
    panel_alpha = get_ui_alpha(player.x, player.y, panel_x + 110, panel_y + 60) if player.alive else 100
    
    # Glowing panel background with dynamic transparency
    temp_panel_bg = pygame.Surface((224, 124), pygame.SRCALPHA)
    pygame.draw.rect(temp_panel_bg, (0, 0, 0, int(140 * panel_alpha / 100)), (2, 2, 220, 120), border_radius=5)
    pygame.draw.rect(temp_panel_bg, (15, 10, 30, int(255 * panel_alpha / 100)), (4, 4, 220, 120), border_radius=4)
    pygame.draw.rect(temp_panel_bg, (*C_PBULLET, int(200 * panel_alpha / 100)), (4, 4, 220, 120), 3, border_radius=4)
    pygame.draw.rect(temp_panel_bg, (150, 100, 255, int(120 * panel_alpha / 100)), (4, 4, 220, 120), 1, border_radius=4)
    surf.blit(temp_panel_bg, (panel_x - 2, panel_y - 2))
    
    surf.blit(_FONT_HUD.render("▶ AI VECTOR", True, C_BOT), (panel_x+6, panel_y+5))
    surf.blit(_FONT_HUD.render(f"  dx: {ai_dx:+.2f}  dy: {ai_dy:+.2f}", True, C_TEXT), (panel_x+6, panel_y+22))
    shoot_col = C_BBULLET if ai_shoot else (80, 80, 80)
    shoot_txt = "► FIRE! ◄" if ai_shoot else "  hold"
    surf.blit(_FONT_HUD.render(f"  shoot: {shoot_txt}", True, shoot_col), (panel_x+6, panel_y+38))
    surf.blit(_FONT_HUD.render("▶ AI THINKING", True, C_THINKING), (panel_x+6, panel_y+56))
    for i, line in enumerate(thinking_lines[-3:]):
        surf.blit(_FONT_SMALL.render(line[:28], True, (120, 200, 120)), (panel_x+6, panel_y+72 + i*14))

    # ── Layout hints (bottom) ─────────────────────────────────────────────────
    hint_y = SCREEN_H - 20
    map_hint = "Press [1][2][3] to switch maps"
    surf.blit(_FONT_SMALL.render(map_hint, True, (100, 140, 100)), (SCREEN_W//2 - 120, hint_y))
    
    surf.blit(_FONT_HUD.render(f"FPS {fps_val:5.1f}", True, (100, 100, 100)), (SCREEN_W - 90, hint_y))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN GAME
# ═════════════════════════════════════════════════════════════════════════════

def game_over_screen(surf: pygame.Surface, winner: str):
    font_big = pygame.font.SysFont("monospace", 72, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 24)
    col = C_PLAYER if winner == "player" else C_BOT
    msg = "YOU WIN!" if winner == "player" else "AI WINS!"
    surf.fill(C_BG)
    
    # Draw semi-transparent overlay
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    surf.blit(overlay, (0, 0))
    
    # Draw glowing background box with enhanced glow
    box_w, box_h = 500, 250
    box_x, box_y = (SCREEN_W - box_w) // 2, (SCREEN_H - box_h) // 2
    
    # Multi-layer glow effect
    for i, alpha in enumerate([20, 30, 40]):
        glow = pygame.Surface((box_w + 20 + i*4, box_h + 20 + i*4), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*col, alpha), (10 + i*2, 10 + i*2, box_w - i*4, box_h - i*4), border_radius=10)
        surf.blit(glow, (box_x - 10 - i*2, box_y - 10 - i*2))
    
    pygame.draw.rect(surf, (12, 12, 25), (box_x, box_y, box_w, box_h), border_radius=8)
    pygame.draw.rect(surf, col, (box_x, box_y, box_w, box_h), 4, border_radius=8)
    pygame.draw.rect(surf, (*col, 100), (box_x, box_y, box_w, box_h), 1, border_radius=8)
    
    # Render text with enhanced outline effect
    txt = font_big.render(msg, True, col)
    txt_outline = font_big.render(msg, True, (0, 0, 0))
    for dx, dy in [(-3,-3), (3,-3), (-3,3), (3,3), (-2,-2), (2,-2), (-2,2), (2,2), (-1,0), (1,0), (0,-1), (0,1)]:
        surf.blit(txt_outline, (box_x + (box_w - txt.get_width())//2 + dx, box_y + 50 + dy))
    surf.blit(txt, (box_x + (box_w - txt.get_width())//2, box_y + 50))
    
    sub = font_sm.render("Press R to restart  |  ESC to quit", True, C_TEXT)
    surf.blit(sub, sub.get_rect(center=(SCREEN_W//2, box_y + box_h - 30)))
    pygame.display.flip()

def difficulty_selection_screen(surf: pygame.Surface) -> int:
    """
    Display difficulty selection screen.
    Returns the index of the selected difficulty (0, 1, 2, or 3).
    """
    font_title = pygame.font.SysFont("monospace", 56, bold=True)
    font_label = pygame.font.SysFont("monospace", 20, bold=True)
    font_hint = pygame.font.SysFont("monospace", 16)
    
    selected = 1  # default to NORMAL
    selecting = True
    
    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 1  # default to normal if quit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 1
                elif event.key == pygame.K_1:
                    selected = 0
                    selecting = False
                elif event.key == pygame.K_2:
                    selected = 1
                    selecting = False
                elif event.key == pygame.K_3:
                    selected = 2
                    selecting = False
                elif event.key == pygame.K_4:
                    selected = 3
                    selecting = False
        
        surf.fill(C_BG)
        
        # Title with glow
        title = font_title.render("SELECT DIFFICULTY", True, C_GOLD)
        title_glow = pygame.Surface((title.get_width() + 20, title.get_height() + 10), pygame.SRCALPHA)
        pygame.draw.rect(title_glow, (*C_GOLD, 30), (10, 5, title.get_width(), title.get_height()), border_radius=4)
        surf.blit(title_glow, (SCREEN_W//2 - title.get_width()//2 - 10, 30))
        surf.blit(title, (SCREEN_W//2 - title.get_width()//2, 35))
        
        # Draw 4 difficulty options
        spacing = 190
        start_x = 20
        
        for i, diff in enumerate(DIFFICULTIES):
            box_x = start_x + i * spacing
            box_y = 140
            box_w, box_h = 170, 280
            
            # Highlight selected
            is_selected = i == selected
            border_col = diff.color if is_selected else (60, 60, 80)
            border_width = 4 if is_selected else 2
            
            # Glowing glow for selected
            if is_selected:
                glow = pygame.Surface((box_w + 12, box_h + 12), pygame.SRCALPHA)
                pygame.draw.rect(glow, (*diff.color, 80), (6, 6, box_w, box_h), border_radius=6)
                pygame.draw.rect(glow, (*diff.color, 30), (4, 4, box_w+4, box_h+4), border_radius=7)
                surf.blit(glow, (box_x - 6, box_y - 6))
            
            # Box background
            pygame.draw.rect(surf, (15, 15, 35), (box_x, box_y, box_w, box_h), border_radius=4)
            pygame.draw.rect(surf, border_col, (box_x, box_y, box_w, box_h), border_width, border_radius=4)
            
            # Difficulty name
            name_txt = font_label.render(diff.name, True, diff.color)
            if is_selected:
                for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
                    name_outline = font_label.render(diff.name, True, (0, 0, 0))
                    surf.blit(name_outline, (box_x + (box_w - name_txt.get_width())//2 + dx, box_y + 20 + dy))
            surf.blit(name_txt, (box_x + (box_w - name_txt.get_width())//2, box_y + 20))
            
            # Stats
            stats = [
                f"Speed: {diff.bot_speed:.0f}",
                f"Cooldown: {diff.bot_cooldown:.2f}s",
                f"Predict: {diff.predict_time:.2f}s"
            ]
            
            y_offset = box_y + 70
            for stat in stats:
                stat_txt = pygame.font.SysFont("monospace", 13).render(stat, True, (150, 150, 180))
                surf.blit(stat_txt, (box_x + 10, y_offset))
                y_offset += 30
            
            # Key hint
            key_txt = font_hint.render(f"Press [{i+1}]", True, border_col)
            surf.blit(key_txt, (box_x + (box_w - key_txt.get_width())//2, box_y + box_h - 35))
        
        # Bottom instruction
        hint = pygame.font.SysFont("monospace", 16).render(
            "Press 1, 2, 3, or 4 to select difficulty  |  ESC to default",
            True, C_GOLD
        )
        surf.blit(hint, (SCREEN_W//2 - hint.get_width()//2, SCREEN_H - 40))
        
        pygame.display.flip()
        pygame.time.wait(16)
    
    return selected

def map_selection_screen(surf: pygame.Surface) -> int:
    """
    Display interactive map selection screen.
    Returns the index of the selected map (0, 1, or 2).
    """
    font_title = pygame.font.SysFont("monospace", 56, bold=True)
    font_label = pygame.font.SysFont("monospace", 22, bold=True)
    font_desc = pygame.font.SysFont("monospace", 16)
    font_hint = pygame.font.SysFont("monospace", 18)
    
    selected = 0
    selecting = True
    
    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0  # default to first map if quit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 0
                elif event.key == pygame.K_1:
                    selected = 0
                    selecting = False
                elif event.key == pygame.K_2:
                    selected = 1
                    selecting = False
                elif event.key == pygame.K_3:
                    selected = 2
                    selecting = False
        
        surf.fill(C_BG)
        
        # Title
        title = font_title.render("SELECT A MAP", True, C_GOLD)
        surf.blit(title, (SCREEN_W//2 - title.get_width()//2, 30))
        
        # Draw 3 map options side by side
        map_spacing = 250
        start_x = 50
        
        for i, map_cfg in enumerate(MAPS):
            box_x = start_x + i * map_spacing
            box_y = 120
            box_w, box_h = 220, 300
            
            # Highlight selected map
            border_col = C_GOLD if i == selected else (80, 80, 80)
            border_width = 5 if i == selected else 2
            
            # Glowing background for selected
            if i == selected:
                glow = pygame.Surface((box_w + 12, box_h + 12), pygame.SRCALPHA)
                pygame.draw.rect(glow, (*C_GOLD, 60), (6, 6, box_w, box_h), border_radius=4)
                pygame.draw.rect(glow, (*C_GOLD, 20), (4, 4, box_w+4, box_h+4), border_radius=5)
                surf.blit(glow, (box_x - 6, box_y - 6))
            
            pygame.draw.rect(surf, (20, 20, 40), (box_x, box_y, box_w, box_h), border_radius=2)
            pygame.draw.rect(surf, border_col, (box_x, box_y, box_w, box_h), border_width, border_radius=2)
            
            # Draw map preview walls
            scale_x = (box_w - 10) / SCREEN_W
            scale_y = (box_h - 50) / SCREEN_H
            preview_x = box_x + 5
            preview_y = box_y + 5
            
            for x, y, w, h in map_cfg.walls:
                px = preview_x + int(x * scale_x)
                py = preview_y + int(y * scale_y)
                pw = int(w * scale_x)
                ph = int(h * scale_y)
                pygame.draw.rect(surf, map_cfg.wall_color, (px, py, max(1, pw), max(1, ph)))
            
            # Draw spawn points on preview
            if map_cfg.player_spawn:
                p_px = preview_x + int(map_cfg.player_spawn[0] * scale_x)
                p_py = preview_y + int(map_cfg.player_spawn[1] * scale_y)
                pygame.draw.circle(surf, C_PLAYER, (int(p_px), int(p_py)), 3)
            
            if map_cfg.bot_spawn:
                b_px = preview_x + int(map_cfg.bot_spawn[0] * scale_x)
                b_py = preview_y + int(map_cfg.bot_spawn[1] * scale_y)
                pygame.draw.circle(surf, C_BOT, (int(b_px), int(b_py)), 3)
            
            # Map name and description
            name_txt = font_label.render(map_cfg.name, True, C_GOLD)
            # Draw name with outline if selected
            if i == selected:
                for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
                    name_outline = font_label.render(map_cfg.name, True, (0, 0, 0))
                    surf.blit(name_outline, (box_x + 10 + dx, box_y + box_h - 40 + dy))
            surf.blit(name_txt, (box_x + 10, box_y + box_h - 40))
            
            # Key hint
            key_txt = font_hint.render(f"Press [{i+1}]", True, border_col)
            surf.blit(key_txt, (box_x + 10, box_y + box_h - 18))
        
        # Bottom instruction with glow
        hint = font_desc.render("Press 1, 2, or 3 to select a map  |  ESC to quit", True, C_GOLD)
        hint_shadow = font_desc.render("Press 1, 2, or 3 to select a map  |  ESC to quit", True, (0, 0, 0, 100))
        surf.blit(hint_shadow, (SCREEN_W//2 - hint.get_width()//2 + 2, SCREEN_H - 38))
        surf.blit(hint, (SCREEN_W//2 - hint.get_width()//2, SCREEN_H - 40))
        
        pygame.display.flip()
        pygame.time.wait(16)
    
    return selected

def map_intro_screen(surf: pygame.Surface, map_config: MapConfig, duration: float = 2.0):
    """Display map intro screen briefly before the game starts."""
    font_title = pygame.font.SysFont("monospace", 44, bold=True)
    font_desc = pygame.font.SysFont("monospace", 18, bold=True)
    
    start_time = pygame.time.get_ticks() / 1000.0
    elapsed = 0.0
    while elapsed < duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        
        elapsed = (pygame.time.get_ticks() / 1000.0) - start_time
        
        # Draw map with preview
        surf.fill(map_config.bg_color)
        
        # Draw walls preview with glow
        for x, y, w, h in map_config.walls:
            # Wall glow
            glow_surf = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*map_config.wall_edge_color, 40), (3, 3, w, h))
            surf.blit(glow_surf, (x - 3, y - 3))
            
            pygame.draw.rect(surf, map_config.wall_color, (x, y, w, h))
            pygame.draw.rect(surf, map_config.wall_edge_color, (x, y, w, h), 2)
        
        # Draw spawn points with pulsing glow
        pulse = 0.5 + 0.5 * math.sin(elapsed * math.pi * 3)
        
        # Player spawn
        glow_r = int(12 + pulse * 4)
        pygame.draw.circle(surf, (*C_PLAYER, int(100 * pulse)), map_config.player_spawn, glow_r)
        pygame.draw.circle(surf, C_PLAYER, map_config.player_spawn, 10)
        pygame.draw.circle(surf, (255, 255, 255), map_config.player_spawn, 4)
        
        # Bot spawn
        pygame.draw.circle(surf, (*C_BOT, int(100 * pulse)), map_config.bot_spawn, glow_r)
        pygame.draw.circle(surf, C_BOT, map_config.bot_spawn, 10)
        pygame.draw.circle(surf, (255, 255, 255), map_config.bot_spawn, 4)
        
        txt_you = font_desc.render("YOU", True, C_PLAYER)
        txt_ai = font_desc.render("AI", True, C_BOT)
        surf.blit(txt_you, (map_config.player_spawn[0] - txt_you.get_width()//2, map_config.player_spawn[1] - 30))
        surf.blit(txt_ai, (map_config.bot_spawn[0] - txt_ai.get_width()//2, map_config.bot_spawn[1] - 30))
        
        # Draw title with outline and glow
        title_txt = font_title.render(map_config.name, True, C_GOLD)
        title_outline = font_title.render(map_config.name, True, (0, 0, 0))
        for dx, dy in [(-2,-2), (2,-2), (-2,2), (2,2), (-1,0), (1,0), (0,-1), (0,1)]:
            surf.blit(title_outline, (SCREEN_W//2 - title_txt.get_width()//2 + dx, 20 + dy))
        surf.blit(title_txt, (SCREEN_W//2 - title_txt.get_width()//2, 20))
        
        # Pulsing loading text
        opacity = int(200 * (0.5 + 0.5 * math.sin(elapsed * math.pi * 2)))
        desc_txt = font_desc.render("Loading...", True, (min(255, opacity + 50), min(255, opacity + 50), min(255, opacity + 100)))
        surf.blit(desc_txt, (SCREEN_W//2 - desc_txt.get_width()//2, SCREEN_H - 50))
        
        pygame.display.flip()
        pygame.time.wait(16)
    
    return True
def build_ai_game_state(player: Entity, bot: Entity,
                        bullets: list[Bullet],
                        walls: list[WallRect],
                        difficulty: DifficultyConfig = None) -> dict:
    """Serialise the dynamic game situation for the Claude pipeline."""
    if difficulty is None:
        difficulty = DIFFICULTY_NORMAL
    
    # Predict player pos PREDICT_T seconds ahead
    pred_px = player.x + player.vx * difficulty.predict_time
    pred_py = player.y + player.vy * difficulty.predict_time
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


def run_game(shared_state: GameState, stop_event: threading.Event, 
             player_memory: str = ""):
    """
    Synchronous game loop running in the main thread at a solid 60 FPS.
    The Claude pipeline runs in a separate daemon thread — completely decoupled.
    
    Returns the AI thread so main can manage it.
    """
    pygame.init()
    pygame.font.init()
    _init_draw_caches()   # build font + grid + bullet surface caches

    surf  = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    
    # ── Difficulty selection screen ────────────────────────────────────────────
    difficulty_index = difficulty_selection_screen(surf)
    difficulty = DIFFICULTIES[difficulty_index]
    
    # ── Map selection screen ───────────────────────────────────────────────────
    current_map_index = map_selection_screen(surf)
    current_map = MAPS[current_map_index]
    walls = make_walls(current_map)

    # ── START AI THREAD ONLY AFTER MAP SELECTION ───────────────────────────────
    def on_thinking(text: str):
        shared_state.latest_thinking = text
    
    ai_thread = AIThread(shared_state, stop_event, on_thinking,
                         player_memory=player_memory, difficulty=difficulty)
    ai_thread.start()
    print(f"[Game] AI pipeline thread started. Playing on {current_map.name} ({difficulty.name})")

    thinking_lines: list[str] = []

    # ── mutable game objects ──────────────────────────────────────────────────
    player         = Entity(x=current_map.player_spawn[0], y=current_map.player_spawn[1])
    bot            = Entity(x=current_map.bot_spawn[0], y=current_map.bot_spawn[1])
    bullets        = []
    particles      = []
    screen_flash   = 0.0
    telemetry_timer = 0.0
    bot_reflex     = BotState(x=bot.x, y=bot.y)
    game_over      = False
    winner         = ""
    fps_val        = 60.0

    def reset(map_config: MapConfig | None = None):
        nonlocal player, bot, bullets, particles, screen_flash, telemetry_timer, bot_reflex, game_over, winner, walls, current_map
        if map_config:
            current_map = map_config
            walls = make_walls(current_map)
        player          = Entity(x=current_map.player_spawn[0], y=current_map.player_spawn[1])
        bot             = Entity(x=current_map.bot_spawn[0], y=current_map.bot_spawn[1])
        bullets         = []
        particles       = []
        screen_flash    = 0.0
        telemetry_timer = 0.0
        bot_reflex      = BotState(x=bot.x, y=bot.y)
        game_over       = False
        winner          = ""
        # Clear observations so the next round starts fresh
        with shared_state._obs_lock:
            shared_state.observations = []

    print("[Game] Loop started — pipeline is live.")

    # Show initial map intro
    map_intro_screen(surf, current_map, duration=1.5)

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
                # ── Map selection (1/2/3 keys) ───────────────────────────────
                if event.key == pygame.K_1 and current_map_index != 0:
                    current_map_index = 0
                    reset(MAPS[current_map_index])
                    map_intro_screen(surf, MAPS[current_map_index], duration=1.0)
                    print(f"[Map] Switched to {MAPS[current_map_index].name}")
                elif event.key == pygame.K_2 and current_map_index != 1:
                    current_map_index = 1
                    reset(MAPS[current_map_index])
                    map_intro_screen(surf, MAPS[current_map_index], duration=1.0)
                    print(f"[Map] Switched to {MAPS[current_map_index].name}")
                elif event.key == pygame.K_3 and current_map_index != 2:
                    current_map_index = 2
                    reset(MAPS[current_map_index])
                    map_intro_screen(surf, MAPS[current_map_index], duration=1.0)
                    print(f"[Map] Switched to {MAPS[current_map_index].name}")

        if stop_event.is_set():
            break

        if game_over:
            # Show game over screen and exit
            game_over_screen(surf, winner)
            pygame.time.wait(2000)  # Show for 2 seconds
            stop_event.set()
            break

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
            bot_reflex, player_bullet_infos, walls, difficulty.bot_speed,
            SCREEN_W, SCREEN_H, dt
        )
        bot.vx = rvx
        bot.vy = rvy

        if rshoot and bot.can_shoot():
            pred_px = player.x + player.vx * difficulty.predict_time
            pred_py = player.y + player.vy * difficulty.predict_time
            b = bot.fire(pred_px, pred_py, "bot", cooldown=difficulty.bot_cooldown)
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
        if not player.alive and not game_over:
            winner = "bot";   game_over = True
            # Save this session's observations to persistent memory (runs ~1-2s in background)
            threading.Thread(
                target=save_session,
                args=(shared_state.get_observations(), winner),
                daemon=True,
            ).start()
        elif not bot.alive and not game_over:
            winner = "player"; game_over = True
            threading.Thread(
                target=save_session,
                args=(shared_state.get_observations(), winner),
                daemon=True,
            ).start()

        # ── Telemetry → pipeline (every 250 ms) ───────────────────────────────
        telemetry_timer += dt
        if telemetry_timer >= 0.25:
            telemetry_timer = 0.0
            shared_state.update(build_ai_game_state(player, bot, bullets, walls, difficulty))

        # ── Render ────────────────────────────────────────────────────────────
        surf.fill(current_map.bg_color)
        surf.blit(_GRID_SURF, (0, 0))   # pre-rendered grid — single blit

        draw_walls(surf, walls, current_map)
        draw_particles(surf, particles)
        for b in bullets:
            draw_bullet(surf, b)
        draw_entity(surf, player, C_PLAYER, C_PLAYER_DARK, "YOU")
        draw_entity(surf, bot,    C_BOT,    C_BOT_DARK,    "AI")
        if player.alive:
            draw_aim_line(surf, player.x, player.y, mx, my)
        pygame.draw.rect(surf, current_map.wall_edge_color, (0, 0, SCREEN_W, SCREEN_H), 3)

        fps_val = clock.get_fps()

        # Pull latest thinking text from the shared state to display + record as observation
        t = shared_state.latest_thinking
        if t and t not in thinking_lines:
            for line in t.split("\n"):
                line = line.strip()
                if line:
                    thinking_lines.append(line)
                    shared_state.add_observation(line)  # persist for end-of-game memory save
            thinking_lines = thinking_lines[-12:]

        draw_hud(surf, player, bot, ai_dx, ai_dy, ai_shoot,
                 thinking_lines, fps_val, screen_flash, current_map, walls)
        pygame.display.flip()

    pygame.quit()
    print("[Game] Exited cleanly.")
    
    return ai_thread


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
                 on_thinking, player_memory: str = "", difficulty: DifficultyConfig = None):
        super().__init__(daemon=True)
        self.shared_state  = shared_state
        self.stop_event    = stop_event
        self.on_thinking   = on_thinking
        self.player_memory = player_memory
        self.difficulty    = difficulty or DIFFICULTY_NORMAL

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
                player_memory     = self.player_memory,
            )

        asyncio.run(_pipeline())


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load cross-session player memory ─────────────────────────────────────
    player_memory = load_player_memory()
    if player_memory:
        print(f"[Main] Loaded player memory ({len(player_memory)} chars) from previous sessions.")
    else:
        print("[Main] No player memory found — starting fresh.")

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

    # Game loop and AI thread start here — AI thread only starts AFTER map selection
    ai_thread = run_game(shared_state, stop_event, player_memory=player_memory)

    stop_event.set()
    ai_thread.join(timeout=3.0)
    print("[Main] Done.")


if __name__ == "__main__":
    main()
