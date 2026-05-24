#!/usr/bin/env python3
"""
ApexAI – F1 winner prediction
"""
import io
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# On Windows the default console codepage is cp1252 which can't encode the
# emoji that f1radio (📻 / 🎙) prints during `load(...)`.  Force stdout/stderr
# to UTF-8 with replacement so third-party packages don't blow up the
# background radio thread.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("tkinter is required.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prediction import (
    run_predictions,
    run_predictions_all_races,
    predict_with_standings,
    F1_POINTS,
    load_last_result,
    acquire_singleton,
    get_event_schedule_cached,
)
from team_colors import TEAM_COLORS
from team_logos import load_logo
from track_layouts import get_track

try:
    import f1radio
    HAS_F1RADIO = True
except ImportError:
    HAS_F1RADIO = False

# Direct FIA-archive loader (preferred): merges TeamRadio.json +
# TeamRadio.jsonStream so we don't lag OpenF1's cache and we get every
# capture published, not just the ones OpenF1 has ingested.
try:
    import radio_fia
    HAS_RADIO = True
except ImportError:
    radio_fia = None
    HAS_RADIO = HAS_F1RADIO

# Cross-platform audio playback.  playsound3 ships with the f1radio
# `[playback]` extra and uses native APIs on Windows/macOS/Linux so we don't
# have to depend on `afplay` or `ffplay` being on PATH.
try:
    from playsound3 import playsound as _playsound
    HAS_PLAYSOUND = True
except ImportError:
    _playsound = None
    HAS_PLAYSOUND = False

# -- Theme: aligned with the official Formula 1 brand palette --
# Source: F1's 2018-present brand guidelines (formula1.com).  Primary
# colour is "F1 Red" #E10600, with a slightly-blue charcoal "F1 Black"
# #15151E sitting under it.  Everything else here is a tonal step in
# the same family so the whole UI reads as an F1 broadcast frame.
BG = "#15151E"          # F1 Black (primary surface)
BG_SURFACE = "#1F1F27"  # Slightly raised surface
BG_CARD = "#2A2A33"     # Cards / panels
BG_HOVER = "#33333D"    # Hover state
BORDER = "#3D3D44"      # Subtle neutral border
# Accent reds (all variants of official F1 Red #E10600).
GOLD = "#E10600"        # Primary accent — official F1 Red
GOLD_DIM = "#5A0300"    # Pressed / dim state
GOLD_GLOW = "#FF1A1A"   # Bright hover / glow
WHITE = "#F6F4F1"       # F1 broadcast off-white (kinder on dark bg than pure white)
GRAY = "#9E9EA8"        # Mid grey (good contrast on F1 Black)
MUTED = "#6A6A75"       # Muted captions
RED = GOLD              # Alias – all "red" accents resolve to official F1 Red
GREEN = "#2ED87A"       # Retained for "go" indicators (DRS zones, MOM markers)

# Carbon-fiber accent: now mid-grey tones over the F1 Black base, no
# longer a tinted near-black weave (which clashed with the new primary
# surface colour).
CF_LIGHT = "#2A2A33"
CF_DARK = "#1A1A22"


def tc(team: str) -> str:
    return TEAM_COLORS.get(team, "#555566")


_CF_TILE_CACHE = {}
_BG_CACHE = {}
_F1_LOGO_CACHE = {}
_TROPHY_CACHE = {}
_AMBIENT_CACHE = {}  # keyed by (kind, w, h, *params)


# Official F1 brand red.  Used for the logo + key accents.
F1_RED = "#E10600"
F1_RED_RGB = (225, 6, 0)


def _try_font(candidates, size, italic=False):
    """Resolve to the first available font from ``candidates``.

    Returns an :class:`ImageFont` instance or ``None`` if nothing is
    available.  Used to find a heavy sans-serif (Arial Black, Helvetica
    Neue Bold, Impact, etc.) for the logo without hard-coding a single
    family.
    """
    if not HAS_PIL:
        return None
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


_F1_LOGO_SOURCE = None  # lazily-loaded PIL image of the official mark


def _load_official_f1_logo():
    """Load the high-res official F1 wordmark from ``logos/f1.png``.

    Cached after the first read.  Returns ``None`` if the file is
    missing so callers can fall back to the procedural homage.
    """
    global _F1_LOGO_SOURCE
    if _F1_LOGO_SOURCE is not None:
        return _F1_LOGO_SOURCE
    if not HAS_PIL:
        return None
    try:
        base = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    except Exception:
        base = Path.cwd()
    path = base / "logos" / "f1.png"
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    _F1_LOGO_SOURCE = img
    return img


def _make_f1_logo(height: int = 40):
    """Return the official F1 wordmark rendered at ``height`` pixels tall.

    When ``logos/f1.png`` is present (downloaded by ``fetch_logos.py``)
    we resize the genuine mark with Lanczos so it stays crisp at every
    size the UI uses.  If the file is missing (a fresh checkout that
    hasn't run the fetcher yet) we fall back to a procedural red "F1 +
    speed strips" homage so the header never goes blank.
    """
    if not HAS_PIL:
        return None
    if height in _F1_LOGO_CACHE:
        return _F1_LOGO_CACHE[height]

    src = _load_official_f1_logo()
    if src is not None:
        sw, sh = src.size
        new_w = max(1, int(round(sw * height / sh)))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = src.resize((new_w, int(height)), resample)
        _F1_LOGO_CACHE[height] = img
        return img

    # ── Fallback (no logos/f1.png on disk) ──────────────────────────────
    font = _try_font(
        ["arialbi.ttf", "arialbd.ttf", "ariblk.ttf",
         "Arial Bold Italic.ttf", "Arial Bold.ttf", "Helvetica.ttc",
         "Impact.ttf"],
        int(height * 0.95),
    )

    pad_x = max(6, height // 4)
    strip_w = int(height * 1.1)
    text = "F1"

    dummy = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    d_meas = ImageDraw.Draw(dummy)
    if hasattr(d_meas, "textbbox"):
        x0, y0, x1, y1 = d_meas.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
        tx_off, ty_off = -x0, -y0
    else:
        tw, th = d_meas.textsize(text, font=font)
        tx_off = ty_off = 0

    canvas_w = pad_x + tw + 6 + strip_w + pad_x
    canvas_h = max(th, height) + 4

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_layer)
    text_y = (canvas_h - th) // 2 + ty_off
    td.text((pad_x + tx_off, text_y), text, font=font, fill=F1_RED_RGB + (255,))

    shear = -0.22
    text_layer = text_layer.transform(
        (canvas_w, canvas_h),
        Image.AFFINE,
        (1, shear, shear * canvas_h * 0.5, 0, 1, 0),
        resample=Image.BICUBIC,
    )
    img.alpha_composite(text_layer)

    strip_x = pad_x + tw + 4
    strip_top = (canvas_h - th) // 2 + int(th * 0.18)
    bar_h = max(2, height // 9)
    bar_gap = max(2, height // 11)
    for i in range(3):
        y = strip_top + i * (bar_h + bar_gap)
        x0 = strip_x + i * (strip_w // 8)
        x1 = strip_x + strip_w
        draw.rectangle([x0, y, x1, y + bar_h], fill=F1_RED_RGB + (255,))

    _F1_LOGO_CACHE[height] = img
    return img


# ---------------------------------------------------------------------------
# Trophy artwork
# ---------------------------------------------------------------------------
# Metallic palettes for each podium tier.  Each tuple is
# (highlight, mid, dark, edge) – four shades let us paint a 3D-looking cup
# + handles + stem + base on a flat 2D canvas.  The bronze palette is a
# warm copper-brown so it reads distinctly from gold; silver is cool grey
# so it doesn't get confused with the off-white card type.
_TROPHY_PALETTES = {
    "gold": (
        (255, 232, 140, 255),
        (240, 195,  82, 255),
        (170, 120,  35, 255),
        (110,  72,  18, 255),
    ),
    "silver": (
        (245, 247, 252, 255),
        (200, 206, 218, 255),
        (138, 145, 158, 255),
        ( 78,  84,  96, 255),
    ),
    "bronze": (
        (240, 178, 112, 255),
        (198, 124,  58, 255),
        (136,  78,  30, 255),
        ( 78,  42,  12, 255),
    ),
}


def _make_trophy_image(height: int = 60, tier: str = "gold"):
    """Render a stylised F1-style podium trophy as a PIL RGBA image.

    Cached by ``(height, tier)`` – the trophy is static once drawn, so
    re-renders are free.  The result is roughly square (slightly taller
    than wide to fit the cup + stem + base proportions of a real F1
    trophy).

    Painted in four metallic tones (highlight → mid → dark → edge) so the
    cup reads as a 3D object on the dark podium card rather than as a
    flat silhouette.  An "F1" engraving on the base ties it back to the
    brand on the gold trophy; silver/bronze get position numerals (2/3)
    instead so the three trophies are unambiguous from a glance.
    """
    if not HAS_PIL:
        return None
    cache_key = (int(height), tier)
    if cache_key in _TROPHY_CACHE:
        return _TROPHY_CACHE[cache_key]

    palette = _TROPHY_PALETTES.get(tier, _TROPHY_PALETTES["gold"])
    gold_hi, gold_mid, gold_dark, gold_edge = palette
    base_label = {"gold": "F1", "silver": "2", "bronze": "3"}.get(tier, "F1")

    # Final output size.
    H_out = max(24, int(height))
    W_out = max(20, int(H_out * 0.85))

    # Supersampling factor — PIL's draw primitives don't anti-alias, which
    # is what makes the trophy look 8-bit/pixelated.  We render at 4x and
    # downsample with LANCZOS so every curve gets free anti-aliasing.
    SS = 4
    H = H_out * SS
    W = W_out * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = W / 2.0

    # ── Cup body ──
    cup_top = int(H * 0.08)
    cup_bot = int(H * 0.50)
    cup_w   = int(W * 0.62)
    draw.ellipse(
        [cx - cup_w / 2, cup_top, cx + cup_w / 2, cup_bot],
        fill=gold_mid, outline=gold_edge, width=SS,
    )
    # Lip (thin band at the top of the cup)
    lip_h = max(2 * SS, int(H * 0.05))
    draw.ellipse(
        [cx - cup_w / 2, cup_top - SS,
         cx + cup_w / 2, cup_top + lip_h],
        fill=gold_dark, outline=gold_edge, width=SS,
    )
    # Inner well (darker oval to suggest cup depth)
    draw.ellipse(
        [cx - cup_w / 2 + 2 * SS, cup_top + SS,
         cx + cup_w / 2 - 2 * SS, cup_top + lip_h - SS],
        fill=gold_edge, outline=None,
    )
    # Highlight – small lighter ellipse on the upper-left curve
    hl_x0 = cx - cup_w / 3
    hl_y0 = cup_top + lip_h + 2 * SS
    hl_x1 = hl_x0 + cup_w * 0.18
    hl_y1 = hl_y0 + (cup_bot - cup_top) * 0.55
    draw.ellipse([hl_x0, hl_y0, hl_x1, hl_y1], fill=gold_hi, outline=None)

    # ── Handles (curved arcs hugging the cup) ──
    handle_w = max(3 * SS, int(W * 0.16))
    h_top = int(H * 0.16)
    h_bot = int(H * 0.42)
    # Left
    draw.arc(
        [cx - cup_w / 2 - handle_w + SS, h_top,
         cx - cup_w / 2 + 4 * SS,        h_bot],
        start=60, end=300, fill=gold_mid, width=3 * SS,
    )
    draw.arc(
        [cx - cup_w / 2 - handle_w + SS, h_top,
         cx - cup_w / 2 + 4 * SS,        h_bot],
        start=60, end=300, fill=gold_edge, width=SS,
    )
    # Right
    draw.arc(
        [cx + cup_w / 2 - 4 * SS,        h_top,
         cx + cup_w / 2 + handle_w - SS, h_bot],
        start=240, end=120, fill=gold_mid, width=3 * SS,
    )
    draw.arc(
        [cx + cup_w / 2 - 4 * SS,        h_top,
         cx + cup_w / 2 + handle_w - SS, h_bot],
        start=240, end=120, fill=gold_edge, width=SS,
    )

    # ── Stem (tapering bridge between cup and base) ──
    stem_top = cup_bot - SS
    stem_bot = int(H * 0.72)
    stem_top_w = int(W * 0.22)
    stem_mid_w = int(W * 0.10)
    stem_bot_w = int(W * 0.14)
    # Two trapezoids for a vase-like profile
    draw.polygon([
        (cx - stem_top_w / 2, stem_top),
        (cx + stem_top_w / 2, stem_top),
        (cx + stem_mid_w / 2, (stem_top + stem_bot) // 2),
        (cx - stem_mid_w / 2, (stem_top + stem_bot) // 2),
    ], fill=gold_mid, outline=gold_edge)
    draw.polygon([
        (cx - stem_mid_w / 2, (stem_top + stem_bot) // 2),
        (cx + stem_mid_w / 2, (stem_top + stem_bot) // 2),
        (cx + stem_bot_w / 2, stem_bot),
        (cx - stem_bot_w / 2, stem_bot),
    ], fill=gold_dark, outline=gold_edge)

    # ── Base plaque (the engraved plinth) ──
    base_top = stem_bot
    base_bot = int(H * 0.94)
    base_top_w = int(W * 0.42)
    base_bot_w = int(W * 0.55)
    draw.polygon([
        (cx - base_top_w / 2, base_top),
        (cx + base_top_w / 2, base_top),
        (cx + base_bot_w / 2, base_bot),
        (cx - base_bot_w / 2, base_bot),
    ], fill=gold_dark, outline=gold_edge)
    # Thin highlight along the top of the base for sheen
    draw.line(
        [(cx - base_top_w / 2 + 2 * SS, base_top + SS),
         (cx + base_top_w / 2 - 2 * SS, base_top + SS)],
        fill=gold_hi, width=SS,
    )
    # Engraving on the base — "F1" for gold, "2"/"3" for silver/bronze.
    # Silver/bronze numerals use the trophy's own dark tone so they read
    # like a tasteful etched engraving instead of fighting the F1 red.
    if H_out >= 36:
        try:
            f1_font = ImageFont.truetype("arialbd.ttf", max(7 * SS, int(H * 0.11)))
        except (OSError, IOError):
            try:
                f1_font = ImageFont.load_default()
            except Exception:
                f1_font = None
        if f1_font is not None:
            text = base_label
            try:
                bbox = draw.textbbox((0, 0), text, font=f1_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = cx - tw / 2 - bbox[0]
                ty = base_top + ((base_bot - base_top) - th) / 2 - bbox[1]
            except Exception:
                tw, th = f1_font.getsize(text) if hasattr(f1_font, "getsize") else (10 * SS, 10 * SS)
                tx = cx - tw / 2
                ty = base_top + ((base_bot - base_top) - th) / 2
            engrave = F1_RED_RGB + (255,) if tier == "gold" else gold_edge
            draw.text((tx, ty), text, font=f1_font, fill=engrave)

    # Down-sample with LANCZOS for smooth, anti-aliased edges.
    img = img.resize((W_out, H_out), Image.LANCZOS)

    _TROPHY_CACHE[cache_key] = img
    return img


# ---------------------------------------------------------------------------
# Laurel wreath – frames the gold trophy on the podium card so the
# winner's slot reads as a proper "champion" presentation rather than a
# bare cup.  Two mirrored sprigs of leaves curve up from a single central
# point, painted in olive-green with gold highlights for that broadcast
# graphic feel.
# ---------------------------------------------------------------------------
def _make_laurel_image(width: int = 110, height: int = 96):
    """Render a laurel wreath as a transparent PIL image.  Cached by size.

    Drawn at 3× resolution and downsampled with LANCZOS so the leaf
    edges and ribbon polygons end up cleanly anti-aliased instead of
    showing PIL's stair-stepped pixel boundaries.
    """
    if not HAS_PIL:
        return None
    key = ("laurel", int(width), int(height))
    cached = _AMBIENT_CACHE.get(key)
    if cached is not None:
        return cached

    W_out, H_out = max(40, int(width)), max(40, int(height))
    SS = 3
    W, H = W_out * SS, H_out * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    leaf_dark = (60,  92,  44, 235)
    leaf_mid  = (98, 142,  60, 240)
    leaf_hi   = (170, 198,  92, 235)
    gold_tip  = (240, 200,  92, 235)

    cx_b = W / 2.0
    cy_b = H * 0.95         # bottom anchor where the two sprigs meet

    # Each sprig is a series of leaves sweeping along a quarter-circle arc
    # from the bottom-centre out to the side and up to the top.
    leaf_count = 9
    for side in (-1, 1):
        for i in range(leaf_count):
            t = i / max(1, leaf_count - 1)
            # Polar position along the wreath's arc (right or left).
            angle = math.pi * 0.05 + t * math.pi * 0.85
            r = W * 0.42
            x = cx_b + side * r * math.sin(angle)
            y = cy_b - r * (math.cos(angle) * 0.55 + 0.15) - H * 0.1

            # Leaf size tapers towards the tips so the wreath silhouette
            # has the classic Greek "chaplet" curve.
            taper = 1.0 - 0.45 * t
            lw = max(4 * SS, int(W * 0.10 * taper))
            lh = max(8 * SS, int(H * 0.18 * taper))

            # Tilt every leaf so the bunch looks layered, not radial.
            tilt_deg = -side * (35 + 25 * (1 - t))

            leaf = Image.new("RGBA", (lw + 2, lh + 2), (0, 0, 0, 0))
            ld = ImageDraw.Draw(leaf)
            ld.ellipse([1, 1, lw, lh],
                       fill=leaf_mid, outline=leaf_dark, width=SS)
            ld.ellipse([2, 1, lw // 2 + 2, lh - 2],
                       fill=leaf_hi, outline=None)
            leaf = leaf.rotate(tilt_deg, resample=Image.BICUBIC, expand=True)
            img.alpha_composite(
                leaf,
                (int(x - leaf.size[0] / 2), int(y - leaf.size[1] / 2)),
            )

    # A small gold ribbon at the bottom where the two sprigs meet.
    ribbon_w = int(W * 0.30)
    ribbon_h = max(4 * SS, int(H * 0.07))
    rx0 = int(cx_b - ribbon_w / 2)
    ry0 = int(cy_b - ribbon_h / 2)
    draw.rectangle([rx0, ry0, rx0 + ribbon_w, ry0 + ribbon_h],
                   fill=gold_tip, outline=(140, 92, 28, 255))
    # Two trailing tails on each side of the ribbon.
    tail_y = ry0 + ribbon_h - 1
    for dx, sign in ((-ribbon_w * 0.45, -1), (ribbon_w * 0.45, 1)):
        tx = int(cx_b + dx)
        draw.polygon(
            [(tx, tail_y),
             (tx + sign * int(ribbon_w * 0.25), tail_y + int(ribbon_h * 1.6)),
             (tx, tail_y + int(ribbon_h * 0.6))],
            fill=gold_tip, outline=(140, 92, 28, 255),
        )

    img = img.resize((W_out, H_out), Image.LANCZOS)
    _AMBIENT_CACHE[key] = img
    return img


# ---------------------------------------------------------------------------
# Champagne / sparkle accents – subtle decorative bits scattered around
# the podium card that give the visualisation a "celebration" feel without
# adding visual noise.
# ---------------------------------------------------------------------------
def _make_sparkle_image(size: int = 14, color=(255, 232, 140, 255)):
    """Tiny 4-point starburst.  Used to pepper the gold trophy with a
    handful of broadcast-style highlights."""
    if not HAS_PIL:
        return None
    key = ("sparkle", int(size), color)
    cached = _AMBIENT_CACHE.get(key)
    if cached is not None:
        return cached

    s = max(6, int(size))
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx_s, cy_s = s / 2, s / 2
    # Long N/S and E/W rays, shorter diagonals.
    draw.line([(cx_s, 1), (cx_s, s - 2)], fill=color, width=1)
    draw.line([(1, cy_s), (s - 2, cy_s)], fill=color, width=1)
    soft = (color[0], color[1], color[2], max(60, color[3] // 3))
    draw.line([(2, 2), (s - 3, s - 3)], fill=soft, width=1)
    draw.line([(s - 3, 2), (2, s - 3)], fill=soft, width=1)
    # Bright core dot.
    draw.ellipse([cx_s - 1, cy_s - 1, cx_s + 1, cy_s + 1], fill=color)

    _AMBIENT_CACHE[key] = img
    return img


# ---------------------------------------------------------------------------
# Ambient decoration artwork (static PIL silhouettes for per-circuit theming)
# ---------------------------------------------------------------------------
def _ambient_cache_get(key):
    return _AMBIENT_CACHE.get(key)


def _ambient_cache_put(key, img):
    _AMBIENT_CACHE[key] = img
    return img


def _make_bonsai_image(height: int = 56, mirror: bool = False):
    """Stylised bonsai silhouette – a curved trunk in a pot with three
    leaf clusters above.  Mirror=True flips it horizontally so corners
    on either side don't feel duplicated."""
    if not HAS_PIL:
        return None
    key = ("bonsai", int(height), bool(mirror))
    cached = _ambient_cache_get(key)
    if cached is not None:
        return cached

    H = max(20, int(height))
    W = max(20, int(H * 1.1))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pot_top = int(H * 0.78)
    pot_w = int(W * 0.55)
    # Pot (warm brown trapezoid)
    pot = [
        (W // 2 - pot_w // 2, pot_top),
        (W // 2 + pot_w // 2, pot_top),
        (W // 2 + pot_w // 2 - 3, H - 2),
        (W // 2 - pot_w // 2 + 3, H - 2),
    ]
    draw.polygon(pot, fill=(78, 50, 28, 230), outline=(40, 24, 12, 255))
    # Pot rim
    draw.rectangle(
        [W // 2 - pot_w // 2 - 1, pot_top - 2,
         W // 2 + pot_w // 2 + 1, pot_top + 1],
        fill=(58, 36, 20, 235),
    )

    # Curved trunk – two short segments forming an S
    trunk_color = (62, 40, 22, 240)
    tx = W // 2
    ty = pot_top
    p1 = (tx, ty)
    p2 = (tx - int(W * 0.08), int(H * 0.55))
    p3 = (tx + int(W * 0.04), int(H * 0.30))
    draw.line([p1, p2], fill=trunk_color, width=3)
    draw.line([p2, p3], fill=trunk_color, width=3)

    # Leaf clusters – two greens for depth
    leaf_dark = (38, 78, 44, 235)
    leaf_mid  = (62, 122, 62, 235)
    for (lx, ly, r) in (
        (int(W * 0.50), int(H * 0.22), int(W * 0.20)),
        (int(W * 0.30), int(H * 0.32), int(W * 0.15)),
        (int(W * 0.68), int(H * 0.30), int(W * 0.16)),
    ):
        draw.ellipse([lx - r, ly - r * 0.6, lx + r, ly + r * 0.6],
                     fill=leaf_dark, outline=None)
        draw.ellipse([lx - r + 2, ly - r * 0.6 + 1,
                      lx + r - 4, ly + r * 0.6 - 3],
                     fill=leaf_mid, outline=None)

    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return _ambient_cache_put(key, img)


# ---------------------------------------------------------------------------
# Canadian maple leaf – the iconic 11-pointed silhouette.  Rendered as a
# single anti-aliased polygon at 4× resolution and downsampled with
# LANCZOS so the lobes / serrations look hand-cut rather than pixelated.
# Used for the Circuit Gilles Villeneuve scene.
# ---------------------------------------------------------------------------
# The right-half outline of the canonical Canadian maple leaf, traced
# top-to-bottom on a unit square.  The polygon is built by emitting these
# points and then mirroring them about x = 0.5 so the silhouette stays
# perfectly symmetric.
_MAPLE_LEAF_HALF = (
    (0.50, 0.04),    # top point
    (0.55, 0.18),    # inset under top
    (0.74, 0.16),    # upper-right shoulder lobe
    (0.66, 0.30),    # inset
    (0.93, 0.40),    # middle-right outer point
    (0.74, 0.52),    # inset
    (0.96, 0.66),    # lower-right outer point
    (0.62, 0.66),    # inner notch above stem
    (0.66, 0.84),    # near-stem shoulder
    (0.55, 0.83),    # stem top-right
    (0.55, 0.96),    # stem bottom-right
)


def _make_maple_leaf_image(size: int = 18, color=(213, 43, 30, 245)):
    """Stylised Canadian maple leaf as an RGBA PIL image."""
    if not HAS_PIL:
        return None
    key = ("maple_leaf", int(size), tuple(color))
    cached = _ambient_cache_get(key)
    if cached is not None:
        return cached

    SS = 4
    S_out = max(8, int(size))
    S = S_out * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pts = [(x * S, y * S) for (x, y) in _MAPLE_LEAF_HALF]
    pts.append((0.50 * S, 0.99 * S))
    for (x, y) in reversed(_MAPLE_LEAF_HALF):
        pts.append(((1.0 - x) * S, y * S))

    edge = (
        max(0, color[0] - 90),
        max(0, color[1] - 25),
        max(0, color[2] - 25),
        255,
    )
    draw.polygon(pts, fill=color, outline=edge)

    # Single hint of a central vein so the leaf doesn't read as a flat
    # silhouette at larger sizes.
    vein_color = (
        max(0, color[0] - 50),
        max(0, color[1] - 14),
        max(0, color[2] - 14),
        200,
    )
    draw.line(
        [(0.50 * S, 0.92 * S), (0.50 * S, 0.20 * S)],
        fill=vein_color,
        width=max(1, SS // 2),
    )

    img = img.resize((S_out, S_out), Image.LANCZOS)
    return _ambient_cache_put(key, img)


def _maple_leaf_rotation_frames(size: int, frame_count: int = 6):
    """Pre-render a small set of rotated maple-leaf frames so tumbling
    leaves can swap images at runtime instead of paying for a fresh
    PIL.Image.rotate every animation tick."""
    if not HAS_PIL:
        return []
    key = ("maple_rot", int(size), int(frame_count))
    cached = _ambient_cache_get(key)
    if cached is not None:
        return cached

    base = _make_maple_leaf_image(size)
    if base is None:
        return []

    frames = []
    for i in range(frame_count):
        deg = 360.0 * i / frame_count
        # ``expand=True`` keeps every rotation frame at the same physical
        # size of the rotated bounding box, so cycling between them
        # doesn't introduce a visible "jump" mid-fall.
        rot = base.rotate(deg, resample=Image.BICUBIC, expand=True)
        frames.append(rot)

    return _ambient_cache_put(key, frames)


def _make_lantern_image(height: int = 26):
    """Paper-lantern silhouette: red oval body with dark caps and a soft
    warm glow.  Used for Japan / China circuits."""
    if not HAS_PIL:
        return None
    key = ("lantern", int(height))
    cached = _ambient_cache_get(key)
    if cached is not None:
        return cached

    H = max(12, int(height))
    W = max(10, int(H * 0.78))
    img = Image.new("RGBA", (W, H + 6), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Hanging cord
    draw.line([(W // 2, 0), (W // 2, 4)], fill=(60, 30, 30, 220), width=1)
    # Top cap
    draw.rectangle([W * 0.30, 3, W * 0.70, 6],
                   fill=(40, 22, 20, 255))
    # Body (red oval) – two-tone for depth
    body_top = 6
    body_bot = H + 1
    draw.ellipse([1, body_top, W - 1, body_bot],
                 fill=(196, 32, 26, 245), outline=(110, 16, 12, 255), width=1)
    # Vertical ribbing (4 thin lines)
    for f in (0.25, 0.50, 0.75):
        x = 1 + (W - 2) * f
        draw.line([(x, body_top + 2), (x, body_bot - 2)],
                  fill=(140, 24, 18, 220), width=1)
    # Warm highlight
    draw.ellipse([2, body_top + 2, W * 0.55, body_top + (body_bot - body_top) * 0.45],
                 fill=(240, 180, 80, 100))
    # Bottom tassel
    draw.rectangle([W * 0.40, body_bot - 1, W * 0.60, body_bot + 4],
                   fill=(60, 30, 30, 255))

    return _ambient_cache_put(key, img)


def _make_palm_image(height: int = 110, mirror: bool = False):
    """Coconut-palm silhouette: thin trunk with a fan of fronds."""
    if not HAS_PIL:
        return None
    key = ("palm", int(height), bool(mirror))
    cached = _ambient_cache_get(key)
    if cached is not None:
        return cached

    H = max(40, int(height))
    W = max(40, int(H * 0.85))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    trunk_color = (44, 30, 18, 220)
    frond_dark  = (32, 70, 38, 235)
    frond_mid   = (52, 116, 60, 230)

    # Trunk: a gentle S-curve made of 3 line segments tapering inward.
    tx0 = W * 0.42
    ty0 = H - 2
    tx1 = W * 0.50
    ty1 = H * 0.55
    tx2 = W * 0.45
    ty2 = H * 0.30
    for (a, b, wd) in (
        ((tx0, ty0), (tx1, ty1), 4),
        ((tx1, ty1), (tx2, ty2), 3),
    ):
        draw.line([a, b], fill=trunk_color, width=wd)

    # Fronds: 7 ovals fanning from the crown
    cx, cy = tx2, ty2
    import math as _m
    for i, ang_deg in enumerate(range(-100, 101, 33)):
        ang = _m.radians(ang_deg - 15)
        fx = cx + _m.cos(ang) * W * 0.30
        fy = cy + _m.sin(ang) * H * 0.18
        # Dark base + lighter top so they overlap nicely
        r = W * 0.22
        draw.ellipse([fx - r, fy - r * 0.30, fx + r, fy + r * 0.30],
                     fill=frond_dark)
    # Lighter top layer slightly offset for sheen
    for i, ang_deg in enumerate(range(-95, 96, 38)):
        ang = _m.radians(ang_deg - 12)
        fx = cx + _m.cos(ang) * W * 0.26
        fy = cy + _m.sin(ang) * H * 0.16
        r = W * 0.18
        draw.ellipse([fx - r, fy - r * 0.28, fx + r, fy + r * 0.28],
                     fill=frond_mid)

    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return _ambient_cache_put(key, img)


def _make_sun_image(diameter: int = 60):
    """Glowing sun: bright disc with a soft outer halo."""
    if not HAS_PIL:
        return None
    key = ("sun", int(diameter))
    cached = _ambient_cache_get(key)
    if cached is not None:
        return cached

    D = max(20, int(diameter))
    img = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = D / 2.0
    # Halo: a series of concentric semitransparent rings
    for i, (r_frac, alpha) in enumerate((
        (1.00,  30),
        (0.85,  55),
        (0.70,  85),
        (0.58, 120),
    )):
        r = c * r_frac
        draw.ellipse([c - r, c - r, c + r, c + r],
                     fill=(255, 200, 80, alpha))
    # Hot core
    rc = c * 0.42
    draw.ellipse([c - rc, c - rc, c + rc, c + rc],
                 fill=(255, 240, 180, 245))
    return _ambient_cache_put(key, img)


def _make_mountain_image(width: int = 280, height: int = 70):
    """Twin-peak mountain silhouette with a snow cap."""
    if not HAS_PIL:
        return None
    key = ("mountain", int(width), int(height))
    cached = _ambient_cache_get(key)
    if cached is not None:
        return cached

    W = max(40, int(width))
    H = max(20, int(height))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rock = (52, 60, 78, 220)
    snow = (220, 230, 245, 235)
    # Two peaks
    peaks = [
        (0, H - 1),
        (W * 0.18, H * 0.35),
        (W * 0.30, H * 0.55),
        (W * 0.42, H * 0.20),
        (W * 0.55, H * 0.55),
        (W * 0.72, H * 0.45),
        (W * 0.85, H * 0.65),
        (W - 1, H - 1),
    ]
    draw.polygon(peaks, fill=rock)
    # Snow caps on each peak
    for (x, y) in ((W * 0.18, H * 0.35), (W * 0.42, H * 0.20)):
        draw.polygon([
            (x - W * 0.05, y + 5),
            (x, y),
            (x + W * 0.05, y + 5),
        ], fill=snow)
    return _ambient_cache_put(key, img)


def _make_dune_image(width: int = 320, height: int = 60):
    """Sandy dune silhouette for desert circuits (Bahrain, Qatar, Saudi)."""
    if not HAS_PIL:
        return None
    key = ("dune", int(width), int(height))
    cached = _ambient_cache_get(key)
    if cached is not None:
        return cached

    W = max(40, int(width))
    H = max(15, int(height))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    sand_dark = (98, 70, 38, 200)
    sand_mid  = (158, 122, 68, 215)
    # Back dune (darker)
    pts = [(0, H - 1)]
    for f in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00):
        y = H * 0.50 + (0.18 if int(f * 10) % 2 == 0 else 0.06) * H * (1 if f < 0.5 else -1)
        pts.append((W * f, y))
    pts.append((W, H - 1))
    draw.polygon(pts, fill=sand_dark)
    # Front dune (lighter, lower)
    pts2 = [(0, H - 1)]
    for f in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00):
        y = H * 0.70 + (0.12 if int(f * 10) % 2 == 1 else 0.05) * H * (-1 if f < 0.5 else 1)
        pts2.append((W * f, y))
    pts2.append((W, H - 1))
    draw.polygon(pts2, fill=sand_mid)
    return _ambient_cache_put(key, img)


def _make_track_bg(w: int, h: int):
    """Render the race-visualisation background.

    A subtle radial vignette (slightly lighter near the centre, darkening
    to deep black at the edges) plus a faint diagonal red glow from the
    top-left.  Cheap to draw because we compute a small 256-pixel
    gradient with NumPy and upsample with Lanczos – the per-pixel loop
    that used to make this prohibitive is gone.
    """
    if not HAS_PIL or w <= 0 or h <= 0:
        return None

    key = (w, h)
    if key in _BG_CACHE:
        return _BG_CACHE[key]

    size = 256
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2.0

    # Radial vignette: 1.0 at the centre, 0.0 at the far corners.
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size * 0.72)
    radial = np.clip(1.0 - d, 0.0, 1.0)

    # Top-left diagonal red glow for a hint of brand colour.
    tl = np.sqrt(x ** 2 + y ** 2) / (size * 1.3)
    glow = np.clip(1.0 - tl, 0.0, 1.0) ** 1.8

    base_v = 10 + radial * 9            # 10 → 19
    r = base_v + glow * 14              # add a touch of red at the corner
    g = base_v
    b = base_v + 1

    rgb = np.stack(
        [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)],
        axis=-1,
    ).astype(np.uint8)

    small = Image.fromarray(rgb, "RGB")
    img = small.resize((w, h), Image.LANCZOS)

    # A whisper of blur softens the corners between Lanczos pixels.
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    _BG_CACHE[key] = img
    return img


def _make_carbon_fiber_img(w, h):
    """Generate a carbon-fiber weave texture by tiling a small 6x6 pattern.
    Caches the tile so repeated calls (eg. on resize) are cheap."""
    if not HAS_PIL:
        return None

    tile = _CF_TILE_CACHE.get(6)
    if tile is None:
        tile = Image.new("RGB", (6, 6), (12, 12, 13))
        pix = tile.load()
        for y in range(6):
            for x in range(6):
                cell_x, cell_y = x, y
                if (cell_x < 3) == (cell_y < 3):
                    pix[x, y] = (15, 15, 16)
                else:
                    pix[x, y] = (11, 11, 12)
                if cell_x == 0 or cell_y == 0:
                    pix[x, y] = (9, 9, 10)
        _CF_TILE_CACHE[6] = tile

    # Use the much faster paste-pattern approach with a pre-built row strip,
    # then paste the row down the height.
    if w <= 0 or h <= 0:
        return tile

    img = Image.new("RGB", (w, h), (10, 10, 11))
    row = Image.new("RGB", (w, 6), (10, 10, 11))
    for tx in range(0, w, 6):
        row.paste(tile, (tx, 0))
    for ty in range(0, h, 6):
        img.paste(row, (0, ty))
    return img


ALGO_TEXT = """\
# ApexAI · Gradient Boosting v2

features = [
  # identity
  "Abbreviation", "TeamName", "DriverNumber",
  # grid / form
  "GridPosition",       # ExpectedGrid for unraced events
  "RecentAvgPos",       # last 5 finishes
  "RecentAvgGrid",      # last 5 qualifying slots
  "RecentWinRate",      # win % over last 10
  "RecentPodiumRate",   # podium % over last 10
  "DNFRate",            # non-finish % over last 10
  "DriverExperience",   # career race count
  "HeadToHead",         # % beating teammate
  "TeamRecentForm",     # team avg finish, last 10
  # circuit affinity
  "DriverCircuitAvg",   # driver avg @ this track
  "TeamCircuitAvg",     # team   avg @ this track
  # 2026 power-unit regs (50/50 ICE/electric)
  "PUBatteryScore",     # hybrid deployment skill
  "PUICEScore",         # combustion efficiency
  # standings
  "DriverPointsBefore", "TeamPointsBefore",
]

model = GradientBoostingClassifier(
  n_estimators ≤ 800   + early stopping,
  max_depth     3–6,
  learning_rate 0.03–0.12,
  sample_weight = rebalance winners,
)

tuning = RandomizedSearchCV(
  n_iter = 20,
  cv     = TimeSeriesSplit(5),
  scoring= "roc_auc",
)

raw = model.predict_proba(X)[:, 1]
P   = softmax(log(raw)/0.18)
# per-race probability distribution\
"""


SCENES = {
    # (vegetation_type, veg_color, veg_spacing, ground_color, [(features...)])
    "Albert Park":
        ("deciduous", "#1a3a1a", 18, "#0a120a",
         [("lake",), ("grandstand", 0.0, 1), ("grandstand", 0.45, -1)]),
    "Sakhir":
        (None, None, 0, "#120f08",
         [("dunes", 4), ("grandstand", 0.0, 1)]),
    "Jeddah Corniche":
        (None, None, 0, "#0a0a12",
         [("water", "left"), ("buildings", "right", 6), ("grandstand", 0.0, 1)]),
    "Suzuka":
        ("cherry", "#3a1a28", 15, "#0a120a",
         [("ferris", 0.92, 0.10), ("grandstand", 0.0, 1), ("grandstand", 0.6, -1)]),
    "Shanghai":
        ("deciduous", "#1a2a1a", 22, "#0a120a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Miami Autodrome":
        ("palm", "#1a5a1a", 22, "#0a120a",
         [("water", "right"), ("skyline", "left", 5), ("grandstand", 0.0, 1)]),
    "Imola":
        ("deciduous", "#1a3a1a", 16, "#0a120a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Monaco":
        (None, None, 0, "#0a0a12",
         [("water", "bottom"), ("yachts", 4), ("buildings", "top", 8), ("grandstand", 0.0, 1)]),
    "Barcelona-Catalunya":
        ("deciduous", "#2a3a1a", 22, "#0a100a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Circuit Gilles Villeneuve":
        ("deciduous", "#1a3a1a", 16, "#0a120a",
         [("water", "right"), ("water", "left"), ("grandstand", 0.0, 1)]),
    "Red Bull Ring":
        ("pine", "#0a2a0a", 14, "#08100a",
         [("mountains", 5), ("grandstand", 0.0, 1), ("grandstand", 0.4, -1)]),
    "Silverstone":
        ("deciduous", "#1a301a", 20, "#0a100a",
         [("grandstand", 0.0, 1), ("grandstand", 0.35, -1), ("grandstand", 0.7, 1)]),
    "Spa-Francorchamps":
        ("pine", "#0a2a0a", 12, "#08100a",
         [("mountains", 4), ("grandstand", 0.0, 1)]),
    "Zandvoort":
        ("deciduous", "#1a301a", 24, "#10100a",
         [("water", "top"), ("dunes", 3), ("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Monza":
        ("deciduous", "#1a3a1a", 14, "#0a120a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Baku City Circuit":
        (None, None, 0, "#0a0a12",
         [("water", "left"), ("buildings", "right", 7), ("grandstand", 0.0, 1)]),
    "Marina Bay":
        ("palm", "#1a4a1a", 25, "#0a0a12",
         [("buildings", "top", 8), ("water", "bottom"), ("grandstand", 0.0, 1)]),
    "COTA":
        (None, None, 0, "#100e08",
         [("tower", 0.50, 0.04), ("cactus_scatter",), ("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Autódromo Hermanos Rodríguez":
        ("deciduous", "#1a3a1a", 22, "#0a100a",
         [("stadium", 0.7, 1), ("grandstand", 0.0, 1)]),
    "Interlagos":
        ("deciduous", "#1a4a1a", 16, "#0a120a",
         [("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Las Vegas Strip":
        (None, None, 0, "#0a0a14",
         [("strip",), ("sphere", 0.90, 0.12), ("grandstand", 0.0, 1)]),
    "Lusail":
        (None, None, 0, "#120f08",
         [("dunes", 3), ("grandstand", 0.0, 1), ("grandstand", 0.5, -1)]),
    "Yas Marina":
        ("palm", "#1a4a1a", 25, "#0a0a10",
         [("water", "bottom"), ("buildings", "right", 4), ("grandstand", 0.0, 1)]),
    "Circuit":
        ("deciduous", "#1a3a1a", 22, "#0a100a",
         [("grandstand", 0.0, 1)]),
}

class ApexAI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ApexAI")
        self.root.configure(bg=BG)
        self.root.minsize(960, 780)
        self.root.geometry("1120x880")
        self.result = None
        self._logos = {}
        self._tk_images = []
        self._current_view = "predictions"
        self._schedule = []
        self._race_idx = -1
        self._season_driver_pts = {}
        self._season_team_pts = {}
        self._build()
        # Stamp the header caption with today's date + 2026 round progress.
        # Done after _build so the caption Label exists.
        self._update_season_caption()

    # -- Header caption (date + season progress) --
    def _update_season_caption(self):
        """Render the header caption as
            F1 RACE PREDICTOR  ·  2026 SEASON  ·  ROUND N / 22  ·  MAY 18, 2026
        using today's date and the FastF1 schedule so it stays accurate as
        the season progresses.  Runs in a background thread so a slow
        schedule fetch doesn't stall window paint."""
        import datetime as _dt
        today = _dt.date.today()
        date_str = today.strftime("%b %d, %Y").upper()
        # Seed with a static caption so the header is never blank.
        try:
            self._caption_lbl.config(
                text=f"F1 RACE PREDICTOR  ·  {today.year} SEASON  ·  {date_str}"
            )
        except Exception:
            return

        def _worker():
            try:
                import fastf1
                sched = fastf1.get_event_schedule(today.year, include_testing=False)
                total = int(sched["RoundNumber"].max())
                past = sched[sched["EventDate"].dt.date <= today]
                done = int(past["RoundNumber"].max()) if not past.empty else 0
                caption = (
                    f"F1 RACE PREDICTOR  ·  {today.year} SEASON  ·  "
                    f"ROUND {done} / {total}  ·  {date_str}"
                )
            except Exception:
                caption = (
                    f"F1 RACE PREDICTOR  ·  {today.year} SEASON  ·  {date_str}"
                )
            try:
                self.root.after(0, lambda: self._caption_lbl.config(text=caption))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    # -- Logo cache --
    def _logo(self, team: str, sz: int = 24):
        k = f"{team}_{sz}"
        if k not in self._logos:
            if HAS_PIL:
                img = load_logo(team, sz)
                if img:
                    if img.mode == "RGBA":
                        bg_rgb = tuple(int(BG_CARD.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                        solid = Image.new("RGBA", img.size, (*bg_rgb, 255))
                        solid.paste(img, (0, 0), img)
                        img = solid
                    self._logos[k] = ImageTk.PhotoImage(img)
                else:
                    self._logos[k] = None
            else:
                self._logos[k] = None
        return self._logos.get(k)

    # -- Build UI --
    def _build(self):
        # ── Header bar (F1 logo · wordmark · model badge · accuracy chip) ──
        hdr = tk.Frame(self.root, bg=BG, padx=36, pady=16)
        hdr.pack(fill=tk.X)

        # Left: official F1-style logo + bold ApexAI wordmark + caption.
        # The logo and wordmark double as a "home" button — clicking
        # either takes the user back to the predictions view (the
        # last predicted race), much like a website logo.
        logo_img = _make_f1_logo(32) if HAS_PIL else None
        if logo_img is not None:
            self._hdr_logo_tk = ImageTk.PhotoImage(logo_img)
            self._tk_images.append(self._hdr_logo_tk)
            self._home_logo_lbl = tk.Label(
                hdr, image=self._hdr_logo_tk, bg=BG, bd=0,
                cursor="hand2",
            )
            self._home_logo_lbl.pack(side=tk.LEFT, padx=(0, 14))
            self._home_logo_lbl.bind("<Button-1>", lambda _e: self._go_home())

        title_box = tk.Frame(hdr, bg=BG, cursor="hand2")
        title_box.pack(side=tk.LEFT, fill=tk.Y)
        title_box.bind("<Button-1>", lambda _e: self._go_home())
        wmark = tk.Frame(title_box, bg=BG, cursor="hand2")
        wmark.pack(anchor="w")
        wmark.bind("<Button-1>", lambda _e: self._go_home())
        self._home_apex_lbl = tk.Label(
            wmark, text="Apex", font=("Helvetica Neue", 24, "bold"),
            fg=F1_RED, bg=BG, cursor="hand2",
        )
        self._home_apex_lbl.pack(side=tk.LEFT)
        self._home_apex_lbl.bind("<Button-1>", lambda _e: self._go_home())
        self._home_ai_lbl = tk.Label(
            wmark, text="AI", font=("Helvetica Neue", 24, "bold"),
            fg=WHITE, bg=BG, cursor="hand2",
        )
        self._home_ai_lbl.pack(side=tk.LEFT)
        self._home_ai_lbl.bind("<Button-1>", lambda _e: self._go_home())

        # Subtle hover dim so the brand mark visibly reads as a button.
        def _brand_hover(entering: bool):
            if entering:
                self._home_apex_lbl.configure(fg=GOLD_GLOW)
                self._home_ai_lbl.configure(fg=GRAY)
            else:
                self._home_apex_lbl.configure(fg=F1_RED)
                self._home_ai_lbl.configure(fg=WHITE)

        for w in (self._home_apex_lbl, self._home_ai_lbl, wmark):
            w.bind("<Enter>", lambda _e: _brand_hover(True))
            w.bind("<Leave>", lambda _e: _brand_hover(False))
        if hasattr(self, "_home_logo_lbl"):
            self._home_logo_lbl.bind(
                "<Enter>", lambda _e: _brand_hover(True))
            self._home_logo_lbl.bind(
                "<Leave>", lambda _e: _brand_hover(False))
        # Caption is built dynamically so it shows current date + season
        # progress, e.g. "F1 RACE PREDICTOR  ·  2026 SEASON  ·  ROUND 4 / 22
        #                 ·  MAY 18, 2026".  Populated lazily after the
        # schedule is fetched; we seed it with a sensible static string so
        # the layout doesn't shift.
        self._caption_lbl = tk.Label(
            title_box, text="F1 RACE PREDICTOR  ·  2026 SEASON",
            font=("Helvetica Neue", 9), fg=MUTED, bg=BG,
        )
        self._caption_lbl.pack(anchor="w", pady=(1, 0))

        # Right: model + accuracy chips, populated after a load.
        chips = tk.Frame(hdr, bg=BG)
        chips.pack(side=tk.RIGHT)
        self.acc_chip = self._make_chip(
            chips, "ACCURACY", "—", muted=True,
        )
        self.acc_chip.pack(side=tk.RIGHT, padx=(8, 0))
        self.model_chip = self._make_chip(
            chips, "MODEL", "GBM v4", muted=True,
        )
        self.model_chip.pack(side=tk.RIGHT, padx=(0, 8))

        # Thin red rule under the header for that broadcast-graphic feel.
        tk.Frame(self.root, bg=F1_RED, height=2).pack(fill=tk.X, padx=36)

        # ── Tab bar (underline-accent navigation) ──
        ctrl = tk.Frame(self.root, bg=BG, padx=36, pady=10)
        ctrl.pack(fill=tk.X)

        self.btn_predict = self._make_tab(ctrl, "Predict Next Race",
                                           self._on_predict)
        self.btn_predict.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_all = self._make_tab(ctrl, "Backtest All Races",
                                       self._on_all_races)
        self.btn_all.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_viz = self._make_tab(ctrl, "Race Visualization",
                                       self._on_show_viz)
        self.btn_viz.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_radio = self._make_tab(ctrl, "Team Radio",
                                         self._on_show_radio)
        self.btn_radio.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_replays = self._make_tab(ctrl, "Race Replays",
                                           self._on_show_replays)
        self.btn_replays.pack(side=tk.LEFT, padx=(0, 6))

        # Trailing refresh button is visually quieter (icon-style ↻).
        self.btn_refresh = self._make_tab(ctrl, "↻  Refresh",
                                           self._on_refresh, secondary=True)
        self.btn_refresh.pack(side=tk.LEFT, padx=(8, 0))

        self._all_btns = [self.btn_predict, self.btn_all, self.btn_viz,
                          self.btn_radio, self.btn_replays, self.btn_refresh]

        # ── Footer status bar (always visible, never truncated) ──
        # Packed at the BOTTOM of the window first so the body fills the
        # remaining space.  Holds the activity message on the left and a
        # tiny version label on the right.
        footer = tk.Frame(self.root, bg=BG_SURFACE, height=28)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        footer.pack_propagate(False)
        # Hairline rule above the footer for a "broadcast lower-third" look.
        tk.Frame(self.root, bg=BORDER, height=1).pack(
            side=tk.BOTTOM, fill=tk.X,
        )
        self.status_lbl = tk.Label(
            footer, text="Ready.", font=("Helvetica Neue", 10),
            fg=GRAY, bg=BG_SURFACE, anchor="w", padx=36,
        )
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            footer,
            text="ApexAI · powered by FastF1 + OpenF1 · 2026 mid-season build",
            font=("Helvetica Neue", 9), fg=MUTED, bg=BG_SURFACE,
            padx=36,
        ).pack(side=tk.RIGHT)

        # Body – two panels (predictions view)
        self.body = tk.Frame(self.root, bg=BG, padx=36, pady=10)
        self.body.pack(fill=tk.BOTH, expand=True)
        body = self.body

        # Track visualization view (hidden initially)
        self.viz_frame = tk.Frame(self.root, bg=BG, padx=18, pady=6)

        # Team radio view (hidden initially)
        self.radio_frame = tk.Frame(self.root, bg=BG, padx=36, pady=8)

        # Race replays view (hidden initially) – links to FullRaces.com
        self.replays_frame = tk.Frame(self.root, bg=BG, padx=36, pady=8)
        self._replays_built = False
        self._replays_year = None
        self._radio_clips = []
        self._radio_clip_meta = []        # parallel: per-clip {lap, event, dt}
        self._radio_event_log = []        # raw f1radio event_log for the race
        self._radio_session_label = ""    # "2025 BAHRAIN GRAND PRIX – RACE"
        self._radio_total_laps = 0
        self._radio_drivers = []          # unique drivers in this race
        self._radio_drivers_silent = []   # [{abbr,name,team}] on grid, no radio
        self._radio_drivers_heard = []    # abbrs that actually had clips
        self._radio_drivers_all = []      # full grid abbrs
        self._radio_filter_driver = "ALL" # "ALL" or driver name
        self._radio_filtered_idx = []     # clip indices currently visible
        self._radio_play_queue = []       # remaining clip indices for sequential
        self._radio_queue_total = 0       # total in current queue
        self._radio_queue_pos = 0         # 1-based position
        self._radio_playing = None
        self._radio_proc = None
        self._radio_sound = None
        self._radio_wave_items = []
        self._radio_anim_running = False

        # Left panel – results
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        self.canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(left, command=self.canvas.yview)
        self.results = tk.Frame(self.canvas, bg=BG)
        self.results.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.results, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for w in (self.canvas, self.results):
            w.bind("<MouseWheel>", self._scroll)

        # ── Right panel: model insight column ──
        # Three clean cards instead of one giant dump:
        #   1. MODEL STATS — accuracy, model id, training scope
        #   2. FEATURE IMPORTANCE — re-rendered chart (short labels)
        #   3. HOW IT WORKS — 4 plain-English bullets, no code
        right = tk.Frame(body, bg=BG, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # Card 1 — model stats (compact 4-row table)
        stats = tk.Frame(right, bg=BG_CARD, padx=14, pady=10)
        stats.pack(fill=tk.X, pady=(0, 10))
        stats.configure(highlightbackground=BORDER, highlightthickness=1)
        tk.Frame(stats, bg=F1_RED, height=2).pack(fill=tk.X, pady=(0, 8))
        tk.Label(stats, text="MODEL", font=("Helvetica Neue", 9, "bold"),
                 fg=F1_RED, bg=BG_CARD).pack(anchor="w")
        tk.Frame(stats, bg=BORDER, height=1).pack(fill=tk.X, pady=(5, 6))

        def _stat_row(label, value_tk_attr):
            row = tk.Frame(stats, bg=BG_CARD)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, font=("Helvetica Neue", 9),
                     fg=MUTED, bg=BG_CARD).pack(side=tk.LEFT)
            v = tk.Label(row, text="—", font=("Helvetica Neue", 10, "bold"),
                         fg=WHITE, bg=BG_CARD)
            v.pack(side=tk.RIGHT)
            setattr(self, value_tk_attr, v)

        _stat_row("Accuracy",            "_stat_accuracy")
        _stat_row("Algorithm",           "_stat_algo")
        _stat_row("Training races",      "_stat_train")
        _stat_row("Features",            "_stat_features")
        self._stat_algo.configure(text="GBM v4", fg=F1_RED)

        # Card 2 — feature importance chart
        chart_card = tk.Frame(right, bg=BG_CARD, padx=12, pady=10)
        chart_card.pack(fill=tk.X, pady=(0, 10))
        chart_card.configure(highlightbackground=BORDER, highlightthickness=1)
        tk.Frame(chart_card, bg=F1_RED, height=2).pack(fill=tk.X, pady=(0, 6))
        head_row = tk.Frame(chart_card, bg=BG_CARD)
        head_row.pack(fill=tk.X)
        tk.Label(head_row, text="FEATURE IMPORTANCE",
                 font=("Helvetica Neue", 9, "bold"),
                 fg=F1_RED, bg=BG_CARD).pack(side=tk.LEFT)
        tk.Label(head_row, text="what drives the prediction",
                 font=("Helvetica Neue", 8, "italic"),
                 fg=MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=(8, 0))
        self.chart_lbl = tk.Label(chart_card, bg=BG_CARD)
        self.chart_lbl.pack(anchor="w", fill=tk.X, pady=(4, 0))

        # Card 3 — plain-English "how it works".  2 bullets instead of 4
        # to keep the whole right rail above the fold at 920 px height.
        how_card = tk.Frame(right, bg=BG_CARD, padx=14, pady=10)
        how_card.pack(fill=tk.X)
        how_card.configure(highlightbackground=BORDER, highlightthickness=1)
        tk.Frame(how_card, bg=F1_RED, height=2).pack(fill=tk.X, pady=(0, 6))
        tk.Label(how_card, text="HOW IT WORKS",
                 font=("Helvetica Neue", 9, "bold"),
                 fg=F1_RED, bg=BG_CARD).pack(anchor="w")
        tk.Frame(how_card, bg=BORDER, height=1).pack(fill=tk.X, pady=(5, 6))
        for bullet in (
            "Gradient-boosted model trained on every F1 race since 2018.",
            "Softmax over per-driver scores so probabilities sum to 100% "
            "and stay calibrated each race.",
        ):
            line = tk.Frame(how_card, bg=BG_CARD)
            line.pack(fill=tk.X, pady=(0, 3))
            tk.Label(line, text="·  ", font=("Helvetica Neue", 11, "bold"),
                     fg=F1_RED, bg=BG_CARD).pack(side=tk.LEFT, anchor="n")
            tk.Label(line, text=bullet, font=("Helvetica Neue", 9),
                     fg=GRAY, bg=BG_CARD, wraplength=280, justify="left",
                     anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # If we have a cached result from a previous session, show it instead
        # of the empty state.  The user can hit "Refresh" to recompute against
        # the latest F1 timing data.
        cached = load_last_result()
        if cached and "next_race" in cached:
            self.result = cached
            self._schedule = cached.get("schedule", []) or []
            nr = cached["next_race"]
            self._race_idx = next(
                (i for i, s in enumerate(self._schedule) if s["round"] == nr["round"]),
                0,
            )
            self._set_active_btn(self.btn_predict)
            self._set_status(
                f"Showing cached predictions  ·  click ↻ Refresh to retrain on latest data"
            )
            self._update_model_stats(cached)
            self._render_chart(cached.get("feature_importance", {}))
            self._clear()
            self._display_prediction_ui(cached)
        else:
            self._show_empty(
                "Ready to predict.\n\nClick "
                "\u201cPredict Next Race\u201d above to start."
            )

    def _update_model_stats(self, r):
        """Refresh the right-rail MODEL stats card + the header chips."""
        acc = r.get("accuracy")
        if acc is not None:
            acc_txt = f"{acc:.1%}"
            self._stat_accuracy.configure(text=acc_txt, fg=F1_RED)
            self._set_chip(self.acc_chip, acc_txt, accent=True)
        train_total = r.get("training_total") or r.get("total")
        if train_total:
            self._stat_train.configure(text=str(train_total))
        fi = r.get("feature_importance") or {}
        if fi:
            self._stat_features.configure(text=str(len(fi)))

    def _make_btn(self, parent, text, bg_c, fg_c, cmd, border=None):
        """Generic pill-style button.  Kept for non-tab use (e.g. inside the
        radio toolbar)."""
        frame = tk.Frame(parent, bg=bg_c, cursor="hand2",
                         highlightbackground=border or bg_c, highlightthickness=1)
        label = tk.Label(frame, text=text, font=("Helvetica Neue", 11, "bold"),
                         fg=fg_c, bg=bg_c, padx=18, pady=8, cursor="hand2")
        label.pack()
        label.bind("<Button-1>", lambda e: cmd())
        frame.bind("<Button-1>", lambda e: cmd())
        frame._label = label
        frame._default_bg = bg_c
        frame._default_fg = fg_c
        frame._current_bg = bg_c
        frame._is_tab = False
        frame.bind("<Enter>", lambda e, f=frame: self._btn_hover(f, True))
        frame.bind("<Leave>", lambda e, f=frame: self._btn_hover(f, False))
        label.bind("<Enter>", lambda e, f=frame: self._btn_hover(f, True))
        label.bind("<Leave>", lambda e, f=frame: self._btn_hover(f, False))
        return frame

    def _make_tab(self, parent, text, cmd, secondary=False):
        """Tab-style nav control: flat label on top of a thin underline that
        switches to F1 red when active.  Far less visually heavy than the old
        pill buttons, and the active state is unambiguous."""
        fg_default = GRAY if secondary else WHITE
        wrap = tk.Frame(parent, bg=BG, cursor="hand2")
        body = tk.Frame(wrap, bg=BG, cursor="hand2")
        body.pack(fill=tk.X)
        lbl = tk.Label(body, text=text,
                        font=("Helvetica Neue", 11, "bold"),
                        fg=fg_default, bg=BG,
                        padx=14, pady=8, cursor="hand2")
        lbl.pack()
        # 2-pixel underline that becomes red when active, dark otherwise.
        underline = tk.Frame(wrap, bg=BG, height=2)
        underline.pack(fill=tk.X)
        for w in (wrap, body, lbl):
            w.bind("<Button-1>", lambda e: cmd())
            w.bind("<Enter>", lambda e, t=wrap: self._tab_hover(t, True))
            w.bind("<Leave>", lambda e, t=wrap: self._tab_hover(t, False))
        wrap._label = lbl
        wrap._underline = underline
        wrap._default_fg = fg_default
        wrap._active = False
        wrap._is_tab = True
        wrap._secondary = secondary
        return wrap

    def _tab_hover(self, tab, entering):
        if tab._active:
            return
        if entering:
            tab._label.configure(fg=WHITE)
            tab._underline.configure(bg=BORDER)
        else:
            tab._label.configure(fg=tab._default_fg)
            tab._underline.configure(bg=BG)

    def _btn_hover(self, btn, entering):
        if getattr(btn, "_is_tab", False):
            return
        c = btn._current_bg
        if entering:
            hover = self._lighten(c, 30)
            btn.configure(bg=hover, highlightbackground=hover)
            btn._label.configure(bg=hover)
        else:
            btn.configure(bg=c, highlightbackground=c)
            btn._label.configure(bg=c)

    def _set_active_btn(self, active_btn):
        for btn in self._all_btns:
            if getattr(btn, "_is_tab", False):
                btn._active = False
                btn._label.configure(fg=btn._default_fg)
                btn._underline.configure(bg=BG)
            else:
                btn._current_bg = btn._default_bg
                btn.configure(bg=btn._default_bg,
                              highlightbackground=btn._default_bg)
                btn._label.configure(fg=btn._default_fg, bg=btn._default_bg)
        if active_btn:
            if getattr(active_btn, "_is_tab", False):
                active_btn._active = True
                active_btn._label.configure(fg=WHITE)
                active_btn._underline.configure(bg=F1_RED)
            else:
                active_btn._current_bg = GOLD_DIM
                active_btn.configure(bg=GOLD_DIM, highlightbackground=GOLD_DIM)
                active_btn._label.configure(fg=GOLD, bg=GOLD_DIM)

    def _make_chip(self, parent, label, value, muted=False, accent=False):
        """Compact label/value chip for the header (e.g. "ACCURACY  82%").
        Returns the outer Frame; the value label is exposed as ``chip._value``
        so the caller can update it later."""
        outer = tk.Frame(parent, bg=BG_SURFACE, padx=10, pady=4)
        outer.configure(highlightbackground=BORDER, highlightthickness=1)
        tk.Label(outer, text=label, font=("Helvetica Neue", 8, "bold"),
                 fg=MUTED, bg=BG_SURFACE).pack(side=tk.LEFT)
        val_color = F1_RED if accent else (MUTED if muted else WHITE)
        val_lbl = tk.Label(outer, text=value,
                           font=("Helvetica Neue", 10, "bold"),
                           fg=val_color, bg=BG_SURFACE)
        val_lbl.pack(side=tk.LEFT, padx=(8, 0))
        outer._value = val_lbl
        return outer

    def _set_chip(self, chip, value, accent=False):
        chip._value.configure(
            text=value, fg=F1_RED if accent else WHITE,
        )

    @staticmethod
    def _lighten(hex_c, amt=20):
        h = hex_c.lstrip("#")
        r, g, b = (min(255, int(h[i:i+2], 16) + amt) for i in (0, 2, 4))
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _wheel_units(event) -> int:
        """Cross-platform mouse-wheel delta normalisation.

        Windows / X11 deliver `event.delta` in multiples of ±120
        (one notch = 120) so we divide.  macOS delivers small
        integers (typically ±1, occasionally ±3) so dividing by 120
        always rounds to 0 — which is why scrolling silently did
        nothing on macOS before.  Use the raw delta directly when
        |delta| < 120, otherwise scale.
        """
        d = int(event.delta)
        if abs(d) >= 120:
            d = d // 120
        return -d

    def _scroll(self, event):
        try:
            self.canvas.yview_scroll(self._wheel_units(event), "units")
        except Exception:
            pass
        return "break"

    def _bind_wheel_recursive(self, widget, scroll_fn):
        """Bind `<MouseWheel>` on `widget` and every descendant.

        Tk's wheel event fires on whichever widget the mouse is
        currently over.  When a scrollable canvas is filled with many
        child rows (e.g. the 96-race backtest table), hovering a row
        means the wheel binding on the canvas itself is never
        triggered.  This helper walks the subtree once and attaches
        the same handler everywhere so the wheel always scrolls the
        outer canvas.
        """
        try:
            widget.bind("<MouseWheel>", scroll_fn, add="+")
        except Exception:
            pass
        for child in widget.winfo_children():
            self._bind_wheel_recursive(child, scroll_fn)

    # -- Status --
    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_lbl.configure(text=msg))

    def _set_busy(self, busy):
        pass

    # -- Refresh: force a full re-run against the latest F1 data + retrain --
    def _on_refresh(self):
        """Invalidate the on-disk caches and start a fresh prediction run."""
        from prediction import LAST_RESULT_PATH
        model_cache = LAST_RESULT_PATH.parent / "model_cache.pkl"
        for p in (LAST_RESULT_PATH, model_cache):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        self.result = None
        self._schedule = []
        self._race_idx = -1
        self._season_driver_pts = {}
        self._season_team_pts = {}
        self._on_predict()

    # -- Predict next race --
    def _on_predict(self):
        self._set_active_btn(self.btn_predict)

        if self.result and self._schedule and "_model" in self.result:
            self._advance_and_predict()
            return

        self._set_status("Loading data...")
        self._clear()
        self._show_empty("Running predictions...")

        def work():
            r = run_predictions(progress_callback=self._set_status)
            self.root.after(0, lambda: self._show_predictions(r))

        threading.Thread(target=work, daemon=True).start()

    def _advance_and_predict(self):
        """Award points from current race prediction, advance to next race, re-predict."""
        r = self.result
        nr = r["next_race"]

        for i, p in enumerate(nr["predictions"]):
            pos = i + 1
            pts = F1_POINTS.get(pos, 0)
            if pts > 0:
                abbr = p["abbreviation"]
                team = p["team"]
                self._season_driver_pts[abbr] = self._season_driver_pts.get(abbr, 0) + pts
                self._season_team_pts[team] = self._season_team_pts.get(team, 0) + pts

        self._race_idx = (self._race_idx + 1) % len(self._schedule)
        race = self._schedule[self._race_idx]

        model = r["_model"]
        features = r["_features"]
        base_lineup = r["_base_lineup"]

        base_driver_pts = dict(r.get("_base_driver_pts", {}))
        base_team_pts = dict(r.get("_base_team_pts", {}))

        driver_pts = {k: base_driver_pts.get(k, 0) + self._season_driver_pts.get(k, 0)
                      for k in set(base_driver_pts) | set(self._season_driver_pts)}
        team_pts = {k: base_team_pts.get(k, 0) + self._season_team_pts.get(k, 0)
                    for k in set(base_team_pts) | set(self._season_team_pts)}

        extra = r.get("_extra_features", {})
        new_preds = predict_with_standings(
            model, features, base_lineup, driver_pts, team_pts,
            extra_features=extra,
            circuit_name=race.get("name"),
            season_year=nr["year"],
        )

        new_nr = {
            "year": nr["year"],
            "round": race["round"],
            "name": race["name"],
            "predicted_winner": new_preds[0]["abbreviation"],
            "top_probability": new_preds[0]["probability"],
            "predictions": new_preds,
        }
        r["next_race"] = new_nr

        self._set_status(f"Accuracy {r['accuracy']:.1%}  ·  Race {self._race_idx + 1}/{len(self._schedule)}")
        self._update_model_stats(r)
        self._render_chart(r.get("feature_importance", {}))
        self._clear()
        self._display_prediction_ui(r)

    def _show_predictions(self, r):
        self._set_active_btn(self.btn_predict)
        if "error" in r:
            self._set_status(f"Error: {r['error'][:80]}")
            self._show_empty(f"Error\n\n{r['error'][:300]}")
            return

        self.result = r
        self._schedule = r.get("schedule", [])
        self._season_driver_pts = {}
        self._season_team_pts = {}

        nr = r["next_race"]
        self._race_idx = next(
            (i for i, s in enumerate(self._schedule) if s["round"] == nr["round"]),
            0
        )
        self._set_status(f"Accuracy {r['accuracy']:.1%}  ·  Race {self._race_idx + 1}/{len(self._schedule)}")
        self._update_model_stats(r)
        self._render_chart(r.get("feature_importance", {}))
        self._clear()
        self._display_prediction_ui(r)

    def _display_prediction_ui(self, r):
        nr = r["next_race"]
        w = nr["predictions"][0]
        team_color = tc(w["team"])

        # -- Winner banner --
        banner = tk.Frame(self.results, bg=BG_CARD, padx=24, pady=22)
        banner.configure(highlightbackground=BORDER, highlightthickness=1)
        banner.pack(fill=tk.X, pady=(0, 16))

        # Race title row: small red label + race name.
        head = tk.Frame(banner, bg=BG_CARD)
        head.pack(fill=tk.X, pady=(0, 14))
        tk.Label(head, text="NEXT RACE", font=("Helvetica Neue", 9, "bold"),
                 fg=F1_RED, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(head, text=f"{nr['name']}  ·  {nr['year']} season",
                 font=("Helvetica Neue", 11), fg=WHITE, bg=BG_CARD
                 ).pack(side=tk.LEFT)

        # ── Predicted-winner hero card ──
        # Inner card uses the predicted team's livery colour as a left
        # stripe, with a deep red gradient backdrop so it reads as the
        # primary "answer" of the whole app.
        wb = tk.Frame(banner, bg=GOLD_DIM, padx=20, pady=18)
        wb.pack(fill=tk.X, pady=(0, 18))
        wb.configure(highlightbackground=team_color, highlightthickness=2)

        # Left: team livery accent column + logo
        accent = tk.Frame(wb, bg=team_color, width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))

        logo = self._logo(w["team"], 56)
        if logo:
            tk.Label(wb, image=logo, bg=GOLD_DIM).pack(
                side=tk.LEFT, padx=(0, 18)
            )

        # Middle: predicted winner text block
        wt = tk.Frame(wb, bg=GOLD_DIM)
        wt.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(wt, text="PREDICTED WINNER",
                 font=("Helvetica Neue", 9, "bold"),
                 fg=GOLD_GLOW, bg=GOLD_DIM).pack(anchor="w")
        tk.Label(wt, text=w["abbreviation"],
                 font=("Helvetica Neue", 36, "bold"),
                 fg=WHITE, bg=GOLD_DIM).pack(anchor="w", pady=(0, 2))
        tk.Label(wt, text=w["team"],
                 font=("Helvetica Neue", 11),
                 fg=team_color, bg=GOLD_DIM).pack(anchor="w")

        # Right: large probability percentage as the "score"
        prob = tk.Frame(wb, bg=GOLD_DIM)
        prob.pack(side=tk.RIGHT)
        tk.Label(prob, text="WIN PROBABILITY",
                 font=("Helvetica Neue", 8, "bold"),
                 fg=GOLD_GLOW, bg=GOLD_DIM).pack(anchor="e")
        tk.Label(prob, text=f"{w['probability']*100:.1f}%",
                 font=("Helvetica Neue", 28, "bold"),
                 fg=WHITE, bg=GOLD_DIM).pack(anchor="e")

        # -- Podium (compact) --
        # 2-3-1 ordering and shorter pedestals so the section feels like a
        # broadcast graphic instead of a stack of coloured blocks.
        pod = tk.Frame(banner, bg=BG_CARD)
        pod.pack(fill=tk.X, pady=(0, 14))
        heights = [34, 48, 28]  # 2nd, 1st, 3rd
        for col_idx, (rank_idx, h) in enumerate(zip([1, 0, 2], heights)):
            p = nr["predictions"][rank_idx]
            color = tc(p["team"])
            c = tk.Frame(pod, bg=BG_CARD)
            c.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

            pl = self._logo(p["team"], 22 if rank_idx == 0 else 18)
            if pl:
                tk.Label(c, image=pl, bg=BG_CARD).pack(pady=(0, 2))

            sz = 18 if rank_idx == 0 else 13
            tk.Label(c, text=p["abbreviation"],
                     font=("Helvetica Neue", sz, "bold"),
                     fg=color, bg=BG_CARD).pack()
            tk.Label(c, text=f"{p['probability']*100:.1f}%",
                     font=("Helvetica Neue", 9),
                     fg=GRAY, bg=BG_CARD).pack(pady=(0, 3))

            pedestal = tk.Frame(c, bg=color, height=h, width=84)
            pedestal.pack(side=tk.BOTTOM)
            pedestal.pack_propagate(False)
            pos_text = ["1st", "2nd", "3rd"][rank_idx]
            tk.Label(pedestal, text=pos_text,
                     font=("Helvetica Neue", 10, "bold"),
                     fg=WHITE, bg=color).pack(expand=True)

        # -- Full grid --
        grid_card = tk.Frame(self.results, bg=BG_CARD, padx=20, pady=16)
        grid_card.configure(highlightbackground=BORDER, highlightthickness=1)
        grid_card.pack(fill=tk.X, pady=(0, 16))
        tk.Label(grid_card, text="FULL GRID", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        for i, p in enumerate(nr["predictions"]):
            self._driver_row(grid_card, i + 1, p["abbreviation"], p["team"], p["probability"], highlight=(i == 0))

        # -- Last race --
        lr = r["last_race"]
        lr_card = tk.Frame(self.results, bg=BG_CARD, padx=20, pady=16)
        lr_card.configure(highlightbackground=BORDER, highlightthickness=1)
        lr_card.pack(fill=tk.X, pady=(0, 16))
        tk.Label(lr_card, text=f"LAST RACE · {lr['name']} ({lr['year']})", font=("Helvetica Neue", 10, "bold"), fg=GOLD, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        pred, actual = lr["predicted_winner"], lr["actual_winner"]
        hit = pred == actual
        tk.Label(lr_card, text=f"Predicted: {pred}  ·  Actual: {actual}  {'✓' if hit else '✗'}", font=("Helvetica Neue", 11, "bold"), fg=GREEN if hit else RED, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        for i, p in enumerate(lr["predictions"][:10]):
            self._driver_row(lr_card, i + 1, p["abbreviation"], p["team"], p["probability"], highlight=(p["abbreviation"] == actual))

        # Accuracy footer
        tk.Label(self.results, text=f"Model accuracy: {r['accuracy']:.1%}", font=("Helvetica Neue", 10), fg=MUTED, bg=BG).pack(anchor="w", pady=(8, 20))

    # -- Backtest all races --
    def _on_all_races(self):
        # Preserve the existing prediction (`self.result`, `_schedule`,
        # season point overlays) so the user can still flip to
        # Visualization / Race Replays / Team Radio while the backtest
        # is running and after it finishes.  The backtest produces its
        # own result UI via `_show_all_races` and does not need to
        # clobber the live prediction state.
        self._set_active_btn(self.btn_all)
        self._set_status("Backtesting every race...")
        self._clear()
        self._show_empty(
            "Running backtest on every race…\n"
            "Takes about 30 seconds on a multi-core Mac."
        )

        def work():
            r = run_predictions_all_races(progress_callback=self._set_status)
            self.root.after(0, lambda: self._show_all_races(r))

        threading.Thread(target=work, daemon=True).start()

    def _show_all_races(self, r):
        self._set_active_btn(self.btn_all)
        if "error" in r:
            self._set_status(f"Error: {r['error'][:80]}")
            self._show_empty(f"Error\n\n{r['error'][:300]}")
            return

        self._set_status(
            f"Backtest: {r['correct']}/{r['total']} ({r['accuracy']:.1%})"
        )
        self._clear()

        # Group races by season so the user can scan year-by-year and
        # the long table stays readable when scrolling.
        races = r["all_races"]
        by_year: dict[int, list] = {}
        for race in races:
            by_year.setdefault(int(race.get("year") or 0), []).append(race)

        years_sorted = sorted(by_year.keys())

        # Headline summary card
        summary = tk.Frame(self.results, bg=BG_CARD, padx=20, pady=14)
        summary.configure(highlightbackground=BORDER, highlightthickness=1)
        summary.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            summary,
            text=f"BACKTEST · {r['correct']}/{r['total']} correct  "
                 f"({r['accuracy']:.1%})",
            font=("Helvetica Neue", 12, "bold"), fg=GOLD, bg=BG_CARD,
        ).pack(anchor="w")
        if years_sorted:
            year_range = (
                f"{years_sorted[0]}–{years_sorted[-1]}"
                if years_sorted[0] != years_sorted[-1]
                else str(years_sorted[0])
            )
            tk.Label(
                summary,
                text=f"Seasons covered: {year_range}  ·  "
                     f"{len(races)} races  ·  scroll to view all",
                font=("Helvetica Neue", 10), fg=MUTED, bg=BG_CARD,
            ).pack(anchor="w", pady=(2, 0))

        # One card per season – each card has its own per-season hit
        # rate so trends across years are obvious.
        for yr in years_sorted:
            year_races = by_year[yr]
            year_correct = sum(1 for x in year_races if x["correct"])
            year_total = len(year_races)
            year_pct = (year_correct / year_total) if year_total else 0

            card = tk.Frame(self.results, bg=BG_CARD, padx=18, pady=14)
            card.configure(highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill=tk.X, pady=(0, 12))

            head = tk.Frame(card, bg=BG_CARD)
            head.pack(fill=tk.X, pady=(0, 8))
            tk.Label(head, text=f"{yr} SEASON",
                     font=("Helvetica Neue", 11, "bold"),
                     fg=F1_RED, bg=BG_CARD).pack(side=tk.LEFT)
            tk.Label(head,
                     text=f"  ·  {year_correct}/{year_total} correct  "
                          f"({year_pct:.0%})",
                     font=("Helvetica Neue", 10),
                     fg=MUTED, bg=BG_CARD).pack(side=tk.LEFT)

            # Column headers
            hdr = tk.Frame(card, bg=BG_SURFACE, padx=10, pady=6)
            hdr.pack(fill=tk.X, pady=(0, 4))
            for txt, w in [("Rd", 4), ("Race", 28),
                           ("Predicted", 11), ("Actual", 11), ("", 3)]:
                tk.Label(hdr, text=txt,
                         font=("Helvetica Neue", 9, "bold"),
                         fg=MUTED, bg=BG_SURFACE, width=w,
                         anchor="w").pack(side=tk.LEFT, padx=2)

            for race in sorted(year_races, key=lambda x: x.get("round", 0)):
                row = tk.Frame(card, bg=BG_CARD, padx=10, pady=4)
                row.pack(fill=tk.X)
                mark_color = GREEN if race["correct"] else RED
                mark = "✓" if race["correct"] else "✗"
                race_name = race.get("name", f"R{race['round']}")
                cells = [
                    (f"R{int(race['round']):02d}", 4, GRAY),
                    (race_name, 28, WHITE),
                    (race["predicted"], 11, GRAY),
                    (race["actual"], 11, GRAY),
                    (mark, 3, mark_color),
                ]
                for txt, w, fg in cells:
                    tk.Label(
                        row, text=txt,
                        font=("Menlo", 10), fg=fg, bg=BG_CARD,
                        width=w, anchor="w",
                    ).pack(side=tk.LEFT, padx=2)

        # Wire mouse-wheel scrolling onto every newly-added row so
        # the wheel keeps scrolling the outer canvas no matter where
        # the cursor is.  Without this the wheel "dies" the moment
        # the cursor crosses one of the 96+ row Frames.
        self._bind_wheel_recursive(self.results, self._scroll)
        # Snap back to the top so the user starts at the summary.
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(0.0)

    # -- Shared helpers --
    def _driver_row(self, parent, rank, abbr, team, prob, highlight=False):
        color = tc(team)
        fg = WHITE if highlight else GRAY
        bg = BG_SURFACE if highlight else BG_CARD

        row = tk.Frame(parent, bg=bg, padx=10, pady=8, cursor="hand2")
        row.pack(fill=tk.X, pady=1)

        # Wider Pn cell so two-digit positions don't shift everything.
        tk.Label(row, text=f"P{rank}",
                 font=("Helvetica Neue", 10, "bold"),
                 fg=F1_RED if rank == 1 else MUTED, bg=bg,
                 width=4, anchor="w"
                 ).pack(side=tk.LEFT, padx=(0, 6))

        # Team livery colour strip
        bar = tk.Frame(row, width=3, height=22, bg=color)
        bar.pack(side=tk.LEFT, padx=(0, 10))
        bar.pack_propagate(False)

        logo = self._logo(team, 22)
        if logo:
            tk.Label(row, image=logo, bg=bg).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(row, text=abbr,
                 font=("Helvetica Neue", 13, "bold"),
                 fg=fg, bg=bg).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(row, text=team,
                 font=("Helvetica Neue", 9),
                 fg=MUTED, bg=bg).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Probability bar + percentage.  Slightly larger track so the leader's
        # bar feels emphatic.
        pf = tk.Frame(row, bg=bg, width=160, height=22)
        pf.pack(side=tk.RIGHT)
        pf.pack_propagate(False)
        track_w = 100
        bw = max(2, int(track_w * min(prob, 1)))
        tk.Frame(pf, bg=BORDER, height=4, width=track_w).place(x=0, y=9)
        tk.Frame(pf, bg=color, height=4, width=bw).place(x=0, y=9)
        tk.Label(pf, text=f"{prob*100:.1f}%",
                 font=("Helvetica Neue", 10, "bold"),
                 fg=fg, bg=bg).place(x=track_w + 8, y=3)

        # Hover effect — subtle elevation across the whole row.
        def _hover(entering, widgets=(row,)):
            target_bg = BG_HOVER if entering else bg
            row.configure(bg=target_bg)
            for w in row.winfo_children():
                try:
                    w.configure(bg=target_bg)
                    for c in w.winfo_children():
                        try:
                            c.configure(bg=target_bg)
                        except tk.TclError:
                            pass
                except tk.TclError:
                    pass
            # The team-livery strip stays its own colour, restore it.
            try:
                bar.configure(bg=color)
            except tk.TclError:
                pass

        row.bind("<Enter>", lambda e: _hover(True))
        row.bind("<Leave>", lambda e: _hover(False))

    # Human-friendly aliases for the cryptic feature names.  Keeps the
    # importance chart readable on a 300-px-wide card without ellipsizing.
    _FEATURE_LABELS = {
        "GridPosition":       "Grid position",
        "RecentAvgPos":       "Recent avg position",
        "RecentAvgGrid":      "Recent avg grid",
        "RecentWinRate":      "Win % (last 10)",
        "RecentPodiumRate":   "Podium % (last 10)",
        "DNFRate":            "DNF rate",
        "DriverExperience":   "Career races",
        "HeadToHead":         "Beats teammate %",
        "TeamRecentForm":     "Team recent form",
        "DriverCircuitAvg":   "Driver @ circuit",
        "TeamCircuitAvg":     "Team @ circuit",
        "PUBatteryScore":     "PU battery score",
        "PUICEScore":         "PU ICE score",
        "DriverPointsBefore": "Driver points",
        "TeamPointsBefore":   "Team points",
        "Abbreviation":       "Driver identity",
        "TeamName":           "Team identity",
        "DriverNumber":       "Driver number",
    }

    def _render_chart(self, fi):
        self.chart_lbl.configure(image="")
        if not fi:
            return
        # Sort descending and keep only the top 10 so the chart doesn't get
        # crushed.  Use plain-English labels.
        items = sorted(fi.items(), key=lambda x: -x[1])[:10]
        labels = [self._FEATURE_LABELS.get(k, k) for k, _ in items]
        vals = [v for _, v in items]

        # Render top-to-bottom (highest importance at top) for natural
        # reading order.
        labels = labels[::-1]
        vals = vals[::-1]

        # Wider figure + explicit left margin so long labels never clip.
        # 50 % of the canvas is dedicated to label text – tk's font metrics
        # are a bit larger than matplotlib's default, so this matches reality.
        # Keep the chart short (2.4 in) so the HOW IT WORKS card below sits
        # above the fold on a 920-px window.
        fig, ax = plt.subplots(figsize=(4.0, 2.4), facecolor=BG_CARD, dpi=110)
        fig.subplots_adjust(left=0.50, right=0.97, top=0.97, bottom=0.04)
        ax.set_facecolor(BG_CARD)
        # Saturate from F1 red on the leader down to a soft maroon.
        n = len(labels)
        def _shade(i):
            t = i / max(1, n - 1)
            r = int(0xE1 - t * 0x90)
            g = int(0x06 + t * 0x10)
            b = int(0x00 + t * 0x10)
            return f"#{r:02x}{g:02x}{b:02x}"
        colors = [_shade(n - 1 - i) for i in range(n)]
        bars = ax.barh(range(n), vals, color=colors, height=0.62,
                        edgecolor="none")
        # Value annotations to the right of each bar so we don't need x-ticks.
        max_v = max(vals) if vals else 1.0
        for bar, v in zip(bars, vals):
            ax.text(v + max_v * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", ha="left",
                    fontsize=7.5, color=GRAY)

        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8.5, color=WHITE)
        ax.tick_params(axis="x", which="both", bottom=False, top=False,
                       labelbottom=False)
        ax.tick_params(axis="y", left=False)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.set_xlim(0, max_v * 1.25)
        ax.margins(y=0.02)
        # NB: don't use tight_layout here – it overrides subplots_adjust and
        # re-introduces the label clipping we just fixed.
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110, bbox_inches=None,
                    facecolor=BG_CARD)
        plt.close()
        buf.seek(0)
        if HAS_PIL:
            img = Image.open(buf).convert("RGB")
            photo = ImageTk.PhotoImage(img)
            self.chart_lbl.configure(image=photo)
            self.chart_lbl.image = photo

    def _clear(self):
        for w in self.results.winfo_children():
            w.destroy()

    def _show_empty(self, text):
        self._clear()
        # Centred empty card so the workspace doesn't look broken when there's
        # no prediction loaded yet.
        wrap = tk.Frame(self.results, bg=BG)
        wrap.pack(expand=True, fill=tk.BOTH, pady=70)
        card = tk.Frame(wrap, bg=BG_CARD, padx=40, pady=32)
        card.configure(highlightbackground=BORDER, highlightthickness=1)
        card.pack(anchor="center")

        if HAS_PIL:
            logo_img = _make_f1_logo(34)
            if logo_img is not None:
                self._empty_logo_tk = ImageTk.PhotoImage(logo_img)
                self._tk_images.append(self._empty_logo_tk)
                tk.Label(card, image=self._empty_logo_tk, bg=BG_CARD,
                         bd=0).pack(pady=(0, 12))

        for i, line in enumerate(text.splitlines()):
            tk.Label(card, text=line,
                     font=("Helvetica Neue", 14 if i == 0 else 11,
                           "bold" if i == 0 else "normal"),
                     fg=WHITE if i == 0 else MUTED, bg=BG_CARD,
                     justify=tk.CENTER).pack(pady=(0, 2))

    # ── View switching ──

    def _go_home(self):
        """Header logo / wordmark click handler – returns to the
        predictions view showing the last-predicted race.  Stops any
        running animations and team-radio playback so the home view
        is clean.
        """
        if getattr(self, "_radio_proc", None) or getattr(self, "_radio_sound", None):
            try:
                self._stop_radio()
            except Exception:
                pass
        self._switch_to_view("predictions")
        if self.result:
            race = ""
            nxt = self.result.get("next_race") or {}
            last = self.result.get("last_race") or {}
            if isinstance(nxt, dict):
                race = nxt.get("name") or ""
            if not race and isinstance(last, dict):
                race = last.get("name") or ""
            if race:
                self._set_status(f"Showing prediction · {race}")
            else:
                self._set_status("Showing last prediction")
        else:
            self._set_status("No prediction yet — click Predict Next Race")

    def _switch_to_view(self, view):
        self._current_view = view
        self._anim_running = False
        self.body.pack_forget()
        self.viz_frame.pack_forget()
        self.radio_frame.pack_forget()
        self.replays_frame.pack_forget()
        if view == "predictions":
            self.body.pack(fill=tk.BOTH, expand=True)
            self._set_active_btn(None)
        elif view == "viz":
            self.viz_frame.pack(fill=tk.BOTH, expand=True)
            self._set_active_btn(self.btn_viz)
        elif view == "radio":
            self.radio_frame.pack(fill=tk.BOTH, expand=True)
            self._set_active_btn(self.btn_radio)
        elif view == "replays":
            self.replays_frame.pack(fill=tk.BOTH, expand=True)
            self._set_active_btn(self.btn_replays)

    def _on_show_viz(self):
        if self._current_view == "viz":
            self._switch_to_view("predictions")
            return
        if not self.result:
            self._set_status("Run a prediction first")
            return
        self._build_viz()
        self._switch_to_view("viz")

    def _on_show_radio(self):
        if self._current_view == "radio":
            self._stop_radio()
            self._switch_to_view("predictions")
            return
        if not HAS_RADIO:
            self._set_status("f1radio package not installed (pip install f1radio)")
            return
        self._build_radio()
        self._switch_to_view("radio")

    def _on_show_replays(self):
        if self._current_view == "replays":
            self._switch_to_view("predictions")
            return
        # Lazy-build: only construct the widget tree the first time
        # the user opens the tab.  Subsequent toggles just re-show
        # the cached widget tree, so flipping in/out is instant.
        # Schedule loading inside `_build_replays` is dispatched to a
        # background thread so the UI never freezes while waiting on
        # disk / network – this prevents subsequent tab clicks from
        # being lost while the first Replays click is loading.
        if not self._replays_built:
            self._build_replays()
            self._replays_built = True
        self._switch_to_view("replays")

    # ── Team Radio ──

    @staticmethod
    def _parse_iso_dt(s):
        """Parse an ISO-8601 timestamp from f1radio (e.g. clip.date or
        event['date']) into a tz-aware datetime.  Returns None on failure."""
        if not s:
            return None
        try:
            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except Exception:
            return None

    @classmethod
    def _enrich_clip_meta(cls, clips, event_log):
        """Walk through every clip and compute, from `event_log` timestamps:
            - lap     : best-effort lap number for the clip
            - event   : short race-event note ("Pit exit open", "SC deployed",
                        "Yellow flag S2", "Lights out" ...) if the clip lands
                        inside a meaningful event window
            - dt      : parsed clip datetime (UTC)
        Returns (meta_list, total_laps).  meta_list is parallel to `clips`.
        """
        evs = []
        for ev in event_log or []:
            d = cls._parse_iso_dt(ev.get("date"))
            if d is None:
                continue
            evs.append({
                "dt": d,
                "lap": ev.get("lap_number"),
                "cat": (ev.get("category") or "").strip(),
                "flag": (ev.get("flag") or "").strip(),
                "scope": (ev.get("scope") or "").strip(),
                "msg": (ev.get("message") or "").strip(),
            })
        evs.sort(key=lambda e: e["dt"])
        max_lap = max((e["lap"] for e in evs if e["lap"]), default=0)

        def _lap_at(dt):
            best = None
            for e in evs:
                if e["dt"] <= dt and e["lap"]:
                    best = e["lap"]
                elif e["dt"] > dt:
                    break
            return best

        def _event_for(dt):
            best = None
            best_delta = None
            for e in evs:
                delta = abs((e["dt"] - dt).total_seconds())
                if delta > 90:
                    continue
                cat = e["cat"].lower()
                msg = e["msg"]
                if not msg:
                    continue
                up = msg.upper()
                interesting = (
                    cat in ("flag", "drs", "safetycar")
                    or "PIT" in up or "SAFETY" in up or "VSC" in up
                    or "LIGHTS" in up or "GREEN" in up or "RED" in up
                    or "YELLOW" in up or "CHEQUERED" in up
                )
                if not interesting:
                    continue
                if best is None or delta < best_delta:
                    best, best_delta = e, delta
            if best is None:
                return ""
            m = best["msg"].strip()
            if len(m) > 42:
                m = m[:39] + "..."
            return m.title() if m.isupper() else m

        meta = []
        for c in clips:
            dt = cls._parse_iso_dt(getattr(c, "date", None))
            lap = None
            ev_label = ""
            if dt is not None:
                lap = _lap_at(dt)
                ev_label = _event_for(dt)
            direct_lap = getattr(c, "lap", None)
            if direct_lap:
                lap = direct_lap
            ctx_lap = None
            if getattr(c, "context", None) is not None:
                ctx_lap = getattr(c.context, "lap_number", None)
            if ctx_lap:
                lap = ctx_lap
            meta.append({"lap": lap, "event": ev_label, "dt": dt})
        return meta, max_lap

    # Full official GP names – f1radio fuzzy-matches short names against
    # OpenF1's `meeting_name` and "Bahrain" alone resolves to *Pre-Season
    # Testing*.  Using the full "<X> Grand Prix" string forces the correct
    # session every time.
    RADIO_RACES = [
        (2025, "Australian Grand Prix"),
        (2025, "Chinese Grand Prix"),
        (2025, "Japanese Grand Prix"),
        (2025, "Bahrain Grand Prix"),
        (2025, "Saudi Arabian Grand Prix"),
        (2025, "Miami Grand Prix"),
        (2025, "Emilia Romagna Grand Prix"),
        (2025, "Monaco Grand Prix"),
        (2025, "Spanish Grand Prix"),
        (2025, "Canadian Grand Prix"),
        (2025, "Austrian Grand Prix"),
        (2025, "British Grand Prix"),
        (2025, "Belgian Grand Prix"),
        (2025, "Hungarian Grand Prix"),
        (2025, "Dutch Grand Prix"),
        (2025, "Italian Grand Prix"),
        (2025, "Azerbaijan Grand Prix"),
        (2025, "Singapore Grand Prix"),
        (2025, "United States Grand Prix"),
        (2025, "Mexico City Grand Prix"),
        (2025, "S\u00e3o Paulo Grand Prix"),
        (2025, "Las Vegas Grand Prix"),
        (2025, "Qatar Grand Prix"),
        (2025, "Abu Dhabi Grand Prix"),
    ]

    def _build_radio(self):
        self._stop_radio()
        for w in self.radio_frame.winfo_children():
            w.destroy()
        self._radio_tk_images = []

        # Header
        top = tk.Frame(self.radio_frame, bg=BG)
        top.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top, text="TEAM RADIO", font=("Helvetica Neue", 16, "bold"),
                 fg=GOLD, bg=BG).pack(side=tk.LEFT)
        tk.Label(top, text="Listen to driver-team communications",
                 font=("Helvetica Neue", 11), fg=MUTED, bg=BG
                 ).pack(side=tk.LEFT, padx=(16, 0), pady=(3, 0))

        tk.Frame(self.radio_frame, bg=BORDER, height=1).pack(fill=tk.X)

        # Two-column layout
        content = tk.Frame(self.radio_frame, bg=BG)
        content.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Left: race selector
        left_panel = tk.Frame(content, bg=BG_CARD, width=220, padx=12, pady=12)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_panel.pack_propagate(False)
        left_panel.configure(highlightbackground=BORDER, highlightthickness=1)

        tk.Label(left_panel, text="SELECT RACE", font=("Helvetica Neue", 9, "bold"),
                 fg=GOLD, bg=BG_CARD).pack(anchor="w")
        tk.Frame(left_panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(6, 8))

        race_scroll_frame = tk.Frame(left_panel, bg=BG_CARD)
        race_scroll_frame.pack(fill=tk.BOTH, expand=True)

        race_canvas = tk.Canvas(race_scroll_frame, bg=BG_CARD, highlightthickness=0)
        race_sb = ttk.Scrollbar(race_scroll_frame, orient="vertical", command=race_canvas.yview)
        race_inner = tk.Frame(race_canvas, bg=BG_CARD)
        race_inner.bind("<Configure>", lambda e: race_canvas.configure(scrollregion=race_canvas.bbox("all")))
        race_canvas.create_window((0, 0), window=race_inner, anchor="nw")
        race_canvas.configure(yscrollcommand=race_sb.set)
        race_sb.pack(side=tk.RIGHT, fill=tk.Y)
        race_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _race_wheel(e, c=race_canvas):
            c.yview_scroll(self._wheel_units(e), "units")
            return "break"

        for w in (race_canvas, race_inner):
            w.bind("<MouseWheel>", _race_wheel)
        self._race_canvas = race_canvas
        self._race_wheel_handler = _race_wheel

        self._race_btns = []
        for yr, race in self.RADIO_RACES:
            rb = tk.Frame(race_inner, bg=BG_SURFACE, cursor="hand2",
                          highlightbackground=BORDER, highlightthickness=1)
            rb.pack(fill=tk.X, pady=2)
            lbl = tk.Label(rb, text=f"  {race}", font=("Helvetica Neue", 10),
                           fg=GRAY, bg=BG_SURFACE, anchor="w", padx=6, pady=5,
                           cursor="hand2")
            lbl.pack(fill=tk.X)
            rb._lbl = lbl
            rb._race = (yr, race)
            for w in (rb, lbl):
                w.bind("<Button-1>", lambda e, r=(yr, race), b=rb: self._load_radio_race(r, b))
                w.bind("<Enter>", lambda e, b=rb: (
                    b.configure(bg=BG_HOVER), b._lbl.configure(bg=BG_HOVER)
                ))
                w.bind("<Leave>", lambda e, b=rb: (
                    b.configure(bg=getattr(b, '_sel_bg', BG_SURFACE)),
                    b._lbl.configure(bg=getattr(b, '_sel_bg', BG_SURFACE))
                ))
            self._race_btns.append(rb)

        # Right: clips panel
        right_panel = tk.Frame(content, bg=BG_CARD, padx=16, pady=16)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_panel.configure(highlightbackground=BORDER, highlightthickness=1)

        # Now-playing bar
        np_bar = tk.Frame(right_panel, bg=BG_SURFACE, padx=12, pady=10)
        np_bar.pack(fill=tk.X, pady=(0, 10))
        np_bar.configure(highlightbackground=BORDER, highlightthickness=1)

        self._np_canvas = tk.Canvas(np_bar, bg=BG_SURFACE, highlightthickness=0,
                                     height=36, width=50)
        self._np_canvas.pack(side=tk.LEFT, padx=(0, 10))

        np_text = tk.Frame(np_bar, bg=BG_SURFACE)
        np_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._np_driver = tk.Label(np_text, text="No clip playing",
                                    font=("Helvetica Neue", 12, "bold"),
                                    fg=WHITE, bg=BG_SURFACE, anchor="w")
        self._np_driver.pack(anchor="w")
        self._np_detail = tk.Label(np_text, text="Select a race to load team radio",
                                    font=("Helvetica Neue", 9), fg=MUTED,
                                    bg=BG_SURFACE, anchor="w")
        self._np_detail.pack(anchor="w")

        stop_btn = self._make_btn(np_bar, "Stop", "#3a1a1a", RED,
                                   self._stop_radio, border="#4a2222")
        stop_btn.pack(side=tk.RIGHT)

        # Clip list header
        hdr = tk.Frame(right_panel, bg=BG_CARD)
        hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(hdr, text="RADIO CLIPS", font=("Helvetica Neue", 9, "bold"),
                 fg=GOLD, bg=BG_CARD).pack(side=tk.LEFT)
        self._clip_count_lbl = tk.Label(hdr, text="", font=("Helvetica Neue", 9),
                                         fg=MUTED, bg=BG_CARD)
        self._clip_count_lbl.pack(side=tk.RIGHT)

        tk.Frame(right_panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 6))

        # Scrollable clip list
        clip_scroll = tk.Frame(right_panel, bg=BG_CARD)
        clip_scroll.pack(fill=tk.BOTH, expand=True)

        self._clip_canvas = tk.Canvas(clip_scroll, bg=BG_CARD, highlightthickness=0)
        clip_sb = ttk.Scrollbar(clip_scroll, orient="vertical", command=self._clip_canvas.yview)
        self._clip_inner = tk.Frame(self._clip_canvas, bg=BG_CARD)
        self._clip_inner.bind("<Configure>",
                               lambda e: self._clip_canvas.configure(
                                   scrollregion=self._clip_canvas.bbox("all")))
        self._clip_canvas.create_window((0, 0), window=self._clip_inner, anchor="nw")
        self._clip_canvas.configure(yscrollcommand=clip_sb.set)
        clip_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._clip_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        def _clip_wheel(e):
            self._clip_canvas.yview_scroll(self._wheel_units(e), "units")
            return "break"

        for w in (self._clip_canvas, self._clip_inner):
            w.bind("<MouseWheel>", _clip_wheel)
        self._clip_wheel_handler = _clip_wheel

        self._radio_loading_lbl = tk.Label(self._clip_inner, text="Select a race from the left panel",
                                            font=("Helvetica Neue", 12), fg=MUTED, bg=BG_CARD,
                                            pady=40)
        self._radio_loading_lbl.pack()

    def _load_radio_race(self, race_info, btn):
        yr, race_name = race_info

        for rb in self._race_btns:
            rb._sel_bg = BG_SURFACE
            rb.configure(bg=BG_SURFACE)
            rb._lbl.configure(bg=BG_SURFACE, fg=GRAY)
        btn._sel_bg = GOLD_DIM
        btn.configure(bg=GOLD_DIM)
        btn._lbl.configure(bg=GOLD_DIM, fg=GOLD)

        self._stop_radio()
        for w in self._clip_inner.winfo_children():
            w.destroy()
        self._radio_loading_lbl = tk.Label(self._clip_inner,
                                            text=f"Loading radio clips for {race_name}...",
                                            font=("Helvetica Neue", 12), fg=GOLD, bg=BG_CARD,
                                            pady=40)
        self._radio_loading_lbl.pack()
        self._clip_count_lbl.configure(text="loading...")

        def fetch():
            try:
                # Prefer the direct FIA-archive loader: it merges static +
                # streaming captures and gives us the full grid (heard +
                # silent) so the UI can show coverage honestly.
                if radio_fia is not None:
                    def _prog(stage, done, total):
                        self.root.after(0, lambda: self._set_status(
                            f"{stage}: {done}/{total}"
                        ))
                    session = radio_fia.load(yr, race_name, "R", progress=_prog)
                elif HAS_F1RADIO:
                    session = f1radio.load(yr, race_name, "R")
                else:
                    raise RuntimeError("No radio backend available")
                clips = list(session.clips)
                event_log = list(getattr(session, "event_log", []) or [])
                session_label = f"{yr} {race_name.upper()} – RACE"
                # FIA loader exposes coverage info + per-driver context maps.
                drivers_silent = list(getattr(session, "drivers_silent", []) or [])
                drivers_heard = list(getattr(session, "drivers_heard", []) or [])
                drivers_all = list(getattr(session, "drivers", []) or [])
                stints_by_drv = dict(getattr(session, "stints_by_drv", {}) or {})
                laps_by_drv = dict(getattr(session, "laps_by_drv", {}) or {})
                self.root.after(
                    0,
                    lambda: self._populate_clips(
                        clips, race_name, event_log, session_label,
                        drivers_silent=drivers_silent,
                        drivers_heard=drivers_heard,
                        drivers_all=drivers_all,
                        stints_by_drv=stints_by_drv,
                        laps_by_drv=laps_by_drv,
                    ),
                )
            except Exception as e:
                self.root.after(0, lambda: self._radio_error(str(e)))

        threading.Thread(target=fetch, daemon=True).start()

    def _radio_error(self, msg):
        for w in self._clip_inner.winfo_children():
            w.destroy()
        tk.Label(self._clip_inner, text=f"Error: {msg}",
                 font=("Helvetica Neue", 11), fg=RED, bg=BG_CARD,
                 wraplength=400, pady=30).pack()
        self._clip_count_lbl.configure(text="error")

    def _populate_clips(self, clips, race_name, event_log=None,
                        session_label="", drivers_silent=None,
                        drivers_heard=None, drivers_all=None,
                        stints_by_drv=None, laps_by_drv=None):
        # Compute lap + event metadata for every clip, then sort the clips
        # chronologically (by their broadcast timestamp) so the visible order
        # actually mirrors how the race unfolded.
        meta, total_laps = self._enrich_clip_meta(clips, event_log or [])

        # Fill in stint/compound/tyre-age/position now that we know the lap.
        if radio_fia is not None and (stints_by_drv or laps_by_drv):
            stints_by_drv = stints_by_drv or {}
            laps_by_drv = laps_by_drv or {}
            for clip, m in zip(clips, meta):
                radio_fia.annotate_clip_context(
                    clip, m.get("lap"), stints_by_drv, laps_by_drv,
                )

        pairs = list(zip(clips, meta))
        pairs.sort(key=lambda p: (
            p[1]["dt"] or datetime.max.replace(tzinfo=timezone.utc),
            p[1]["lap"] or 0,
        ))
        clips_sorted = [p[0] for p in pairs]
        meta_sorted = [p[1] for p in pairs]

        self._radio_clips = clips_sorted
        self._radio_clip_meta = meta_sorted
        self._radio_event_log = event_log or []
        self._radio_session_label = session_label
        self._radio_total_laps = total_laps
        # Coverage info from the FIA loader.
        self._radio_drivers_silent = list(drivers_silent or [])
        self._radio_drivers_heard = list(drivers_heard or [])
        self._radio_drivers_all = list(drivers_all or [])
        # Sort drivers by number of clips (most chatty first) for the filter
        drv_counts = {}
        for c in clips_sorted:
            drv = c.driver_name or c.driver or "Unknown"
            drv_counts[drv] = drv_counts.get(drv, 0) + 1
        self._radio_drivers = [
            d for d, _ in sorted(drv_counts.items(), key=lambda x: (-x[1], x[0]))
        ]
        self._radio_filter_driver = "ALL"

        # Update the now-playing header with the race + lap span summary
        # and coverage info ("11 of 20 drivers heard").
        if hasattr(self, "_np_detail"):
            span = f"{len(clips_sorted)} clips"
            if total_laps:
                span += f"  ·  spanning {total_laps} laps"
            if self._radio_drivers_all:
                span += (
                    f"  ·  {len(self._radio_drivers_heard)} of "
                    f"{len(self._radio_drivers_all)} drivers heard"
                )
            if session_label:
                span += f"  ·  {session_label}"
            self._np_detail.configure(text=span)

        self._refresh_clip_list()

    def _refresh_clip_list(self):
        """Re-render the right-hand clip list using the current driver filter."""
        for w in self._clip_inner.winfo_children():
            w.destroy()

        # Render the toolbar (driver filter + "play all" button) at the top.
        self._build_clip_toolbar()
        # Coverage banner: how many drivers of the grid we have radio for,
        # plus the explicit list of drivers with zero clips this race.
        self._build_coverage_banner()

        if not self._radio_clips:
            tk.Label(self._clip_inner, text="No radio clips available for this race.",
                     font=("Helvetica Neue", 12), fg=MUTED, bg=BG_CARD, pady=30).pack()
            self._radio_filtered_idx = []
            self._clip_count_lbl.configure(text="0 clips")
            return

        flt = self._radio_filter_driver
        visible = []
        for i, c in enumerate(self._radio_clips):
            drv = c.driver_name or c.driver or "Unknown"
            if flt == "ALL" or drv == flt:
                visible.append(i)
        self._radio_filtered_idx = visible

        # Header count line.  Show heard/total drivers when we have it so
        # the user can see at a glance which drivers contributed radio.
        bits = [f"{len(visible)} of {len(self._radio_clips)} clips"]
        if self._radio_drivers_all:
            bits.append(
                f"{len(self._radio_drivers_heard)}/"
                f"{len(self._radio_drivers_all)} drivers"
            )
        if self._radio_total_laps:
            bits.append(f"{self._radio_total_laps} laps")
        self._clip_count_lbl.configure(text="  ·  ".join(bits))

        if not visible:
            tk.Label(self._clip_inner,
                     text=f"No clips for {flt} in this race.",
                     font=("Helvetica Neue", 11), fg=MUTED,
                     bg=BG_CARD, pady=30).pack()
            return

        # Group rows by lap so the user can see the race timeline clearly.
        last_lap = object()
        for vi, ci in enumerate(visible):
            clip = self._radio_clips[ci]
            cmeta = self._radio_clip_meta[ci]
            lap = cmeta.get("lap")
            if lap != last_lap:
                self._make_lap_divider(lap, cmeta.get("event", ""))
                last_lap = lap
            self._make_clip_row(ci, clip, cmeta, vi)

    def _build_clip_toolbar(self):
        """Driver filter dropdown + Play-all/Stop-all controls at the top of
        the clip list."""
        bar = tk.Frame(self._clip_inner, bg=BG_CARD, pady=6)
        bar.pack(fill=tk.X)

        tk.Label(bar, text="DRIVER",
                 font=("Helvetica Neue", 8, "bold"), fg=GOLD, bg=BG_CARD
                 ).pack(side=tk.LEFT, padx=(0, 6))

        options = ["ALL"] + list(self._radio_drivers)
        var = tk.StringVar(value=self._radio_filter_driver)
        self._radio_filter_var = var

        # Use a themed Combobox – ttk picks up the dark style elsewhere in
        # the app, but we override colors locally for safety.
        cb = ttk.Combobox(bar, values=options, textvariable=var,
                           state="readonly", width=22,
                           font=("Helvetica Neue", 9))
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_radio_filter_change())

        play_all = self._make_btn(
            bar,
            "▶  Play Full Race" if self._radio_filter_driver == "ALL"
            else f"▶  Play {self._radio_filter_driver.split()[0].title()}'s Race",
            GOLD_DIM, GOLD, self._on_radio_play_all, border=GOLD_DIM,
        )
        play_all.pack(side=tk.RIGHT, padx=(8, 0))

        stop_btn = self._make_btn(bar, "■  Stop", BG_SURFACE, GRAY,
                                   self._stop_radio, border=BORDER)
        stop_btn.pack(side=tk.RIGHT, padx=(8, 0))

        tk.Frame(self._clip_inner, bg=BORDER, height=1).pack(fill=tk.X, pady=(2, 4))

    def _on_radio_filter_change(self):
        self._radio_filter_driver = self._radio_filter_var.get()
        self._radio_play_queue = []
        self._refresh_clip_list()

    def _build_coverage_banner(self):
        """A thin, dismissable banner under the toolbar that lists which
        drivers the FIA *didn't* publish any radio for, so absence is
        explicit instead of silent."""
        silent = getattr(self, "_radio_drivers_silent", []) or []
        heard = getattr(self, "_radio_drivers_heard", []) or []
        if not silent and not heard:
            return  # legacy loader (f1radio) – no coverage info

        # Skip the banner entirely when every driver was heard – nothing
        # useful to communicate.
        if not silent:
            return

        banner = tk.Frame(self._clip_inner, bg=BG_SURFACE,
                          padx=10, pady=8)
        banner.pack(fill=tk.X, pady=(2, 4))
        banner.configure(highlightbackground=BORDER, highlightthickness=1)

        # Left: heard summary line in F1 red.
        left = tk.Frame(banner, bg=BG_SURFACE)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            left,
            text=(
                f"BROADCAST RADIO ONLY  ·  {len(heard)} of "
                f"{len(heard) + len(silent)} drivers heard this race"
            ),
            font=("Helvetica Neue", 9, "bold"),
            fg=F1_RED, bg=BG_SURFACE, anchor="w",
        ).pack(anchor="w")

        # Body: comma-separated abbreviations of silent drivers, wrapped.
        silent_abbrs = "  ·  ".join(
            (d.get("abbr") or "??") if isinstance(d, dict) else str(d)
            for d in silent
        )
        tk.Label(
            left,
            text=f"No radio published for: {silent_abbrs}",
            font=("Helvetica Neue", 9),
            fg=MUTED, bg=BG_SURFACE, anchor="w",
            wraplength=520, justify="left",
        ).pack(anchor="w", pady=(2, 0))

        # Footnote so the user understands this is a data-source limit,
        # not a bug in the app.
        tk.Label(
            left,
            text=(
                "Only radio the FIA broadcast on the world feed is publicly "
                "archived – private engineer-to-driver comms aren't available."
            ),
            font=("Helvetica Neue", 8, "italic"),
            fg=MUTED, bg=BG_SURFACE, anchor="w",
            wraplength=520, justify="left",
        ).pack(anchor="w", pady=(2, 0))

    def _make_lap_divider(self, lap, event_label=""):
        row = tk.Frame(self._clip_inner, bg=BG, pady=2)
        row.pack(fill=tk.X, pady=(8, 2))
        lap_txt = f"LAP {lap}" if lap else "PRE-RACE / FORMATION"
        tk.Label(row, text=lap_txt,
                 font=("Helvetica Neue", 9, "bold"), fg=GOLD,
                 bg=BG, padx=4).pack(side=tk.LEFT)
        if event_label:
            tk.Label(row, text=f"  ·  {event_label}",
                     font=("Helvetica Neue", 9, "italic"),
                     fg=MUTED, bg=BG).pack(side=tk.LEFT)
        tk.Frame(self._clip_inner, bg=BORDER, height=1).pack(fill=tk.X, padx=2)

    def _on_radio_play_all(self):
        """Queue every clip in the current filter and play them in order."""
        if not self._radio_filtered_idx:
            return
        self._radio_play_queue = list(self._radio_filtered_idx)
        self._radio_queue_total = len(self._radio_play_queue)
        self._radio_queue_pos = 0
        self._radio_advance_queue()

    def _radio_advance_queue(self):
        if not self._radio_play_queue:
            return
        next_idx = self._radio_play_queue.pop(0)
        self._radio_queue_pos = self._radio_queue_total - len(self._radio_play_queue)
        self._play_radio_clip(next_idx, from_queue=True)

    def _make_clip_row(self, idx, clip, meta=None, visible_idx=None):
        color = tc(clip.team) if clip.team else GRAY
        # Stripe by *visible* index so alternating bands stay even after
        # filtering, not by the original clip index.
        stripe = visible_idx if visible_idx is not None else idx
        bg = BG_SURFACE if stripe % 2 == 0 else BG_CARD

        row = tk.Frame(self._clip_inner, bg=bg, padx=10, pady=7, cursor="hand2")
        row.pack(fill=tk.X, pady=1)

        accent = tk.Frame(row, bg=color, width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        play_lbl = tk.Label(row, text="\u25B6", font=("Helvetica Neue", 14),
                            fg=color, bg=bg, cursor="hand2")
        play_lbl.pack(side=tk.LEFT, padx=(0, 10))

        info = tk.Frame(row, bg=bg)
        info.pack(side=tk.LEFT, fill=tk.X, expand=True)

        driver_text = clip.driver_name or clip.driver or "Unknown"
        team_text = clip.team or ""

        # Top line: driver  ·  lap
        top = tk.Frame(info, bg=bg)
        top.pack(anchor="w", fill=tk.X)
        tk.Label(top, text=driver_text,
                 font=("Helvetica Neue", 11, "bold"), fg=WHITE, bg=bg,
                 anchor="w").pack(side=tk.LEFT)
        lap = (meta or {}).get("lap")
        if lap:
            tk.Label(top, text=f"  ·  Lap {lap}" + (
                f" / {self._radio_total_laps}" if self._radio_total_laps else ""
            ), font=("Helvetica Neue", 9), fg=GOLD, bg=bg).pack(side=tk.LEFT)

        # Second line: team · position · compound · stint · tyre-age
        ctx = getattr(clip, "context", None)
        detail_parts = [team_text]
        if ctx is not None:
            pos = getattr(ctx, "position", None)
            if pos:
                detail_parts.append(f"P{pos}")
            comp = getattr(ctx, "compound", None)
            if comp:
                detail_parts.append(comp)
            stint = getattr(ctx, "stint_number", None)
            if stint:
                detail_parts.append(f"Stint {stint}")
            age = getattr(ctx, "tyre_age", None)
            if age is not None:
                detail_parts.append(f"{age} lap{'s' if age != 1 else ''} on tyres")
            gap = getattr(ctx, "gap_to_leader", None)
            if gap:
                detail_parts.append(f"+{gap}")
        detail = "  |  ".join(p for p in detail_parts if p)
        tk.Label(info, text=detail, font=("Helvetica Neue", 9),
                 fg=MUTED, bg=bg, anchor="w").pack(anchor="w")

        # Third line (only when we matched a race event): event context
        ev_label = (meta or {}).get("event") or ""
        if ev_label:
            tk.Label(info, text=ev_label,
                     font=("Helvetica Neue", 8, "italic"),
                     fg=F1_RED, bg=bg, anchor="w").pack(anchor="w", pady=(1, 0))

        tk.Label(row, text=f"#{idx + 1}", font=("Helvetica Neue", 9),
                 fg=MUTED, bg=bg).pack(side=tk.RIGHT, padx=(10, 0))

        row._clip_idx = idx
        row._play_lbl = play_lbl
        row._orig_bg = bg
        # Bind on the leaves only (cheap) – the row + nested labels all share
        # the same click handler.
        leaves = [row, play_lbl, info, accent, top]
        leaves.extend(info.winfo_children())
        leaves.extend(top.winfo_children())
        for w in leaves:
            w.bind("<Button-1>", lambda e, ci=idx: self._play_radio_clip(ci))
            w.bind("<Enter>", lambda e, r=row: self._clip_hover(r, True))
            w.bind("<Leave>", lambda e, r=row: self._clip_hover(r, False))

    def _clip_hover(self, row, entering):
        bg = BG_HOVER if entering else row._orig_bg
        row.configure(bg=bg)
        for w in row.winfo_children():
            if isinstance(w, (tk.Label, tk.Frame)):
                try:
                    w.configure(bg=bg)
                    for c in w.winfo_children():
                        if isinstance(c, tk.Label):
                            c.configure(bg=bg)
                except Exception:
                    pass

    def _play_radio_clip(self, idx, from_queue=False):
        if idx >= len(self._radio_clips):
            return
        clip = self._radio_clips[idx]
        meta = self._radio_clip_meta[idx] if idx < len(self._radio_clip_meta) else {}

        # When a manual clip is clicked, drop any pending queue so we don't
        # surprise the user by auto-advancing.
        if not from_queue:
            self._radio_play_queue = []
            self._radio_queue_total = 0
            self._radio_queue_pos = 0

        # _stop_radio resets the now-playing labels too, so call it first then
        # repopulate them with the new clip's metadata.
        self._stop_radio(_keep_queue=True)

        path = clip.local_path
        if not path or not os.path.exists(path):
            self._set_status("Audio file not available")
            return

        self._radio_playing = idx
        driver_text = clip.driver_name or clip.driver or "Unknown"
        team_text = clip.team or ""
        self._np_driver.configure(
            text=f"{driver_text}",
            fg=tc(team_text) if team_text else WHITE,
        )
        detail_bits = [team_text]
        lap = meta.get("lap") if meta else None
        if lap:
            if self._radio_total_laps:
                detail_bits.append(f"Lap {lap} / {self._radio_total_laps}")
            else:
                detail_bits.append(f"Lap {lap}")
        if from_queue and self._radio_queue_total:
            detail_bits.append(
                f"Clip {self._radio_queue_pos}/{self._radio_queue_total} in race"
            )
        else:
            detail_bits.append(f"Clip #{idx + 1}")
        ev = (meta or {}).get("event")
        if ev:
            detail_bits.append(ev)
        self._np_detail.configure(text="  |  ".join(detail_bits))

        # Prefer the playsound3 backend (cross-platform, ships with the
        # `f1radio[playback]` extra).  Fall back to the legacy CLI players
        # only if it isn't available.
        self._radio_sound = None
        self._radio_proc = None
        started = False

        if HAS_PLAYSOUND:
            try:
                self._radio_sound = _playsound(str(path), block=False)
                started = True
            except Exception as e:
                self._set_status(f"Audio backend error: {e}")
                self._radio_sound = None

        if not started:
            for cmd in (
                ["afplay", str(path)],
                ["ffplay", "-nodisp", "-autoexit", str(path)],
            ):
                try:
                    self._radio_proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    started = True
                    break
                except FileNotFoundError:
                    continue
            if not started:
                self._set_status("No audio backend available (install playsound3 or ffmpeg)")
                return

        self._radio_anim_running = True
        self._radio_wave_frame = 0
        self._radio_wave_tick()

        def wait_for_end():
            if self._radio_sound is not None:
                try:
                    self._radio_sound.wait()
                except Exception:
                    pass
            elif self._radio_proc is not None:
                try:
                    self._radio_proc.wait()
                except Exception:
                    pass
            self.root.after(0, self._on_radio_ended)
        threading.Thread(target=wait_for_end, daemon=True).start()

    def _stop_radio(self, _keep_queue=False):
        self._radio_anim_running = False
        snd = getattr(self, "_radio_sound", None)
        if snd is not None:
            try:
                snd.stop()
            except Exception:
                pass
            self._radio_sound = None
        if self._radio_proc:
            try:
                self._radio_proc.terminate()
                self._radio_proc.wait(timeout=2)
            except Exception:
                try:
                    self._radio_proc.kill()
                except Exception:
                    pass
            self._radio_proc = None
        # User-initiated stop should also clear the queue so we don't suddenly
        # auto-advance on the next clip.
        if not _keep_queue:
            self._radio_play_queue = []
            self._radio_queue_total = 0
            self._radio_queue_pos = 0
        self._radio_playing = None
        if hasattr(self, "_np_driver"):
            self._np_driver.configure(text="No clip playing", fg=WHITE)
            self._np_detail.configure(text="Select a clip to play")
        if hasattr(self, "_np_canvas"):
            self._np_canvas.delete("all")

    def _on_radio_ended(self):
        self._radio_anim_running = False
        self._radio_playing = None
        if hasattr(self, "_np_canvas"):
            self._np_canvas.delete("all")
        # Auto-advance if we're inside a "Play full race" queue.
        if self._radio_play_queue:
            # Small gap between clips so they don't feel mashed together.
            self.root.after(350, self._radio_advance_queue)
            return
        if hasattr(self, "_np_driver"):
            self._np_driver.configure(text="No clip playing", fg=WHITE)
            self._np_detail.configure(text="Playback finished")

    def _radio_wave_tick(self):
        if not self._radio_anim_running or self._current_view != "radio":
            return
        c = self._np_canvas
        c.delete("all")
        w = 50
        h = 36
        # Drive the wave from wall-clock time so the bars animate at the same
        # speed regardless of scheduling jitter, and bump the redraw rate up
        # to ~30 fps for visibly smoother motion.
        t = time.perf_counter()
        bars = 8
        bar_w = w / bars
        for i in range(bars):
            amp = 0.3 + 0.7 * abs(math.sin(t * 7.5 + i * 0.8))
            bh = amp * (h - 4)
            x0 = i * bar_w + 2
            x1 = x0 + bar_w - 2
            y0 = (h - bh) / 2
            y1 = y0 + bh
            color_val = int(180 + 75 * amp)
            green_val = int(160 * amp)
            bar_color = f"#{min(255, color_val):02x}{green_val:02x}20"
            c.create_rectangle(x0, y0, x1, y1, fill=bar_color, outline="")
        self.root.after(33, self._radio_wave_tick)

    # ── Race Replays (FullRaces.com) ──

    # Friendly aliases so the search URL hits the way fullraces.com
    # actually titles its uploads (e.g. "Sao Paulo" rather than "São
    # Paulo", which trips up basic WordPress search).
    REPLAY_RACE_ALIASES = {
        "São Paulo Grand Prix": "Sao Paulo Grand Prix",
        "Emilia Romagna Grand Prix": "Emilia Romagna Grand Prix",
        "70th Anniversary Grand Prix": "70th Anniversary Grand Prix",
    }

    # The full set of session links FullRaces.com publishes per round.
    # Keyword is what we feed into the WordPress `?s=` search so we
    # land on the matching post for that round + year.
    REPLAY_SESSIONS = [
        ("Race",          "race"),
        ("Qualifying",    "qualifying"),
        ("Sprint",        "sprint"),
        ("Sprint Quali",  "sprint qualifying"),
        ("Practice",      "practice"),
    ]

    @staticmethod
    def _replay_search_url(year: int, race_name: str, keyword: str) -> str:
        """Build a FullRaces.com search URL for a given race + session.

        Using the WordPress `?s=` query is more durable than guessing
        the post slug — every upload's title contains the session
        keyword + year + race name, so search hits the right post even
        if the title casing or punctuation drifts.
        """
        race = ApexAIApp.REPLAY_RACE_ALIASES.get(race_name, race_name)
        terms = f"{keyword} f1 {year} {race}".strip()
        return "https://fullraces.com/?s=" + urllib.parse.quote_plus(terms)

    def _open_replay(self, year: int, race_name: str, keyword: str):
        url = self._replay_search_url(year, race_name, keyword)
        try:
            webbrowser.open(url, new=2)
            self._set_status(
                f"Opened FullRaces.com – {race_name} {year} ({keyword})"
            )
        except Exception as exc:
            self._set_status(f"Could not open browser: {exc}")

    def _build_replays(self):
        """Render the Race Replays tab.  Schedule is fetched lazily
        the first time the tab opens, then cached on the widget tree
        until the next refresh."""
        for w in self.replays_frame.winfo_children():
            w.destroy()

        # Header row
        top = tk.Frame(self.replays_frame, bg=BG)
        top.pack(fill=tk.X, pady=(0, 8))
        tk.Label(top, text="RACE REPLAYS",
                 font=("Helvetica Neue", 16, "bold"),
                 fg=GOLD, bg=BG).pack(side=tk.LEFT)
        tk.Label(top, text="Watch every session – powered by FullRaces.com",
                 font=("Helvetica Neue", 11), fg=MUTED, bg=BG
                 ).pack(side=tk.LEFT, padx=(16, 0), pady=(3, 0))

        # Year selector lives on the right of the header.
        right = tk.Frame(top, bg=BG)
        right.pack(side=tk.RIGHT)

        current_year = datetime.now().year
        years = [str(y) for y in range(current_year, current_year - 8, -1)]
        if self._replays_year is None:
            self._replays_year = str(current_year)

        tk.Label(right, text="SEASON",
                 font=("Helvetica Neue", 9, "bold"),
                 fg=MUTED, bg=BG).pack(side=tk.LEFT, padx=(0, 8))
        self._replays_year_var = tk.StringVar(value=self._replays_year)
        cb = ttk.Combobox(right, values=years, state="readonly",
                          textvariable=self._replays_year_var, width=6,
                          font=("Helvetica Neue", 11))
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._reload_replays())

        # Quick link to the FullRaces home page.
        home_btn = self._make_btn(right, "Open FullRaces.com",
                                   BG_SURFACE, WHITE,
                                   lambda: webbrowser.open(
                                       "https://fullraces.com/", new=2),
                                   border=BORDER)
        home_btn.pack(side=tk.LEFT, padx=(12, 0))

        tk.Frame(self.replays_frame, bg=F1_RED, height=2).pack(
            fill=tk.X, pady=(2, 10))

        # Scrollable race list.
        body = tk.Frame(self.replays_frame, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        self._replays_canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(body, orient="vertical",
                           command=self._replays_canvas.yview)
        self._replays_inner = tk.Frame(self._replays_canvas, bg=BG)
        self._replays_inner.bind(
            "<Configure>",
            lambda e: self._replays_canvas.configure(
                scrollregion=self._replays_canvas.bbox("all")),
        )
        self._replays_canvas.create_window(
            (0, 0), window=self._replays_inner, anchor="nw"
        )
        self._replays_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._replays_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _wheel(e, c=self._replays_canvas):
            c.yview_scroll(self._wheel_units(e), "units")
            return "break"

        for w in (self._replays_canvas, self._replays_inner):
            w.bind("<MouseWheel>", _wheel)
        self._replays_wheel_handler = _wheel

        # Make the inner column stretch the full canvas width so each
        # race row fills the available space instead of bunching up
        # on the left.  Guarded with `find_all()` length check because
        # the canvas may emit Configure events while it's empty (e.g.
        # while it's being torn down or before the inner window is
        # added) which would otherwise raise IndexError into Tk's
        # event handler.
        def _resize_inner(e):
            items = self._replays_canvas.find_all()
            if items:
                self._replays_canvas.itemconfigure(items[0], width=e.width)

        self._replays_canvas.bind("<Configure>", _resize_inner)

        # Show a placeholder while the schedule loads in the background.
        self._replays_loading_lbl = tk.Label(
            self._replays_inner, text="Loading schedule…",
            font=("Helvetica Neue", 11), fg=MUTED, bg=BG,
        )
        self._replays_loading_lbl.pack(anchor="w", pady=20)

        self._fetch_replays_schedule(int(self._replays_year))

    def _fetch_replays_schedule(self, year: int):
        """Load the schedule on a worker thread so the UI never
        freezes on the first Replays tab open.  When done, the
        callback marshals back to the Tk thread via `after(0, ...)`.
        """
        def work():
            try:
                schedule = get_event_schedule_cached(year)
                err = None
            except Exception as exc:
                schedule = None
                err = exc
            self.root.after(
                0, lambda: self._populate_replays(year, schedule, err)
            )

        threading.Thread(target=work, daemon=True).start()

    def _populate_replays(self, year, schedule, err):
        """Tk-thread callback: render the year's races into the
        already-built replays scroll container."""
        # The user may have switched the year selector in the meantime –
        # if so, ignore this stale result.
        try:
            if int(self._replays_year_var.get()) != year:
                return
        except Exception:
            pass

        # Wipe the loading placeholder / any prior render.
        for w in self._replays_inner.winfo_children():
            w.destroy()

        if err is not None:
            tk.Label(self._replays_inner,
                     text=f"Could not load {year} schedule: {err}",
                     font=("Helvetica Neue", 11), fg=MUTED, bg=BG
                     ).pack(anchor="w", pady=20)
            return

        if schedule is None or schedule.empty:
            tk.Label(self._replays_inner,
                     text=f"No schedule available for {year}.",
                     font=("Helvetica Neue", 11), fg=MUTED, bg=BG
                     ).pack(anchor="w", pady=20)
            return

        rows = schedule.sort_values("RoundNumber").to_dict("records")
        today = datetime.now().date()
        for row in rows:
            try:
                rnd = int(row.get("RoundNumber") or 0)
            except (TypeError, ValueError):
                rnd = 0
            name = str(row.get("EventName") or "Unknown Grand Prix")
            ed = row.get("EventDate")
            try:
                if hasattr(ed, "date"):
                    ev_date = ed.date()
                else:
                    ev_date = datetime.fromisoformat(str(ed)[:10]).date()
            except Exception:
                ev_date = None
            self._replays_render_card(year, rnd, name, ev_date, today)

        # Re-bind wheel on every newly-rendered card so scrolling
        # works no matter which row the cursor is hovering.
        if hasattr(self, "_replays_wheel_handler"):
            self._bind_wheel_recursive(
                self._replays_inner, self._replays_wheel_handler
            )

    def _reload_replays(self):
        """Re-render the race list when the user changes the season –
        loads the new year's schedule on a background thread so the
        Combobox doesn't freeze the UI."""
        try:
            self._replays_year = self._replays_year_var.get()
        except Exception:
            return
        for w in self._replays_inner.winfo_children():
            w.destroy()
        self._replays_loading_lbl = tk.Label(
            self._replays_inner, text="Loading schedule…",
            font=("Helvetica Neue", 11), fg=MUTED, bg=BG,
        )
        self._replays_loading_lbl.pack(anchor="w", pady=20)
        self._fetch_replays_schedule(int(self._replays_year))

    def _replays_render_card(self, year, rnd, name, ev_date, today):
        """One card per round: round + race name on the left, session
        buttons on the right.  Future races render greyed-out so it's
        obvious which weekends don't have replays online yet."""
        is_future = ev_date is not None and ev_date > today

        card = tk.Frame(self._replays_inner, bg=BG_CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill=tk.X, pady=4)

        inner = tk.Frame(card, bg=BG_CARD, padx=14, pady=10)
        inner.pack(fill=tk.X)

        # Left column: round badge + race title + date
        left = tk.Frame(inner, bg=BG_CARD)
        left.pack(side=tk.LEFT, fill=tk.Y)

        rnd_text = f"R{rnd:02d}" if rnd else "  "
        tk.Label(left, text=rnd_text,
                 font=("Helvetica Neue", 10, "bold"),
                 fg=F1_RED, bg=BG_CARD, width=4
                 ).pack(side=tk.LEFT, padx=(0, 12))

        text_col = tk.Frame(left, bg=BG_CARD)
        text_col.pack(side=tk.LEFT)
        tk.Label(text_col, text=name,
                 font=("Helvetica Neue", 12, "bold"),
                 fg=WHITE, bg=BG_CARD, anchor="w"
                 ).pack(anchor="w")
        date_text = ev_date.strftime("%a %d %b %Y") if ev_date else "TBD"
        if is_future:
            date_text += "  ·  upcoming"
        tk.Label(text_col, text=date_text,
                 font=("Helvetica Neue", 9),
                 fg=MUTED, bg=BG_CARD, anchor="w"
                 ).pack(anchor="w")

        # Right column: session buttons.  Disabled-look for upcoming
        # rounds (still clickable — search may still find practice or
        # qualifying clips for ongoing weekends).
        right = tk.Frame(inner, bg=BG_CARD)
        right.pack(side=tk.RIGHT)

        for label, keyword in self.REPLAY_SESSIONS:
            bg_c = BG_HOVER if not is_future else BG_SURFACE
            fg_c = WHITE if not is_future else GRAY
            border = BORDER
            btn = self._make_btn(
                right, label, bg_c, fg_c,
                lambda y=year, n=name, k=keyword: self._open_replay(y, n, k),
                border=border,
            )
            btn.pack(side=tk.LEFT, padx=4)

    # ── Track visualization ──

    def _build_viz(self):
        self._anim_running = False
        for w in self.viz_frame.winfo_children():
            w.destroy()
        self._tk_images = []
        # Reset incremental-state caches used by the optimised _anim_tick so
        # they're re-initialised when this run's drivers are created.
        self._trails_shown = None
        self._glow_shown = False

        r = self.result
        nr = r["next_race"]
        preds = nr["predictions"]
        race_name = nr.get("name", "")

        track_pts, circuit_name = get_track(race_name)

        # ── Sleek header: F1 logo · race title · circuit subtitle · back btn
        header = tk.Frame(self.viz_frame, bg=BG)
        header.pack(fill=tk.X, pady=(2, 6))

        # F1 logo as the leftmost element – immediately reads as F1 branding.
        logo_img = _make_f1_logo(36) if HAS_PIL else None
        if logo_img is not None:
            self._viz_logo_tk = ImageTk.PhotoImage(logo_img)
            self._tk_images.append(self._viz_logo_tk)
            tk.Label(header, image=self._viz_logo_tk, bg=BG, bd=0).pack(
                side=tk.LEFT, padx=(0, 14)
            )
        else:
            tk.Label(header, text="F1", font=("Helvetica Neue", 22, "bold"),
                     fg=F1_RED, bg=BG).pack(side=tk.LEFT, padx=(0, 14))

        # Two-line title block: race name on top, circuit on bottom.  Tighter
        # and more elegant than the previous single-line "·"-separated layout.
        title_box = tk.Frame(header, bg=BG)
        title_box.pack(side=tk.LEFT, fill=tk.Y)

        race_title = f"{nr['name']}".upper()
        season_str = f"  ·  {nr['year']} SEASON"
        race_row = tk.Frame(title_box, bg=BG)
        race_row.pack(anchor="w")
        tk.Label(race_row, text=race_title,
                 font=("Helvetica Neue", 15, "bold"), fg=WHITE, bg=BG).pack(side=tk.LEFT)
        tk.Label(race_row, text=season_str,
                 font=("Helvetica Neue", 10), fg=F1_RED, bg=BG).pack(side=tk.LEFT, padx=(0, 0))

        tk.Label(title_box, text=f"{circuit_name}  ·  RACE VISUALISATION",
                 font=("Helvetica Neue", 9), fg=MUTED, bg=BG,
                 anchor="w").pack(anchor="w", pady=(1, 0))

        # Back button stays on the right as a slim pill so it doesn't fight
        # with the title for attention.
        self._make_btn(header, "← Back", BG_CARD, GRAY,
                       lambda: self._switch_to_view("predictions"),
                       border=BORDER).pack(side=tk.RIGHT)

        # Thin accent rule under the header for a magazine-style break.
        accent = tk.Frame(self.viz_frame, bg=F1_RED, height=2)
        accent.pack(fill=tk.X, pady=(0, 6))

        self.track_canvas = tk.Canvas(self.viz_frame, bg=BG, highlightthickness=0)
        self.track_canvas.pack(fill=tk.BOTH, expand=True)

        self._viz_preds = preds
        self._viz_raw_pts = track_pts
        self._viz_circuit = circuit_name
        self._viz_drawn = False
        self._viz_last_size = (0, 0)
        self._viz_resize_job = None
        self.track_canvas.bind("<Configure>", self._on_viz_resize)

    def _on_viz_resize(self, event=None):
        # Debounce: while the user is dragging the window the Configure event fires
        # dozens of times per second.  Defer the (expensive) full redraw until the
        # size has settled.
        cw = self.track_canvas.winfo_width()
        ch = self.track_canvas.winfo_height()
        if cw < 200 or ch < 200:
            return
        if (cw, ch) == self._viz_last_size:
            return
        if self._viz_resize_job is not None:
            try:
                self.root.after_cancel(self._viz_resize_job)
            except Exception:
                pass
        self._viz_resize_job = self.root.after(120, self._do_viz_resize)

    def _do_viz_resize(self):
        self._viz_resize_job = None
        cw = self.track_canvas.winfo_width()
        ch = self.track_canvas.winfo_height()
        if cw < 200 or ch < 200:
            return
        self._viz_last_size = (cw, ch)
        self._anim_running = False
        self.track_canvas.delete("all")
        self._tk_images = []
        self._draw_real_track(cw, ch, self._viz_raw_pts, self._viz_preds)

    # ── Interpolation helpers ──

    @staticmethod
    def _catmull_rom(p0, p1, p2, p3, t):
        t2, t3 = t * t, t * t * t
        x = 0.5 * ((2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                    (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
        y = 0.5 * ((2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                    (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
        return (x, y)

    def _interpolate_track(self, raw_pts, num_out=400):
        n = len(raw_pts)
        pts = []
        seg_steps = max(2, num_out // n)
        for i in range(n):
            p0 = raw_pts[(i - 1) % n]
            p1 = raw_pts[i]
            p2 = raw_pts[(i + 1) % n]
            p3 = raw_pts[(i + 2) % n]
            for s in range(seg_steps):
                t = s / seg_steps
                pts.append(self._catmull_rom(p0, p1, p2, p3, t))
        return pts

    def _track_normal(self, pts, idx):
        # Use the pre-computed normal table when it matches the active track.
        cache = getattr(self, "_track_normals", None)
        if cache is not None and pts is getattr(self, "_track_pts", None):
            return cache[idx % len(cache)]
        p0 = pts[idx]
        p1 = pts[(idx + 1) % len(pts)]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.sqrt(dx * dx + dy * dy) or 1
        return (-dy / length, dx / length)

    # ── Smooth position helpers ──

    def _pos_at(self, t):
        """Get smooth (x, y) at float position t along self._track_pts."""
        track = self._track_pts
        num = len(track)
        t = t % num
        i0 = int(t) % num
        i1 = (i0 + 1) % num
        frac = t - int(t)
        return (track[i0][0] + (track[i1][0] - track[i0][0]) * frac,
                track[i0][1] + (track[i1][1] - track[i0][1]) * frac)

    def _normal_at(self, t):
        """Get smooth normal at float position t."""
        cache = getattr(self, "_track_normals", None)
        if cache is not None:
            num = len(cache)
            t = t % num
            i0 = int(t) % num
            i1 = (i0 + 1) % num
            frac = t - int(t)
            n0 = cache[i0]
            n1 = cache[i1]
            # Linear interpolation of pre-normalized normals – good enough for
            # rendering and dramatically faster than recomputing a sqrt each call.
            nx = n0[0] + (n1[0] - n0[0]) * frac
            ny = n0[1] + (n1[1] - n0[1]) * frac
            length = math.sqrt(nx * nx + ny * ny) or 1.0
            return (nx / length, ny / length)
        track = self._track_pts
        num = len(track)
        t = t % num
        i0 = int(t) % num
        i1 = (i0 + 1) % num
        dx = track[i1][0] - track[i0][0]
        dy = track[i1][1] - track[i0][1]
        length = math.sqrt(dx * dx + dy * dy) or 1
        return (-dy / length, dx / length)

    # ── Scene drawing ──

    def _draw_scene(self, canvas, cw, ch, track, hw, circuit_name):
        scene = SCENES.get(circuit_name, SCENES.get("Circuit"))
        if not scene:
            return
        veg_type, veg_color, veg_spacing, ground_color, features = scene
        s = min(cw, ch) / 800.0
        num = len(track)
        tw = int(hw * 2)

        flat = []
        for p in track:
            flat.extend(p)
        # `smooth=True` on a 500-vertex polygon with this wide an outline
        # is a heavy per-frame cost (Tk re-tessellates whenever a dot moves
        # over it).  The interpolated track is already smooth enough at
        # this density.
        canvas.create_polygon(flat, outline=ground_color, fill="",
                              width=tw * 3)

        for feat in features:
            try:
                kind = feat[0]
                if kind == "grandstand":
                    continue
                elif kind == "water":
                    self._s_water_edge(canvas, cw, ch, feat[1], s)
                elif kind == "mountains":
                    self._s_mountain_range(canvas, cw, ch * 0.06, feat[1], s)
                elif kind == "dunes":
                    self._s_dune_field(canvas, cw, ch, feat[1], s)
                elif kind in ("buildings", "skyline"):
                    self._s_building_row(canvas, cw, ch, feat[1], feat[2], s)
                elif kind == "strip":
                    self._s_vegas_strip(canvas, cw, ch, s)
                elif kind == "sphere":
                    self._s_sphere(canvas, feat[1] * cw, feat[2] * ch, 25 * s)
                elif kind == "ferris":
                    self._s_ferris(canvas, feat[1] * cw, feat[2] * ch, 30 * s)
                elif kind == "lake":
                    xs = [p[0] for p in track]
                    ys = [p[1] for p in track]
                    cx = (min(xs) + max(xs)) / 2
                    cy = (min(ys) + max(ys)) / 2
                    rw = (max(xs) - min(xs)) * 0.2
                    rh = (max(ys) - min(ys)) * 0.15
                    self._s_lake(canvas, cx, cy - 15 * s, rw, rh)
                elif kind == "yachts":
                    self._s_yacht_harbor(canvas, cw, ch, feat[1], s)
                elif kind == "tower":
                    self._s_tower(canvas, feat[1] * cw, feat[2] * ch, s)
                elif kind == "stadium":
                    self._s_stadium(canvas, track, hw, feat[1], feat[2], s)
                elif kind == "cactus_scatter":
                    self._s_cactus_scatter(canvas, track, hw, s)
            except Exception:
                pass

        if veg_type and veg_spacing > 0:
            self._draw_tree_line(canvas, track, hw, veg_type, veg_spacing,
                                 veg_color or "#1a3a1a", s)

        for feat in features:
            if feat[0] == "grandstand":
                try:
                    self._s_grandstand_at(canvas, track, hw, feat[1], feat[2], s)
                except Exception:
                    pass

    def _s_tree(self, c, x, y, s, kind, color):
        tw = max(1, 2 * s)
        if kind == "deciduous":
            c.create_line(x, y, x, y - 14 * s, fill="#2a1a0a", width=tw)
            r = 10 * s
            c.create_oval(x - r, y - 14 * s - r * 1.2, x + r,
                          y - 14 * s + r * 0.4, fill=color, outline="")
        elif kind == "pine":
            c.create_line(x, y, x, y - 8 * s, fill="#2a1a0a", width=tw)
            for hw_v, off in [(4, 0), (6, 6), (8, 12)]:
                by = y - 8 * s - off * s
                c.create_polygon(x - hw_v * s, by, x, by - 10 * s,
                                 x + hw_v * s, by, fill=color, outline="")
        elif kind == "palm":
            c.create_line(x, y, x + 2 * s, y - 35 * s, x + 1 * s,
                          y - 45 * s, fill="#5c3a14",
                          width=max(2, 3 * s), smooth=True)
            tx, ty = x + 1 * s, y - 45 * s
            for dx, dy in [(-22, -8), (-16, -15), (-8, -18), (6, -17),
                           (14, -12), (20, -5), (-18, 2), (16, 4)]:
                c.create_line(tx, ty, tx + dx * s, ty + dy * s,
                              fill="#1a5a1a", width=max(1, 2 * s))
        elif kind == "cherry":
            c.create_line(x, y, x, y - 12 * s, fill="#3a2014", width=tw)
            r = 9 * s
            c.create_oval(x - r, y - 12 * s - r * 1.2, x + r,
                          y - 12 * s + r * 0.4, fill="#3a1a28", outline="")
            for dx, dy in [(-5, -3), (3, -6), (6, 0), (-3, 2), (0, -8)]:
                pr = 2 * s
                c.create_oval(x + dx * s - pr, y - 16 * s + dy * s - pr,
                              x + dx * s + pr, y - 16 * s + dy * s + pr,
                              fill="#ffb7c5", outline="")

    def _draw_tree_line(self, c, track, hw, kind, spacing, color, s):
        num = len(track)
        offset_base = hw * 2.5
        for i in range(0, num, spacing):
            nx, ny = self._track_normal(track, i)
            for side in (1, -1):
                if (i // spacing + (1 if side > 0 else 0)) % 5 == 0:
                    continue
                dist = offset_base + ((i * 7) % 11 - 5) * s
                tree_s = s * (0.6 + ((i * 13) % 7) * 0.08)
                tx = track[i][0] + nx * dist * side
                ty = track[i][1] + ny * dist * side
                self._s_tree(c, tx, ty, tree_s, kind, color)

    def _s_grandstand_at(self, c, track, hw, frac, side, s):
        num = len(track)
        base = int(frac * num) % num
        span = max(8, int(num * 0.06))
        for row in range(4):
            inner_d = hw * 2.4 + row * 5 * s
            outer_d = inner_d + 4 * s
            pts = []
            for j in range(-span // 2, span // 2 + 1, 2):
                idx = (base + j) % num
                nx, ny = self._track_normal(track, idx)
                pts.extend([track[idx][0] + nx * inner_d * side,
                            track[idx][1] + ny * inner_d * side])
            for j in range(span // 2, -span // 2 - 1, -2):
                idx = (base + j) % num
                nx, ny = self._track_normal(track, idx)
                pts.extend([track[idx][0] + nx * outer_d * side,
                            track[idx][1] + ny * outer_d * side])
            if len(pts) >= 6:
                gray = 0x14 + row * 4
                c.create_polygon(pts, fill=f"#{gray:02x}{gray:02x}{gray + 8:02x}",
                                 outline="#222230")

    def _s_water_edge(self, c, cw, ch, side, s):
        # The water edge used to paint a dark-blue rectangle (and ripple
        # lines) along one side of the canvas to suggest a harbour or
        # waterfront.  In practice it read as a thick blue *border*
        # framing the visualisation, so the canvas now keeps a single
        # uniform background colour.  The shimmer / sparkle particles
        # configured for water-themed circuits (Monaco, Miami, Yas
        # Marina, Baku, …) still convey the watery atmosphere via
        # `sparkle_water` ambient items.
        return

    def _s_mountain_range(self, c, cw, base_y, count, s):
        spacing = cw / (count + 1)
        for i in range(count):
            mx = spacing * (i + 1) + ((i * 37) % 20 - 10) * s
            mh = (60 + (i * 23) % 40) * s
            mw = (40 + (i * 17) % 30) * s
            c.create_polygon(mx - mw * 1.3, base_y, mx, base_y - mh,
                             mx + mw * 1.3, base_y, fill="#0a140a", outline="")
            c.create_polygon(mx - mw, base_y, mx + mw * 0.1,
                             base_y - mh * 0.8, mx + mw, base_y,
                             fill="#0e1a0e", outline="")
            c.create_polygon(mx - mw * 0.15, base_y - mh * 0.65,
                             mx + mw * 0.1, base_y - mh * 0.8,
                             mx + mw * 0.25, base_y - mh * 0.65,
                             fill="#c0c0c8", outline="")

    def _s_building_row(self, c, cw, ch, side, count, s):
        if side in ("top", "bottom"):
            is_top = side == "top"
            spacing = cw / (count + 1)
            for i in range(count):
                bw = (14 + (i * 13) % 12) * s
                bh = (25 + (i * 31) % 45) * s
                bx = spacing * (i + 1) - bw / 2
                y1 = 0 if is_top else ch - bh
                y2 = bh if is_top else ch
                c.create_rectangle(bx, y1, bx + bw, y2,
                                   fill="#0e0e18", outline="#1a1a28")
                for wy in range(max(1, int(bh / (7 * s)))):
                    for wx in range(max(1, int(bw / (5 * s)))):
                        if (wx + wy + i) % 3 != 0:
                            c.create_rectangle(
                                bx + wx * 5 * s + 2 * s,
                                y1 + wy * 7 * s + 2 * s,
                                bx + wx * 5 * s + 4 * s,
                                y1 + wy * 7 * s + 5 * s,
                                fill="#252540", outline="")
        elif side in ("left", "right"):
            is_left = side == "left"
            spacing = ch / (count + 1)
            for i in range(count):
                bw = (20 + (i * 31) % 35) * s
                bh = (10 + (i * 13) % 8) * s
                by = spacing * (i + 1) - bh / 2
                x1 = 0 if is_left else cw - bw
                x2 = bw if is_left else cw
                c.create_rectangle(x1, by, x2, by + bh,
                                   fill="#0e0e18", outline="#1a1a28")

    def _s_dune_field(self, c, cw, ch, count, s):
        for i in range(count):
            dx = cw * (0.1 + 0.8 * i / max(count, 1))
            dx += ((i * 47) % 30 - 15) * s
            dy = ch * (0.85 + (i * 13) % 5 * 0.02)
            dw = (60 + (i * 31) % 40) * s
            dh = (15 + (i * 17) % 10) * s
            c.create_arc(dx - dw, dy - dh, dx + dw, dy + dh,
                         start=0, extent=180, fill="#1a1608",
                         outline="#201c0a", style=tk.PIESLICE)

    def _s_vegas_strip(self, c, cw, ch, s):
        neon = ["#ff1060", "#4040ff", "#ff8020",
                "#20ff60", "#ff20ff", "#20ffff"]
        strip_y = ch * 0.04
        for i in range(8):
            bw = (15 + (i * 7) % 12) * s
            bh = (30 + (i * 23) % 50) * s
            bx = cw * 0.08 + i * cw * 0.11
            c.create_rectangle(bx, strip_y, bx + bw, strip_y + bh,
                               fill="#0e0e1a", outline="#1a1a30")
            nc = neon[i % len(neon)]
            c.create_line(bx, strip_y, bx + bw, strip_y,
                          fill=nc, width=max(1, 2 * s))
            for wy in range(max(1, int(bh / (8 * s)))):
                for wx in range(max(1, int(bw / (5 * s)))):
                    if (wx + wy + i) % 2 == 0:
                        wc = neon[(wx + wy + i) % len(neon)]
                        c.create_rectangle(
                            bx + wx * 5 * s + 1 * s,
                            strip_y + wy * 8 * s + 2 * s,
                            bx + wx * 5 * s + 3 * s,
                            strip_y + wy * 8 * s + 5 * s,
                            fill=wc, outline="")

    def _s_sphere(self, c, x, y, r):
        for i in range(4, 0, -1):
            gr = r + i * 5
            c.create_oval(x - gr, y - gr, x + gr, y + gr,
                          fill="", outline="#2020a0", width=1)
        c.create_oval(x - r, y - r, x + r, y + r,
                      fill="#0a0a30", outline="#4040c0", width=2)
        for f in [-0.6, -0.3, 0, 0.3, 0.6]:
            dy = r * f
            hw_v = math.sqrt(max(0, r * r - dy * dy))
            c.create_line(x - hw_v, y + dy, x + hw_v, y + dy,
                          fill="#3030a0", width=1)
        for f in [-0.3, 0, 0.3]:
            dx = r * f
            hh = math.sqrt(max(0, r * r - dx * dx))
            c.create_line(x + dx, y - hh, x + dx, y + hh,
                          fill="#3030a0", width=1)
        c.create_text(x, y + r + 8, text="SPHERE",
                      font=("Helvetica Neue", max(6, int(r * 0.3)), "bold"),
                      fill="#4040c0")

    def _s_ferris(self, c, x, y, r):
        c.create_line(x, y + r + 10, x - r * 0.3, y,
                      fill="#2a2a34", width=2)
        c.create_line(x, y + r + 10, x + r * 0.3, y,
                      fill="#2a2a34", width=2)
        c.create_oval(x - r, y - r, x + r, y + r,
                      fill="", outline="#3a3a4a", width=2)
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            c.create_line(x, y, x + math.cos(rad) * r,
                          y + math.sin(rad) * r,
                          fill="#2a2a3a", width=1)
            gx = x + math.cos(rad) * r
            gy = y + math.sin(rad) * r
            gr = max(2, r * 0.08)
            c.create_oval(gx - gr, gy - gr, gx + gr, gy + gr,
                          fill="#3a3a5a", outline="#4a4a6a")
        hr = max(3, r * 0.1)
        c.create_oval(x - hr, y - hr, x + hr, y + hr,
                      fill="#3a3a4a", outline="")

    def _s_yacht_harbor(self, c, cw, ch, count, s):
        # Yachts only made sense floating on the water-edge rectangles
        # (now disabled to keep the canvas a single colour), so this is
        # a no-op too.
        return

    def _s_lake(self, c, cx, cy, rw, rh):
        # Skipped for the same reason as `_s_water_edge` – the blue
        # Albert Park lake silhouette inside the track conflicted with
        # the goal of a uniform background.
        return

    def _s_tower(self, c, x, y, s):
        c.create_polygon(x - 8 * s, y + 50 * s, x - 3 * s, y,
                         x + 3 * s, y, x + 8 * s, y + 50 * s,
                         fill="#14141e", outline="#1e1e2a")
        c.create_rectangle(x - 12 * s, y - 3 * s, x + 12 * s, y + 3 * s,
                           fill="#1a1a28", outline="#2a2a3a")
        c.create_line(x, y - 3 * s, x, y - 12 * s,
                      fill="#2a2a3a", width=max(1, 2 * s))

    def _s_stadium(self, c, track, hw, frac, side, s):
        num = len(track)
        base = int(frac * num) % num
        span = max(15, int(num * 0.10))
        for row in range(6):
            inner_d = hw * 2.4 + row * 7 * s
            outer_d = inner_d + 6 * s
            pts = []
            for j in range(-span // 2, span // 2 + 1, 2):
                idx = (base + j) % num
                nx, ny = self._track_normal(track, idx)
                pts.extend([track[idx][0] + nx * inner_d * side,
                            track[idx][1] + ny * inner_d * side])
            for j in range(span // 2, -span // 2 - 1, -2):
                idx = (base + j) % num
                nx, ny = self._track_normal(track, idx)
                pts.extend([track[idx][0] + nx * outer_d * side,
                            track[idx][1] + ny * outer_d * side])
            if len(pts) >= 6:
                gray = 0x10 + row * 3
                c.create_polygon(pts,
                                 fill=f"#{gray:02x}{gray:02x}{gray + 6:02x}",
                                 outline="#1e1e28")

    def _s_cactus_scatter(self, c, track, hw, s):
        num = len(track)
        g = "#1a3a1a"
        for i in range(0, num, 35):
            nx, ny = self._track_normal(track, i)
            for side in (1, -1):
                if (i + (1 if side > 0 else 0)) % 3 == 0:
                    continue
                dist = hw * 3.0 + ((i * 13) % 20) * s
                cx = track[i][0] + nx * dist * side
                cy = track[i][1] + ny * dist * side
                cs = s * (0.5 + ((i * 7) % 5) * 0.1)
                c.create_line(cx, cy, cx, cy - 15 * cs,
                              fill=g, width=max(2, 3 * cs))
                c.create_line(cx - 6 * cs, cy - 12 * cs,
                              cx - 6 * cs, cy - 8 * cs,
                              fill=g, width=max(1, 2 * cs))
                c.create_line(cx - 6 * cs, cy - 8 * cs, cx, cy - 8 * cs,
                              fill=g, width=max(1, 2 * cs))
                c.create_line(cx + 5 * cs, cy - 10 * cs,
                              cx + 5 * cs, cy - 6 * cs,
                              fill=g, width=max(1, 2 * cs))
                c.create_line(cx + 5 * cs, cy - 6 * cs, cx, cy - 6 * cs,
                              fill=g, width=max(1, 2 * cs))

    # ── Draw real track ──

    def _draw_real_track(self, cw, ch, raw_pts, preds):
        canvas = self.track_canvas

        # ── Track fitting ──
        # Each circuit's raw points are authored in a roughly normalised
        # space, but the *real* bounding box varies wildly between tracks
        # (Baku and Spa are long and skinny, Red Bull Ring is squat).
        # Stretching every track to fill an arbitrary canvas distorts the
        # actual shape.  Instead, compute the points' real bbox, fit the
        # bbox into the canvas with `min(scale_x, scale_y)` so aspect
        # ratio is preserved, and centre it.
        #
        # We also reserve room for the *track surface itself* (`hw`) plus
        # the kerb/border cushion (`+4`) so a corner that brushes the
        # bbox edge doesn't get clipped by the canvas frame.
        tw = max(28, min(48, int(min(cw, ch) * 0.045)))
        hw = tw / 2

        if not raw_pts:
            return
        xs = [x for (x, _y) in raw_pts]
        ys = [y for (_x, y) in raw_pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bw = max(1e-6, xmax - xmin)
        bh = max(1e-6, ymax - ymin)

        # Reserve space for the track surface and a small breathing
        # margin on all four sides.
        pad = int(hw + 6)
        avail_w = max(80, cw - 2 * pad)
        avail_h = max(80, ch - 2 * pad)
        scale = min(avail_w / bw, avail_h / bh)

        ox = pad + (avail_w - bw * scale) / 2 - xmin * scale
        oy = pad + (avail_h - bh * scale) / 2 - ymin * scale

        scaled = [(x * scale + ox, y * scale + oy) for (x, y) in raw_pts]

        # Drop point density: 500 → 200.  Each redraw pass through Tk
        # rasterises every line segment that overlaps a dirty rect, and the
        # track polygons + border lines + ground outline all run along the
        # full path.  At 200 points the Catmull-Rom curve is still visually
        # smooth (segments are <8 px apart at typical canvas sizes) and the
        # raster cost per frame drops ~2.5x.
        track = self._interpolate_track(scaled, 200)
        self._track_pts = track
        num = len(track)

        # Pre-compute per-point unit normals once.  Both the scene drawing and the
        # animation loop hit `_track_normal` many times per frame – caching avoids
        # an `sqrt` for every driver every tick.
        normals = []
        for i in range(num):
            p0 = track[i]
            p1 = track[(i + 1) % num]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = math.sqrt(dx * dx + dy * dy) or 1.0
            normals.append((-dy / length, dx / length))
        self._track_normals = normals

        # The "nice background" used to be a ~1 megapixel PIL gradient image
        # at z=0.  Tk's canvas is software-rasterised and recomposites every
        # dirty rect through every overlapping item, so that single image
        # added a real cost to *every* moving thing on the canvas.  Drop
        # it: the canvas's solid `bg=BG` is dramatically cheaper, and the
        # typography + podium card + accent rule are doing most of the
        # "sleek" work anyway.
        #
        # Two cheap-but-effective accent items remain (both small, both
        # static, both at the *edges* of the canvas so they almost never
        # overlap any moving driver dot):
        #   1) a thin F1-red corner stripe in the top-left, and
        #   2) the faded F1 watermark in the bottom-right.
        canvas.create_line(8, 8, 60, 8, fill=F1_RED, width=2)
        canvas.create_line(8, 8, 8, 60, fill=F1_RED, width=2)
        canvas.create_line(cw - 60, ch - 8, cw - 8, ch - 8,
                           fill=F1_RED, width=2)
        canvas.create_line(cw - 8, ch - 60, cw - 8, ch - 8,
                           fill=F1_RED, width=2)

        if HAS_PIL:
            wm_size = max(28, int(min(cw, ch) * 0.07))
            wm = _make_f1_logo(wm_size)
            if wm is not None:
                faded = wm.copy()
                alpha = faded.split()[-1].point(lambda v: int(v * 0.18))
                faded.putalpha(alpha)
                self._viz_watermark_tk = ImageTk.PhotoImage(faded)
                self._tk_images.append(self._viz_watermark_tk)
                canvas.create_image(
                    cw - 18, ch - 18,
                    image=self._viz_watermark_tk, anchor="se",
                )

        self._draw_scene(canvas, cw, ch, track, hw, self._viz_circuit)

        # ── Track surface ──
        flat = []
        for p in track:
            flat.extend(p)
        # NOTE: `smooth=True` forces Tk's canvas to *re-tessellate* the Bezier
        # curve on every dirty-rect redraw.  With ~500 vertices that's the
        # single biggest per-frame cost the canvas pays whenever a driver
        # dot moves over the track surface (i.e. every frame).  Our
        # Catmull-Rom interpolation already produced 500 smooth points, so
        # the extra smoothing is purely overhead – draw the polygons as
        # straight-segment polylines and let the density carry the curve.
        canvas.create_polygon(flat, outline="#16161e", fill="", width=tw + 4)
        canvas.create_polygon(flat, outline="#111119", fill="", width=tw)

        # ── Borders ──
        for side_sign in (1, -1):
            border = []
            for i in range(num):
                nx, ny = self._track_normal(track, i)
                border.extend([track[i][0] + nx * hw * side_sign,
                               track[i][1] + ny * hw * side_sign])
            canvas.create_line(border, fill="#2a2a38", width=1.5)

        # ── Curvature analysis ──
        # Cross-product magnitude of consecutive segment vectors gives a
        # cheap proxy for local curvature.  A wider `step` smooths out
        # interpolation noise so we don't see spurious "kinks" between
        # adjacent straight points.
        curvatures = []
        step = max(4, num // 100)
        for i in range(num):
            p_prev = track[(i - step) % num]
            p_cur = track[i]
            p_next = track[(i + step) % num]
            dx1, dy1 = p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]
            dx2, dy2 = p_next[0] - p_cur[0], p_next[1] - p_cur[1]
            curvatures.append(abs(dx1 * dy2 - dy1 * dx2))

        curv_sorted = sorted(curvatures, reverse=True)
        curv_top = curv_sorted[min(30, num - 1)]

        # ── Kerbs at tight corners ──
        for i, c in enumerate(curvatures):
            if c >= curv_top and i % 6 == 0:
                nx, ny = self._track_normal(track, i)
                for ss in (1, -1):
                    kx = track[i][0] + nx * hw * ss
                    ky = track[i][1] + ny * hw * ss
                    canvas.create_oval(kx - 2, ky - 2, kx + 2, ky + 2,
                                       fill=RED, outline="")

        # ── Start / finish ──
        sf = track[0]
        nx, ny = self._track_normal(track, 0)
        canvas.create_line(sf[0] + nx * hw, sf[1] + ny * hw,
                           sf[0] - nx * hw, sf[1] - ny * hw,
                           fill=WHITE, width=3, dash=(4, 4))
        canvas.create_text(sf[0] + nx * (hw + 16), sf[1] + ny * (hw + 16),
                           text="START / FINISH", font=("Helvetica Neue", 7, "bold"),
                           fill=MUTED)

        # ── MOM zones — find the two longest straight stretches ──
        # Threshold: a point is "straight" if its curvature is below
        # the 55th percentile.  The previous median threshold
        # promoted slightly-less-curvy corners to MOM zones; this
        # tightening keeps every track's actual straights eligible
        # without becoming so strict that twisty circuits (Imola,
        # Lusail) end up with no eligible points at all.
        threshold_idx = int(num * 0.55)
        low_curv_threshold = (
            curv_sorted[threshold_idx] if num > 10 else 0
        )
        positions = [c < low_curv_threshold for c in curvatures]

        # Walk the loop starting from the *first non-straight point* so
        # any straight that wraps the seam is captured contiguously.
        # The previous version used `range(num * 2)` and computed a
        # closing length of zero when the entire track passed the
        # threshold — that effectively dropped the longest run.
        runs = []
        if all(positions):
            runs.append((0, num))
        elif any(positions):
            start = next(i for i, p in enumerate(positions) if not p)
            in_run = False
            run_offset = 0
            run_idx = 0
            for offset in range(num + 1):
                idx = (start + offset) % num
                if offset < num and positions[idx]:
                    if not in_run:
                        in_run = True
                        run_offset = offset
                        run_idx = idx
                else:
                    if in_run:
                        run_len = offset - run_offset
                        if run_len > 0:
                            runs.append((run_idx, run_len))
                        in_run = False

        # Only count straights long enough to actually overtake into.
        # Scale the minimum with track length so we don't accept
        # micro-segments that look like straights but only span a
        # couple of interpolation steps.
        min_run_len = max(10, num // 18)
        runs_clean = [(s, l) for s, l in runs if l >= min_run_len]
        runs_clean.sort(key=lambda x: -x[1])

        # Place up to 2 MOM zones.  We try a generous separation first
        # and progressively relax it so a track with two distinct
        # but adjacent straights (e.g. Sakhir's main + back straight)
        # still gets both flagged instead of just one.
        chosen = []
        for sep in (num // 4, num // 6, num // 9):
            chosen = []
            for s, l in runs_clean:
                mid = (s + l // 2) % num
                if all(
                    min(abs(mid - prev), num - abs(mid - prev)) >= sep
                    for prev in chosen
                ):
                    chosen.append(mid)
                    if len(chosen) >= 2:
                        break
            if len(chosen) >= 2:
                break

        # Render whichever ones we got (1 or 2).
        chosen_set = set(chosen)
        for s, l in runs_clean:
            mid = (s + l // 2) % num
            if mid not in chosen_set:
                continue
            chosen_set.discard(mid)
            zone_flat = []
            for j in range(l):
                zone_flat.extend(track[(s + j) % num])
            if len(zone_flat) >= 4:
                canvas.create_line(zone_flat, fill=GREEN, width=3, dash=(8, 4))
            mnx, mny = self._normal_at(float(mid))
            mp = self._pos_at(float(mid))
            canvas.create_text(mp[0] + mnx * (hw + 16),
                               mp[1] + mny * (hw + 16),
                               text="MOM",
                               font=("Helvetica Neue", 8, "bold"),
                               fill=GREEN)

        # ── Center podium with race trophy ──
        xs = [p[0] for p in track]
        ys = [p[1] for p in track]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        # The redesigned podium card is a full broadcast-style podium with
        # three trophies — silver (P2, left), gold (P1, centre, taller and
        # framed by a laurel wreath) and bronze (P3, right) — sitting on
        # top of physical "podium step" pedestals stamped with their
        # finishing position.  Each pedestal carries the driver's
        # abbreviation in their team's livery colour and the predicted
        # win share, so the card alone tells the entire story.
        #
        # All podium items get a "podium" tag so we can `tag_raise` them
        # in one shot after the driver labels are created – the card
        # should always be the topmost layer so transient driver-card
        # labels never partly obscure the winner readout.
        card_w, card_h = 300, 168
        x0, y0 = cx - card_w / 2, cy - card_h / 2
        x1, y1 = cx + card_w / 2, cy + card_h / 2
        PT = "podium"

        # Soft drop-shadow halo so the card lifts off the background.
        for off, alpha_color in ((6, "#0a0203"), (4, "#160505"), (2, "#240608")):
            canvas.create_rectangle(
                x0 - off, y0 - off, x1 + off, y1 + off,
                outline=alpha_color, width=1, tags=(PT,),
            )
        # Card body + thin red accent rule on top.
        canvas.create_rectangle(x0, y0, x1, y1,
                                fill="#0d0d0e", outline=F1_RED, width=1,
                                tags=(PT,))
        canvas.create_rectangle(x0, y0, x1, y0 + 3,
                                fill=F1_RED, outline="", tags=(PT,))

        # Header strip — small red label, then circuit name + subtitle.
        canvas.create_text(
            x0 + 12, y0 + 14,
            text="NEXT RACE",
            font=("Helvetica Neue", 7, "bold"),
            fill=F1_RED, anchor="w", tags=(PT,),
        )
        canvas.create_text(
            cx, y0 + 14,
            text=self._viz_circuit.upper(),
            font=("Helvetica Neue", 9, "bold"),
            fill=WHITE, tags=(PT,),
        )
        canvas.create_text(
            cx, y0 + 28,
            text="PREDICTED PODIUM",
            font=("Helvetica Neue", 7, "bold"),
            fill=MUTED, tags=(PT,),
        )

        # ── Three-trophy podium layout ──
        # P2 left, P1 centre (tallest), P3 right.  Trophy heights mirror
        # the step heights so the silhouettes match a real broadcast
        # podium.
        positions = []
        if len(preds) >= 1:
            positions.append(("gold",   1, preds[0]))
        if len(preds) >= 2:
            positions.insert(0, ("silver", 2, preds[1]))
        if len(preds) >= 3:
            positions.append(("bronze", 3, preds[2]))

        # Three-column layout: silver / gold / bronze
        n_cols = max(1, len(positions))
        col_w = (card_w - 24) / n_cols  # 12px padding either side

        # Step heights per tier — taller centre step is the gold
        # winner's box.  Trophy height matches so it visually scales
        # with the podium step.
        step_for_tier = {"gold": 46, "silver": 34, "bronze": 26}
        trophy_for_tier = {"gold": 70, "silver": 54, "bronze": 44}

        steps_baseline = y1 - 16  # bottom of the steps
        for col_idx, (tier, rank, p) in enumerate(positions):
            col_cx = x0 + 12 + col_w * (col_idx + 0.5)
            step_h = step_for_tier[tier]
            step_top = steps_baseline - step_h
            step_w = col_w * 0.78
            sx0 = col_cx - step_w / 2
            sx1 = col_cx + step_w / 2

            # Pedestal — solid charcoal with a subtle gradient suggested
            # by a brighter top edge.
            canvas.create_rectangle(
                sx0, step_top, sx1, steps_baseline,
                fill=BG_CARD, outline=BORDER, width=1, tags=(PT,),
            )
            canvas.create_rectangle(
                sx0, step_top, sx1, step_top + 3,
                fill=BG_HOVER, outline="", tags=(PT,),
            )
            # Position numeral big on the front of the step.
            tier_color_for_num = {
                "gold":   "#f0c352",
                "silver": "#c8ced8",
                "bronze": "#c67c3a",
            }[tier]
            canvas.create_text(
                col_cx, step_top + step_h / 2 + 3,
                text=str(rank),
                font=("Helvetica Neue", 18, "bold"),
                fill=tier_color_for_num, tags=(PT,),
            )

            # Trophy + (gold only) laurel wreath sitting on the step.
            trophy_h = trophy_for_tier[tier]
            trophy_cx = col_cx
            trophy_cy = step_top - trophy_h / 2 + 4

            if HAS_PIL:
                if tier == "gold":
                    laurel = _make_laurel_image(
                        int(col_w * 0.92), trophy_h + 14
                    )
                    if laurel is not None:
                        tk_laurel = ImageTk.PhotoImage(laurel)
                        self._tk_images.append(tk_laurel)
                        canvas.create_image(
                            trophy_cx, trophy_cy + 4,
                            image=tk_laurel, anchor="center", tags=(PT,),
                        )

                trophy_img = _make_trophy_image(trophy_h, tier=tier)
                if trophy_img is not None:
                    tk_trophy = ImageTk.PhotoImage(trophy_img)
                    self._tk_images.append(tk_trophy)
                    if tier == "gold":
                        self._viz_trophy_tk = tk_trophy
                    canvas.create_image(
                        trophy_cx, trophy_cy,
                        image=tk_trophy, anchor="center", tags=(PT,),
                    )

                # A few sparkle highlights around the gold trophy so the
                # winner's slot reads as a real "celebration" beat.
                if tier == "gold":
                    sparkle = _make_sparkle_image(11)
                    if sparkle is not None:
                        tk_spark = ImageTk.PhotoImage(sparkle)
                        self._tk_images.append(tk_spark)
                        for sx, sy in (
                            (trophy_cx - trophy_h * 0.34, trophy_cy - trophy_h * 0.30),
                            (trophy_cx + trophy_h * 0.36, trophy_cy - trophy_h * 0.10),
                            (trophy_cx + trophy_h * 0.20, trophy_cy - trophy_h * 0.42),
                        ):
                            canvas.create_image(
                                sx, sy, image=tk_spark,
                                anchor="center", tags=(PT,),
                            )

            # Driver readout: abbreviation in team livery + win share.
            team_color = tc(p["team"])
            abbr_y = steps_baseline + 10
            canvas.create_text(
                col_cx, abbr_y,
                text=p["abbreviation"],
                font=("Helvetica Neue", 11 if tier == "gold" else 10, "bold"),
                fill=team_color, tags=(PT,),
            )
            canvas.create_text(
                col_cx, abbr_y + 12,
                text=f"{p['probability']*100:.1f}%",
                font=("Helvetica Neue", 8, "bold"),
                fill=GOLD_GLOW if tier == "gold" else WHITE,
                tags=(PT,),
            )

        # WINNER caption above the centre step so the eye lands on the
        # gold trophy first.
        if positions:
            for tier, _rank, _p in positions:
                if tier == "gold":
                    canvas.create_text(
                        cx, y0 + 44,
                        text="WINNER",
                        font=("Helvetica Neue", 7, "bold"),
                        fill=GOLD_GLOW, tags=(PT,),
                    )
                    break

        # ── Animation — only top 8 drivers for performance ──
        max_anim = min(8, len(preds))
        n = max_anim
        lead_gap = num * 0.65
        spacing = lead_gap / max(n, 1)
        self._anim_targets = [lead_gap - i * spacing for i in range(n)]
        self._anim_pos = [0.0] * n
        self._anim_frame = 0
        self._anim_time = 0.0           # virtual seconds since start
        self._anim_last_wall = time.perf_counter()
        # Absolute target time for the next tick.  We schedule each frame
        # against this clock instead of saying "after 33ms from now" so the
        # animation doesn't drift when Tk is briefly busy (resize, focus,
        # paint).  This is what makes 30fps feel smoother than 60fps with
        # jittery pacing.
        self._anim_next_tick = self._anim_last_wall + self._FRAME_INTERVAL_MS / 1000.0
        self._anim_running = True
        self._anim_hw = hw
        self._anim_num = float(num)
        self._anim_count = n

        self._viz_logos = []
        for i, p in enumerate(preds[:max_anim]):
            if HAS_PIL:
                logo_img = load_logo(p["team"], 18)
                if logo_img:
                    tkimg = ImageTk.PhotoImage(logo_img)
                    self._tk_images.append(tkimg)
                    self._viz_logos.append(tkimg)
                else:
                    self._viz_logos.append(None)
            else:
                self._viz_logos.append(None)

        sf = track[0]
        self._dot_ids = []
        self._dot_txt_ids = []
        self._trail_ids = []
        self._label_ids = []
        self._label_visible = [False] * n

        self._glow_id = canvas.create_oval(0, 0, 0, 0, fill="", outline=GOLD_GLOW,
                                            width=2, state="hidden")

        for i in range(max_anim):
            p = preds[i]
            color = tc(p["team"])
            dot_r = 8 if i == 0 else 6 if i < 3 else 4

            dot = canvas.create_oval(
                sf[0] - dot_r, sf[1] - dot_r, sf[0] + dot_r, sf[1] + dot_r,
                fill=color, outline=WHITE if i == 0 else color,
                width=2 if i == 0 else 1)
            self._dot_ids.append(dot)

            txt = canvas.create_text(sf[0], sf[1], text=str(i + 1),
                                     font=("Helvetica Neue", 7, "bold"),
                                     fill=WHITE if i < 3 else BG)
            self._dot_txt_ids.append(txt)

            # Trails only for top 3
            trails = []
            if i < 3:
                for t_off in range(1, 3):
                    tr_r = dot_r * (1 - t_off * 0.3)
                    tr = canvas.create_oval(0, 0, 0, 0, fill=color, outline="",
                                            stipple="gray25" if t_off == 1 else "gray12",
                                            state="hidden")
                    trails.append(tr_r)
                    trails.append(tr)
            self._trail_ids.append(trails)

            bg_fill = GOLD_DIM if i == 0 else BG_CARD
            border_w = 2 if i < 3 else 1
            name_color = WHITE if i < 3 else GRAY
            prob_color = GOLD_GLOW if i == 0 else MUTED

            connector = canvas.create_line(0, 0, 0, 0, fill=color, width=1,
                                           dash=(2, 2), state="hidden")
            card = canvas.create_rectangle(0, 0, 0, 0, fill=bg_fill, outline=color,
                                           width=border_w, state="hidden")
            logo_item = None
            if i < len(self._viz_logos) and self._viz_logos[i]:
                logo_item = canvas.create_image(0, 0, image=self._viz_logos[i],
                                                anchor="center", state="hidden")
            name_txt = canvas.create_text(0, 0, text=p["abbreviation"],
                                          font=("Helvetica Neue", 10 if i == 0 else 9, "bold"),
                                          fill=name_color, anchor="w", state="hidden")
            prob_txt = canvas.create_text(0, 0,
                                          text=f"P{i+1}  {p['probability']*100:.1f}%",
                                          font=("Helvetica Neue", 8),
                                          fill=prob_color, anchor="w", state="hidden")
            self._label_ids.append({
                "connector": connector, "card": card, "logo": logo_item,
                "name": name_txt, "prob": prob_txt
            })

        self._create_scene_anims(canvas, cw, ch)
        # Keep the podium card on top of dot/label items so transient
        # driver labels never partly obscure the trophy + winner readout.
        try:
            canvas.tag_raise("podium")
        except tk.TclError:
            pass
        self._anim_tick()

    # ── Circuit-specific animated decorations ──

    # Counts kept modest – each item is a Tk canvas item being repositioned
    # every frame.  The eye doesn't notice 15 vs 10 blossoms but Tk's canvas
    # recompositor definitely does.
    # Per-circuit ambient theming.  Each entry is a list of (atom, count)
    # tuples that get instantiated when the visualisation opens.  Atoms are
    # divided into two groups:
    #   • Animated atoms (blossom, lantern, leaf, confetti, …) update every
    #     tick — keep counts modest.
    #   • Decor atoms (bonsai, palm, sun, moon, mountain, dune, …) are
    #     painted once as static PIL silhouettes — they cost nothing per
    #     frame, so we can use them more freely to set the mood.
    SCENE_ANIMS = {
        # Australia · eucalyptus drift + roo
        "Albert Park":      [("kangaroo", 2), ("leaf", 6)],
        # Japan · sakura petals, paper lanterns + bonsai garden corners
        "Suzuka":           [("blossom", 12), ("lantern", 4),
                              ("bonsai_l", 1), ("bonsai_r", 1)],
        # China · spring petals + red lanterns
        "Shanghai":         [("blossom", 8), ("lantern", 4)],
        # Miami · sun-baked sparkle on the bay
        "Miami Autodrome":  [("sparkle_water", 5), ("sun", 1)],
        # Imola · Italian summer drift
        "Imola":            [("leaf", 6), ("sun", 1)],
        # Monaco · harbour shimmer + Mediterranean sun
        "Monaco":           [("sparkle_water", 8), ("sun", 1)],
        # Barcelona · sun-baked Catalan summer
        "Barcelona-Catalunya": [("sun", 1), ("leaf", 5)],
        # Canada · iconic 11-point maple leaves drifting down the
        # tree-lined Île Notre-Dame, with St Lawrence shimmer underneath.
        "Circuit Gilles Villeneuve": [("maple", 9), ("sparkle_water", 3)],
        # Spielberg · Austrian sky (Alps already in SCENES)
        "Red Bull Ring":    [("seagull", 2), ("leaf", 3)],
        # Hungary · sunflower-field summer (golden leaves + warm sun)
        "Hungaroring":      [("leaf", 6), ("sun", 1)],
        # Silverstone · classic British rain
        "Silverstone":      [("rain", 12)],
        # Spa · Ardennes downpour (forest already in SCENES)
        "Spa-Francorchamps": [("rain", 12), ("leaf", 4)],
        # Zandvoort · seagulls over the dunes
        "Zandvoort":        [("seagull", 3)],
        # Monza · royal park tricolour confetti
        "Monza":            [("confetti", 8), ("leaf", 4)],
        # Baku · Caspian breeze, seagulls + shimmer
        "Baku City Circuit": [("seagull", 2), ("sparkle_water", 4)],
        # Singapore · Marina Bay neon fireworks
        "Marina Bay":       [("firework", 2), ("sparkle_water", 5),
                              ("neon_flash", 4)],
        # COTA · Lone Star night
        "COTA":             [("star", 8)],
        # Mexico City · festival confetti
        "Autódromo Hermanos Rodríguez": [("confetti", 7), ("star", 4)],
        # Brazil · Interlagos blossoms + carnival confetti
        "Interlagos":       [("blossom", 5), ("confetti", 6)],
        # Las Vegas · neon strip + fireworks
        "Las Vegas Strip":  [("firework", 3), ("neon_flash", 5),
                              ("star", 6)],
        # Qatar · desert sky (dunes already in SCENES)
        "Lusail":           [("star", 10)],
        # Saudi Arabia · desert stars
        "Jeddah Corniche":  [("star", 8)],
        # Abu Dhabi · Yas marina shimmer + stars
        "Yas Marina":       [("sparkle_water", 5), ("star", 6)],
        # Bahrain · desert sky (dunes already in SCENES)
        "Sakhir":           [("star", 9)],
    }

    def _create_scene_anims(self, canvas, cw, ch):
        import random as _rng
        self._rng = _rng
        self._scene_items = []
        anims = self.SCENE_ANIMS.get(self._viz_circuit, [])

        for atype, count in anims:
            for _ in range(count):
                x = _rng.uniform(40, cw - 40)
                y = _rng.uniform(40, ch - 40)

                if atype == "blossom":
                    size = _rng.uniform(3, 6)
                    petal = canvas.create_oval(x, y, x + size, y + size,
                                               fill="#ffb7c5", outline="#ff8fa3", width=1)
                    self._scene_items.append({
                        "type": "blossom", "id": petal,
                        "x": x, "y": y, "size": size,
                        "vx": _rng.uniform(-0.3, 0.3),
                        "vy": _rng.uniform(0.4, 1.0),
                        "sway": _rng.uniform(0, 6.28),
                        "cw": cw, "ch": ch,
                    })

                elif atype == "kangaroo":
                    x = _rng.uniform(60, cw - 60)
                    y = _rng.uniform(ch * 0.3, ch * 0.7)
                    body = canvas.create_oval(x - 8, y - 5, x + 8, y + 5,
                                               fill="#8B6914", outline="#6B4F12")
                    head = canvas.create_oval(x + 5, y - 10, x + 13, y - 2,
                                               fill="#8B6914", outline="#6B4F12")
                    ear1 = canvas.create_oval(x + 7, y - 14, x + 10, y - 9,
                                               fill="#A07818", outline="#6B4F12")
                    ear2 = canvas.create_oval(x + 10, y - 14, x + 13, y - 9,
                                               fill="#A07818", outline="#6B4F12")
                    tail = canvas.create_line(x - 8, y, x - 18, y - 6,
                                               fill="#6B4F12", width=2)
                    self._scene_items.append({
                        "type": "kangaroo",
                        "ids": [body, head, ear1, ear2, tail],
                        "base_x": x, "base_y": y,
                        "phase": _rng.uniform(0, 6.28),
                        "speed": _rng.uniform(0.03, 0.06),
                        "hop_h": _rng.uniform(8, 16),
                        "direction": _rng.choice([-1, 1]),
                    })

                elif atype == "firework":
                    sparks = []
                    cx = _rng.uniform(cw * 0.2, cw * 0.8)
                    cy = _rng.uniform(ch * 0.1, ch * 0.4)
                    color = _rng.choice(["#ff4444", "#44ff44", "#ffaa00",
                                          "#ff66ff", "#44aaff", GOLD_GLOW])
                    for j in range(8):
                        angle = j * 0.785
                        sid = canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2,
                                                  fill=color, outline="", state="hidden")
                        sparks.append({"id": sid, "angle": angle})
                    self._scene_items.append({
                        "type": "firework", "sparks": sparks,
                        "cx": cx, "cy": cy, "color": color,
                        "phase": _rng.uniform(0, 200),
                        "period": _rng.uniform(120, 250),
                        "cw": cw, "ch": ch,
                    })

                elif atype == "star":
                    x = _rng.uniform(20, cw - 20)
                    y = _rng.uniform(10, ch * 0.35)
                    star = canvas.create_oval(x - 1, y - 1, x + 1, y + 1,
                                               fill="#ffffff", outline="")
                    self._scene_items.append({
                        "type": "star", "id": star,
                        "x": x, "y": y,
                        "phase": _rng.uniform(0, 6.28),
                        "speed": _rng.uniform(0.02, 0.08),
                        "base_size": _rng.uniform(0.5, 2.0),
                    })

                elif atype == "rain":
                    x = _rng.uniform(0, cw)
                    y = _rng.uniform(-20, ch)
                    drop = canvas.create_line(x, y, x - 1, y + 8,
                                               fill="#4488aa", width=1)
                    self._scene_items.append({
                        "type": "rain", "id": drop,
                        "x": x, "y": y,
                        "speed": _rng.uniform(3, 6),
                        "cw": cw, "ch": ch,
                    })

                elif atype == "sparkle_water":
                    x = _rng.uniform(20, cw - 20)
                    y = _rng.uniform(ch * 0.6, ch - 30)
                    sp = canvas.create_oval(x - 1, y - 1, x + 1, y + 1,
                                             fill="#aaddff", outline="")
                    self._scene_items.append({
                        "type": "sparkle_water", "id": sp,
                        "x": x, "y": y,
                        "phase": _rng.uniform(0, 6.28),
                        "speed": _rng.uniform(0.04, 0.10),
                    })

                elif atype == "seagull":
                    x = _rng.uniform(20, cw - 80)
                    y = _rng.uniform(20, ch * 0.3)
                    wing1 = canvas.create_line(x, y, x - 8, y - 4, fill="white", width=1)
                    wing2 = canvas.create_line(x, y, x + 8, y - 4, fill="white", width=1)
                    self._scene_items.append({
                        "type": "seagull",
                        "ids": [wing1, wing2],
                        "x": x, "y": y,
                        "vx": _rng.uniform(0.5, 1.2),
                        "phase": _rng.uniform(0, 6.28),
                        "cw": cw, "ch": ch,
                    })

                elif atype == "neon_flash":
                    x = _rng.uniform(cw * 0.15, cw * 0.85)
                    y = _rng.uniform(ch * 0.15, ch * 0.55)
                    color = _rng.choice(["#ff0066", "#00ffcc", "#ff6600",
                                          "#cc00ff", "#00ff66", "#ffcc00"])
                    neon = canvas.create_rectangle(x, y, x + _rng.uniform(12, 25),
                                                    y + _rng.uniform(3, 6),
                                                    fill=color, outline="", state="hidden")
                    self._scene_items.append({
                        "type": "neon_flash", "id": neon,
                        "phase": _rng.uniform(0, 200),
                        "on_time": _rng.uniform(15, 40),
                        "off_time": _rng.uniform(30, 80),
                        "color": color,
                    })

                # ── Japan / China: paper lanterns sway from the top ──
                elif atype == "lantern":
                    if not HAS_PIL:
                        continue
                    h = _rng.randint(22, 30)
                    img = _make_lantern_image(h)
                    if img is None:
                        continue
                    tkimg = ImageTk.PhotoImage(img)
                    self._tk_images.append(tkimg)
                    # Lanterns hang from the top edge of the canvas
                    lx = _rng.uniform(50, cw - 50)
                    ly = _rng.uniform(20, ch * 0.18)
                    iid = canvas.create_image(lx, ly, image=tkimg,
                                              anchor="n")
                    canvas.lower(iid)  # behind the track
                    self._scene_items.append({
                        "type": "lantern", "id": iid,
                        "base_x": lx, "base_y": ly,
                        "phase": _rng.uniform(0, 6.28),
                        "speed": _rng.uniform(0.03, 0.06),
                        "amp": _rng.uniform(4, 9),
                    })

                # ── Canadian maple leaves (Circuit Gilles Villeneuve) ──
                # A pre-rendered set of 6 rotation frames lets every leaf
                # tumble realistically as it falls without paying the cost
                # of a per-frame PIL.Image.rotate.
                elif atype == "maple":
                    if not HAS_PIL:
                        continue
                    base_size = int(_rng.uniform(14, 22))
                    frames = _maple_leaf_rotation_frames(base_size, 6)
                    if not frames:
                        continue
                    tk_frames = [ImageTk.PhotoImage(f) for f in frames]
                    self._tk_images.extend(tk_frames)

                    mx = _rng.uniform(20, cw - 20)
                    my = _rng.uniform(-30, ch * 0.4)
                    iid = canvas.create_image(
                        mx, my, image=tk_frames[0], anchor="center",
                    )
                    self._scene_items.append({
                        "type": "maple", "id": iid,
                        "x": mx, "y": my,
                        "vx": _rng.uniform(-0.4, 0.4),
                        "vy": _rng.uniform(0.5, 1.0),
                        "sway": _rng.uniform(0, 6.28),
                        "rot_speed": _rng.uniform(0.05, 0.13),
                        "rot_phase": _rng.uniform(0, 6.28),
                        "frames": tk_frames,
                        "frame_count": len(tk_frames),
                        "cur_frame": 0,
                        "cw": cw, "ch": ch,
                    })

                # ── Falling leaves (autumn red/gold/green) ──
                elif atype == "leaf":
                    lx = _rng.uniform(20, cw - 20)
                    ly = _rng.uniform(-30, ch * 0.4)
                    color = _rng.choice([
                        "#c94a2a", "#d97a1f", "#e8b923",
                        "#7a9a3a", "#a04a18",
                    ])
                    size = _rng.uniform(3, 5)
                    leaf = canvas.create_oval(
                        lx - size, ly - size * 0.55,
                        lx + size, ly + size * 0.55,
                        fill=color, outline="",
                    )
                    self._scene_items.append({
                        "type": "leaf", "id": leaf,
                        "x": lx, "y": ly, "size": size,
                        "vx": _rng.uniform(-0.5, 0.5),
                        "vy": _rng.uniform(0.5, 1.1),
                        "sway": _rng.uniform(0, 6.28),
                        "cw": cw, "ch": ch,
                    })

                # ── Carnival / Monza tricolour confetti ──
                elif atype == "confetti":
                    palette = ["#E10600", "#FFFFFF", "#2ED87A",
                               "#FFD400", "#3FA8FF", "#FF6FA1"]
                    color = _rng.choice(palette)
                    cx_ = _rng.uniform(0, cw)
                    cy_ = _rng.uniform(-40, ch * 0.4)
                    w_ = _rng.uniform(3, 5)
                    h_ = _rng.uniform(2, 4)
                    conf = canvas.create_rectangle(
                        cx_, cy_, cx_ + w_, cy_ + h_,
                        fill=color, outline="",
                    )
                    self._scene_items.append({
                        "type": "confetti", "id": conf,
                        "x": cx_, "y": cy_, "w": w_, "h": h_,
                        "vy": _rng.uniform(0.7, 1.4),
                        "sway": _rng.uniform(0, 6.28),
                        "swirl": _rng.uniform(0.04, 0.10),
                        "cw": cw, "ch": ch,
                    })

                # ── Static PIL silhouettes pushed behind the track ──
                elif atype in ("bonsai_l", "bonsai_r"):
                    if not HAS_PIL:
                        continue
                    h = max(50, int(min(cw, ch) * 0.13))
                    img = _make_bonsai_image(h, mirror=(atype == "bonsai_r"))
                    if img is None:
                        continue
                    tkimg = ImageTk.PhotoImage(img)
                    self._tk_images.append(tkimg)
                    bx = 14 if atype == "bonsai_l" else cw - 14
                    by = ch - 14
                    anchor = "sw" if atype == "bonsai_l" else "se"
                    iid = canvas.create_image(bx, by, image=tkimg,
                                              anchor=anchor)
                    canvas.lower(iid)

                elif atype in ("palm_l", "palm_r"):
                    if not HAS_PIL:
                        continue
                    h = max(80, int(min(cw, ch) * 0.22))
                    img = _make_palm_image(h, mirror=(atype == "palm_r"))
                    if img is None:
                        continue
                    tkimg = ImageTk.PhotoImage(img)
                    self._tk_images.append(tkimg)
                    px = 6 if atype == "palm_l" else cw - 6
                    py = ch - 4
                    anchor = "sw" if atype == "palm_l" else "se"
                    iid = canvas.create_image(px, py, image=tkimg,
                                              anchor=anchor)
                    canvas.lower(iid)

                elif atype == "sun":
                    if not HAS_PIL:
                        continue
                    d = max(48, int(min(cw, ch) * 0.10))
                    img = _make_sun_image(d)
                    if img is None:
                        continue
                    tkimg = ImageTk.PhotoImage(img)
                    self._tk_images.append(tkimg)
                    # Top-right corner so it doesn't fight the F1 logo
                    iid = canvas.create_image(cw - 20, 20, image=tkimg,
                                              anchor="ne")
                    canvas.lower(iid)

                elif atype == "mountain":
                    if not HAS_PIL:
                        continue
                    w_ = int(cw * 0.85)
                    h_ = max(50, int(ch * 0.12))
                    img = _make_mountain_image(w_, h_)
                    if img is None:
                        continue
                    tkimg = ImageTk.PhotoImage(img)
                    self._tk_images.append(tkimg)
                    iid = canvas.create_image(cw / 2, 4, image=tkimg,
                                              anchor="n")
                    canvas.lower(iid)

                elif atype == "dune":
                    if not HAS_PIL:
                        continue
                    w_ = int(cw)
                    h_ = max(40, int(ch * 0.10))
                    img = _make_dune_image(w_, h_)
                    if img is None:
                        continue
                    tkimg = ImageTk.PhotoImage(img)
                    self._tk_images.append(tkimg)
                    iid = canvas.create_image(0, ch, image=tkimg,
                                              anchor="sw")
                    canvas.lower(iid)

    # ── Pre-computed colour palettes for star / sparkle twinkle ──
    #
    # Both star and sparkle_water animations used to format a hex string from
    # a `brightness` float every single frame for every item.  At 60 fps with
    # ~15 items that's ~900 f-string allocations per second just for these
    # two effects.  We bake the 32-step palettes here once and look them up.
    _STAR_PALETTE = tuple(
        f"#{(150 + int(b * 105)):02x}{(150 + int(b * 105)):02x}{min(255, 150 + int(b * 105) + 20):02x}"
        for b in (i / 31 for i in range(32))
    )
    _SPARKLE_PALETTE = tuple(
        f"#{max(80, (100 + int(b * 155)) - 30):02x}{(100 + int(b * 155)):02x}{min(255, (100 + int(b * 155)) + 30):02x}"
        for b in (i / 31 for i in range(32))
    )

    def _tick_scene_anims(self, frame, dt_scale=1.0):
        """Update all ambient scene items.

        `frame` is a *virtual* frame counter that advances at the legacy ~12.5
        ticks/sec rate (so cycle/period math is unchanged).  `dt_scale` is the
        fraction of one legacy tick that has actually elapsed this wall frame
        and is multiplied into per-tick deltas so the visual speed stays the
        same at any framerate.
        """
        canvas = self.track_canvas
        coords = canvas.coords
        itemcfg = canvas.itemconfigure
        sin = math.sin
        cos = math.cos
        star_palette = self._STAR_PALETTE
        sparkle_palette = self._SPARKLE_PALETTE

        # Twinkle/flicker doesn't need to update its colour every frame – the
        # eye barely sees the difference at >20Hz.  We refresh colour every
        # third tick which cuts itemconfigure traffic to a third.
        update_twinkle = (int(frame * 3) % 3) == 0

        for item in self._scene_items:
            t = item["type"]

            if t == "blossom":
                item["x"] += (item["vx"] + sin(item["sway"] + frame * 0.03) * 0.4) * dt_scale
                item["y"] += item["vy"] * dt_scale
                if item["y"] > item["ch"] + 10:
                    item["y"] = -10
                    item["x"] = self._rng.uniform(20, item["cw"] - 20)
                s = item["size"]
                wobble = sin(item["sway"] + frame * 0.05) * 2
                coords(item["id"], item["x"] + wobble, item["y"],
                       item["x"] + s + wobble, item["y"] + s)

            elif t == "kangaroo":
                item["phase"] += item["speed"] * dt_scale
                hop = abs(sin(item["phase"])) * item["hop_h"]
                drift = sin(item["phase"] * 0.3) * 25
                dx = drift * item["direction"]
                bx, by = item["base_x"] + dx, item["base_y"] - hop
                ids = item["ids"]
                coords(ids[0], bx - 8, by - 5, bx + 8, by + 5)
                coords(ids[1], bx + 5, by - 10, bx + 13, by - 2)
                coords(ids[2], bx + 7, by - 14, bx + 10, by - 9)
                coords(ids[3], bx + 10, by - 14, bx + 13, by - 9)
                coords(ids[4], bx - 8, by, bx - 18, by - 6 + hop * 0.3)

            elif t == "firework":
                cycle = (frame + item["phase"]) % item["period"]
                burst_dur = 40
                if cycle < burst_dur:
                    t_frac = cycle / burst_dur
                    radius = t_frac * 30
                    alpha_state = "normal" if t_frac < 0.8 else "hidden"
                    prev_state = item.get("_state")
                    for spark in item["sparks"]:
                        ang = spark["angle"]
                        sx = item["cx"] + cos(ang) * radius
                        sy = item["cy"] + sin(ang) * radius
                        sr = 3 * (1 - t_frac * 0.5)
                        coords(spark["id"], sx - sr, sy - sr, sx + sr, sy + sr)
                        if prev_state != alpha_state:
                            itemcfg(spark["id"], state=alpha_state)
                    item["_state"] = alpha_state
                else:
                    if item.get("_state") != "hidden":
                        for spark in item["sparks"]:
                            itemcfg(spark["id"], state="hidden")
                        item["_state"] = "hidden"

            elif t == "star":
                # Twinkle: the position barely changes (just radius) so most of
                # the cost is `itemconfigure(fill=...)`.  Update colour at ~20Hz
                # instead of 60Hz.
                brightness = 0.3 + 0.7 * abs(sin(item["phase"] + frame * item["speed"]))
                r = item["base_size"] * (0.5 + brightness * 0.5)
                coords(item["id"], item["x"] - r, item["y"] - r,
                       item["x"] + r, item["y"] + r)
                if update_twinkle:
                    pal_idx = int(brightness * 31)
                    if pal_idx > 31: pal_idx = 31
                    if pal_idx < 0:  pal_idx = 0
                    new_color = star_palette[pal_idx]
                    if item.get("_color") != new_color:
                        itemcfg(item["id"], fill=new_color)
                        item["_color"] = new_color

            elif t == "rain":
                item["y"] += item["speed"] * dt_scale
                item["x"] -= item["speed"] * 0.3 * dt_scale
                if item["y"] > item["ch"]:
                    item["y"] = -10
                    item["x"] = self._rng.uniform(0, item["cw"])
                coords(item["id"], item["x"], item["y"],
                       item["x"] - 1, item["y"] + 8)

            elif t == "sparkle_water":
                brightness = abs(sin(item["phase"] + frame * item["speed"]))
                r = 1 + brightness * 2
                coords(item["id"], item["x"] - r, item["y"] - r,
                       item["x"] + r, item["y"] + r)
                if update_twinkle:
                    pal_idx = int(brightness * 31)
                    if pal_idx > 31: pal_idx = 31
                    if pal_idx < 0:  pal_idx = 0
                    new_color = sparkle_palette[pal_idx]
                    if item.get("_color") != new_color:
                        itemcfg(item["id"], fill=new_color)
                        item["_color"] = new_color

            elif t == "seagull":
                item["x"] += item["vx"] * dt_scale
                item["phase"] += 0.08 * dt_scale
                wing_y = sin(item["phase"]) * 5
                if item["x"] > item["cw"] + 20:
                    item["x"] = -20
                x, y = item["x"], item["y"]
                ids = item["ids"]
                coords(ids[0], x, y, x - 8, y - 4 + wing_y)
                coords(ids[1], x, y, x + 8, y - 4 + wing_y)

            elif t == "neon_flash":
                cycle = (frame + item["phase"]) % (item["on_time"] + item["off_time"])
                desired = "normal" if cycle < item["on_time"] else "hidden"
                if item.get("_state") != desired:
                    itemcfg(item["id"], state=desired)
                    item["_state"] = desired

            elif t == "lantern":
                # Paper lantern pendulum sway – the canvas image item only
                # needs its (x, y) anchor updated, no per-frame redraw of
                # the bitmap itself.
                item["phase"] += item["speed"] * dt_scale
                dx = sin(item["phase"]) * item["amp"]
                # Tiny vertical lift to mimic a rope length change
                dy = cos(item["phase"] * 0.5) * 1.5
                coords(item["id"], item["base_x"] + dx,
                       item["base_y"] + dy)

            elif t == "leaf":
                item["x"] += (item["vx"] + sin(item["sway"] + frame * 0.04) * 0.55) * dt_scale
                item["y"] += item["vy"] * dt_scale
                if item["y"] > item["ch"] + 12:
                    item["y"] = -12
                    item["x"] = self._rng.uniform(20, item["cw"] - 20)
                s = item["size"]
                # Subtle horizontal stretch so it looks like a leaf
                # flipping as it falls (cheap pseudo-rotation).
                stretch = 0.45 + 0.55 * abs(sin(item["sway"] + frame * 0.08))
                coords(item["id"],
                       item["x"] - s * stretch, item["y"] - s * 0.55,
                       item["x"] + s * stretch, item["y"] + s * 0.55)

            elif t == "maple":
                # Drift + sway like a leaf, but tumble by cycling through
                # pre-rendered rotation frames instead of stretching.
                item["x"] += (item["vx"] + sin(item["sway"] + frame * 0.04) * 0.55) * dt_scale
                item["y"] += item["vy"] * dt_scale
                if item["y"] > item["ch"] + 14:
                    item["y"] = -14
                    item["x"] = self._rng.uniform(20, item["cw"] - 20)
                coords(item["id"], item["x"], item["y"])
                # Index into the rotation frames; only itemconfig when
                # the frame index actually changes so we don't generate
                # unnecessary canvas dirty-rects every tick.
                fi = int(
                    (item["rot_phase"] + frame * item["rot_speed"])
                ) % item["frame_count"]
                if fi != item["cur_frame"]:
                    item["cur_frame"] = fi
                    itemcfg(item["id"], image=item["frames"][fi])

            elif t == "confetti":
                # Falling confetti with a swirling horizontal drift and a
                # pseudo-rotation that flattens / re-widens the rectangle.
                item["y"] += item["vy"] * dt_scale
                item["x"] += sin(item["sway"] + frame * item["swirl"]) * 0.9 * dt_scale
                if item["y"] > item["ch"] + 10:
                    item["y"] = -10
                    item["x"] = self._rng.uniform(0, item["cw"])
                # Pseudo-rotation by oscillating the width
                w_eff = item["w"] * (0.35 + 0.65 * abs(cos(item["sway"] + frame * 0.18)))
                coords(item["id"],
                       item["x"], item["y"],
                       item["x"] + w_eff, item["y"] + item["h"])

    # ── Smooth animation loop ──

    # Logical timeline is expressed in *seconds*, not frames, so the animation
    # looks identical regardless of how fast Tk actually schedules us.  The
    # constants below replace the old per-frame values:
    #   spread_duration  90 frames @ 25fps  -> 3.6 s
    #   orbit_speed     0.35 idx / frame    -> ~8.75 idx / s
    _SPREAD_SECONDS = 3.6
    _ORBIT_SPEED = 8.75
    # 30 fps target. Tk's software canvas is single-threaded and recomposites
    # dirty rects through every overlapping item; pushing 60 fps with ~150
    # canvas items per frame ends up feeling more labored than 30 fps with
    # consistent pacing.  Combined with absolute-time scheduling (see below)
    # this gives a much smoother perceived motion.
    _FRAME_INTERVAL_MS = 33
    # Driver labels follow their cars at ~10 fps (every 3rd frame at the
    # 30 fps base).  We *stagger* across drivers (`(frame + i) % N`) so
    # only ~2-3 of the 8 cards refresh in any single tick — this keeps the
    # dirty-rect cost almost flat across frames instead of producing one
    # heavy "all-cards" frame every Nth tick.
    _LABEL_FOLLOW_EVERY_N = 3

    def _anim_tick(self):
        if not self._anim_running or self._current_view != "viz":
            return

        canvas = self.track_canvas
        # Hoist hot method lookups to locals – inside a per-frame loop the
        # attribute resolution for `canvas.coords` etc. becomes a measurable
        # share of the work.
        coords = canvas.coords
        itemcfg = canvas.itemconfigure

        now = time.perf_counter()
        dt = now - self._anim_last_wall
        self._anim_last_wall = now
        # Clamp dt: if the window is occluded/dragged, Tk may pause us for a
        # long time; we don't want the cars to teleport when we resume.
        if dt > 0.1:
            dt = 0.1
        elif dt < 0:
            dt = 0.0
        self._anim_time += dt
        t_total = self._anim_time
        self._anim_frame += 1
        frame = self._anim_frame

        hw = self._anim_hw
        num = self._anim_num
        n = self._anim_count

        spread = self._SPREAD_SECONDS
        orbit_speed = self._ORBIT_SPEED

        in_orbit = t_total >= spread

        # (Labels are now fully static after their reveal frame; nothing
        # to throttle on a per-frame basis.)

        # Pre-cache pos lookups: each driver call uses _pos_at on its own pos
        # plus 1–2 trail positions; we resolve _pos_at to a local once.
        pos_at = self._pos_at
        normal_at = self._normal_at
        dot_ids = self._dot_ids
        txt_ids = self._dot_txt_ids
        trail_ids = self._trail_ids
        label_ids = self._label_ids
        label_visible = self._label_visible
        trails_shown = getattr(self, "_trails_shown", None)
        if trails_shown is None:
            trails_shown = [False] * n
            self._trails_shown = trails_shown

        for i in range(n):
            target = self._anim_targets[i]

            if not in_orbit:
                p = t_total / spread
                ease = 1.0 - (1.0 - p) ** 3
                pos = target * ease
            else:
                pos = target + (t_total - spread) * orbit_speed

            pos = pos % num
            self._anim_pos[i] = pos
            px, py = pos_at(pos)
            dot_r = 8 if i == 0 else 6 if i < 3 else 4

            coords(dot_ids[i],
                   px - dot_r, py - dot_r, px + dot_r, py + dot_r)
            coords(txt_ids[i], px, py)

            trails = trail_ids[i]
            if trails and t_total > 0.32:
                # State only needs to flip to "normal" once.  Skipping the
                # itemconfigure on subsequent frames is the single biggest
                # per-frame saving for top-3 cars.
                if not trails_shown[i]:
                    trails_shown[i] = True
                    for t_idx in range(1, len(trails), 2):
                        itemcfg(trails[t_idx], state="normal")
                for t_idx in range(0, len(trails), 2):
                    tr_r = trails[t_idx]
                    tr_id = trails[t_idx + 1]
                    trail_pos = (pos - (t_idx // 2 + 1) * 2.5) % num
                    tx, ty = pos_at(trail_pos)
                    coords(tr_id, tx - tr_r, ty - tr_r, tx + tr_r, ty + tr_r)

            if i == 0 and in_orbit:
                pulse = 0.5 + 0.5 * math.sin(t_total * 3.75)
                glow_r = 14 + pulse * 6
                coords(self._glow_id,
                       px - glow_r, py - glow_r, px + glow_r, py + glow_r)
                if not getattr(self, "_glow_shown", False):
                    itemcfg(self._glow_id, state="normal")
                    self._glow_shown = True

            reveal_at = spread + i * 0.08
            if t_total >= reveal_at:
                cw2 = 44.0
                ch2 = 19.0
                ids = label_ids[i]

                if not label_visible[i]:
                    # First reveal – place everything against the track
                    # normal so the card sits cleanly *outside* the loop.
                    # Cache the resulting (dx, dy) offset between dot and
                    # card so subsequent follow ticks can skip the
                    # normal-vector math.
                    label_visible[i] = True
                    nx, ny = normal_at(pos)
                    side = 1.0 if i % 2 == 0 else -1.0
                    dist = hw + 30 + (10 if i < 3 else 0)
                    lx = px + nx * dist * side
                    ly = py + ny * dist * side
                    ids["_offset"] = (lx - px, ly - py)

                    coords(ids["card"], lx - cw2, ly - ch2, lx + cw2, ly + ch2)
                    if ids["logo"] is not None:
                        coords(ids["logo"], lx - cw2 + 14, ly)
                    coords(ids["name"], lx + 2, ly - 7)
                    coords(ids["prob"], lx + 2, ly + 8)
                    coords(ids["connector"], px, py, lx, ly)
                    for item in ids.values():
                        if isinstance(item, int):
                            itemcfg(item, state="normal")
                elif (frame + i) % self._LABEL_FOLLOW_EVERY_N == 0:
                    # Follow the car at a throttled, *staggered* rate.
                    # The offset is the fixed (dx, dy) computed at reveal,
                    # so this is just two adds + five coord calls.
                    ox, oy = ids["_offset"]
                    lx = px + ox
                    ly = py + oy
                    coords(ids["card"], lx - cw2, ly - ch2, lx + cw2, ly + ch2)
                    if ids["logo"] is not None:
                        coords(ids["logo"], lx - cw2 + 14, ly)
                    coords(ids["name"], lx + 2, ly - 7)
                    coords(ids["prob"], lx + 2, ly + 8)
                    coords(ids["connector"], px, py, lx, ly)

        # Scene animations now run every frame – the higher framerate makes the
        # extra updates almost free and the motion noticeably smoother.  The
        # original deltas were calibrated for an 80ms cadence; pass dt_scale so
        # the on-screen speed matches the legacy version regardless of fps.
        if hasattr(self, "_scene_items") and self._scene_items:
            virtual_frame = t_total * 12.5  # legacy "frames" (1 per 80ms)
            self._tick_scene_anims(virtual_frame, dt / 0.08)

        # Absolute-time scheduling: target the next tick relative to a fixed
        # clock rather than "X ms from when this tick finishes".  If a tick
        # runs long, the next one fires sooner to catch up.  If we're ahead,
        # we wait the proper amount.  Either way, the average framerate stays
        # locked to _FRAME_INTERVAL_MS instead of drifting under load.
        interval = self._FRAME_INTERVAL_MS / 1000.0
        target = self._anim_next_tick + interval
        # If we fell more than 3 frames behind (background tab, drag, etc.)
        # snap the target forward so we don't try to "catch up" forever.
        wall_now = time.perf_counter()
        if target < wall_now - 2 * interval:
            target = wall_now + interval
        self._anim_next_tick = target
        delay_ms = max(1, int((target - wall_now) * 1000))
        self.root.after(delay_ms, self._anim_tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Single-instance enforcement: if another ApexAI window is already open,
    # kill it before we claim the window.  This keeps the user from ending up
    # with duplicate Tk windows fighting over the foreground.
    acquire_singleton()
    ApexAI().run()
