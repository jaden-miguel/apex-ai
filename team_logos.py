"""
Team logo loading – uses local assets or generates styled badges.
"""
from pathlib import Path

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from team_colors import TEAM_COLORS

# Map display names to logo filenames
TEAM_LOGO_KEYS = {
    "Red Bull Racing": "redbull",
    "Racing Bulls": "racingbulls",
    "RB": "racingbulls",
    "AlphaTauri": "racingbulls",
    "Ferrari": "ferrari",
    "Mercedes": "mercedes",
    "McLaren": "mclaren",
    "Aston Martin": "astonmartin",
    "Alpine": "alpine",
    "Haas F1 Team": "haas",
    "Williams": "williams",
    "Audi": "audi",
    "Kick Sauber": "audi",
    "Alfa Romeo": "audi",
    "Cadillac": "cadillac",
}

# Short names for badge fallback
TEAM_INITIALS = {
    "Red Bull Racing": "RB",
    "Racing Bulls": "RB",
    "RB": "RB",
    "Ferrari": "SF",
    "Mercedes": "ME",
    "McLaren": "MC",
    "Aston Martin": "AM",
    "Alpine": "AL",
    "Haas F1 Team": "HA",
    "Williams": "WI",
    "Audi": "AU",
    "Cadillac": "CA",
}

# Per-team rectangular crop, in normalised coordinates (left, top, right,
# bottom) of the source PNG, used at SMALL display sizes (<= 32px) to
# pick out the iconic symbol portion of an otherwise wide wordmark.  The
# 2024+ team logos all bundle big title-sponsor wordmarks (MoneyGram,
# Atlassian, Oracle, etc.) which become illegible smudges at grid-row
# scale – cropping to just the brand crest restores readability while
# keeping each team instantly identifiable.
#
# Hand-tuned by visually inspecting each ``logos/<team>.png``.  Coords
# are normalised so the crops survive any later re-download at a
# different resolution.
LOGO_ICON_CROP = {
    # Wings emblem with "ASTON MARTIN" text – leftmost half
    "Aston Martin":    (0.02, 0.00, 0.48, 0.55),
    # Yellow shield + prancing horse, ignore the "/HP" partnership block
    "Ferrari":         (0.00, 0.00, 0.42, 1.00),
    # Red "H" crest on the left of the MoneyGram wordmark
    "Haas F1 Team":    (0.00, 0.00, 0.24, 1.00),
    # Orange speedmark sits in the top-right of the McLaren wordmark
    "McLaren":         (0.78, 0.00, 1.00, 0.45),
    # BWT/Alpine top half – pink shield + ALPINE wordmark
    "Alpine":          (0.00, 0.00, 1.00, 0.55),
    # Cadillac heritage crest sits stacked above the wordmark
    "Cadillac":        (0.18, 0.00, 0.82, 0.48),
    # Williams' stylised W mark – central band, skipping the Atlassian
    # banner up top and the "F1 TEAM" caption beneath.
    "Williams":        (0.00, 0.30, 1.00, 0.85),
    # Charging bulls + yellow sun – right half of the Red Bull lockup.
    "Red Bull Racing": (0.50, 0.25, 1.00, 0.75),
    # Stylised RB silhouette sits dead-centre in the VCARB lockup.
    "Racing Bulls":    (0.27, 0.00, 0.69, 0.80),
    # Audi's four rings are the entire mark – keep full width.  Mercedes
    # is already square.  F1 stays as a wordmark at all sizes.
}

# Above this display size we render the full logo (sponsor wordmarks
# included) so the hero-card and empty-state look "official"; below it
# we crop to the icon so the grid-row and podium stay readable.
LOGO_ICON_CROP_MAX_SIZE = 32


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def create_badge(team_name: str, size: int = 28) -> "Image.Image":
    """Create a shield-shaped badge with team color and initials."""
    if not HAS_PIL:
        return None
    color = TEAM_COLORS.get(team_name, "#555566")
    rgb = _hex_to_rgb(color)
    initial = TEAM_INITIALS.get(team_name, team_name[:2].upper() if team_name else "?")
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    m = 1
    cx, cy = size // 2, size // 2
    pts = [
        (m, m + size // 6),
        (cx, m),
        (size - m - 1, m + size // 6),
        (size - m - 1, cy + size // 6),
        (cx, size - m - 1),
        (m, cy + size // 6),
    ]
    draw.polygon(pts, fill=rgb, outline=None)

    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(8, size // 3))
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), initial, font=font)
    except (AttributeError, TypeError):
        try:
            w, h = draw.textsize(initial, font=font)
            bbox = (0, 0, w, h)
        except Exception:
            bbox = (0, 0, size // 2, size // 2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), initial, fill="#ffffff", font=font)
    return img


def _trim_transparent(img: "Image.Image") -> "Image.Image":
    """Strip fully-transparent borders so the actual artwork fills the
    target box.  Returns the original image if there's nothing to trim
    (or no alpha channel)."""
    if img.mode != "RGBA":
        return img
    bbox = img.split()[3].getbbox()
    if bbox is None or bbox == (0, 0, *img.size):
        return img
    return img.crop(bbox)


def load_logo(team_name: str, size: int = 28) -> "Image.Image":
    """Load a team logo from ``logos/`` and fit it into ``size``×``size``.

    Pipeline (in order):
      1. Open the file and normalise to ``RGBA`` (files come in P, LA
         and RGBA modes – we want a uniform format).
      2. At small display sizes (``size <= LOGO_ICON_CROP_MAX_SIZE``),
         crop to the team's ``LOGO_ICON_CROP`` rectangle so the brand
         crest – not the sponsor wordmark – fills the available pixels.
         This is what's visible in the grid rows and podium.
      3. Trim any fully-transparent border that remains.
      4. Scale to *fit inside* a ``size``×``size`` box preserving the
         aspect ratio, using Lanczos resampling so everything stays
         sharp.
      5. Centre the result on a transparent square so callers can drop
         the result into any cell without further alignment.
    """
    if not HAS_PIL:
        return None
    base = Path(__file__).parent
    key = TEAM_LOGO_KEYS.get(team_name, team_name.lower().replace(" ", "").replace("f1team", ""))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for ext in ("png", "jpg", "webp"):
        for folder in ("logos", "assets/logos"):
            path = base / folder / f"{key}.{ext}"
            if not path.exists():
                continue
            try:
                src = Image.open(path).convert("RGBA")
            except Exception:
                continue
            sw, sh = src.size
            if sw <= 0 or sh <= 0:
                continue

            # Step 2 – icon crop for small renders
            if size <= LOGO_ICON_CROP_MAX_SIZE and team_name in LOGO_ICON_CROP:
                cl, ct, cr, cb = LOGO_ICON_CROP[team_name]
                box = (
                    max(0, int(round(sw * cl))),
                    max(0, int(round(sh * ct))),
                    min(sw, int(round(sw * cr))),
                    min(sh, int(round(sh * cb))),
                )
                if box[2] > box[0] and box[3] > box[1]:
                    src = src.crop(box)

            # Step 3 – trim transparent padding
            src = _trim_transparent(src)
            sw, sh = src.size
            if sw <= 0 or sh <= 0:
                continue

            # Step 4 – aspect-preserving fit
            scale = min(size / sw, size / sh)
            new_w = max(1, int(round(sw * scale)))
            new_h = max(1, int(round(sh * scale)))
            resized = src.resize((new_w, new_h), resample)

            # Step 5 – centre on a square transparent canvas
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            canvas.paste(
                resized,
                ((size - new_w) // 2, (size - new_h) // 2),
                resized,
            )
            return canvas
    return create_badge(team_name, size)
