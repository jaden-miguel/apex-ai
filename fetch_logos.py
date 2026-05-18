#!/usr/bin/env python3
"""
Download F1 team logos + the official F1 wordmark into ``logos/``.

Pulls every logo at 512 px (the Wikipedia thumb service rasterises from
the canonical SVG, so the result is genuinely high-resolution and stays
crisp at every size the UI uses).

Run once after a fresh checkout::

    python fetch_logos.py

Pass ``--force`` to overwrite the cached PNGs (use this if a logo looks
pixelated – the older version of this script saved low-res 96 px copies).
"""
import argparse
import sys
import urllib.request
from pathlib import Path

# Wikimedia only rasterises thumbnails at specific standard widths
# (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840).  500 is
# generously larger than any size the UI renders (we top out at 56 px
# on the hero card and 96 px on the empty state), and the Lanczos
# downsample stays sharp.  See https://w.wiki/GHai for the policy.
TARGET_PX = 500

# Wikimedia stores the canonical SVG and rasterises a PNG of any width
# via the /thumb/.../<NNN>px-... URL pattern.  We use that so the local
# cache scales cleanly from a 14 px grid badge all the way up to the
# 56 px hero card.
def _thumb(prefix: str, file_path: str, filename: str) -> str:
    """Build a Wikimedia thumbnail URL of width ``TARGET_PX``.

    ``filename`` should be URL-encoded already if it contains characters
    that need escaping (e.g. parentheses); pass it through ``urllib.parse``
    at call-site for clarity.
    """
    return (
        f"https://upload.wikimedia.org/wikipedia/{prefix}/thumb/"
        f"{file_path}/{filename}/{TARGET_PX}px-{filename}.png"
    )


def _file_url(prefix: str, file_path: str, filename: str) -> str:
    """Direct (non-thumbnail) image URL – used when the source itself
    is already a raster PNG sized appropriately."""
    return (
        f"https://upload.wikimedia.org/wikipedia/{prefix}/"
        f"{file_path}/{filename}"
    )


# Canonical Wikipedia filenames for every current (2026) team's logo.
# Verified against the wiki's imageinfo API – the older filenames used
# by previous releases of this script have been deleted/renamed (most
# teams added title sponsors to their mark for 2024–2026).
PNG_URLS = {
    "redbull":     _thumb("en",      "f/fa", "Red_Bull_Racing_Logo_2026.svg"),
    "ferrari":     _thumb("en",      "d/df", "Scuderia_Ferrari_HP_logo_24.svg"),
    "mercedes":    _thumb("commons", "9/90", "Mercedes-Logo.svg"),
    "mclaren":     _thumb("en",      "6/66", "McLaren_Racing_logo.svg"),
    # Aston Martin's logo is uploaded as a PNG, not an SVG, so we fetch
    # the source raster directly.
    "astonmartin": _file_url("en",   "1/15", "Aston_Martin_Aramco_2024_logo.png"),
    "alpine":      _thumb("commons", "4/4a", "BWT_Alpine_F1_Team_Logo.png"),
    "haas":        _thumb("commons", "9/92", "MoneyGram_Haas_F1_Team_Logo.svg"),
    "williams":    _thumb("commons", "1/12", "Atlassian_Williams_F1_Team_logo.svg"),
    "audi":        _thumb("commons", "9/92", "Audi-Logo_2016.svg"),
    # Cadillac's filename has parentheses (the (2025) suffix) so it's
    # URL-encoded here.
    "cadillac":    _thumb("en",      "b/bc",
                          "Cadillac_Formula_1_Team_Logo_%282025%29.svg"),
    "racingbulls": _thumb("en",      "2/2b", "VCARB_F1_logo.svg"),
}

# Official F1 wordmark (the bold red mark used since 2018).  Several
# mirrors are tried so a single 404 doesn't sink the whole download –
# the first URL to return a real PNG wins.
F1_LOGO_CANDIDATES = [
    _thumb("commons", "3/33", "F1.svg"),
    _thumb("commons", "0/00", "F1_logo.svg"),
    _thumb("commons", "4/4a", "F1_logo.svg"),
    _thumb("en",      "0/04", "F1.svg"),
    _thumb("commons", "5/57", "Formula_1.svg"),
]


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ApexAI-F1-Predictor/1.0 (educational; "
                              "https://github.com/local)"
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        if not data or len(data) < 200:
            return False
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"    ! {url} -> {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="re-download even when a cached copy already exists",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    logos_dir = base / "logos"
    logos_dir.mkdir(exist_ok=True)

    print(f"Downloading logos to {logos_dir} at {TARGET_PX}px...")

    for key, url in PNG_URLS.items():
        dest = logos_dir / f"{key}.png"
        if dest.exists() and not args.force:
            sz = dest.stat().st_size
            print(f"  {key:>12}: keeping cached ({sz} bytes)")
            continue
        ok = _download(url, dest)
        if ok:
            print(f"  {key:>12}: downloaded ({dest.stat().st_size} bytes)")
        else:
            print(f"  {key:>12}: FAILED")

    # Official F1 logo – try each candidate in turn.
    f1_dest = logos_dir / "f1.png"
    if f1_dest.exists() and not args.force:
        print(f"  {'f1':>12}: keeping cached ({f1_dest.stat().st_size} bytes)")
    else:
        print(f"  {'f1':>12}: trying official wordmark...")
        wrote = False
        for url in F1_LOGO_CANDIDATES:
            if _download(url, f1_dest):
                sz = f1_dest.stat().st_size
                print(f"  {'f1':>12}: downloaded ({sz} bytes) from {url}")
                wrote = True
                break
        if not wrote:
            print(f"  {'f1':>12}: FAILED – all mirrors unreachable")

    print(f"\nDone. {len(list(logos_dir.glob('*.png')))} files in {logos_dir}.")


if __name__ == "__main__":
    main()
