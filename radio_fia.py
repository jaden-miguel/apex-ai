"""Direct FIA livetiming team-radio loader.

f1radio (and OpenF1, which it wraps) sometimes lag the FIA's own archive by
a few clips, and the static `TeamRadio.json` index occasionally misses
late-added captures that *do* appear in `TeamRadio.jsonStream`.  This
module hits the FIA archive directly, replays the jsonStream so we capture
every late add, downloads MP3s into our own cache, and cross-references
clip metadata against OpenF1's /drivers and /race_control endpoints so the
UI can lap-map and group clips just like before.

Resulting clip dicts are deliberately shaped to look like f1radio.Clip
duck-typed: the UI only touches a small set of attributes
(`driver`, `driver_name`, `team`, `local_path`, `date`, `context.*`),
so wrapping the raw FIA data in a simple class keeps the integration
trivial.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# -- HTTP plumbing -----------------------------------------------------------

_FIA_BASE = "https://livetiming.formula1.com/static"
_OPENF1_BASE = "https://api.openf1.org/v1"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36 ApexAI/1.0"
)


def _http_get(url: str, timeout: int = 25, retries: int = 3) -> bytes:
    """GET with retry + UA spoofing.  Raises the last exception on failure."""
    # FIA archive URLs contain non-ASCII characters (e.g. São Paulo); the
    # default `urlopen` rejects non-ASCII paths.  Percent-encode just the
    # path/query so the rest of the URL (scheme/netloc) stays clean.
    parsed = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(parsed.path, safe="/")
    safe_query = urllib.parse.quote(parsed.query, safe="=&")
    url_safe = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, safe_path, safe_query, parsed.fragment)
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url_safe, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            # Don't retry 404s – the resource just isn't there.
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            time.sleep(0.4 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _json_get(url: str, timeout: int = 25) -> Any:
    return json.loads(_http_get(url, timeout=timeout).decode("utf-8-sig",
                                                              errors="ignore"))


# -- Session-path resolution -------------------------------------------------

# We need the FIA archive `Path` (e.g. "2025/2025-04-13_Bahrain_Grand_Prix/
# 2025-04-13_Race/") to fetch TeamRadio.{json,jsonStream}.  fastf1 already
# knows how to resolve this – we just call into it – but we also keep a tiny
# cache so repeated loads don't re-hit fastf1's heavy session pipeline.

_PATH_CACHE: dict[tuple[int, str, str], dict] = {}


def _resolve_session_meta(year: int, race: str, session_type: str = "R") -> dict:
    """Returns {'path': '<FIA archive sub-path>', 'session_key': int,
    'meeting_key': int, 'name': str}.

    Tries OpenF1's `/sessions` first (cheap) and falls back to fastf1 when
    that doesn't return a row.  Result is cached in-memory for the process
    lifetime."""
    cache_key = (year, race.lower(), session_type.upper())
    if cache_key in _PATH_CACHE:
        return _PATH_CACHE[cache_key]

    session_name = {
        "R": "Race", "Q": "Qualifying", "S": "Sprint",
        "SQ": "Sprint Qualifying", "SS": "Sprint Shootout",
        "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
    }.get(session_type.upper(), "Race")

    # --- Try OpenF1 first (gives us the session_key + meeting_key) ---
    info = {"path": None, "session_key": None, "meeting_key": None,
            "name": race}
    # Country / city aliases for the requested race so we can match against
    # OpenF1's `country_name` and `location` fields (its `meeting_name`
    # tends to come back null).
    _ALIASES = {
        "australian":     ["australia", "melbourne", "albert park"],
        "chinese":        ["china", "shanghai"],
        "japanese":       ["japan", "suzuka"],
        "bahrain":        ["bahrain", "sakhir"],
        "saudi arabian":  ["saudi arabia", "jeddah"],
        "saudi":          ["saudi arabia", "jeddah"],
        "miami":          ["united states", "miami", "miami gardens"],
        "emilia romagna": ["italy", "imola"],
        "monaco":         ["monaco"],
        "spanish":        ["spain", "barcelona", "catalunya"],
        "canadian":       ["canada", "montréal", "montreal"],
        "austrian":       ["austria", "spielberg", "red bull ring"],
        "british":        ["united kingdom", "silverstone", "great britain"],
        "great britain":  ["united kingdom", "silverstone"],
        "belgian":        ["belgium", "spa", "spa-francorchamps"],
        "hungarian":      ["hungary", "budapest", "hungaroring"],
        "dutch":          ["netherlands", "zandvoort"],
        "italian":        ["italy", "monza"],
        "azerbaijan":     ["azerbaijan", "baku"],
        "singapore":      ["singapore", "marina bay"],
        "united states":  ["united states", "austin", "circuit of the americas"],
        "us":             ["united states", "austin"],
        "mexico city":    ["mexico", "mexico city"],
        "mexican":        ["mexico", "mexico city"],
        "são paulo":      ["brazil", "são paulo", "sao paulo", "interlagos"],
        "sao paulo":      ["brazil", "são paulo", "sao paulo", "interlagos"],
        "brazilian":      ["brazil", "interlagos"],
        "las vegas":      ["united states", "las vegas"],
        "qatar":          ["qatar", "lusail"],
        "abu dhabi":      ["united arab emirates", "yas island", "yas marina"],
    }
    try:
        sessions = _json_get(
            f"{_OPENF1_BASE}/sessions?year={year}"
            f"&session_name={urllib.parse.quote(session_name)}"
        )
        race_lower = race.lower()
        # Strip the "grand prix" / "gp" suffix to get just the locator.
        race_token = re.sub(r"\b(grand\s*prix|gp)\b", "",
                             race_lower).strip()
        candidates = list({race_lower, race_token})
        # Append any alias hits to broaden our match.
        for key, aliases in _ALIASES.items():
            if key == race_token or key in race_lower:
                candidates.extend(aliases)
        candidates = [c for c in candidates if c]

        def _match(s):
            mn = (s.get("meeting_name") or "").lower()
            cn = (s.get("country_name") or "").lower()
            loc = (s.get("location") or "").lower()
            for cand in candidates:
                # Substring match against non-empty haystacks only – the bug
                # we used to have was `"" in <anything>` returning True for
                # the first row (meeting_name is null on OpenF1).
                if mn and (cand == mn or cand in mn):
                    return True
                if cn and (cand == cn or cand in cn):
                    return True
                if loc and (cand == loc or cand in loc):
                    return True
            return False

        hit = next((s for s in sessions if _match(s)), None)
        if hit:
            info["session_key"] = hit.get("session_key")
            info["meeting_key"] = hit.get("meeting_key")
            info["name"] = hit.get("meeting_name") or race
    except Exception:
        pass

    # --- Fall back to fastf1 for the FIA archive path ---
    try:
        import fastf1
        try:
            fastf1.Cache.enable_cache("cache")
        except Exception:
            pass
        ses = fastf1.get_session(year, race, session_type)
        # `session_info` raises DataNotLoadedError until `load()` is called.
        try:
            ses_info = ses.session_info
        except Exception:
            ses.load(telemetry=False, weather=False, laps=False, messages=False)
            ses_info = ses.session_info
        if ses_info and ses_info.get("Path"):
            info["path"] = ses_info["Path"].rstrip("/")
            info["name"] = (ses_info.get("Meeting") or {}).get("Name") or info["name"]
            # Trust fastf1's session_key over the OpenF1 fuzzy-match – fastf1
            # resolves against the FIA archive directly so its `Key` is the
            # authoritative session_key.  This fixes the Las Vegas / Miami /
            # USGP collision where OpenF1's `/sessions` returned `meeting_name=None`
            # and the country-only match picked the first US race.
            ff1_key = ses_info.get("Key")
            if ff1_key:
                info["session_key"] = ff1_key
                ff1_meeting = (ses_info.get("Meeting") or {}).get("Key")
                if ff1_meeting:
                    info["meeting_key"] = ff1_meeting
    except Exception as e:
        # Without an archive path we can't do anything; surface the failure.
        raise RuntimeError(
            f"Could not resolve FIA archive path for {year} {race} "
            f"{session_type}: {e}"
        ) from e

    if not info["path"]:
        raise RuntimeError(
            f"FIA archive path not found for {year} {race} {session_type}"
        )

    _PATH_CACHE[cache_key] = info
    return info


# -- Capture enumeration -----------------------------------------------------

_STREAM_LINE_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}\.\d+)(\{.*\})$")


def _parse_team_radio_stream(text: str) -> list[dict]:
    """Replay TeamRadio.jsonStream to build the final canonical capture list.

    The first line is `<ts>{"Captures":[...]}` – an absolute snapshot.
    Subsequent lines are `<ts>{"Captures":{"<idx>": {<partial>}}}` – indexed
    updates that mutate the snapshot.  We replay them in order so the result
    matches what the FIA's live timing client would show at the end of the
    session."""
    idx_array: list[dict] = []
    for line in text.splitlines():
        m = _STREAM_LINE_RE.match(line.strip())
        if not m:
            continue
        try:
            obj = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        caps = obj.get("Captures")
        if isinstance(caps, list):
            idx_array = [dict(c) for c in caps]
        elif isinstance(caps, dict):
            for idx_s, partial in caps.items():
                try:
                    idx = int(idx_s)
                except (TypeError, ValueError):
                    continue
                while len(idx_array) <= idx:
                    idx_array.append({})
                if isinstance(partial, dict):
                    idx_array[idx].update(partial)
    return [c for c in idx_array if c]


def _fetch_fia_captures(archive_path: str) -> list[dict]:
    """Returns the merged list of all radio captures from the FIA archive.

    Combines:
      * static TeamRadio.json (the snapshot at session end)
      * TeamRadio.jsonStream  (incremental, sometimes has late adds)
    """
    merged: dict[tuple, dict] = {}

    base = f"{_FIA_BASE}/{archive_path}"
    # Static snapshot
    try:
        j = _json_get(f"{base}/TeamRadio.json")
        for c in j.get("Captures", []):
            key = (c.get("Utc"), c.get("RacingNumber"), c.get("Path"))
            merged[key] = c
    except Exception:
        pass

    # Stream replay
    try:
        stream_raw = _http_get(f"{base}/TeamRadio.jsonStream").decode(
            "utf-8-sig", errors="ignore"
        )
        for c in _parse_team_radio_stream(stream_raw):
            key = (c.get("Utc"), c.get("RacingNumber"), c.get("Path"))
            merged[key] = c
    except Exception:
        pass

    return list(merged.values())


# -- Driver lookup -----------------------------------------------------------

def _fetch_drivers_map(session_key: int | None,
                       archive_path: str | None = None) -> dict[str, dict]:
    """Returns {racing_number_str: {name, abbr, team, color}}.

    Prefers OpenF1 (rich) and falls back to FIA's DriverList.json."""
    out: dict[str, dict] = {}

    if session_key:
        try:
            for d in _json_get(
                f"{_OPENF1_BASE}/drivers?session_key={session_key}"
            ):
                num = str(d.get("driver_number") or "")
                if not num:
                    continue
                out[num] = {
                    "name": d.get("full_name") or d.get("broadcast_name") or "",
                    "abbr": d.get("name_acronym") or "",
                    "team": d.get("team_name") or "",
                    "color": d.get("team_colour") or "",
                }
        except Exception:
            pass

    # If OpenF1 didn't return everyone, top up from the FIA's DriverList.
    if archive_path and (not out or len(out) < 20):
        try:
            j = _json_get(f"{_FIA_BASE}/{archive_path}/DriverList.json")
            for num, d in j.items():
                if num in out:
                    continue
                if not isinstance(d, dict):
                    continue
                out[num] = {
                    "name": d.get("FullName") or d.get("BroadcastName") or "",
                    "abbr": d.get("Tla") or "",
                    "team": d.get("TeamName") or "",
                    "color": d.get("TeamColour") or "",
                }
        except Exception:
            pass

    return out


# -- Race-control event log --------------------------------------------------

def _fetch_event_log(session_key: int | None) -> list[dict]:
    """Returns OpenF1 /race_control rows shaped like f1radio's event_log."""
    if not session_key:
        return []
    try:
        rows = _json_get(
            f"{_OPENF1_BASE}/race_control?session_key={session_key}"
        )
    except Exception:
        return []
    return rows or []


# -- Local cache + MP3 download ----------------------------------------------

def _cache_dir() -> Path:
    base = Path(__file__).resolve().parent / "radio_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def _download_audio(year: int, race_slug: str, archive_path: str,
                    captures: list[dict], progress=None) -> None:
    """Download every capture's MP3 into our cache.  Updates each capture
    dict in-place with `_local_path`."""
    out_dir = _cache_dir() / f"{year}_{race_slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parallel-ish downloads with a small thread pool keep the cold-cache
    # load fast on long races (e.g. Vegas has 80+ clips).
    sem = threading.Semaphore(8)
    threads: list[threading.Thread] = []
    done = [0]
    total = len(captures)
    lock = threading.Lock()

    def _one(cap: dict):
        try:
            rel = cap.get("Path") or ""
            if not rel:
                return
            filename = rel.rsplit("/", 1)[-1]
            local = out_dir / filename
            cap["_local_path"] = str(local)
            if local.exists() and local.stat().st_size > 0:
                return
            url = f"{_FIA_BASE}/{archive_path}/{rel}"
            data = _http_get(url, timeout=30)
            local.write_bytes(data)
        except Exception:
            cap["_local_path"] = None
        finally:
            sem.release()
            with lock:
                done[0] += 1
                if progress:
                    try:
                        progress(done[0], total)
                    except Exception:
                        pass

    for cap in captures:
        sem.acquire()
        t = threading.Thread(target=_one, args=(cap,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# -- Public shaped objects ---------------------------------------------------

@dataclass
class _Ctx:
    position: int | None = None
    compound: str | None = None
    lap_number: int | None = None
    stint_number: int | None = None
    tyre_age: int | None = None
    gap_to_leader: str | None = None


@dataclass
class _Clip:
    driver: str          # 3-letter abbr ("VER")
    driver_name: str     # "Max VERSTAPPEN"
    driver_number: int | None
    team: str
    local_path: str | None
    recording_url: str
    date: str            # ISO-8601 UTC
    lap: int | None = None
    context: _Ctx = field(default_factory=_Ctx)


@dataclass
class _Session:
    year: int
    race: str
    session_type: str
    clips: list[_Clip]
    event_log: list[dict]
    drivers: list[str]              # all abbreviations on the grid
    drivers_heard: list[str]        # abbreviations actually heard
    drivers_silent: list[dict]      # [{abbr, name, team}] – on grid, no radio
    total_clips: int
    # Lap-indexed lookups so the UI can fill in stint/compound/position
    # *after* the clip's lap has been derived from event_log timestamps.
    stints_by_drv: dict[int, list[dict]] = field(default_factory=dict)
    laps_by_drv: dict[int, list[dict]] = field(default_factory=dict)


def annotate_clip_context(clip: _Clip, lap: int | None,
                          stints_by_drv: dict[int, list[dict]],
                          laps_by_drv: dict[int, list[dict]] | None = None
                          ) -> None:
    """Fill in `clip.context` with stint/compound/tyre-age/position for the
    given (already-derived) lap.  Safe to call multiple times."""
    if clip.driver_number is None:
        return
    dnum = clip.driver_number
    if lap:
        clip.lap = lap
        clip.context.lap_number = lap
    # Stint match
    stints = stints_by_drv.get(dnum) or []
    matched_stint = None
    if lap and stints:
        for s in stints:
            ls = s.get("lap_start")
            le = s.get("lap_end") or 99
            if ls is None:
                continue
            if ls <= lap <= le:
                matched_stint = s
                break
    if matched_stint is None and stints:
        # Best-effort: take the latest stint whose start <= lap, else the
        # first stint of the race.
        if lap:
            cand = [s for s in stints if (s.get("lap_start") or 1) <= lap]
            if cand:
                matched_stint = max(cand, key=lambda s: s.get("lap_start") or 0)
        if matched_stint is None:
            matched_stint = stints[0]
    if matched_stint:
        clip.context.compound = matched_stint.get("compound") or clip.context.compound
        clip.context.stint_number = (
            matched_stint.get("stint_number") or clip.context.stint_number
        )
        # tyre_age = (current lap) - (stint lap_start) + initial age
        ls = matched_stint.get("lap_start") or 1
        init_age = matched_stint.get("tyre_age_at_start") or 0
        if lap:
            clip.context.tyre_age = max(0, lap - ls) + init_age

    # Position match
    laps_for = (laps_by_drv or {}).get(dnum) or []
    if lap and laps_for:
        for L in laps_for:
            if L.get("lap_number") == lap:
                pos = L.get("position")
                if pos:
                    clip.context.position = pos
                break


def load(year: int, race: str, session_type: str = "R",
         progress=None) -> _Session:
    """Drop-in alternative to f1radio.load that talks to the FIA archive
    directly and merges OpenF1 metadata."""

    meta = _resolve_session_meta(year, race, session_type)
    archive_path = meta["path"]
    session_key = meta["session_key"]
    race_name = meta["name"]

    if progress:
        progress("Fetching captures", 0, 1)

    captures = _fetch_fia_captures(archive_path)
    drivers_map = _fetch_drivers_map(session_key, archive_path)
    event_log = _fetch_event_log(session_key)

    # Pre-index stints (compound, tyre age, stint number) and laps
    # (position per lap) by driver_number.  The UI matches stints/positions
    # to clips by lap *after* lap-mapping completes.
    stints_by_drv: dict[int, list[dict]] = {}
    laps_by_drv: dict[int, list[dict]] = {}
    try:
        if session_key:
            for s in _json_get(
                f"{_OPENF1_BASE}/stints?session_key={session_key}"
            ):
                num = s.get("driver_number")
                if isinstance(num, int):
                    stints_by_drv.setdefault(num, []).append(s)
            for L in _json_get(
                f"{_OPENF1_BASE}/laps?session_key={session_key}"
            ):
                num = L.get("driver_number")
                if isinstance(num, int):
                    laps_by_drv.setdefault(num, []).append(L)
    except Exception:
        pass
    # Sort stints by lap_start so binary-search-style matching works.
    for v in stints_by_drv.values():
        v.sort(key=lambda s: s.get("lap_start") or 0)
    for v in laps_by_drv.values():
        v.sort(key=lambda L: L.get("lap_number") or 0)

    if progress:
        progress("Downloading audio", 0, len(captures))
    _download_audio(year, _slug(race_name), archive_path, captures,
                    progress=lambda d, t: (
                        progress and progress("Downloading audio", d, t)
                    ))

    clips: list[_Clip] = []
    for cap in captures:
        num = str(cap.get("RacingNumber") or "")
        drv_info = drivers_map.get(num, {})
        rel_path = cap.get("Path") or ""
        date = cap.get("Utc") or ""

        # Leave stint/compound/tyre-age/position unset here – the UI fills
        # them in via `annotate_clip_context()` once it has derived the lap
        # number from event_log timestamps.
        ctx = _Ctx()

        clip = _Clip(
            driver=drv_info.get("abbr") or num,
            driver_name=drv_info.get("name") or f"#{num}",
            driver_number=int(num) if num.isdigit() else None,
            team=drv_info.get("team") or "",
            local_path=cap.get("_local_path"),
            recording_url=f"{_FIA_BASE}/{archive_path}/{rel_path}",
            date=date,
            context=ctx,
        )
        clips.append(clip)

    # Coverage: who's on the grid vs. who actually had a clip.
    on_grid = sorted(drivers_map.keys(), key=lambda n: int(n) if n.isdigit() else 999)
    heard_nums = {str(c.driver_number) for c in clips if c.driver_number is not None}
    drivers_heard = [drivers_map[n].get("abbr") or n
                     for n in on_grid if n in heard_nums]
    drivers_silent = [
        {"abbr": drivers_map[n].get("abbr") or n,
         "name": drivers_map[n].get("name") or f"#{n}",
         "team": drivers_map[n].get("team") or ""}
        for n in on_grid if n not in heard_nums
    ]
    drivers_all = [drivers_map[n].get("abbr") or n for n in on_grid]

    return _Session(
        year=year,
        race=race_name,
        session_type=session_type,
        clips=clips,
        event_log=event_log,
        drivers=drivers_all,
        drivers_heard=drivers_heard,
        drivers_silent=drivers_silent,
        total_clips=len(clips),
        stints_by_drv=stints_by_drv,
        laps_by_drv=laps_by_drv,
    )


if __name__ == "__main__":
    # Tiny smoke test
    s = load(2025, "Bahrain Grand Prix", "R",
             progress=lambda *a: print("  ", *a, file=sys.stderr))
    print(f"\n{s.year} {s.race} R")
    print(f"  {s.total_clips} clips")
    print(f"  heard ({len(s.drivers_heard)}): {s.drivers_heard}")
    print(f"  silent ({len(s.drivers_silent)}): "
          f"{[d['abbr'] for d in s.drivers_silent]}")
    print(f"  first clip: {s.clips[0] if s.clips else 'none'}")
