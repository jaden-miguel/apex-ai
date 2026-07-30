#!/usr/bin/env python3
"""
ApexAI – telemetry & session replay data layer.

This module is the data half of the two telemetry features:

  * **Telemetry comparison** – pull a single lap's car data (speed,
    throttle, brake, gear, RPM, DRS) for any driver in any session of any
    season, align two laps on *distance around the lap*, and derive the
    time delta plus mini-sector dominance.  Because everything is keyed on
    distance rather than wall-clock, the two laps do not have to come from
    the same session: lap-vs-lap, compound-vs-compound and year-vs-year
    overlays all fall out of the same code path.

  * **Session replay** – resample every driver's position + car telemetry
    onto one common time grid, project each car onto the track centreline
    to get a continuous "laps completed + fraction of the current lap"
    progress value, and hand the UI a compact set of numpy arrays it can
    scrub through at any playback speed.

Everything here is pure data: no Tk, no matplotlib.  `app.py` owns the
rendering.  All heavy results are memoised in-process and mirrored to a
pickle on disk so re-opening a session you have already watched is
instant.

Data source is FastF1 throughout, which is the same timing feed the rest
of ApexAI already runs on.
"""
from __future__ import annotations

import hashlib
import logging
import math
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

import fastf1

# Base path mirrors prediction.py so a frozen build writes to the same
# writable app-support directory instead of inside the .app bundle.
if getattr(sys, "frozen", False):
    _BASE = Path.home() / "Library" / "Application Support" / "F1 Winner Predictor"
    _BASE.mkdir(parents=True, exist_ok=True)
else:
    _BASE = Path(__file__).parent

CACHE_DIR = _BASE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
REPLAY_CACHE_DIR = _BASE / "replay_cache"
REPLAY_CACHE_DIR.mkdir(exist_ok=True)

logging.getLogger("fastf1").setLevel(logging.WARNING)

try:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
except Exception:
    # prediction.py may have enabled it already; enabling twice is a no-op
    # in recent FastF1 but older versions raise.
    pass

# Bump when the shape of anything we pickle changes so stale replay caches
# are discarded rather than unpickled into the wrong dataclass.
REPLAY_CACHE_VERSION = "rp1"

# ---------------------------------------------------------------------------
# Session catalogue
# ---------------------------------------------------------------------------

# FastF1 accepts these identifiers directly.  Order is "most interesting
# first" so the UI's default selection lands on the race.
SESSION_CHOICES: list[tuple[str, str]] = [
    ("R", "Race"),
    ("Q", "Qualifying"),
    ("S", "Sprint"),
    ("SQ", "Sprint Qualifying"),
    ("SS", "Sprint Shootout"),
    ("FP1", "Practice 1"),
    ("FP2", "Practice 2"),
    ("FP3", "Practice 3"),
]

SESSION_LABELS = dict(SESSION_CHOICES)

# Sessions where running order is a live race order (progress round the
# track).  Everything else is ranked by best lap time, exactly like a real
# timing screen.
RACE_LIKE = {"R", "S"}

# Tyre compound colours, matching the official Pirelli marking colours.
COMPOUND_COLORS = {
    "SOFT": "#E10600",
    "MEDIUM": "#FFD12E",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
    "TEST-UNKNOWN": "#8F8F98",
    "UNKNOWN": "#8F8F98",
}


def compound_color(compound: Optional[str]) -> str:
    if not compound:
        return COMPOUND_COLORS["UNKNOWN"]
    return COMPOUND_COLORS.get(str(compound).upper(), COMPOUND_COLORS["UNKNOWN"])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _to_seconds(series) -> np.ndarray:
    """Timedelta series -> float seconds, NaT -> nan."""
    if series is None or len(series) == 0:
        return np.zeros(0, dtype=np.float64)
    vals = pd.to_timedelta(pd.Series(series).values)
    return vals.total_seconds().to_numpy(dtype=np.float64)


def fmt_laptime(seconds: Optional[float]) -> str:
    """1:23.456 style lap time.  Returns '—' for missing values."""
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(s) or s <= 0:
        return "—"
    minutes = int(s // 60)
    rem = s - minutes * 60
    if minutes:
        return f"{minutes}:{rem:06.3f}"
    return f"{rem:.3f}"


def fmt_delta(seconds: Optional[float]) -> str:
    """Signed gap, e.g. '+0.312' / '-1.204'."""
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(s):
        return "—"
    return f"{s:+.3f}"


def fmt_gap(seconds: Optional[float], laps_down: int = 0) -> str:
    if laps_down > 0:
        return f"+{laps_down}L"
    if seconds is None or not math.isfinite(seconds):
        return "—"
    if seconds <= 0.0005:
        return "LEADER"
    return f"+{seconds:.3f}"


def _nan_to_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _interp_continuous(grid: np.ndarray, t: np.ndarray,
                       vals: np.ndarray) -> np.ndarray:
    """Linear interpolation with edge clamping, nan-safe."""
    if len(t) == 0:
        return np.full(grid.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(t) & np.isfinite(vals)
    if not ok.any():
        return np.full(grid.shape, np.nan, dtype=np.float32)
    t = t[ok]
    vals = vals[ok]
    order = np.argsort(t, kind="stable")
    return np.interp(grid, t[order], vals[order]).astype(np.float32)


def _interp_discrete(grid: np.ndarray, t: np.ndarray,
                     vals: np.ndarray) -> np.ndarray:
    """Sample-and-hold for channels that must not be blended (gear, DRS,
    brake).  Interpolating a gear between 3 and 5 would invent a 4 that was
    never selected, so we take the most recent actual sample instead."""
    if len(t) == 0:
        return np.zeros(grid.shape, dtype=np.float32)
    order = np.argsort(t, kind="stable")
    t = t[order]
    vals = vals[order]
    idx = np.searchsorted(t, grid, side="right") - 1
    np.clip(idx, 0, len(t) - 1, out=idx)
    return vals[idx].astype(np.float32)


def _resample_path(x: np.ndarray, y: np.ndarray,
                   n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample an (x, y) polyline to `n_out` evenly-arc-spaced points.

    Returns (xs, ys, cumulative_distance) where the distance array is in
    the same units as the input coordinates.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok].astype(np.float64)
    y = y[ok].astype(np.float64)
    if len(x) < 4:
        return x, y, np.arange(len(x), dtype=np.float64)

    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return x, y, s
    # Drop duplicate arc-length samples; np.interp needs a strictly
    # increasing x-array to behave.
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    s = s[keep]
    x = x[keep]
    y = y[keep]

    grid = np.linspace(0.0, total, n_out)
    return np.interp(grid, s, x), np.interp(grid, s, y), grid


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass
class DriverInfo:
    abbr: str
    number: str
    name: str
    team: str
    color: str

    @property
    def label(self) -> str:
        return f"{self.abbr} · {self.name}"


@dataclass
class LapInfo:
    """One row of a driver's lap list, pre-formatted for a picker."""
    number: int
    lap_time: Optional[float]
    compound: Optional[str]
    tyre_life: Optional[float]
    stint: Optional[int]
    position: Optional[int]
    is_personal_best: bool
    is_accurate: bool
    deleted: bool

    @property
    def label(self) -> str:
        comp = (self.compound or "?")[:1].upper()
        star = " ★" if self.is_personal_best else ""
        return f"L{self.number:>2}  {fmt_laptime(self.lap_time)}  [{comp}]{star}"


@dataclass
class LapTelemetry:
    """A single lap's car trace, resampled onto an even distance grid.

    Everything downstream (overlay charts, delta, mini-sectors) works off
    the distance grid, which is what lets two laps from completely
    different sessions be compared.
    """
    year: int
    round_no: int
    event: str
    session_code: str
    session_name: str
    driver: str
    driver_name: str
    team: str
    color: str
    lap_number: int
    lap_time: Optional[float]
    compound: Optional[str]
    tyre_life: Optional[float]
    circuit: str

    distance: np.ndarray          # metres from the start line
    elapsed: np.ndarray           # seconds since the lap started
    speed: np.ndarray             # km/h
    throttle: np.ndarray          # %
    brake: np.ndarray             # 0/1
    gear: np.ndarray
    rpm: np.ndarray
    drs: np.ndarray               # raw DRS status code
    x: np.ndarray                 # track map coords
    y: np.ndarray

    @property
    def label(self) -> str:
        return (f"{self.driver} · {self.year} {self.event} "
                f"{self.session_name} · L{self.lap_number}")

    @property
    def short_label(self) -> str:
        return f"{self.driver} {self.year} L{self.lap_number}"

    # -- derived stats, all cheap enough to compute on demand --

    @property
    def top_speed(self) -> float:
        return float(np.nanmax(self.speed)) if len(self.speed) else float("nan")

    @property
    def avg_speed(self) -> float:
        return float(np.nanmean(self.speed)) if len(self.speed) else float("nan")

    @property
    def wot_pct(self) -> float:
        """Share of the lap *distance* spent at wide-open throttle (95 %+).

        Measured against distance rather than time, so a slower lap isn't
        flattered by spending longer pinned at the stop.
        """
        if not len(self.throttle):
            return float("nan")
        return float(np.mean(self.throttle >= 95.0) * 100.0)

    @property
    def braking_pct(self) -> float:
        if not len(self.brake):
            return float("nan")
        return float(np.mean(self.brake > 0.5) * 100.0)


@dataclass
class Comparison:
    """Two laps aligned on a shared distance grid."""
    ref: LapTelemetry
    other: LapTelemetry
    distance: np.ndarray
    ref_elapsed: np.ndarray
    other_elapsed: np.ndarray
    delta: np.ndarray             # other - ref, seconds (positive = other slower)
    minisector_edges: np.ndarray  # distance boundaries, len = n+1
    minisector_winner: np.ndarray # 0 = ref, 1 = other, per mini-sector
    minisector_delta: np.ndarray  # per-sector time difference
    same_circuit: bool

    @property
    def final_delta(self) -> Optional[float]:
        if not len(self.delta):
            return None
        return float(self.delta[-1])


@dataclass
class ReplayDriver:
    info: DriverInfo
    x: np.ndarray            # float32, per frame
    y: np.ndarray
    progress: np.ndarray     # float32 laps completed + fraction of current lap
    on_track: np.ndarray     # bool, per frame
    speed: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    gear: np.ndarray
    rpm: np.ndarray
    drs: np.ndarray
    lap_number: np.ndarray   # int16, current lap (1-based)
    compound: list           # per frame index -> compound string (run-length decoded)
    best_lap: np.ndarray     # float32, best lap time set *so far* at each frame
    finished_at: Optional[float]   # session-time seconds when they stopped


@dataclass
class Replay:
    """Everything the replay UI needs, as flat arrays on one time grid."""
    year: int
    round_no: int
    event: str
    session_code: str
    session_name: str
    circuit: str
    hz: float
    t: np.ndarray                 # session-time seconds, len = n_frames
    drivers: list                 # list[ReplayDriver], results order
    track_x: np.ndarray           # centreline for drawing the map
    track_y: np.ndarray
    track_length: float
    corners: list                 # list[(x, y, label)]
    total_laps: Optional[int]
    race_like: bool

    @property
    def n_frames(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [self.track_x] + [d.x for d in self.drivers]
        ys = [self.track_y] + [d.y for d in self.drivers]
        xs = np.concatenate([a[np.isfinite(a)] for a in xs if len(a)])
        ys = np.concatenate([a[np.isfinite(a)] for a in ys if len(a)])
        if not len(xs) or not len(ys):
            return (0.0, 0.0, 1.0, 1.0)
        return (float(xs.min()), float(ys.min()),
                float(xs.max()), float(ys.max()))

    # -- live timing screen ------------------------------------------------

    def order_at(self, frame: int) -> list:
        """Running order at `frame`, as a list of dicts ready to render.

        Race-like sessions are ordered by track progress (the classic
        "who is physically ahead" timing screen).  Qualifying and practice
        are ordered by the best lap time set so far, which is what those
        screens actually show.
        """
        frame = int(np.clip(frame, 0, max(0, self.n_frames - 1)))
        rows = []

        if self.race_like:
            live = [(d, float(d.progress[frame])) for d in self.drivers]
            live.sort(key=lambda p: -p[1] if math.isfinite(p[1]) else 1e9)
            leader, leader_prog = live[0] if live else (None, 0.0)
            for pos, (d, prog) in enumerate(live, start=1):
                laps_down = 0
                # The leader's gap is zero by definition, which is what
                # renders as "LEADER"; None is reserved for a gap we
                # genuinely could not compute.
                gap = 0.0
                if leader is not None and d is not leader:
                    diff = leader_prog - prog
                    laps_down = int(diff) if diff >= 1.0 else 0
                    gap = self._time_gap(leader, prog, frame)
                rows.append(self._row(d, pos, frame, gap, laps_down))
        else:
            def _key(d):
                b = float(d.best_lap[frame])
                return b if math.isfinite(b) and b > 0 else 1e9
            live = sorted(self.drivers, key=_key)
            best = _key(live[0]) if live else 1e9
            for pos, d in enumerate(live, start=1):
                own = _key(d)
                gap = None
                if own < 1e9 and best < 1e9:
                    # Zero for whoever holds the session best, so P1 reads
                    # as "LEADER" rather than "+0.000".
                    gap = own - best
                rows.append(self._row(d, pos, frame, gap, 0))
        return rows

    def _time_gap(self, leader: ReplayDriver, prog: float,
                  frame: int) -> Optional[float]:
        """How long ago the current leader was at this car's track position.

        Progress is monotonic, so a plain interpolation of the leader's
        progress-vs-time curve inverts cleanly into a time gap — the same
        definition a real timing screen uses.
        """
        lp = leader.progress[:frame + 1]
        if len(lp) < 2 or not math.isfinite(prog):
            return None
        # np.interp needs an increasing x; progress is built monotonic.
        crossed = np.interp(prog, lp, self.t[:frame + 1],
                            left=float(self.t[0]), right=float(self.t[frame]))
        gap = float(self.t[frame] - crossed)
        return gap if math.isfinite(gap) and gap >= 0 else None

    def _row(self, d: ReplayDriver, pos: int, frame: int,
             gap: Optional[float], laps_down: int) -> dict:
        best = float(d.best_lap[frame])
        return {
            "pos": pos,
            "abbr": d.info.abbr,
            "name": d.info.name,
            "team": d.info.team,
            "color": d.info.color,
            "lap": int(d.lap_number[frame]),
            "compound": d.compound[frame] if d.compound else None,
            "gap": gap,
            "laps_down": laps_down,
            "best": best if math.isfinite(best) and best > 0 else None,
            "speed": float(d.speed[frame]),
            "on_track": bool(d.on_track[frame]),
            "retired": not bool(d.on_track[frame]) and (
                d.finished_at is not None and self.t[frame] >= d.finished_at),
        }

    def telemetry_at(self, abbr: str, frame: int) -> Optional[dict]:
        """Live channel readout for one driver — the replay HUD."""
        frame = int(np.clip(frame, 0, max(0, self.n_frames - 1)))
        for d in self.drivers:
            if d.info.abbr == abbr:
                return {
                    "speed": _nan_to_none(d.speed[frame]),
                    "throttle": _nan_to_none(d.throttle[frame]),
                    "brake": _nan_to_none(d.brake[frame]),
                    "gear": _nan_to_none(d.gear[frame]),
                    "rpm": _nan_to_none(d.rpm[frame]),
                    "drs": _nan_to_none(d.drs[frame]),
                    "lap": int(d.lap_number[frame]),
                    "compound": d.compound[frame] if d.compound else None,
                    "color": d.info.color,
                    "team": d.info.team,
                }
        return None

    def trace(self, abbr: str, channel: str, frame: int,
              window: int) -> tuple[np.ndarray, np.ndarray]:
        """A rolling slice of one channel, for the HUD's scrolling trace."""
        frame = int(np.clip(frame, 0, max(0, self.n_frames - 1)))
        lo = max(0, frame - window)
        for d in self.drivers:
            if d.info.abbr == abbr:
                arr = getattr(d, channel, None)
                if arr is None:
                    break
                return self.t[lo:frame + 1], arr[lo:frame + 1]
        return np.zeros(0), np.zeros(0)


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

_SESSION_MEM: dict = {}
_SESSION_LOCK = threading.Lock()


def available_sessions(year: int, round_no: int) -> list[tuple[str, str]]:
    """Session identifiers that actually exist for this event.

    Falls back to the standard non-sprint weekend if the schedule can't be
    resolved, so the picker is never empty.
    """
    default = [("R", "Race"), ("Q", "Qualifying"),
               ("FP1", "Practice 1"), ("FP2", "Practice 2"),
               ("FP3", "Practice 3")]
    try:
        event = fastf1.get_event(year, round_no)
    except Exception:
        return default

    names = []
    for i in range(1, 6):
        try:
            nm = event.get(f"Session{i}")
        except Exception:
            nm = None
        if isinstance(nm, str) and nm.strip() and nm.strip().lower() != "none":
            names.append(nm.strip())
    if not names:
        return default

    # Map the schedule's human names back onto FastF1 identifiers.
    name_to_code = {
        "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
        "Qualifying": "Q", "Sprint": "S",
        "Sprint Qualifying": "SQ", "Sprint Shootout": "SS",
        "Race": "R",
    }
    out = []
    for nm in names:
        code = name_to_code.get(nm)
        if code:
            out.append((code, nm))
    if not out:
        return default
    # Race first, then quali, then the rest — most-wanted at the top.
    priority = {"R": 0, "S": 1, "Q": 2, "SQ": 3, "SS": 3,
                "FP3": 4, "FP2": 5, "FP1": 6}
    out.sort(key=lambda c: priority.get(c[0], 9))
    return out


def load_session(year: int, round_no: int, code: str,
                 with_telemetry: bool = False,
                 progress: Optional[Callable[[str], None]] = None):
    """Load (and memoise) a FastF1 session.

    `with_telemetry` controls whether car + position streams are pulled.
    They are a much larger download than laps alone, so the comparison tab
    asks for them but the lap-list picker does not.  A session already
    loaded *with* telemetry satisfies a later request without telemetry.
    """
    key = (int(year), int(round_no), str(code))

    with _SESSION_LOCK:
        cached = _SESSION_MEM.get(key)
    if cached is not None:
        session, had_tel = cached
        if had_tel or not with_telemetry:
            return session

    def _say(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    _say(f"Fetching {SESSION_LABELS.get(code, code)} · {year} round {round_no}…")
    session = fastf1.get_session(year, round_no, code)
    _say("Loading timing data…" if not with_telemetry
         else "Loading timing + telemetry (first time can take a minute)…")
    session.load(laps=True, telemetry=with_telemetry,
                 weather=False, messages=False)
    _assert_loaded(session, year, round_no, code, with_telemetry)

    with _SESSION_LOCK:
        _SESSION_MEM[key] = (session, with_telemetry)
    return session


def _assert_loaded(session, year: int, round_no: int, code: str,
                   with_telemetry: bool) -> None:
    """Fail loudly when `Session.load()` quietly came back empty.

    FastF1 wraps each loader in `@soft_exceptions`, so a network failure or
    a session the feed has no data for is logged as a warning and `load()`
    returns normally with nothing attached.  Every later access then dies
    on `DataNotLoadedError`, which surfaces to the user as either a bare
    "no drivers found" or FastF1's own "see Session.load" jargon — neither
    of which says what actually went wrong.  Check once, here, and say it
    plainly.
    """
    label = f"{SESSION_LABELS.get(code, code)} · {year} round {round_no}"
    try:
        laps = session.laps
    except Exception:
        raise ValueError(
            f"No timing data came back for {label}. The F1 timing feed was "
            f"unreachable, or it has no data for this session yet."
        ) from None
    if laps is None or not len(laps):
        raise ValueError(
            f"{label} returned an empty timing sheet — the session may not "
            f"have run yet."
        )
    if with_telemetry and not (_safe_stream(session, "pos_data")
                               or _safe_stream(session, "car_data")):
        raise ValueError(
            f"Timing data for {label} loaded, but its car telemetry stream "
            f"did not. Telemetry only exists from 2018 onwards, and very "
            f"recent sessions can take a few hours to be published."
        )


def _safe_stream(session, name: str):
    """`session.pos_data` / `car_data` raise when unloaded rather than
    returning None, so a plain getattr default is not enough."""
    try:
        return getattr(session, name, None)
    except Exception:
        return None


def session_display_name(session) -> str:
    try:
        return str(session.name)
    except Exception:
        return "Session"


def _safe_total_laps(session) -> Optional[int]:
    # `total_laps` raises unless the lap-count endpoint came back, which it
    # doesn't for practice and occasionally fails for older races.
    try:
        tl = session.total_laps
        if tl:
            return int(tl)
    except Exception:
        pass
    try:
        return int(session.laps["LapNumber"].max())
    except Exception:
        return None


def driver_table(session, team_colors: Optional[dict] = None) -> list[DriverInfo]:
    """Drivers in the session, in classification order."""
    out: list[DriverInfo] = []
    try:
        results = session.results
    except Exception:
        results = None

    if results is not None and len(results):
        for _, row in results.iterrows():
            abbr = str(row.get("Abbreviation") or "").strip()
            if not abbr:
                continue
            team = str(row.get("TeamName") or "").strip()
            name = str(row.get("FullName") or "").strip() or abbr
            color = None
            if team_colors:
                color = team_colors.get(team)
            if not color:
                tc = str(row.get("TeamColor") or "").strip()
                color = f"#{tc}" if tc and not tc.startswith("#") else (tc or None)
            out.append(DriverInfo(
                abbr=abbr,
                number=str(row.get("DriverNumber") or "").strip(),
                name=name,
                team=team,
                color=color or "#8F8F98",
            ))

    if out:
        return out

    # Results can be empty for some practice sessions — fall back to laps.
    try:
        laps = session.laps
    except Exception:
        return out
    seen = set()
    for _, row in laps.iterrows():
        abbr = str(row.get("Driver") or "").strip()
        if not abbr or abbr in seen:
            continue
        seen.add(abbr)
        team = str(row.get("Team") or "").strip()
        color = (team_colors or {}).get(team, "#8F8F98")
        out.append(DriverInfo(abbr=abbr,
                              number=str(row.get("DriverNumber") or "").strip(),
                              name=abbr, team=team, color=color))
    return out


def lap_table(session, abbr: str) -> list[LapInfo]:
    """Every lap the driver completed, newest data first formatted for a
    picker.  Laps without a time (in/out laps, aborted runs) are kept —
    you may well want to look at an out-lap's telemetry."""
    try:
        laps = session.laps.pick_drivers(abbr)
    except Exception:
        return []
    if laps is None or not len(laps):
        return []

    out: list[LapInfo] = []
    for _, row in laps.iterrows():
        num = row.get("LapNumber")
        if pd.isna(num):
            continue
        lt = row.get("LapTime")
        lt_s = None if pd.isna(lt) else float(pd.Timedelta(lt).total_seconds())
        comp = row.get("Compound")
        comp = None if (comp is None or pd.isna(comp)) else str(comp)
        out.append(LapInfo(
            number=int(num),
            lap_time=lt_s,
            compound=comp,
            tyre_life=_nan_to_none(row.get("TyreLife")),
            stint=int(row["Stint"]) if not pd.isna(row.get("Stint")) else None,
            position=int(row["Position"]) if not pd.isna(row.get("Position")) else None,
            is_personal_best=bool(row.get("IsPersonalBest") is True),
            is_accurate=bool(row.get("IsAccurate") is True),
            deleted=bool(row.get("Deleted") is True),
        ))
    return out


def _pick_fastest(laps):
    """`Laps.pick_fastest()` with a sane fallback.

    By default FastF1 only considers laps the timing feed flagged as a
    personal best.  That flag is routinely absent in practice sessions and
    in older seasons, in which case the default call returns None even
    though the driver has perfectly good timed laps.  Fall back to picking
    purely on lap time so we don't tell the user "no timed lap" when there
    plainly is one.
    """
    if laps is None or not len(laps):
        return None
    try:
        fast = laps.pick_fastest()
    except Exception:
        fast = None
    if fast is None:
        try:
            fast = laps.pick_fastest(only_by_time=True)
        except Exception:
            fast = None
    return fast


def fastest_lap_number(session, abbr: str) -> Optional[int]:
    try:
        fast = _pick_fastest(session.laps.pick_drivers(abbr))
    except Exception:
        return None
    if fast is None:
        return None
    try:
        num = fast["LapNumber"]
    except Exception:
        return None
    return None if pd.isna(num) else int(num)


# ---------------------------------------------------------------------------
# Lap telemetry
# ---------------------------------------------------------------------------

# Distance-grid resolution for a single lap.  ~5 m per sample on a typical
# 5 km circuit, which is finer than the ~4 Hz raw car-data feed but keeps
# the delta curve smooth on screen.
LAP_GRID_POINTS = 1000


def lap_telemetry(session, abbr: str, lap_number: Optional[int] = None,
                  team_colors: Optional[dict] = None) -> LapTelemetry:
    """Pull one lap and resample it onto an even distance grid.

    `lap_number=None` picks the driver's fastest lap.  Raises ValueError
    with a human-readable message if the lap or its telemetry is missing,
    which the UI surfaces directly.
    """
    try:
        laps = session.laps.pick_drivers(abbr)
    except Exception as exc:
        raise ValueError(f"No lap data for {abbr}: {exc}") from exc
    if laps is None or not len(laps):
        raise ValueError(f"{abbr} has no laps in this session.")

    if lap_number is None:
        lap = _pick_fastest(laps)
        if lap is None:
            raise ValueError(f"{abbr} set no timed lap in this session.")
    else:
        sel = laps[laps["LapNumber"] == float(lap_number)]
        if not len(sel):
            raise ValueError(f"{abbr} has no lap {lap_number}.")
        lap = sel.iloc[0]

    try:
        car = lap.get_car_data()
    except Exception as exc:
        raise ValueError(
            f"No car telemetry for {abbr} lap {lap_number or 'fastest'}: {exc}"
        ) from exc
    if car is None or not len(car):
        raise ValueError(f"No car telemetry for {abbr} on that lap.")

    car = car.add_distance()

    dist = car["Distance"].to_numpy(dtype=np.float64)
    elapsed = _to_seconds(car["Time"])
    # add_distance() integrates speed from the first sample, so a lap whose
    # telemetry starts mid-lap would carry an offset; normalise both axes.
    if len(dist):
        dist = dist - dist[0]
    if len(elapsed):
        elapsed = elapsed - elapsed[0]

    total = float(dist[-1]) if len(dist) else 0.0
    if total <= 0:
        raise ValueError(f"Telemetry for {abbr} on that lap covers no distance.")

    grid = np.linspace(0.0, total, LAP_GRID_POINTS)

    def _cont(col):
        if col not in car:
            return np.full(grid.shape, np.nan, dtype=np.float32)
        return _interp_continuous(grid, dist,
                                  car[col].to_numpy(dtype=np.float64))

    def _disc(col):
        if col not in car:
            return np.zeros(grid.shape, dtype=np.float32)
        vals = car[col].to_numpy()
        if vals.dtype == bool:
            vals = vals.astype(np.float64)
        return _interp_discrete(grid, dist, vals.astype(np.float64))

    elapsed_grid = _interp_continuous(grid, dist, elapsed).astype(np.float64)

    # Track map coordinates for the mini-sector overlay.  Position data is
    # a separate stream at a different rate, so map it onto the same
    # distance grid via the shared lap-relative time base.
    xs = np.full(grid.shape, np.nan, dtype=np.float32)
    ys = np.full(grid.shape, np.nan, dtype=np.float32)
    try:
        pos = lap.get_pos_data()
    except Exception:
        pos = None
    if pos is not None and len(pos):
        pos_t = _to_seconds(pos["Time"])
        if len(pos_t):
            pos_t = pos_t - pos_t[0]
        xs = _interp_continuous(elapsed_grid, pos_t,
                                pos["X"].to_numpy(dtype=np.float64))
        ys = _interp_continuous(elapsed_grid, pos_t,
                                pos["Y"].to_numpy(dtype=np.float64))

    lt = lap.get("LapTime")
    lt_s = None if lt is None or pd.isna(lt) else float(
        pd.Timedelta(lt).total_seconds())
    comp = lap.get("Compound")
    comp = None if (comp is None or pd.isna(comp)) else str(comp)
    team = str(lap.get("Team") or "")

    drivers = {d.abbr: d for d in driver_table(session, team_colors)}
    info = drivers.get(abbr)

    event_name, year, rnd, circuit = _event_meta(session)

    return LapTelemetry(
        year=year, round_no=rnd, event=event_name,
        session_code=_session_code(session),
        session_name=session_display_name(session),
        driver=abbr,
        driver_name=info.name if info else abbr,
        team=team or (info.team if info else ""),
        color=(info.color if info else None) or
              (team_colors or {}).get(team, "#8F8F98"),
        lap_number=int(lap["LapNumber"]) if not pd.isna(lap.get("LapNumber")) else 0,
        lap_time=lt_s,
        compound=comp,
        tyre_life=_nan_to_none(lap.get("TyreLife")),
        circuit=circuit,
        distance=grid.astype(np.float32),
        elapsed=elapsed_grid.astype(np.float32),
        speed=_cont("Speed"),
        throttle=_cont("Throttle"),
        brake=_disc("Brake"),
        gear=_disc("nGear"),
        rpm=_cont("RPM"),
        drs=_disc("DRS"),
        x=xs, y=ys,
    )


def _event_meta(session) -> tuple[str, int, int, str]:
    name, year, rnd, circuit = "Grand Prix", 0, 0, ""
    try:
        ev = session.event
        name = str(ev.get("EventName") or name)
        circuit = str(ev.get("Location") or ev.get("Country") or "")
        rnd = int(ev.get("RoundNumber") or 0)
    except Exception:
        pass
    try:
        year = int(session.event.year)
    except Exception:
        try:
            year = int(session.date.year)
        except Exception:
            year = 0
    return name, year, rnd, circuit


def _session_code(session) -> str:
    nm = session_display_name(session)
    for code, label in SESSION_CHOICES:
        if label == nm:
            return code
    return nm


DEFAULT_MINISECTORS = 25


def compare_laps(ref: LapTelemetry, other: LapTelemetry,
                 minisectors: int = DEFAULT_MINISECTORS) -> Comparison:
    """Align two laps on distance and derive the delta + sector dominance.

    The laps do not need to share a session — only a circuit.  We clip to
    the shorter of the two distance traces so a lap whose telemetry starts
    slightly late doesn't drag the comparison off the end.
    """
    max_d = min(float(ref.distance[-1]), float(other.distance[-1]))
    if max_d <= 0:
        raise ValueError("Laps have no overlapping distance to compare.")

    grid = np.linspace(0.0, max_d, LAP_GRID_POINTS)
    ref_t = np.interp(grid, ref.distance, ref.elapsed)
    oth_t = np.interp(grid, other.distance, other.elapsed)
    delta = oth_t - ref_t

    edges = np.linspace(0.0, max_d, minisectors + 1)
    ref_at = np.interp(edges, ref.distance, ref.elapsed)
    oth_at = np.interp(edges, other.distance, other.elapsed)
    ref_seg = np.diff(ref_at)
    oth_seg = np.diff(oth_at)
    seg_delta = oth_seg - ref_seg
    winner = (seg_delta > 0).astype(np.int8)  # other slower -> ref wins (0)

    same = _same_circuit(ref, other)

    return Comparison(
        ref=ref, other=other,
        distance=grid.astype(np.float32),
        ref_elapsed=ref_t.astype(np.float32),
        other_elapsed=oth_t.astype(np.float32),
        delta=delta.astype(np.float32),
        minisector_edges=edges.astype(np.float32),
        minisector_winner=winner,
        minisector_delta=seg_delta.astype(np.float32),
        same_circuit=same,
    )


def _same_circuit(a: LapTelemetry, b: LapTelemetry) -> bool:
    """Distance-aligned comparison is only meaningful on the same track.

    Circuits get resurfaced and occasionally re-profiled between seasons,
    so we compare the location name and allow a few percent of lap-length
    drift rather than demanding an exact match.
    """
    if a.circuit and b.circuit and a.circuit.strip().lower() != b.circuit.strip().lower():
        return False
    la, lb = float(a.distance[-1]), float(b.distance[-1])
    if la <= 0 or lb <= 0:
        return False
    return abs(la - lb) / max(la, lb) < 0.05


# ---------------------------------------------------------------------------
# Session replay
# ---------------------------------------------------------------------------

# 4 Hz gives visibly smooth motion once the UI interpolates between frames,
# while keeping a two-hour race under ~30 k frames per driver.
DEFAULT_REPLAY_HZ = 4.0

# Centreline resolution used to project cars onto track progress.  240
# points is ~20 m on a typical circuit — fine enough to order cars
# correctly, coarse enough that the projection stays a couple of seconds
# of numpy rather than a couple of minutes.
CENTERLINE_POINTS = 240


def _replay_cache_path(year: int, round_no: int, code: str, hz: float) -> Path:
    key = f"{REPLAY_CACHE_VERSION}:{year}:{round_no}:{code}:{hz:g}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return REPLAY_CACHE_DIR / f"replay_{year}_{round_no}_{code}_{digest}.pkl"


def load_cached_replay(year: int, round_no: int, code: str,
                       hz: float = DEFAULT_REPLAY_HZ) -> Optional[Replay]:
    path = _replay_cache_path(year, round_no, code, hz)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        # A truncated or stale pickle should never be fatal — just rebuild.
        try:
            path.unlink()
        except Exception:
            pass
        return None
    return obj if isinstance(obj, Replay) else None


def _store_replay(rp: Replay) -> None:
    path = _replay_cache_path(rp.year, rp.round_no, rp.session_code, rp.hz)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(rp, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass


def _build_centerline(session, pos_data: dict) -> tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, float]:
    """Build a track centreline from the fastest lap's position trace.

    The fastest lap is the cleanest single loop available: no pit entry, no
    slow-down lap, no off-track excursion.
    """
    xs = ys = None
    try:
        fast = _pick_fastest(session.laps)
        if fast is not None:
            pos = fast.get_pos_data()
            if pos is not None and len(pos) > 10:
                xs = pos["X"].to_numpy(dtype=np.float64)
                ys = pos["Y"].to_numpy(dtype=np.float64)
    except Exception:
        xs = ys = None

    if xs is None or ys is None or len(xs) < 10:
        # Fall back to the longest available position stream.
        best = None
        for df in pos_data.values():
            if df is not None and len(df) > (0 if best is None else len(best)):
                best = df
        if best is None or not len(best):
            return (np.zeros(0), np.zeros(0), np.zeros(0), 0.0)
        xs = best["X"].to_numpy(dtype=np.float64)
        ys = best["Y"].to_numpy(dtype=np.float64)

    cx, cy, s = _resample_path(xs, ys, CENTERLINE_POINTS)
    length = float(s[-1]) if len(s) else 0.0
    return cx, cy, s, length


def _project_progress(x: np.ndarray, y: np.ndarray,
                      cx: np.ndarray, cy: np.ndarray,
                      s: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Arc-length position of each (x, y) sample along the centreline.

    Done as a chunked full nearest-neighbour search: a windowed search
    would be cheaper but breaks the moment a car leaves the track and
    rejoins somewhere else, which happens in every race.
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or len(cx) == 0:
        return out
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        dx = x[lo:hi, None] - cx[None, :]
        dy = y[lo:hi, None] - cy[None, :]
        idx = np.argmin(dx * dx + dy * dy, axis=1)
        out[lo:hi] = s[idx]
    return out


def _run_length_compounds(grid: np.ndarray, lap_start: np.ndarray,
                          compounds: Sequence) -> list:
    """Per-frame compound, taken from the stint that owns each frame."""
    if not len(lap_start) or not len(compounds):
        return [None] * len(grid)
    idx = np.searchsorted(lap_start, grid, side="right") - 1
    np.clip(idx, 0, len(compounds) - 1, out=idx)
    return [compounds[i] for i in idx]


def build_replay(session, hz: float = DEFAULT_REPLAY_HZ,
                 team_colors: Optional[dict] = None,
                 progress: Optional[Callable[[str], None]] = None) -> Replay:
    """Turn a loaded session into a scrubbable replay.

    The session must have been loaded with `with_telemetry=True`.
    """
    def _say(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    pos_data = _safe_stream(session, "pos_data") or {}
    car_data = _safe_stream(session, "car_data") or {}
    if not pos_data:
        raise ValueError(
            "This session has no position data — replay needs the car "
            "telemetry stream, which F1 only publishes for 2018 onwards."
        )

    laps = session.laps
    infos = driver_table(session, team_colors)
    if not infos:
        raise ValueError("No drivers found in this session.")

    _say("Building time grid…")

    # -- common time grid -------------------------------------------------
    starts, ends = [], []
    for df in pos_data.values():
        if df is None or not len(df) or "SessionTime" not in df:
            continue
        st = _to_seconds(df["SessionTime"])
        st = st[np.isfinite(st)]
        if len(st):
            starts.append(st[0])
            ends.append(st[-1])
    if not starts:
        raise ValueError("Position data carries no usable session timestamps.")

    # Trim to the actual green-flag window where we can: the raw streams
    # start well before the session does (formation, garage running) and a
    # replay that opens on twenty stationary cars is a poor first frame.
    t_start = float(min(starts))
    t_end = float(max(ends))
    try:
        lap_starts = _to_seconds(laps["LapStartTime"])
        lap_starts = lap_starts[np.isfinite(lap_starts)]
        lap_ends = _to_seconds(laps["Time"])
        lap_ends = lap_ends[np.isfinite(lap_ends)]
        if len(lap_starts) and len(lap_ends):
            t_start = max(t_start, float(np.min(lap_starts)) - 5.0)
            t_end = min(t_end, float(np.max(lap_ends)) + 15.0)
    except Exception:
        pass
    if not (t_end > t_start):
        raise ValueError("Could not determine the session's time window.")

    step = 1.0 / float(hz)
    grid = np.arange(t_start, t_end + step, step, dtype=np.float64)

    _say("Tracing the circuit…")
    cx, cy, cs, track_len = _build_centerline(session, pos_data)
    race_like = _session_code(session) in RACE_LIKE

    drivers: list[ReplayDriver] = []
    total = len(infos)
    for i, info in enumerate(infos, start=1):
        _say(f"Replaying {info.abbr} ({i}/{total})…")
        num = info.number
        pdf = pos_data.get(num)
        if pdf is None or not len(pdf):
            continue

        pt = _to_seconds(pdf["SessionTime"])
        px = pdf["X"].to_numpy(dtype=np.float64)
        py = pdf["Y"].to_numpy(dtype=np.float64)

        gx = _interp_continuous(grid, pt, px)
        gy = _interp_continuous(grid, pt, py)

        # Status is a string channel; hold the last known value.
        if "Status" in pdf:
            status = pdf["Status"].to_numpy()
            on_num = (status == "OnTrack").astype(np.float64)
            on_track = _interp_discrete(grid, pt, on_num) > 0.5
        else:
            on_track = np.ones(grid.shape, dtype=bool)
        # Outside the driver's own data window they are not running.
        if len(pt):
            on_track &= (grid >= pt[0] - 1.0) & (grid <= pt[-1] + 1.0)

        cdf = car_data.get(num)
        if cdf is not None and len(cdf):
            ct = _to_seconds(cdf["SessionTime"])

            def _cc(col):
                if col not in cdf:
                    return np.full(grid.shape, np.nan, dtype=np.float32)
                return _interp_continuous(
                    grid, ct, cdf[col].to_numpy(dtype=np.float64))

            def _cd(col):
                if col not in cdf:
                    return np.zeros(grid.shape, dtype=np.float32)
                vals = cdf[col].to_numpy()
                if vals.dtype == bool:
                    vals = vals.astype(np.float64)
                return _interp_discrete(grid, ct, vals.astype(np.float64))

            speed, throttle, rpm = _cc("Speed"), _cc("Throttle"), _cc("RPM")
            brake, gear, drs = _cd("Brake"), _cd("nGear"), _cd("DRS")
        else:
            nanv = np.full(grid.shape, np.nan, dtype=np.float32)
            zerov = np.zeros(grid.shape, dtype=np.float32)
            speed = throttle = rpm = nanv
            brake = gear = drs = zerov

        # -- laps completed, current lap, rolling best ---------------------
        dl = laps[laps["Driver"] == info.abbr] if "Driver" in laps else laps.iloc[0:0]
        lap_end = _to_seconds(dl["Time"]) if len(dl) else np.zeros(0)
        lap_start = _to_seconds(dl["LapStartTime"]) if len(dl) else np.zeros(0)
        lap_times = _to_seconds(dl["LapTime"]) if len(dl) else np.zeros(0)
        comps = [None if pd.isna(c) else str(c)
                 for c in (dl["Compound"] if len(dl) else [])]

        valid = np.isfinite(lap_end)
        ends_sorted = np.sort(lap_end[valid]) if valid.any() else np.zeros(0)
        completed = np.searchsorted(ends_sorted, grid, side="right").astype(
            np.float64)

        # Rolling personal best: the best lap time among laps *finished* by
        # each frame, so the quali timing screen updates as laps land.
        if valid.any():
            order = np.argsort(lap_end[valid], kind="stable")
            lt_sorted = lap_times[valid][order]
            lt_sorted = np.where(np.isfinite(lt_sorted), lt_sorted, np.inf)
            running_best = np.minimum.accumulate(lt_sorted)
            idx = np.searchsorted(ends_sorted, grid, side="right") - 1
            best = np.where(idx >= 0,
                            running_best[np.clip(idx, 0, len(running_best) - 1)],
                            np.inf)
        else:
            best = np.full(grid.shape, np.inf)
        best = np.where(np.isfinite(best), best, np.nan).astype(np.float32)

        # -- track progress ------------------------------------------------
        if track_len > 0 and race_like:
            s_pos = _project_progress(gx.astype(np.float64),
                                      gy.astype(np.float64), cx, cy, cs)
            frac = s_pos / track_len
            prog = completed + frac
            # Projection noise around the start line can momentarily read as
            # "nearly a full lap ahead" the instant after the counter ticks
            # over.  Cars only ever move forwards, so clamp to monotonic.
            prog = np.maximum.accumulate(prog)
        else:
            prog = completed.copy()
        prog = prog.astype(np.float32)

        lap_no = np.clip(completed + 1, 1, None).astype(np.int16)

        # Pair each stint compound with the lap it started on, then sort the
        # pair together — searchsorted needs an ascending key array and the
        # compound list has to stay aligned with it.
        if len(lap_start) and len(comps) == len(lap_start):
            ok = np.isfinite(lap_start)
            starts_valid = lap_start[ok]
            comps_valid = [c for c, keep in zip(comps, ok) if keep]
            order = np.argsort(starts_valid, kind="stable")
            starts_valid = starts_valid[order]
            comps_valid = [comps_valid[j] for j in order]
        else:
            starts_valid = np.zeros(0)
            comps_valid = []
        comp_seq = _run_length_compounds(grid, starts_valid, comps_valid)

        finished_at = float(pt[-1]) if len(pt) else None

        drivers.append(ReplayDriver(
            info=info, x=gx, y=gy, progress=prog, on_track=on_track,
            speed=speed, throttle=throttle, brake=brake, gear=gear,
            rpm=rpm, drs=drs, lap_number=lap_no, compound=comp_seq,
            best_lap=best, finished_at=finished_at,
        ))

    if not drivers:
        raise ValueError("No driver had usable position data for this session.")

    _say("Marking corners…")
    corners = []
    try:
        ci = session.get_circuit_info()
        if ci is not None and ci.corners is not None:
            for _, row in ci.corners.iterrows():
                letter = row.get("Letter") or ""
                corners.append((float(row["X"]), float(row["Y"]),
                                f"{int(row['Number'])}{letter}"))
    except Exception:
        corners = []

    event_name, year, rnd, circuit = _event_meta(session)

    rp = Replay(
        year=year, round_no=rnd, event=event_name,
        session_code=_session_code(session),
        session_name=session_display_name(session),
        circuit=circuit, hz=float(hz),
        t=grid.astype(np.float32), drivers=drivers,
        track_x=cx.astype(np.float32), track_y=cy.astype(np.float32),
        track_length=track_len, corners=corners,
        total_laps=_safe_total_laps(session), race_like=race_like,
    )

    _say("Caching replay…")
    _store_replay(rp)
    return rp


def get_replay(year: int, round_no: int, code: str,
               hz: float = DEFAULT_REPLAY_HZ,
               team_colors: Optional[dict] = None,
               progress: Optional[Callable[[str], None]] = None) -> Replay:
    """Cached replay for a session, building it if we've not seen it."""
    cached = load_cached_replay(year, round_no, code, hz)
    if cached is not None:
        if progress:
            try:
                progress("Loaded replay from cache.")
            except Exception:
                pass
        return cached
    session = load_session(year, round_no, code, with_telemetry=True,
                           progress=progress)
    return build_replay(session, hz=hz, team_colors=team_colors,
                        progress=progress)
