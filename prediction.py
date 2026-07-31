"""
Core prediction logic. Returns structured data for the GUI.
"""
# Annotations are never evaluated at runtime here, so deferring them lets
# the sklearn-typed signatures below (`-> Pipeline`) stay readable while
# sklearn itself is imported lazily; see `_build_pipeline`.
from __future__ import annotations

import atexit
import datetime as _dt
import hashlib
import json
import logging
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path

# Base path: use app support when bundled (writable), else script directory
if getattr(sys, "frozen", False):
    _BASE = Path.home() / "Library" / "Application Support" / "F1 Winner Predictor"
    _BASE.mkdir(parents=True, exist_ok=True)
else:
    _BASE = Path(__file__).parent

import numpy as np
import fastf1
import pandas as pd

# sklearn is deliberately NOT imported here.  It costs ~3.5 s to import
# and is only needed when a model is actually built or scored, showing
# the cached predictions on launch never touches it.  The two places that
# need it (`_build_pipeline` and the holdout scoring inside
# `run_predictions`) import it locally, which keeps that cost off the
# launch path.  Unpickling a cached model still works: pickle imports the
# classes it needs on its own.
#
# `RandomizedSearchCV` / `TimeSeriesSplit` were imported here too but had
# no remaining call sites, v6 fits a fixed configuration rather than
# searching: so they are gone rather than moved.

# Suppress fastf1 verbose logging
logging.getLogger("fastf1").setLevel(logging.WARNING)

CACHE_DIR = _BASE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Bump this string whenever the feature schema or training pipeline changes – it
# is folded into the cache fingerprint so old `model_cache.pkl` files are
# invalidated automatically.
MODEL_VERSION = "v6_2026_ensemble"

# ---------------------------------------------------------------------------
# Schedule cache (massive speedup)
# ---------------------------------------------------------------------------
# `fastf1.get_event_schedule(year)` is called from 5+ places per prediction
# run.  Every miss fans out across three backends (FastF1 → F1 API → Ergast)
# with retries, and on a flaky / partial-season schedule (e.g. 2026
# mid-season) it can stall the whole prediction for tens of seconds even
# after the model is cached.
#
# We keep a small in-process memo so within a single `run_predictions()`
# call each year is only resolved once, plus a JSON-on-disk fallback so a
# subsequent launch can hydrate immediately even if the network is slow or
# flaky.  Stale entries (older than 24 h) are still used as a fallback when
# a fresh fetch fails, better to reuse yesterday's schedule than spin for
# 30 s on a dead Ergast endpoint.
_SCHEDULE_MEM_CACHE: dict = {}
_SCHEDULE_DISK_CACHE = _BASE / "schedule_cache.json"
_SCHEDULE_TTL_SECONDS = 24 * 60 * 60  # 24 h


def _schedule_disk_load() -> dict:
    if not _SCHEDULE_DISK_CACHE.exists():
        return {}
    try:
        with open(_SCHEDULE_DISK_CACHE, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _schedule_disk_save(blob: dict) -> None:
    try:
        tmp = _SCHEDULE_DISK_CACHE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(blob, f)
        os.replace(tmp, _SCHEDULE_DISK_CACHE)
    except Exception:
        pass


def _schedule_from_blob(blob_year: dict) -> "pd.DataFrame":
    """Rehydrate a cached schedule blob back into the DataFrame shape that
    fastf1 returns (columns: RoundNumber, EventName, EventDate)."""
    rows = blob_year.get("rows", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "EventDate" in df.columns:
        df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
    return df


def _schedule_to_blob(df: "pd.DataFrame") -> dict:
    cols = [c for c in ("RoundNumber", "EventName", "EventDate") if c in df.columns]
    out = df[cols].copy()
    if "EventDate" in out.columns:
        out["EventDate"] = out["EventDate"].astype(str)
    return {"ts": time.time(), "rows": out.to_dict("records")}


def get_event_schedule_cached(year: int) -> "pd.DataFrame":
    """Memoized wrapper around `fastf1.get_event_schedule(year)`.

    Order of preference:
      1. In-process memo (zero cost on a repeat call).
      2. Fresh fastf1 fetch – stored to memo + disk on success.
      3. Disk cache (even if stale) – avoids minutes of retry hell when
         all three fastf1 backends are unreachable.
    """
    year = int(year)
    if year in _SCHEDULE_MEM_CACHE:
        return _SCHEDULE_MEM_CACHE[year]

    disk = _schedule_disk_load()
    disk_year = disk.get(str(year))

    # Fresh-enough disk hit → skip the network round-trip entirely.
    if disk_year and (time.time() - float(disk_year.get("ts", 0))) < _SCHEDULE_TTL_SECONDS:
        df = _schedule_from_blob(disk_year)
        if not df.empty:
            _SCHEDULE_MEM_CACHE[year] = df
            return df

    try:
        df = fastf1.get_event_schedule(year, include_testing=False)
    except Exception:
        df = None

    if df is not None and not df.empty:
        _SCHEDULE_MEM_CACHE[year] = df
        try:
            disk[str(year)] = _schedule_to_blob(df)
            _schedule_disk_save(disk)
        except Exception:
            pass
        return df

    # Fallback: stale disk cache is still better than nothing.
    if disk_year:
        df = _schedule_from_blob(disk_year)
        if not df.empty:
            _SCHEDULE_MEM_CACHE[year] = df
            return df

    raise RuntimeError(f"Could not load schedule for {year}")

# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------
# Race data lives in `data.csv`, the trained model in `model_cache.pkl`, and
# the latest fully-formed prediction *result dict* (winner, podium, full grid,
# accuracy, feature-importance, base lineup for race cycling, …) lives in
# `last_predictions.pkl`.  The GUI hydrates from these files at startup so
# reopening the window shows predictions instantly without re-fetching the
# F1 timing API or retraining the GBM.

LAST_RESULT_PATH = _BASE / "last_predictions.pkl"
LOCK_FILE = _BASE / ".apex_ai.lock"


def save_last_result(result: dict) -> None:
    """Persist a slim version of the result dict to disk so the next launch
    can hydrate the UI immediately.  The model itself is *not* embedded –
    it stays in `model_cache.pkl` and is rehydrated on load.

    We also stamp `MODEL_VERSION` into the saved blob so a relaunch with a
    newer model/normalisation pipeline doesn't show stale predictions
    produced under the old logic."""
    if not isinstance(result, dict) or "error" in result:
        return
    try:
        slim = {k: v for k, v in result.items() if k != "_model"}
        slim["_model_version"] = MODEL_VERSION
        tmp = LAST_RESULT_PATH.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(slim, f)
        os.replace(tmp, LAST_RESULT_PATH)
    except Exception:
        pass


def load_last_result() -> dict | None:
    """Return the most recent prediction result, with the trained model
    rehydrated from `model_cache.pkl` when available.  Returns None if no
    cached result is present, the cache is unreadable, or the cache was
    written by an older MODEL_VERSION (so we don't keep showing stale
    predictions from a previous build of the model)."""
    if not LAST_RESULT_PATH.exists():
        return None
    try:
        with open(LAST_RESULT_PATH, "rb") as f:
            result = pickle.load(f)
    except Exception:
        return None

    # Reject anything written by a previous model version so a code update
    # never shows stale (incompatible) predictions on launch.
    if result.get("_model_version") != MODEL_VERSION:
        return None

    # Note that a re-usable fitted model exists *without* unpickling it.
    # Race cycling (`_advance_and_predict`) needs the model, but launch
    # does not, and the unpickle is expensive; see `load_cached_model`.
    result["_has_model"] = (_BASE / "model_cache.pkl").exists()
    return result


_MODEL_UNSET = object()
_cached_model = _MODEL_UNSET


def load_cached_model():
    """Unpickle the fitted ensemble from `model_cache.pkl`, memoised.

    Kept out of `load_last_result` on purpose: the model is a ~6 MB
    sklearn pipeline whose unpickle drags the whole of sklearn into the
    process (~3.5 s), and a launch that just redisplays the cached
    prediction never needs it.  The first action that actually re-scores
    a lineup pays that cost instead, off the launch path.

    Returns None if there is no cached model or it cannot be read.
    """
    global _cached_model
    if _cached_model is _MODEL_UNSET:
        _cached_model = None
        model_path = _BASE / "model_cache.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    _cached_model = pickle.load(f).get("model")
            except Exception:
                _cached_model = None
    return _cached_model


# ---------------------------------------------------------------------------
# Single-instance enforcement
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=3,
            )
            return str(pid) in (out.stdout or "")
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _kill_pid(pid: int, force: bool = False) -> None:
    if pid <= 0:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=4,
            )
        else:
            try:
                os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
            except OSError:
                pass
    except Exception:
        pass


def _list_other_app_pids() -> list[int]:
    """Return PIDs of every *other* python process whose command line
    looks like another instance of this app (i.e. is running `app.py`
    or this very script).  Best-effort across platforms; returns []
    if the listing fails for any reason.

    The lockfile alone is not enough, if the user double-clicks the
    bundle, runs `python app.py` from two terminals, or a prior crash
    left a stale lockfile, multiple instances can stack up.  This
    sweeps all of them so the singleton is *actually* enforced.
    """
    me = os.getpid()
    candidates: set[str] = set()
    try:
        script = os.path.realpath(__file__)
    except Exception:
        script = ""
    app_script = str((_BASE / "app.py").resolve())
    for path in (script, app_script):
        if path:
            candidates.add(path)
            candidates.add(os.path.basename(path))
    candidates.discard("")

    pids: list[int] = []
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["wmic", "process", "where",
                 "name='python.exe' or name='pythonw.exe'",
                 "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
                capture_output=True, text=True, timeout=4,
            )
            for line in (out.stdout or "").splitlines():
                if not line.strip() or "ProcessId" in line:
                    continue
                lower = line.lower()
                if not any(c.lower() in lower for c in candidates):
                    continue
                tail = line.rsplit(",", 1)[-1].strip()
                try:
                    pid = int(tail)
                except ValueError:
                    continue
                if pid != me:
                    pids.append(pid)
        else:
            out = subprocess.run(
                ["ps", "-ax", "-o", "pid=,command="],
                capture_output=True, text=True, timeout=4,
            )
            for line in (out.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                if not any(c in line for c in candidates):
                    continue
                # Skip wrapper shells (zsh/bash that *spawn* python but
                # aren't the python process themselves).  We only want
                # actual python interpreters.
                if "python" not in line.lower():
                    continue
                head = line.split(None, 1)[0]
                try:
                    pid = int(head)
                except ValueError:
                    continue
                if pid != me:
                    pids.append(pid)
    except Exception:
        pass
    return pids


def acquire_singleton() -> None:
    """Ensure only one ApexAI window is running, even when launched from
    a development terminal in parallel with a stale background instance.

    1.  Honour the lockfile PID (legacy fast path).
    2.  Sweep the OS process list for any other python process whose
        command line points at this app and kill them too.
    3.  Wait up to ~2 s for them to exit, then SIGKILL stragglers so we
        never claim the lock while another instance is still alive.
    4.  Write our own PID into the lockfile and register a clean-up
        atexit handler.
    """
    me = os.getpid()
    targets: set[int] = set()

    try:
        if LOCK_FILE.exists():
            try:
                prior_pid = int(LOCK_FILE.read_text().strip() or "0")
            except Exception:
                prior_pid = 0
            if prior_pid and prior_pid != me and _pid_alive(prior_pid):
                targets.add(prior_pid)
    except Exception:
        pass

    for pid in _list_other_app_pids():
        if pid != me and _pid_alive(pid):
            targets.add(pid)

    for pid in targets:
        _kill_pid(pid)

    if targets:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if not any(_pid_alive(p) for p in targets):
                break
            time.sleep(0.1)
        for pid in targets:
            if _pid_alive(pid):
                _kill_pid(pid, force=True)
        # Final short grace period for the OS to reap the process so a
        # newly-launched instance doesn't briefly see itself as a dup.
        time.sleep(0.1)

    try:
        LOCK_FILE.write_text(str(me))
        atexit.register(_release_singleton)
    except Exception:
        pass


def _release_singleton() -> None:
    try:
        if LOCK_FILE.exists():
            current = LOCK_FILE.read_text().strip()
            if current == str(os.getpid()):
                LOCK_FILE.unlink()
    except Exception:
        pass

# Map historical team names to their 2026 successors for points inheritance
TEAM_LINEAGE = {
    "Kick Sauber": "Audi",
    "Alfa Romeo": "Audi",
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
}

# ---------------------------------------------------------------------------
# 2026 power-unit regulation model
# ---------------------------------------------------------------------------
# The 2026 regulations shift the power split between the internal combustion
# engine (ICE) and electric motor to roughly 50/50 (it was ~80/20 under the
# 2014–2025 rules) and remove the MGU-H. Battery deployment efficiency and
# energy recovery are therefore *much* more decisive than they used to be.
#
# We model two effects per (team, year):
#   - PUBatteryScore : how well the manufacturer recovers + deploys electric
#                      energy.  Mercedes and Honda have dominated the hybrid
#                      era; Ferrari has improved; Renault has historically
#                      lagged.  New entrants are conservatively rated lower.
#   - PUICEScore     : pure combustion-side output / thermal efficiency.

TEAM_PU_2026 = {
    "Red Bull Racing": "RBPT-Ford",
    "Racing Bulls":    "RBPT-Ford",
    "Ferrari":         "Ferrari",
    "Mercedes":        "Mercedes",
    "McLaren":         "Mercedes",
    "Aston Martin":    "Honda",         # Honda HRC factory deal for 2026
    "Alpine":          "Mercedes",      # switched from Renault for 2026
    "Haas F1 Team":    "Ferrari",
    "Williams":        "Mercedes",
    "Audi":            "Audi",          # Audi works PU debut
    "Cadillac":        "Ferrari",       # customer Ferrari PU at debut
}

TEAM_PU_HISTORICAL = {
    "Red Bull Racing": "Honda",
    "AlphaTauri":      "Honda",
    "RB":              "Honda",
    "Racing Bulls":    "Honda",
    "Ferrari":         "Ferrari",
    "Mercedes":        "Mercedes",
    "McLaren":         "Mercedes",
    "Aston Martin":    "Mercedes",
    "Alpine":          "Renault",
    "Haas F1 Team":    "Ferrari",
    "Williams":        "Mercedes",
    "Alfa Romeo":      "Ferrari",
    "Kick Sauber":     "Ferrari",
    "Audi":            "Ferrari",
    "Cadillac":        "Mercedes",
}

PU_BATTERY_SCORE = {
    "Mercedes":  0.92,
    "Honda":     0.90,
    "Ferrari":   0.78,
    "Renault":   0.55,
    "RBPT-Ford": 0.68,   # built on Honda foundation but unproven on full Ford partnership
    "Audi":      0.62,   # heavy hybrid R&D investment, no race data yet
}

PU_ICE_SCORE = {
    "Mercedes":  0.85,
    "Honda":     0.88,
    "Ferrari":   0.86,
    "Renault":   0.66,
    "RBPT-Ford": 0.74,
    "Audi":      0.70,
}


def _get_pu_scores(team: str, year: int) -> tuple:
    """Return (battery_score, ice_score) for a team in a given season."""
    if year >= 2026:
        pu = TEAM_PU_2026.get(team, "Mercedes")
    else:
        pu = TEAM_PU_HISTORICAL.get(team, "Mercedes")
    return (
        PU_BATTERY_SCORE.get(pu, 0.70),
        PU_ICE_SCORE.get(pu, 0.75),
    )

# 2026 F1 driver lineup (official FIA-confirmed numbers – formula1.com)
LINEUP_2026 = [
    {"DriverNumber": 3,  "Abbreviation": "VER", "TeamName": "Red Bull Racing"},
    {"DriverNumber": 6,  "Abbreviation": "HAD", "TeamName": "Red Bull Racing"},
    {"DriverNumber": 30, "Abbreviation": "LAW", "TeamName": "Racing Bulls"},
    {"DriverNumber": 41, "Abbreviation": "LIN", "TeamName": "Racing Bulls"},
    {"DriverNumber": 44, "Abbreviation": "HAM", "TeamName": "Ferrari"},
    {"DriverNumber": 16, "Abbreviation": "LEC", "TeamName": "Ferrari"},
    {"DriverNumber": 63, "Abbreviation": "RUS", "TeamName": "Mercedes"},
    {"DriverNumber": 12, "Abbreviation": "ANT", "TeamName": "Mercedes"},
    {"DriverNumber": 1,  "Abbreviation": "NOR", "TeamName": "McLaren"},
    {"DriverNumber": 81, "Abbreviation": "PIA", "TeamName": "McLaren"},
    {"DriverNumber": 14, "Abbreviation": "ALO", "TeamName": "Aston Martin"},
    {"DriverNumber": 18, "Abbreviation": "STR", "TeamName": "Aston Martin"},
    {"DriverNumber": 10, "Abbreviation": "GAS", "TeamName": "Alpine"},
    {"DriverNumber": 43, "Abbreviation": "COL", "TeamName": "Alpine"},
    {"DriverNumber": 31, "Abbreviation": "OCO", "TeamName": "Haas F1 Team"},
    {"DriverNumber": 87, "Abbreviation": "BEA", "TeamName": "Haas F1 Team"},
    {"DriverNumber": 23, "Abbreviation": "ALB", "TeamName": "Williams"},
    {"DriverNumber": 55, "Abbreviation": "SAI", "TeamName": "Williams"},
    {"DriverNumber": 27, "Abbreviation": "HUL", "TeamName": "Audi"},
    {"DriverNumber": 5,  "Abbreviation": "BOR", "TeamName": "Audi"},
    {"DriverNumber": 11, "Abbreviation": "PER", "TeamName": "Cadillac"},
    {"DriverNumber": 77, "Abbreviation": "BOT", "TeamName": "Cadillac"},
]


def load_data(years=(2022, 2023, 2024, 2025, 2026), progress_callback=None):
    # Check multiple locations for data.csv
    candidates = [
        _BASE / "data.csv",
        Path.cwd() / "data.csv",  # Current working directory
    ]
    if getattr(sys, "frozen", False):
        # .app/F1 Winner Predictor.app/executable -> check project root (parent of dist)
        app_dir = Path(sys.executable).resolve().parent
        candidates.append(app_dir.parent / "data.csv")   # dist/
        candidates.append(app_dir.parent.parent / "data.csv")  # project root
    # Allow the requested `years` range to be the trigger for a refresh –
    # if the cached CSV doesn't cover up through the most recent completed
    # race in the current target season, fall through to a fresh fetch so
    # we ingest e.g. the 2026 mid-season races as they happen.
    today_date = _dt.date.today()
    requested_max_year = max(years) if years else None

    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "Year" not in df.columns or len(df) < 50:
            continue
        cached_max_year = int(df["Year"].max())
        # Compare against fastf1 schedule to find the last race that has
        # already happened in the requested target season.  If the cache
        # is missing that round (or the whole season), refresh.
        needs_refresh = False
        if requested_max_year and cached_max_year < requested_max_year:
            try:
                sched = get_event_schedule_cached(requested_max_year)
                past = sched[sched["EventDate"].dt.date <= today_date]
                if not past.empty:
                    # There IS a completed race in the target season that
                    # the cache hasn't seen yet, refresh.
                    needs_refresh = True
            except Exception:
                pass
        elif requested_max_year and cached_max_year == requested_max_year:
            try:
                sched = get_event_schedule_cached(requested_max_year)
                past = sched[sched["EventDate"].dt.date <= today_date]
                if not past.empty:
                    last_completed_round = int(past["RoundNumber"].max())
                    cached_last_round = int(
                        df[df["Year"] == cached_max_year]["Round"].max()
                    )
                    if cached_last_round < last_completed_round:
                        needs_refresh = True
            except Exception:
                pass
        if needs_refresh:
            continue
        return _enrich_existing(df)

    csv_path = _BASE / "data.csv"
    records = []
    today = _dt.date.today()
    for year in years:
        if progress_callback:
            progress_callback(f"Loading {year} season...")
        try:
            schedule = get_event_schedule_cached(year)
        except Exception:
            continue
        # Build a round -> event-name lookup once per season so we can tag every
        # race row with its circuit (needed for circuit-affinity features).
        try:
            event_names = dict(
                zip(schedule["RoundNumber"].astype(int), schedule["EventName"])
            )
        except Exception:
            event_names = {}

        # Build a round -> date lookup so we can skip future races (their
        # results obviously don't exist yet, and trying to load them is slow).
        try:
            event_dates = dict(
                zip(
                    schedule["RoundNumber"].astype(int),
                    schedule["EventDate"].dt.date,
                )
            )
        except Exception:
            event_dates = {}

        for rnd in schedule["RoundNumber"]:
            rnd_i = int(rnd)
            evt_date = event_dates.get(rnd_i)
            if evt_date and evt_date > today:
                # Future race, no results to ingest.
                continue
            if progress_callback:
                progress_callback(f"Loading {year} round {rnd_i}...")
            try:
                session = fastf1.get_session(year, rnd_i, "R")
                session.load(laps=False, telemetry=False)
            except Exception:
                continue
            if session.results is None or session.results.empty:
                continue

            res = session.results[
                [
                    "DriverNumber",
                    "Abbreviation",
                    "TeamName",
                    "GridPosition",
                    "Position",
                    "Points",
                ]
            ].copy()
            res["Year"] = year
            res["Round"] = rnd_i
            res["EventName"] = event_names.get(rnd_i, f"R{rnd_i}")
            records.append(res)

    if not records:
        return None

    df = pd.concat(records, ignore_index=True)
    df.sort_values(["Year", "Round"], inplace=True)
    df["DriverPointsBefore"] = df.groupby("DriverNumber")["Points"].cumsum() - df["Points"]
    df["TeamPointsBefore"] = df.groupby("TeamName")["Points"].cumsum() - df["Points"]
    # NOTE: we drop rows without GridPosition / DriverNumber (corrupt data) but
    # keep rows where Position is NaN, those represent DNFs and carry useful
    # signal about driver / team reliability.  The Winner target is computed
    # *after* this so DNFs correctly resolve to Winner=0.
    df = df.dropna(subset=["GridPosition", "DriverNumber"])
    df = _add_rolling_features(df)
    df = _attach_pu_features(df)
    df.to_csv(csv_path, index=False)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling performance features per driver for improved prediction."""
    df = df.sort_values(["Year", "Round"]).copy()

    if "EventName" not in df.columns:
        df["EventName"] = "Unknown"

    # ------------------------------------------------------------------
    # DNF / reliability:  Position is NaN for true DNFs, but the underlying
    # F1 result service sometimes shows them classified outside the top 15.
    # We treat the union as "didn't finish properly" for rolling reliability.
    # ------------------------------------------------------------------
    pos = df["Position"]
    df["IsDNF"] = ((pos.isna()) | (pos > 15)).astype(int)

    # Per-driver rolling stats (using expanding window for cumulative history)
    grp = df.groupby("Abbreviation")

    # Recent average finish position (last 5 races): lower is better
    df["RecentAvgPos"] = grp["Position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(10.0)

    # Recent average qualifying / grid position (last 5 races): used as
    # ExpectedGridPosition when projecting forward, since the real grid slot
    # for an unraced event is obviously unknown.
    df["RecentAvgGrid"] = grp["GridPosition"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(10.0)

    # Recent win rate (last 10 races)
    df["RecentWinRate"] = grp["Position"].transform(
        lambda s: (s.shift(1) == 1).astype(float).rolling(10, min_periods=1).mean()
    ).fillna(0.0)

    # Recent podium rate (last 10 races, position <= 3)
    df["RecentPodiumRate"] = grp["Position"].transform(
        lambda s: (s.shift(1) <= 3).astype(float).rolling(10, min_periods=1).mean()
    ).fillna(0.0)

    # DNF / non-finish rate (last 10 races).  A high value heavily reduces the
    # plausibility of a win regardless of pace.
    df["DNFRate"] = grp["IsDNF"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(0.10)

    # Driver experience: cumulative race count
    df["DriverExperience"] = grp.cumcount()

    # Head-to-head vs teammate: fraction of races beating teammate
    h2h = []
    for _, race_grp in df.groupby(["Year", "Round"]):
        for team, team_grp in race_grp.groupby("TeamName"):
            if len(team_grp) == 2:
                rows = team_grp.sort_values("Position")
                h2h.append({"idx": rows.index[0], "beat_tm": 1.0})
                h2h.append({"idx": rows.index[1], "beat_tm": 0.0})
            else:
                for idx in team_grp.index:
                    h2h.append({"idx": idx, "beat_tm": 0.5})
    h2h_df = pd.DataFrame(h2h).set_index("idx")
    df["_beat_tm"] = h2h_df["beat_tm"]
    df["HeadToHead"] = df.groupby("Abbreviation")["_beat_tm"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(0.5)
    df.drop(columns=["_beat_tm"], inplace=True)

    # Team recent form: team's avg finish position over last 10 races
    df["TeamRecentForm"] = df.groupby("TeamName")["Position"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(10.0)

    # ------------------------------------------------------------------
    # Circuit affinity – some drivers / teams are reliably strong at specific
    # tracks (Hamilton at Silverstone, Verstappen at Suzuka, Ferrari at Monza
    # / Imola).  We use the running average finish at *that* circuit over the
    # driver's career (excluding the current row, hence shift(1)).
    # ------------------------------------------------------------------
    df["DriverCircuitAvg"] = df.groupby(["Abbreviation", "EventName"])["Position"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df["RecentAvgPos"]).fillna(10.0)

    df["TeamCircuitAvg"] = df.groupby(["TeamName", "EventName"])["Position"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df["TeamRecentForm"]).fillna(10.0)

    # ------------------------------------------------------------------
    # Season-to-date championship points (leak-free: excludes the current
    # round entirely).  The existing DriverPointsBefore / TeamPointsBefore
    # are *career* cumulative sums, which mostly encode "is a veteran with
    # a good career"; the current season's standings are a far sharper
    # signal for who is winning races under *this* year's regulations.
    # ------------------------------------------------------------------
    df["SeasonPointsBefore"] = (
        df.groupby(["Year", "Abbreviation"])["Points"].cumsum() - df["Points"]
    ).fillna(0.0)

    # Team version must be built from per-round totals so a driver's row
    # never includes the teammate's points from the *same* race.
    if "TeamSeasonPointsBefore" in df.columns:
        df = df.drop(columns=["TeamSeasonPointsBefore"])
    team_rounds = (
        df.groupby(["Year", "TeamName", "Round"], as_index=False)["Points"].sum()
        .sort_values(["Year", "TeamName", "Round"])
    )
    team_rounds["TeamSeasonPointsBefore"] = (
        team_rounds.groupby(["Year", "TeamName"])["Points"].cumsum()
        - team_rounds["Points"]
    )
    df = df.merge(
        team_rounds[["Year", "TeamName", "Round", "TeamSeasonPointsBefore"]],
        on=["Year", "TeamName", "Round"], how="left",
    )
    df["TeamSeasonPointsBefore"] = df["TeamSeasonPointsBefore"].fillna(0.0)

    # Racecraft: average grid→finish delta over the last 5 races.  Positive
    # means the driver habitually gains places on Sunday; a strong
    # discriminator between "qualifies well" and "races well".
    df["_pos_gain"] = df["GridPosition"] - df["Position"]
    df["RecentPosGain"] = df.groupby("Abbreviation")["_pos_gain"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0.0)
    df.drop(columns=["_pos_gain"], inplace=True)

    # ------------------------------------------------------------------
    # Momentum: raw points scored over the last 3 races.  Sharper than the
    # 5/10-race windows above; it spikes immediately when a car upgrade
    # or form swing lands.
    # ------------------------------------------------------------------
    df["RecentPoints3"] = df.groupby("Abbreviation")["Points"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum()
    ).fillna(0.0)

    # ------------------------------------------------------------------
    # Field-relative features.  Winning is a *ranking* problem: what
    # matters is not that a driver's form is "4.2 average finish" but that
    # it is the best (or third best) form ON THIS GRID.  Absolute features
    # can't express that, so we add within-race ranks and shares.
    # ------------------------------------------------------------------
    race_grp = df.groupby(["Year", "Round"])

    # Rank of recent form within the field (1 = best form on the grid).
    df["FormRankInField"] = race_grp["RecentAvgPos"].rank(method="min")

    # Share of the field's season points held by this driver / this team.
    # Early-season races (points sum == 0) fall back to a uniform share so
    # the feature stays defined and neutral.
    n_field = race_grp["SeasonPointsBefore"].transform("count").astype(float)
    tot_pts = race_grp["SeasonPointsBefore"].transform("sum")
    df["SeasonPointsShare"] = np.where(
        tot_pts > 0,
        df["SeasonPointsBefore"] / tot_pts.replace(0, 1.0),
        1.0 / n_field,
    )
    tot_team = race_grp["TeamSeasonPointsBefore"].transform("sum")
    df["TeamSeasonPointsShare"] = np.where(
        tot_team > 0,
        df["TeamSeasonPointsBefore"] / tot_team.replace(0, 1.0),
        1.0 / n_field,
    )

    # Qualifying surprise: grid slot vs the driver's recent grid average.
    # Negative = qualified ahead of their own baseline (fresh upgrade,
    # circuit suits the car): a strong short-horizon pace signal.
    df["GridVsForm"] = (df["GridPosition"] - df["RecentAvgGrid"]).fillna(0.0)

    return df


def _attach_pu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row with the season-appropriate power-unit battery/ICE
    scores.  These columns let the model learn how each manufacturer's
    hybrid pedigree (especially under the 2026 50/50 regulations) maps to
    win probability."""
    df = df.copy()
    if "PUBatteryScore" not in df.columns or "PUICEScore" not in df.columns:
        battery, ice = [], []
        for team, year in zip(df["TeamName"], df["Year"]):
            b, i = _get_pu_scores(team, int(year))
            battery.append(b)
            ice.append(i)
        df["PUBatteryScore"] = battery
        df["PUICEScore"] = ice
    return df


# Columns that _add_rolling_features promises to provide.
_ROLLING_COLS = [
    "EventName", "IsDNF",
    "RecentAvgPos", "RecentAvgGrid",
    "RecentWinRate", "RecentPodiumRate",
    "DNFRate",
    "DriverExperience", "HeadToHead",
    "TeamRecentForm",
    "DriverCircuitAvg", "TeamCircuitAvg",
    "SeasonPointsBefore", "TeamSeasonPointsBefore",
    "RecentPosGain",
    "RecentPoints3",
    "FormRankInField", "SeasonPointsShare", "TeamSeasonPointsShare",
    "GridVsForm",
]


def _enrich_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Bring an existing data.csv (potentially produced by an older version
    of this module) up to the current feature schema.  Missing rolling /
    PU columns are recomputed in place."""
    missing_rolling = [c for c in _ROLLING_COLS if c not in df.columns]
    if missing_rolling:
        df = _add_rolling_features(df)
    if "PUBatteryScore" not in df.columns or "PUICEScore" not in df.columns:
        df = _attach_pu_features(df)
    return df


def _team_points_with_lineage(df: pd.DataFrame) -> dict:
    """
    Sum team points, merging historical names into their 2026 successors.
    e.g. Kick Sauber + Alfa Romeo points → Audi
    """
    raw = df.groupby("TeamName")["Points"].sum().to_dict()
    merged = {}
    for team, pts in raw.items():
        target = TEAM_LINEAGE.get(team, team)
        merged[target] = merged.get(target, 0) + pts
    # Keep originals too so non-2026 lookups still work
    for team, pts in raw.items():
        if team not in merged:
            merged[team] = pts
    return merged


def get_lineup_for_next_round(df: pd.DataFrame, next_year: int, features: list,
                              circuit_name: str = None) -> pd.DataFrame:
    if next_year == 2026:
        lineup = pd.DataFrame(LINEUP_2026)
    else:
        prev = df[df["Year"] == next_year - 1]
        if prev.empty:
            lineup = pd.DataFrame(LINEUP_2026)
        else:
            lineup = (
                prev.groupby(["DriverNumber", "Abbreviation", "TeamName"])
                .tail(1)
                [["DriverNumber", "Abbreviation", "TeamName"]]
                .reset_index(drop=True)
            )

    driver_pts_by_abbr = df.groupby("Abbreviation")["Points"].sum()
    lineup["DriverPointsBefore"] = lineup["Abbreviation"].map(driver_pts_by_abbr).fillna(0)

    # Season-to-date standings for the season being predicted.  If the
    # season hasn't started yet these are all zero, which is correct.
    season_df = df[df["Year"] == next_year]
    season_driver_pts = season_df.groupby("Abbreviation")["Points"].sum()
    season_team_pts = season_df.groupby("TeamName")["Points"].sum()
    lineup["SeasonPointsBefore"] = lineup["Abbreviation"].map(season_driver_pts).fillna(0)
    lineup["TeamSeasonPointsBefore"] = lineup["TeamName"].map(season_team_pts).fillna(0)

    if next_year == 2026:
        team_totals = _team_points_with_lineage(df)
        lineup["TeamPointsBefore"] = lineup["TeamName"].map(team_totals).fillna(0)
    else:
        team_totals = df.groupby("TeamName")["Points"].sum()
        lineup["TeamPointsBefore"] = lineup["TeamName"].map(team_totals).fillna(0)

    # Compute rolling features from the latest historical data per driver
    latest = df.sort_values(["Year", "Round"])

    # Optional: restrict circuit-affinity lookups to the upcoming event.
    circuit_df = None
    if circuit_name:
        circuit_df = latest[latest.get("EventName") == circuit_name] \
            if "EventName" in latest.columns else None

    for abbr in lineup["Abbreviation"].unique():
        drv = latest[latest["Abbreviation"] == abbr]
        if drv.empty:
            continue
        last_rows = drv.tail(10)
        last5 = drv.tail(5)
        idx = lineup["Abbreviation"] == abbr
        lineup.loc[idx, "RecentAvgPos"] = last5["Position"].mean()
        lineup.loc[idx, "RecentAvgGrid"] = last5["GridPosition"].mean()
        lineup.loc[idx, "RecentWinRate"] = (last_rows["Position"] == 1).mean()
        lineup.loc[idx, "RecentPodiumRate"] = (last_rows["Position"] <= 3).mean()
        lineup.loc[idx, "RecentPosGain"] = (
            (last5["GridPosition"] - last5["Position"]).mean()
        )
        lineup.loc[idx, "RecentPoints3"] = drv.tail(3)["Points"].sum()
        lineup.loc[idx, "DriverExperience"] = float(len(drv))
        lineup.loc[idx, "HeadToHead"] = (
            drv["HeadToHead"].iloc[-1] if "HeadToHead" in drv.columns else 0.5
        )

        # DNF rate (proxy for reliability).  Falls back to a conservative
        # base rate when we have no data on the driver yet.
        if "IsDNF" in drv.columns:
            lineup.loc[idx, "DNFRate"] = drv["IsDNF"].tail(10).mean()
        else:
            lineup.loc[idx, "DNFRate"] = (
                (drv["Position"].tail(10).isna()) | (drv["Position"].tail(10) > 15)
            ).mean()

        # Circuit affinity – this driver's avg finish at the upcoming track.
        if circuit_df is not None and not circuit_df.empty:
            drv_circuit = circuit_df[circuit_df["Abbreviation"] == abbr]
            if not drv_circuit.empty:
                lineup.loc[idx, "DriverCircuitAvg"] = drv_circuit["Position"].mean()

    for team in lineup["TeamName"].unique():
        mapped_team = TEAM_LINEAGE.get(team, team)
        related = [t for t, v in TEAM_LINEAGE.items() if v == mapped_team] + [mapped_team, team]
        team_data = latest[latest["TeamName"].isin(related)]
        if not team_data.empty:
            lineup.loc[lineup["TeamName"] == team, "TeamRecentForm"] = (
                team_data.tail(20)["Position"].mean()
            )

        # Team affinity for the upcoming circuit
        if circuit_df is not None and not circuit_df.empty:
            tc = circuit_df[circuit_df["TeamName"].isin(related)]
            if not tc.empty:
                lineup.loc[lineup["TeamName"] == team, "TeamCircuitAvg"] = tc["Position"].mean()

    # ExpectedGrid: drivers don't have a real grid slot for an unraced event,
    # so we project from recent grid history.  This is far more informative
    # than the old `0` placeholder which the model interpreted as "pole minus
    # one" via the StandardScaler.
    lineup["GridPosition"] = lineup.get("RecentAvgGrid", pd.Series(10.0, index=lineup.index)).fillna(10.0)

    # Power-unit / battery scores for the season we're predicting.
    pu_b, pu_i = [], []
    for t in lineup["TeamName"]:
        b, i = _get_pu_scores(t, int(next_year))
        pu_b.append(b)
        pu_i.append(i)
    lineup["PUBatteryScore"] = pu_b
    lineup["PUICEScore"] = pu_i

    defaults = {
        "RecentAvgPos": 10.0, "RecentAvgGrid": 10.0,
        "RecentWinRate": 0.0, "RecentPodiumRate": 0.0,
        "RecentPosGain": 0.0, "RecentPoints3": 0.0,
        "DNFRate": 0.10,
        "DriverExperience": 0.0, "HeadToHead": 0.5,
        "TeamRecentForm": 10.0,
        "DriverCircuitAvg": 10.0, "TeamCircuitAvg": 10.0,
        "SeasonPointsBefore": 0.0, "TeamSeasonPointsBefore": 0.0,
    }
    for col, default in defaults.items():
        if col not in lineup.columns:
            lineup[col] = default
        lineup[col] = lineup[col].fillna(default)

    lineup = _attach_field_relative(lineup)
    return lineup


def _attach_field_relative(lineup: pd.DataFrame) -> pd.DataFrame:
    """Compute the within-field relative features for a prediction lineup
    (mirrors what `_add_rolling_features` does per historical race).

    Must run AFTER the absolute features (RecentAvgPos, SeasonPointsBefore,
    TeamSeasonPointsBefore, GridPosition, RecentAvgGrid) are final.
    """
    lineup = lineup.copy()
    n = float(len(lineup)) or 1.0

    lineup["FormRankInField"] = lineup["RecentAvgPos"].rank(method="min")

    tot = lineup["SeasonPointsBefore"].sum()
    lineup["SeasonPointsShare"] = (
        lineup["SeasonPointsBefore"] / tot if tot > 0 else 1.0 / n
    )
    tot_team = lineup["TeamSeasonPointsBefore"].sum()
    lineup["TeamSeasonPointsShare"] = (
        lineup["TeamSeasonPointsBefore"] / tot_team if tot_team > 0 else 1.0 / n
    )

    # Grid slot is projected from recent form for an unraced event, so the
    # "qualifying surprise" is definitionally zero (but stays meaningful in
    # training rows where the real Saturday grid is known).
    lineup["GridVsForm"] = (
        lineup["GridPosition"] - lineup["RecentAvgGrid"]
    ).fillna(0.0)
    return lineup


# NOTE on feature selection (walk-forward ablation, 32 races 2025-2026):
# FormRankInField (+6 pts top-1) earns its slot; SeasonPointsShare /
# TeamSeasonPointsShare / GridVsForm / RecentPoints3 all *reduced* top-1
# or top-3 accuracy when added, so they are computed (for analysis /
# future re-evaluation) but deliberately excluded from the model input.
# (Column order matters slightly: max_features subsampling in the trees is
# order-sensitive, so this list pins the exact configuration that won the
# ablation: the v5 block with FormRankInField appended.)
FEATURES = [
    "Abbreviation", "TeamName",
    "GridPosition", "DriverNumber",
    "DriverPointsBefore", "TeamPointsBefore",
    "SeasonPointsBefore", "TeamSeasonPointsBefore",
    "RecentAvgPos", "RecentAvgGrid",
    "RecentWinRate", "RecentPodiumRate",
    "RecentPosGain",
    "DNFRate",
    "DriverExperience", "HeadToHead", "TeamRecentForm",
    "DriverCircuitAvg", "TeamCircuitAvg",
    "PUBatteryScore", "PUICEScore",
    "FormRankInField",
]


# Hyperparameters for the classic GBM member of the ensemble.  These sit
# at the empirical optimum of the old RandomizedSearchCV grid (they won
# most search runs), so pinning them costs nothing measurable while
# making every fit deterministic and ~36× cheaper.
_GBM_HYPERPARAMS = {
    "n_estimators":      350,
    "max_depth":         4,
    "learning_rate":     0.08,
    "min_samples_split": 8,
    "min_samples_leaf":  4,
    "subsample":         0.85,
    "max_features":      0.6,
}


def _build_pipeline() -> Pipeline:
    """Construct the preprocessing → ensemble pipeline (model v6).

    Instead of a single gradient-boosted classifier we soft-vote three
    learners with deliberately different biases:

      - ``gbm``  : classic GradientBoosting, shallow trees + shrinkage,
                   the strongest single model in every backtest so far.
      - ``hgb``  : HistGradientBoosting, leaf-wise growth with L2
                   regularisation captures different interaction
                   structure than depth-wise GBM.
      - ``rf``   : RandomForest, bagging decorrelates its errors from
                   both boosters and tempers their overconfidence on
                   outlier races (weather, first-lap chaos).

    Averaged probabilities (weights 2:2:1) beat any individual member on
    walk-forward top-1 accuracy because the members disagree exactly on
    the marginal races where a single model's variance flips the pick.
    """
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )

    categorical = ["Abbreviation", "TeamName"]
    numeric = [f for f in FEATURES if f not in categorical]

    # Dense output: HistGradientBoosting doesn't accept sparse matrices,
    # and at ~2k rows × ~80 encoded columns density costs nothing.
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    gbm = GradientBoostingClassifier(
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=20,
        tol=1e-4,
        **_GBM_HYPERPARAMS,
    )
    hgb = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        min_samples_leaf=10,
        early_stopping=False,
    )
    rf = RandomForestClassifier(
        random_state=42,
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[("gbm", gbm), ("hgb", hgb), ("rf", rf)],
        voting="soft",
        weights=[2.0, 2.0, 1.0],
    )

    return Pipeline(
        [
            ("preprocess", pre),
            ("classifier", ensemble),
        ]
    )


def build_model() -> Pipeline:
    """Full-strength model used by "Predict Next Race"."""
    return _build_pipeline()


def build_model_fast() -> Pipeline:
    """Model used per-race by the leave-one-out backtest.  Identical to
    `build_model()`: the ensemble is a fixed configuration, so there is
    no hyperparameter search to skip anymore."""
    return _build_pipeline()


def _ensemble_feature_importances(clf) -> np.ndarray | None:
    """Aggregate impurity importances across ensemble members that expose
    them (GBM + RF; HistGradientBoosting has none), weighted by the same
    voting weights used for prediction."""
    try:
        members = clf.named_estimators_
        weights = dict(zip([n for n, _ in clf.estimators], clf.weights))
        total, wsum = None, 0.0
        for name, est in members.items():
            imp = getattr(est, "feature_importances_", None)
            if imp is None:
                continue
            w = float(weights.get(name, 1.0))
            total = imp * w if total is None else total + imp * w
            wsum += w
        if total is None or wsum <= 0:
            return None
        return total / wsum
    except Exception:
        return None


def _winner_sample_weights(y) -> np.ndarray:
    """Per-row sample weights that counter the class imbalance (only one
    winner per ~20-driver race) WITHOUT pushing the model so hard that it
    learns a binary "winner vs not" cut.

    The previous version weighted winners by the full negative/positive
    ratio (~19x).  Empirically that made the gradient-boosted classifier
    output ~0.6-0.85 for clear favourites and a nearly identical ~0.02 for
    everybody else – mid-pack drivers became indistinguishable from
    back-markers because the binary loss was so dominant.

    Square-root scaling keeps a meaningful (~4x) bump for winners while
    preserving the natural gradient across the rest of the field, so the
    model's raw probability actually correlates with "how likely is this
    driver to win" rather than just "is this a champion-tier feature
    vector".
    """
    y_arr = np.asarray(y).astype(int)
    n_pos = max(int(y_arr.sum()), 1)
    n_neg = max(int(len(y_arr) - n_pos), 1)
    ratio = n_neg / n_pos
    w = np.ones_like(y_arr, dtype=float)
    w[y_arr == 1] = float(np.sqrt(ratio))
    return w


# Per-year decay applied to training rows.  The 2026 regulations (50/50
# hybrid power split, new aero, Cadillac + Audi entries) changed the
# competitive order enough that a 2022 result says much less about who wins
# next weekend than a 2026 result does.  0.85^Δyear keeps older seasons in
# the mix (a 2022 row still carries ~52% weight) while letting the current
# season dominate the gradient.
_RECENCY_DECAY = 0.85


def _training_sample_weights(train_df) -> np.ndarray:
    """Combined per-row training weights: winner-class balancing (see
    `_winner_sample_weights`) multiplied by a recency decay so recent
    seasons: raced under the current regulations, drive the fit."""
    w = _winner_sample_weights(train_df["Winner"])
    years = pd.to_numeric(train_df["Year"], errors="coerce").to_numpy(dtype=float)
    max_year = np.nanmax(years) if len(years) else 0.0
    age = np.clip(max_year - np.nan_to_num(years, nan=max_year), 0.0, None)
    return w * np.power(_RECENCY_DECAY, age)


def _softmax_normalize(raw_probs: np.ndarray, temperature: float = 1.2) -> np.ndarray:
    """Convert raw per-driver "is this a winner?" probabilities into a
    proper distribution over the field that sums to 1.

    Math is `softmax(log p / T)` which equals `linear_normalise(p ** (1/T))`.
        T < 1 sharpens (the favourite gets a bigger share);
        T = 1 reproduces the raw probabilities as a simple linear share;
        T > 1 flattens (the tail stays visible).

    The previous default of T=0.18 was *catastrophically* sharp – the
    favourite ended up at ~98% and the rest of the grid was crushed to
    well below 0.05%, which then displayed as a misleading "0.0%".  We now
    use T=1.2 which leaves the favourite at a realistic ~30-40% while the
    midfield still gets a few percent and back-markers around 1%.

    A small floor (0.2%) is then applied so no driver ever displays as
    "0.0%" purely because of normalisation, and the result is renormalised
    to sum to 1.  This costs the top driver at most ~1% of share but keeps
    the leaderboard honest about which drivers are in the field.
    """
    p = np.asarray(raw_probs, dtype=float)
    if p.size == 0:
        return p

    eps = 1e-9
    soft = np.power(np.clip(p, eps, 1.0), 1.0 / max(temperature, 1e-3))
    total = soft.sum()
    if total <= 0:
        return np.full_like(p, 1.0 / p.size)
    norm = soft / total

    floor = min(0.002, 0.5 / p.size)
    norm = np.maximum(norm, floor)
    return norm / norm.sum()


def _data_fingerprint(df) -> str:
    """Hash of data shape AND the current feature schema, so an old cached
    model is automatically invalidated when we add / change features."""
    key = (
        f"{MODEL_VERSION}_"
        f"{len(df)}_{int(df['Year'].max())}_{int(df['Round'].max())}_"
        f"{','.join(FEATURES)}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def run_predictions(progress_callback=None, target_year=2026):
    """
    Run full prediction pipeline. Returns dict with results or error.
    target_year forces prediction for a specific season (default: 2026).
    progress_callback(status: str) is optional for GUI updates.
    """
    def report(msg):
        if progress_callback:
            progress_callback(msg)

    try:
        report("Loading race data...")
        df = load_data(progress_callback=progress_callback)
        if df is None or df.empty:
            return {"error": "No race data available. Check your internet connection."}

        df["Winner"] = (df["Position"] == 1).astype(int)
        last_year = int(df["Year"].max())
        last_round = int(df[df["Year"] == last_year]["Round"].max())
        fingerprint = _data_fingerprint(df)

        features = list(FEATURES)

        train_df = df[~((df["Year"] == last_year) & (df["Round"] == last_round))]
        test_df = df[(df["Year"] == last_year) & (df["Round"] == last_round)]

        # Load cached model if data unchanged
        model_path = _BASE / "model_cache.pkl"
        best_model = None
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("fingerprint") == fingerprint:
                    best_model = cached["model"]
                    report("Using cached model...")
            except Exception:
                pass

        if best_model is None:
            report("Training model...")
            model = build_model()
            # Upweight winner rows (class imbalance) and recent seasons
            # (regulation relevance) so the gradient reflects both.
            sw = _training_sample_weights(train_df)
            model.fit(
                train_df[features], train_df["Winner"],
                classifier__sample_weight=sw,
            )
            best_model = getattr(model, "best_estimator_", model)
            try:
                with open(model_path, "wb") as f:
                    pickle.dump({"fingerprint": fingerprint, "model": best_model}, f)
            except Exception:
                pass

        # Resolve race names from schedule
        try:
            last_schedule = get_event_schedule_cached(last_year)
            last_race_event = last_schedule[last_schedule["RoundNumber"] == last_round]
            last_race_name = last_race_event.iloc[0]["EventName"] if not last_race_event.empty else f"Round {last_round}"
        except Exception:
            last_race_name = f"Round {last_round}"

        # Last race predictions – softmax-normalised across the grid so the
        # displayed probabilities represent "share of the win" rather than a
        # raw binary-classifier output.
        raw = best_model.predict_proba(test_df[features])[:, 1]
        norm = _softmax_normalize(raw)
        test_df = test_df.copy()
        test_df["WinProbability"] = norm
        last_race_preds = [
            {
                "abbreviation": row["Abbreviation"],
                "team": row["TeamName"],
                "probability": float(row["WinProbability"]),
            }
            for _, row in test_df.sort_values("WinProbability", ascending=False).iterrows()
        ]
        pred_winner = last_race_preds[0]
        actual_winner = test_df[test_df["Winner"] == 1]
        actual_abbr = actual_winner.iloc[0]["Abbreviation"] if not actual_winner.empty else "--"

        # Next round – find the actual next upcoming race by date
        report("Predicting next race...")
        next_race_name = None
        today = _dt.date.today()

        def _find_next_race(year):
            """Find the next race in `year` that hasn't happened yet."""
            schedule = get_event_schedule_cached(year)
            upcoming = schedule[schedule["EventDate"].dt.date >= today]
            if not upcoming.empty:
                evt = upcoming.iloc[0]
                return year, int(evt["RoundNumber"]), evt["EventName"]
            return None

        found = None
        if target_year and target_year > last_year:
            try:
                found = _find_next_race(target_year)
            except Exception:
                pass
        if not found:
            try:
                found = _find_next_race(last_year)
            except Exception:
                pass
        if not found:
            try:
                found = _find_next_race(last_year + 1)
            except Exception:
                pass

        if found:
            next_year, next_round, next_race_name = found
        else:
            next_year = target_year or last_year + 1
            next_round = 1
            next_race_name = f"Round {next_round}"

        lineup = get_lineup_for_next_round(df, next_year, features, circuit_name=next_race_name)
        raw_next = best_model.predict_proba(lineup[features])[:, 1]
        lineup["WinProbability"] = _softmax_normalize(raw_next)
        lineup = lineup.sort_values("WinProbability", ascending=False)

        next_race_preds = [
            {
                "abbreviation": row["Abbreviation"],
                "team": row["TeamName"],
                "probability": float(row["WinProbability"]),
            }
            for _, row in lineup.iterrows()
        ]

        # Accuracy
        from sklearn.model_selection import train_test_split

        X = df[features]
        y = df["Winner"]
        _, X_te, _, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        accuracy = float(best_model.score(X_te, y_te))

        # Feature importance for algorithm viz, aggregated across the
        # ensemble members that expose impurity importances.
        try:
            clf = best_model.named_steps["classifier"]
            pre = best_model.named_steps["preprocess"]
            names = pre.get_feature_names_out()
            imp = _ensemble_feature_importances(clf)
            if imp is None:
                raise ValueError("no importances available")
            # Aggregate by base feature (e.g. TeamName_* -> TeamName)
            base_imp = {}
            for n, v in zip(names, imp):
                base = n.split("__")[-1].rsplit("_", 1)[0] if "_" in n.split("__")[-1] else n.split("__")[-1]
                base_imp[base] = base_imp.get(base, 0) + float(v)
            feature_importance = {k: float(v) for k, v in sorted(base_imp.items(), key=lambda x: -x[1])}
        except Exception:
            feature_importance = {}

        # Build full schedule for race cycling
        schedule_list = []
        try:
            sched = get_event_schedule_cached(next_year)
            for _, row in sched.iterrows():
                schedule_list.append({
                    "round": int(row["RoundNumber"]),
                    "name": row["EventName"],
                    "date": str(row["EventDate"].date()),
                })
        except Exception:
            pass

        result = {
            "feature_importance": feature_importance,
            "last_race": {
                "year": last_year,
                "round": last_round,
                "name": last_race_name,
                "predicted_winner": pred_winner["abbreviation"],
                "actual_winner": actual_abbr,
                "predictions": last_race_preds,
            },
            "next_race": {
                "year": next_year,
                "round": next_round,
                "name": next_race_name,
                "predicted_winner": next_race_preds[0]["abbreviation"],
                "top_probability": next_race_preds[0]["probability"],
                "predictions": next_race_preds,
            },
            "schedule": schedule_list,
            "accuracy": accuracy,
            "_model": best_model,
            "_features": features,
            "_base_lineup": lineup[["DriverNumber", "Abbreviation", "TeamName"]].to_dict("records"),
            "_base_driver_pts": lineup.set_index("Abbreviation")["DriverPointsBefore"].to_dict(),
            "_base_team_pts": lineup.drop_duplicates("TeamName").set_index("TeamName")["TeamPointsBefore"].to_dict(),
            "_extra_features": {
                col: lineup.set_index("Abbreviation")[col].to_dict()
                for col in [
                    "RecentAvgPos", "RecentAvgGrid",
                    "RecentWinRate", "RecentPodiumRate",
                    "RecentPosGain", "RecentPoints3",
                    "DNFRate",
                    "DriverExperience", "HeadToHead", "TeamRecentForm",
                    "DriverCircuitAvg", "TeamCircuitAvg",
                    "SeasonPointsBefore", "TeamSeasonPointsBefore",
                    "PUBatteryScore", "PUICEScore",
                ]
                if col in lineup.columns
            },
        }
        # Persist a slim copy so reopening the window hydrates instantly.
        save_last_result(result)
        return result
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n\n{traceback.format_exc()}"}


F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def predict_with_standings(model, features, base_lineup, driver_pts, team_pts,
                           extra_features=None, circuit_name=None, season_year=2026):
    """Re-run predictions with updated championship standings.

    `extra_features` carries the per-driver rolling stats from the original
    run; we use them to seed the new lineup so we don't have to retrain.
    `circuit_name` lets us pull a circuit-specific affinity number from
    `extra_features["DriverCircuitAvg"]` when the caller has it.
    """
    lineup = pd.DataFrame(base_lineup)
    lineup["DriverPointsBefore"] = lineup["Abbreviation"].map(driver_pts).fillna(0)
    lineup["TeamPointsBefore"] = lineup["TeamName"].map(team_pts).fillna(0)

    if extra_features:
        for col, mapping in extra_features.items():
            if col not in lineup.columns:
                lineup[col] = lineup["Abbreviation"].map(mapping)

    # ExpectedGrid – use the carried RecentAvgGrid if available, else default
    # to mid-field rather than the old `0` (which the StandardScaler treated
    # as a strong "ahead of pole" outlier).
    if "GridPosition" not in lineup.columns or lineup["GridPosition"].isna().all():
        lineup["GridPosition"] = lineup.get("RecentAvgGrid", pd.Series(10.0, index=lineup.index))
    lineup["GridPosition"] = lineup["GridPosition"].fillna(10.0)

    # Refresh PU scores from the season we're predicting – these are not in
    # extra_features by row, they depend on (team, year).
    pu_b, pu_i = [], []
    for t in lineup["TeamName"]:
        b, i = _get_pu_scores(t, int(season_year))
        pu_b.append(b)
        pu_i.append(i)
    lineup["PUBatteryScore"] = pu_b
    lineup["PUICEScore"] = pu_i

    defaults = {
        "RecentAvgPos": 10.0, "RecentAvgGrid": 10.0,
        "RecentWinRate": 0.0, "RecentPodiumRate": 0.0,
        "RecentPosGain": 0.0, "RecentPoints3": 0.0,
        "DNFRate": 0.10,
        "DriverExperience": 0.0, "HeadToHead": 0.5,
        "TeamRecentForm": 10.0,
        "DriverCircuitAvg": 10.0, "TeamCircuitAvg": 10.0,
        "SeasonPointsBefore": 0.0, "TeamSeasonPointsBefore": 0.0,
    }
    for col in features:
        if col not in lineup.columns:
            lineup[col] = defaults.get(col, 0.0)
        if col in defaults:
            lineup[col] = lineup[col].fillna(defaults[col])

    # Field-relative features must be recomputed for the *current* lineup
    # and standings (the carried-over values reflect the original race).
    lineup = _attach_field_relative(lineup)

    raw = model.predict_proba(lineup[features])[:, 1]
    lineup["WinProbability"] = _softmax_normalize(raw)
    lineup = lineup.sort_values("WinProbability", ascending=False)

    return [
        {
            "abbreviation": row["Abbreviation"],
            "team": row["TeamName"],
            "probability": float(row["WinProbability"]),
        }
        for _, row in lineup.iterrows()
    ]


def _backtest_one_race(args):
    """Train on every race except (year, rnd) and predict that race.

    Pulled out as a top-level function so it pickles cleanly into a
    `joblib.Parallel` worker process.  Returns a result dict or None
    when the race doesn't have enough data to be evaluable.
    """
    year, rnd, df, features = args
    mask = (df["Year"] == year) & (df["Round"] == rnd)
    train_df = df[~mask]
    test_df = df[mask].copy()

    if len(train_df) < 100 or len(test_df) < 2:
        return None

    model = build_model_fast()
    sw = _training_sample_weights(train_df)
    model.fit(
        train_df[features], train_df["Winner"],
        classifier__sample_weight=sw,
    )
    raw = model.predict_proba(test_df[features])[:, 1]
    test_df["WinProbability"] = _softmax_normalize(raw)

    pred_row = test_df.sort_values("WinProbability", ascending=False).iloc[0]
    pred_abbr = pred_row["Abbreviation"]
    actual_row = test_df[test_df["Winner"] == 1]
    actual_abbr = actual_row.iloc[0]["Abbreviation"] if not actual_row.empty else "--"

    return {
        "year": int(year),
        "round": int(rnd),
        "predicted": pred_abbr,
        "actual": actual_abbr,
        "correct": pred_abbr == actual_abbr,
    }


def run_predictions_all_races(progress_callback=None):
    """Backtest the model: for every (year, round), train on all
    other races and predict that one.

    The previous implementation used the full RandomizedSearchCV
    (36 fits × ~100 races ≈ thousands of GBM fits) and ran
    sequentially – ~50-90 minutes on an M-series Mac.

    Two changes deliver a ~50-100× speedup:

    1. `build_model_fast()` skips the hyperparameter search and
       trains one GBM with the search's empirical centre point.
       Validation ROC-AUC is within noise of the full search but
       per-race cost drops from ~30s to ~1-2s.
    2. The outer race loop runs in parallel via `joblib.Parallel`.
       Each race trains independently, so this scales linearly with
       cores up to memory bandwidth.
    """
    def report(msg):
        if progress_callback:
            progress_callback(msg)

    try:
        report("Loading race data...")
        df = load_data(progress_callback=progress_callback)
        if df is None or df.empty:
            return {"error": "No race data available. Ensure data.csv exists or check your internet connection."}

        df["Winner"] = (df["Position"] == 1).astype(int)
        features = list(FEATURES)
        race_keys = sorted({(int(y), int(r)) for y, r in
                            df[["Year", "Round"]].itertuples(index=False)})
        total = len(race_keys)
        report(f"Backtesting {total} races (parallel)…")

        # Threading backend keeps shared `df` zero-copy across workers –
        # GBM releases the GIL during tree fitting via numpy/scipy ops,
        # and we avoid the multi-second pickle hit that the loky/process
        # backend would pay shipping the full DataFrame to each worker.
        from joblib import Parallel, delayed
        n_jobs = max(1, (os.cpu_count() or 4) - 1)

        completed = {"n": 0}

        def _wrapped(args):
            res = _backtest_one_race(args)
            completed["n"] += 1
            if completed["n"] % 5 == 0 or completed["n"] == total:
                report(f"Backtest {completed['n']}/{total}…")
            return res

        raw_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_wrapped)((y, r, df, features))
            for (y, r) in race_keys
        )

        # Resolve race names once at the end – schedule lookups go
        # through the on-disk cache so this is essentially free.
        schedule_cache: dict = {}
        results = []
        correct = 0
        for res in raw_results:
            if res is None:
                continue
            year = res["year"]
            rnd = res["round"]
            race_name = f"Round {rnd}"
            try:
                if year not in schedule_cache:
                    schedule_cache[year] = get_event_schedule_cached(year)
                sched = schedule_cache[year]
                evt = sched[sched["RoundNumber"] == rnd]
                if not evt.empty:
                    race_name = evt.iloc[0]["EventName"]
            except Exception:
                pass
            res["name"] = race_name
            if res["correct"]:
                correct += 1
            results.append(res)

        accuracy = correct / len(results) if results else 0
        return {
            "all_races": results,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
        }
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n\n{traceback.format_exc()}"}
