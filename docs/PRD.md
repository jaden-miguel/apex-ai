# ApexAI — Product Requirements Document

> Last updated: July 9, 2026 (mid-season build, model v5)

## 1. Overview

**ApexAI** is a desktop application that predicts Formula 1 race
results using a Gradient Boosting Classifier trained on five seasons of
race history (2022 – 2026 mid-season). The user opens the app, hits one
button, and gets calibrated win probabilities for the next race plus a
broadcast-style visualization, the full race radio, and one-click
replays of every session of every round.

The product is a single-binary, offline-friendly Python desktop app
(Tk + PIL) — no servers, no accounts, no telemetry.

## 2. Problem Statement

F1 prediction tooling on the open web is fragmented:

- Sportsbook odds are calibrated for action, not for accuracy.
- Pure ML notebooks predict winners but have no UX, no race context,
  and no ergonomic way to flip between rounds or see *why* the model
  picked a driver.
- F1 TV's broadcast graphics are gorgeous but locked behind a
  subscription and never expose model probabilities or backtests.

ApexAI's bet is that the most useful product is one that combines
**transparent ML**, **broadcast-grade visuals**, and **live session
artefacts** (team radio, replays) in a single window so an enthusiast
can sit down 30 minutes before lights-out and have everything they
need to follow the race.

## 3. Goals

### 3.1 Primary goals

- **G1.** Calibrated win probabilities for the next race in <30 s on a
  warm cache, <2 min on cold start.
- **G2.** Full leave-one-race-out backtest of the model in <60 s so
  users can sanity-check accuracy before trusting a prediction.
- **G3.** Animated, broadcast-quality circuit visualization with the
  predicted podium and circuit-specific ambient theming.
- **G4.** First-class access to **team radio** (full race, lap-mapped)
  and **race replays** for every session of every round, without
  leaving the app.
- **G5.** Single-instance enforcement so the app never accidentally
  stacks duplicate windows.

### 3.2 Non-goals

- Live in-race telemetry overlays (qualifying / race timing).
- Live betting integration.
- Mobile / web client.
- Multi-user / cloud accounts.
- Sponsor / FOM-licensed assets that we'd need to license.

## 4. Target Users

| Persona | Need |
|---|---|
| **F1 enthusiast** | Wants a smarter pre-race pick than chat-room consensus and a way to enjoy the race weekend in one app. |
| **Fantasy F1 player** | Needs probability rankings (not just a winner) to allocate fantasy budget. |
| **ML / data nerd** | Wants to read the feature importance, run a backtest, and audit the pipeline. |
| **Casual viewer** | Just wants to watch a replay or hear team radio without hunting through Reddit links. |

## 5. Architecture

```
┌─────────────────── app.py (Tk + PIL) ──────────────────┐
│  Header (F1 wordmark — clickable home button)          │
│  Tab bar:  Predict · Backtest · Visualization ·         │
│            Team Radio · Race Replays · Telemetry ·      │
│            Session Replay · Refresh                     │
│                                                        │
│  ┌──── Predictions panel ───┐ ┌──── Insight panel ───┐ │
│  │  Podium card + grid      │ │ Model stats          │ │
│  │  (canvas + PIL trophies) │ │ Feature importance   │ │
│  └──────────────────────────┘ │ How it works         │ │
│                                └──────────────────────┘ │
│  Footer: status bar (live activity log)                 │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
            ┌──── prediction.py ───────┐
            │  load_data()             │
            │  feature engineering     │
            │  build_model() (search)  │
            │  build_model_fast()      │
            │  run_predictions()       │
            │  run_predictions_all_…() │
            │  schedule cache          │
            │  singleton enforcement   │
            └──────────────────────────┘
                              │
            ┌──── telemetry.py ────────┐
            │  load_session()          │
            │  lap_telemetry()         │
            │  compare_laps() + delta  │
            │  build_replay()          │
            │  Replay.order_at()       │
            │  replay disk cache       │
            └──────────────────────────┘
                              │
                              ▼
                  ┌──── External APIs ────┐
                  │  FastF1 (timing)       │
                  │  FIA livetiming archive│
                  │  OpenF1 (radio fallback)│
                  │  fullraces.com (replays)│
                  └────────────────────────┘
```

## 6. Functional Requirements

### 6.1 Predict Next Race

- **F1.1.** On click, train (or hydrate from cache) a Gradient Boosting
  Classifier and produce calibrated win probabilities for every driver
  on the active grid.
- **F1.2.** Probabilities sum to 100 % across the grid (softmax-calibrated
  with a temperature term so no driver collapses to 0 %).
- **F1.2a.** (v5) Features must include leak-free *season-to-date*
  championship standings for driver and team (the team variant built
  from per-round totals so a row never sees its teammate's points from
  the same race) and a grid→finish racecraft delta, alongside the
  career-points, rolling-form, circuit-affinity, and power-unit
  features.
- **F1.2b.** (v5) Training rows are weighted by winner-class balancing
  × a per-season recency decay (`0.85^age`) so races run under the
  current regulations dominate the fit.
- **F1.3.** UI displays a podium card (P1/P2/P3) with procedural gold,
  silver, and bronze trophies, plus the full grid sorted by probability.
- **F1.4.** Cached predictions hydrate instantly on relaunch
  (`last_predictions.pkl`); model artefact (`model_cache.pkl`) is
  versioned by `MODEL_VERSION` so a code update auto-invalidates stale
  caches.
- **F1.5.** "Predict" while a prediction already exists *advances* to
  the next race in the schedule using the cached model and updated
  standings (no retraining required).

### 6.2 Backtest All Races

- **F2.1.** Walk every (year, round) in the dataset, train on all *other*
  races, predict the held-out race.
- **F2.2.** Aggregate accuracy displayed in the chip (`X / Y · NN.N%`).
- **F2.3.** Per-season breakdown card with year-level hit rate so users
  can see if the model trends up or down by season.
- **F2.4.** Backtest must complete in **≤ 60 s** on a multi-core Mac
  (currently ~28 s for 96 races).
- **F2.5.** Running backtest does NOT clobber the active prediction —
  Visualization, Replays, and Radio remain available throughout.
- **F2.6.** Backtest results scroll cleanly via mousewheel anywhere on
  the page (cross-platform delta normalisation).

### 6.3 Race Visualization

- **F3.1.** Animated 30 fps track-map canvas with one dot per driver,
  trailing motion blur, and per-driver team colour.
- **F3.2.** **MOM zones** highlighted on the two longest detected
  straights (cross-product curvature analysis with progressive
  separation fallback so single-straight tracks like Imola still get a
  zone).
- **F3.3.** **Per-circuit ambient theming**:
  - Suzuka: cherry blossom petals
  - Montréal: maple leaves
  - Monaco: Mediterranean sun, harbour silhouette
  - Las Vegas / Mexico / Interlagos: confetti
  - Silverstone, Spa: rain
  - Desert circuits: starlit night sky
- **F3.4.** **Real circuit silhouettes** — each track is hand-traced
  in `track_layouts.py` (Suzuka figure-8, Baku L-shape, Spa triangle,
  etc.) and rendered with aspect-preserving fit + dynamic padding so
  the track is never cropped.
- **F3.5.** Center podium card with three procedural trophies
  (super-sampled at 4× then anti-aliased with `Image.LANCZOS` so the
  trophies don't look 8-bit).

### 6.4 Team Radio

- **F4.1.** For any selectable past race, fetch every clip directly
  from the FIA livetiming archive (`TeamRadio.json` +
  `TeamRadio.jsonStream`) with OpenF1 as a fallback.
- **F4.2.** Map every clip to its lap number by aligning capture
  timestamp against the race event log.
- **F4.3.** Filter clips by driver; play sequentially with a now-playing
  bar and waveform animation; stop on tab switch.

### 6.5 Race Replays (NEW)

- **F5.1.** Tab listing every round of the selected season — round
  badge, race name, date, "upcoming" flag for unraced weekends.
- **F5.2.** Per-row session buttons: **Race · Qualifying · Sprint ·
  Sprint Quali · Practice**.
- **F5.3.** Clicking a session button opens a deep-linked search on
  [fullraces.com](https://fullraces.com) in the user's default browser.
- **F5.4.** Year selector for browsing prior seasons (current year
  + 7 previous seasons).
- **F5.5.** Schedule loads on a background thread so the UI never
  freezes while the schedule cache is populating.

### 6.6 Telemetry Overlay (NEW)

- **F6.1.** Two independent lap selectors (**Car A** / **Car B**), each
  a season → round → session → driver → lap chain. Lap defaults to the
  driver's fastest.
- **F6.2.** *LINK SESSIONS* (on by default) pins Car B to Car A's
  session for the common two-team-mates case, and locks B's session
  combos. Unlinking enables **cross-session and cross-season**
  comparison — the year-vs-year use case.
- **F6.3.** Laps are aligned on **distance around the lap**, not on
  wall-clock, which is what makes lap-vs-lap, compound-vs-compound and
  year-vs-year the same code path.
- **F6.4.** Rendered output:
  - Summary cards per lap: lap time, event/session/lap, tyre compound
    + age, top speed, average speed, full-throttle %, braking %.
  - Headline **lap delta** with the faster driver named.
  - Trace stack against distance: **speed**, **running delta**
    (filled toward whoever is ahead), **throttle**, **brake + DRS**
    bands, and **gear**.
  - **Mini-sector dominance**: the lap's track map coloured by which
    driver owns each of 25 mini-sectors, beside a bar chart of the
    per-sector time swing that sums to the final delta.
- **F6.5.** Team-mates share a livery colour; when the two selected
  drivers resolve to the same colour, Car B is shifted to a light tint
  so the traces stay distinguishable.
- **F6.6.** Comparing laps from different circuits is permitted but
  flagged with an inline warning, since the distance axis then only
  lines up by length.
- **F6.7.** All loading happens on worker threads with progress
  reported inline; stale results are discarded if the selection moves
  on while a fetch is in flight.

### 6.7 Session Replay (NEW)

- **F7.1.** Season → round → session picker, restricted to **2018 and
  later** because FastF1 carries no car/position telemetry before then.
- **F7.2.** Every driver's position and car-telemetry streams are
  resampled onto one common 4 Hz time grid, so any frame can be
  rendered directly without re-reading the source streams.
- **F7.3.** **Track map** traced from the session's own position data
  (the fastest lap is used as the cleanest single loop), with numbered
  corners, a start/finish marker, and one live marker per car.
- **F7.4.** **Timing screen**: position, driver, current lap, gap,
  tyre compound and rolling personal best.
  - Race-like sessions are ordered by **track progress** — each car is
    projected onto the circuit centreline to give a continuous
    "laps completed + fraction of the current lap" value, and the gap
    is the time since the leader was at that same point.
  - Qualifying and practice are ordered by **best lap set so far**,
    matching the real broadcast screens.
- **F7.5.** **Telemetry HUD** for the focused car: speed, gear,
  throttle/brake bars, RPM, DRS state and tyre. Clicking any timing row
  moves the focus.
- **F7.6.** **Transport**: play/pause, 0.5× → 16× playback, a scrub bar,
  ±1 lap jumps, jump to start/end, plus a session clock and lap counter.
  Playback auto-pauses at the end and restarts from the top if played
  again.
- **F7.7.** Leaving the tab pauses playback rather than tearing the
  replay down, so returning resumes in place.
- **F7.8.** Built replays are pickled to `replay_cache/` under a
  version-stamped key; a corrupt or stale cache silently rebuilds.
- **F7.9.** Car markers update every frame (~25 fps); the timing screen
  and HUD refresh at ~5 Hz, since re-configuring two dozen label rows at
  full frame rate is pure overhead.

### 6.8 Header / Home

- **F8.1.** F1 logo and "Apex" / "AI" wordmark act as a **home button**
  — clicking returns to the predictions view of the last predicted
  race, stops any running radio playback and replay playback, and
  resets the view state.
- **F8.2.** Hover affordance: brand mark dims/brightens to telegraph
  that it's interactive.

### 6.9 Singleton

- **F9.1.** Launching `app.py` while another instance is already
  running kills the prior PID via `SIGTERM` (escalates to `SIGKILL`
  after a 2 s grace period) and claims an exclusive lockfile
  (`.apex_ai.lock`).
- **F9.2.** Sweep the OS process list for any *other* python process
  whose command line points at `app.py`, not just the lockfile PID,
  so stale instances from terminals or IDE runs are also reaped.

## 7. Non-Functional Requirements

| Category | Target |
|---|---|
| **Cold start (first launch, no caches)** | ≤ 3 min including data download |
| **Warm start (cache hits)** | ≤ 5 s to interactive |
| **Predict next race (warm)** | ≤ 30 s total |
| **Backtest 96 races** | ≤ 60 s (currently ~28 s) |
| **Visualization frame rate** | 30 fps sustained on M-series Macs |
| **Replays tab open** | ≤ 200 ms perceived latency |
| **Telemetry comparison (session cached)** | ≤ 3 s from *Compare Laps* to rendered traces |
| **Session replay build (first time)** | ≤ 90 s for a race, dominated by the telemetry download |
| **Session replay open (cached)** | ≤ 2 s from `replay_cache/` |
| **Replay playback frame rate** | 25 fps for car markers, 5 Hz for timing screen + HUD |
| **Memory** | ≤ 800 MB resident; a built race replay adds ≈ 30 MB |
| **Offline tolerance** | App must load from caches if FastF1 backends are down — schedule disk cache + stale-OK fallback |
| **Accessibility** | Hand cursors on every clickable element, ≥ 11 pt fonts, WCAG-AA contrast on text |
| **Singleton** | Exactly one bot instance can run at any time |

## 8. UX Principles

1. **One window, no modals.** Tabs swap the body; everything else
   (header, footer) stays put.
2. **Broadcast graphics first.** Trophy podium, F1 red accent rules,
   official wordmark — should look like an F1 lower-third, not a
   research notebook.
3. **Lazy work.** Schedule loads, model training, radio fetches all
   happen on background threads behind a status bar.
4. **No silent failures.** Every error path writes to the status bar.
5. **Cache aggressively.** Schedule, model, predictions, and radio
   clips are all cached on disk and rehydrated on launch.

## 9. Success Metrics

- **Accuracy on backtest:** ≥ 60 % winner-pick rate across the
  2022 – present span (currently 61.5 %).
- **Walk-forward accuracy on the current season** (train on every race
  before round N, predict round N — mirrors production use): ≥ 60 %
  top-1 and ≥ 90 % top-3. Model v5 currently hits 6/8 (75 %) top-1 and
  8/8 (100 %) top-3 through 2026 round 9. Regressions against this
  benchmark block model changes.
- **Cold-to-podium time:** ≤ 3 min on a fresh install.
- **Warm-to-podium time:** ≤ 5 s.
- **Backtest run time:** ≤ 60 s.
- **Crash-free sessions:** ≥ 99 % of launches reach the predictions
  view without an unhandled exception.

## 10. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| GUI | Tk + PIL (Pillow) | Zero-deps cross-platform; PIL gives us anti-aliased custom graphics (trophies, maple leaves, F1 logo). |
| ML | scikit-learn `GradientBoostingClassifier` | Strong baseline for tabular feature data; in-fit early stopping. |
| Hyperparameter search | `RandomizedSearchCV` + `TimeSeriesSplit` | Causal CV; never validates on a race the model hasn't seen yet. |
| Parallelism | `joblib.Parallel` (threading backend) | GBM releases the GIL during numpy ops; threading avoids the pickle hit of process workers. |
| Data | FastF1 + FIA livetiming + OpenF1 + fullraces.com | All free; FastF1 cache lives in `cache/`. |
| Audio | `playsound3` (via `f1radio[playback]`) | Native APIs, no `afplay`/`ffplay` PATH dependency. |
| Telemetry charts | Matplotlib rendered to PIL, shown as a Tk image | Multi-panel trace stacks and the dominance map are static once drawn; rasterising once beats re-drawing on a Tk canvas. |
| Replay playback | Plain Tk canvas item updates on `after()` | Only the car markers move per frame, so moving ~20 canvas ovals is far cheaper than re-rendering a figure. |
| Replay cache | `pickle` of flat numpy arrays in `replay_cache/` | A built replay is just arrays on a common time grid; version-stamped so a schema change invalidates rather than mis-loads. |
| Packaging | PyInstaller (`build_mac.sh`) | Produces a standalone `.app` bundle. |

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| FastF1 backends flaky / down | On-disk schedule cache (24 h TTL) + stale-OK fallback. Predictions can hydrate from `last_predictions.pkl` for visualisation. |
| Model regression after a refactor | `MODEL_VERSION` constant invalidates stale caches. Backtest accuracy is shown in the UI as a constant trust signal. |
| FullRaces.com URL pattern change | Use WordPress `?s=` search instead of hard-coded slugs — survives title and casing changes. |
| Tk freezes on slow synchronous I/O | All network / heavy work runs in background threads with `root.after(0, ...)` callbacks for UI updates. |
| User accidentally launches two instances | Singleton sweep kills any other python `app.py` process at startup. |
| Mac vs Windows mousewheel delta differences | `_wheel_units` static helper normalises delta cross-platform. |

## 12. Roadmap (post-MVP)

- **Live qualifying integration** — show grid-position adjusted
  predictions as quali results come in.
- **Weather features** — both 2026 walk-forward misses (Hamilton R7,
  Leclerc R9) were upset wins the model ranked 2nd–3rd; wet-race
  signals are the most promising lever for catching them.
- **Lap-time forecasts** — extend the model from "who wins" to
  "what's the expected race time".
- **Driver-vs-driver head-to-head card** — pick any two drivers and
  see their feature delta.
- **In-app embedded video** — investigate `tkinterweb` / Chromium
  Embedded Framework for in-window FullRaces playback.
- **Cloud sync** — opt-in sync of season standings + cached
  predictions across machines.
- **Windows / Linux executables** — extend `build_mac.sh` to PyInstaller
  Spec for cross-platform builds.

## 13. Open Questions

- Should "Predict Next Race" auto-refresh nightly during a race week?
- Is there room for a "what changed since last run" diff card after
  every retrain?
- Should the Replays tab include direct embed links (Mixdrop, etc.)
  rather than the WP search? Trade-off: cleaner UX vs. more breakage.
- Should we license F1's broadcast assets formally to ship a public
  build?

## 14. Glossary

- **MOM zone** — *Maximum Overtake Moment* — a long straight on the
  track where DRS overtakes typically happen. Detected automatically
  via curvature analysis.
- **Backtest** — Out-of-sample historical evaluation. For each past
  race, train on every *other* race and predict the held-out one;
  aggregate the hit rate.
- **MODEL_VERSION** — Constant in `prediction.py` bumped whenever the
  feature schema or training pipeline changes; used as the cache
  fingerprint so stale caches are auto-invalidated.
- **Singleton enforcement** — Mechanism that guarantees at most one
  instance of `app.py` runs at any time. Combines a PID lockfile with
  an OS-wide process sweep.
