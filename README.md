# ApexAI — F1 Race Predictor

ApexAI is an end-to-end Formula 1 race prediction suite. It ingests
timing data with [FastF1](https://github.com/theOehrly/Fast-F1), trains
a three-model soft-voting ensemble on five seasons of race history
(2022 – 2026 mid-season), and presents calibrated win probabilities for
the next race inside a custom Tk + PIL desktop app styled like a
pit-wall racing console — complete with a broadcast-style podium,
per-circuit ambient theming, live FIA team-radio playback, and
one-click race replays for every session of every round.

![ApexAI GUI](screenshot.png)

> Looking for the full product spec? See [`docs/PRD.md`](docs/PRD.md).

## Highlights

- **Three-model soft-voting ensemble** (`prediction.py`, model v6) —
  a classic Gradient Boosting Classifier, a HistGradientBoosting model
  (leaf-wise trees + L2 regularisation), and a Random Forest vote on
  every driver's win score with 2:2:1 weights. The members disagree
  exactly on the marginal races where a single model's variance flips
  the pick, so the average beats any individual member on walk-forward
  accuracy. Raw scores are softmax-calibrated with a temperature term
  so the grid sums to 100 % and no driver collapses to a 0 % outlier.
- **Ablation-validated features** — every candidate feature is accepted
  or rejected by a leak-free walk-forward benchmark over all 32
  completed 2025 – 2026 races. The v6 winner: a within-race *form rank*
  (is this the best recent form **on this grid**, not just "good
  form"), which lifted top-1 accuracy from 53 % to 56 % on 2025 – 2026.
  Candidates that hurt accuracy (points shares, 3-race momentum,
  quali-surprise) are computed in the dataset but deliberately kept
  out of the model input.
- **Validated on the 2026 season** — in the same walk-forward backtest
  (train only on races before each round, predict that round) the model
  picks the actual winner in 6 of the 8 completed 2026 rounds
  (75 % top-1) and has the winner inside its top 3 for all 8
  (100 % top-3).
- **Rich feature engineering** — season-to-date championship standings
  (driver + team, leak-free), career points, rolling win/podium rates,
  average finish, grid→finish racecraft delta, within-race form rank,
  head-to-head vs teammate, DNF rate, driver experience, circuit
  affinity, and team / power-unit form, all derived on the fly from
  FastF1 results.
- **Recency-weighted training** — training rows decay at `0.85^age` per
  season, so races run under the 2026 regulations (50/50 hybrid power
  split, new aero, Cadillac + Audi) dominate the fit while older
  ground-effect seasons still contribute signal.
- **Fast leave-one-race-out backtest** — the full 2022 → 2026 backtest
  (~100 races) runs each race as a single fixed-configuration ensemble
  fit (no per-race hyperparameter search) and parallelises the outer
  race loop with `joblib`, finishing in a few minutes on a multi-core
  machine instead of the better part of an hour. Predictions stay warm
  during and after the backtest, so you can flip straight to
  Visualization or Replays.
- **2026 mid-season build** — driver lineup, team roster, and the new
  Cadillac and Audi power units are all wired in. The training pipeline
  automatically re-fetches if `data.csv` is missing rounds from the
  latest completed weekend.
- **Racing-console GUI** (`app.py`) — Tk + PIL interface styled like a
  steering-wheel / pit-wall console: a telemetry header cluster with a
  pulsing SYS LED, live session clock, and model/accuracy readouts; five
  numbered console-button tabs (*01 Predict*, *02 Backtest*,
  *03 Visualization*, *04 Team Radio*, *05 Replays*) with red engaged
  light-bars; monospace telemetry fonts throughout; a carbon-fiber cap
  strip; and a timing-bar footer — plus a one-click *Refresh* to
  retrain on the latest data.
  - **Home button:** the F1 logo + "ApexAI" wordmark in the header
    return you to the last predicted race from anywhere in the app.
  - **30 fps animated track-map visualisation** with per-driver dots,
    automatic MOM (Maximum Overtake Moment) zone detection on the
    longest straights, hand-traced real circuit silhouettes, and a
    podium card featuring procedural anti-aliased gold / silver /
    bronze trophies plus a laurel wreath.
  - **Per-race ambient theming:** sakura petals at Suzuka, maple leaves
    at Montréal, Mediterranean sun over Monaco, carnival confetti at
    Interlagos and Mexico City, neon + fireworks on the Vegas Strip,
    rain over Silverstone and Spa, starlit skies over the desert
    circuits, and more.
  - **Singleton enforcement** — launching `app.py` while another
    instance is open kills the prior PID (with `SIGKILL` fallback) and
    sweeps the OS process list so stale instances from terminals or
    IDE runs are also reaped. Exactly one bot instance ever runs.
  - **Prediction + model caching** (`model_cache.pkl`,
    `last_predictions.pkl`) is stamped with a `MODEL_VERSION` so a
    relaunch on a bumped model rebuilds automatically and a relaunch
    on the same version is instantaneous.
- **Full-race team radio** (`radio_fia.py`) — radio clips are fetched
  directly from the FIA livetiming archive (with OpenF1 as a fallback)
  and each clip is mapped to its lap number by matching its capture
  timestamp against the race event log. Plays back through
  `playsound3`.
- **Race Replays** — every session of every round, deep-linked to
  [fullraces.com](https://fullraces.com). Pick a season, click *Race* /
  *Qualifying* / *Sprint* / *Sprint Quali* / *Practice* and the
  replay opens in your default browser.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): FastF1, pandas, scikit-learn, numpy,
matplotlib, Pillow, `f1radio[playback]`, `playsound3`, `joblib`.

The first launch downloads timing data through the official F1 API and
caches it in `cache/`. If `data.csv` is missing or stale, race results
from 2022 through the most recent completed 2026 round are pulled to
rebuild the training dataset (cold-start can take a few minutes the
first time; subsequent launches hydrate from the cache and are
near-instant).

## Usage

### GUI (recommended)

```bash
python app.py
```

Click **Predict Next Race** to fetch data, train the model, and view
the podium card + win probabilities for the next round. From there:

- **Backtest All Races** — Aggregate accuracy across 2022 – present
  with a per-season breakdown card (~28 s on a multi-core Mac).
- **Race Visualization** — Animated track map with MOM zones, podium
  trophies, and circuit-specific ambience.
- **Team Radio** — Browse and play back full-race radio clips for any
  driver, lap-mapped from the FIA archive.
- **Race Replays** — One-click links to FullRaces.com for every
  session of every round, by season.
- **F1 / ApexAI logo (header)** — Click anywhere on the brand mark to
  jump back to the predictions view.

### Command line

```bash
python predict_winner.py
```

Prints the predicted winner of the most recent race, the upcoming
round, and the model's overall validation accuracy.

## Team logos

To use the official team logos instead of coloured initials badges:

```bash
python fetch_logos.py
```

This downloads up-to-date PNGs (Wikimedia thumbnails, 500 px wide) to
`logos/`. `--force` re-downloads existing files. Logos are auto-cropped
and aspect-preservingly resized at runtime, with icon-only crops used
at small sizes (grid rows, podium) for legibility.

## Project layout

```
app.py            Tk + PIL desktop app (GUI, viz, radio, replays)
prediction.py     Data ingest, feature engineering, model training,
                  caching, singleton enforcement, and inference
predict_winner.py Headless CLI entry point
radio_fia.py      FIA livetiming archive client for full-race radio
team_colors.py    Official team-colour palette
team_logos.py     Logo loading, icon cropping, alpha-aware resizing
track_layouts.py  Hand-traced silhouettes for every 2026 circuit
fetch_logos.py    Wikimedia logo downloader
docs/PRD.md       Full product requirements document
```

## Building a Mac executable (optional)

```bash
pip install pyinstaller
./build_mac.sh
```

Produces `dist/F1 Winner Predictor.app`. Data and cache live in
`~/Library/Application Support/F1 Winner Predictor/`.

## Data sources

- **Race + timing data** — `fastf1`, talking to the official F1 live
  timing API.
- **Team radio** — FIA livetiming archive (`TeamRadio.json` +
  `TeamRadio.jsonStream`) with OpenF1 as a fallback.
- **Race replays** — [fullraces.com](https://fullraces.com), reached
  through deep-linked WordPress search URLs so the integration
  survives post-title changes.

## License

[MIT](LICENSE)
