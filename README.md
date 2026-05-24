# ApexAI — F1 Race Predictor

ApexAI is an end-to-end Formula 1 race prediction suite. It ingests
timing data with [FastF1](https://github.com/theOehrly/Fast-F1), trains
a Gradient Boosting Classifier on five seasons of race history
(2022 – 2026 mid-season), and presents calibrated win probabilities for
the next race inside a custom Tk + PIL desktop app — complete with a
broadcast-style podium, per-circuit ambient theming, live FIA team-radio
playback, and one-click race replays for every session of every round.

![ApexAI GUI](screenshot.png)

> Looking for the full product spec? See [`docs/PRD.md`](docs/PRD.md).

## Highlights

- **Calibrated win probabilities** — Gradient Boosting Classifier
  (`prediction.py`) tuned with `RandomizedSearchCV` over a
  `TimeSeriesSplit` (ROC-AUC scoring) so the model never trains on
  races run after the validation set. Raw logits are softmax-calibrated
  with a temperature term so the grid sums to 100 % and no driver
  collapses to a 0 % outlier.
- **Rich feature engineering** — cumulative championship points,
  rolling win/podium rates, average finish, head-to-head vs teammate,
  DNF rate, driver experience, team / power-unit form, all derived on
  the fly from FastF1 results.
- **Fast leave-one-race-out backtest** — the full 2022 → 2026 backtest
  (96 races) finishes in ~28 seconds on a multi-core Mac (down from
  ~50 minutes) by skipping the per-race hyperparameter search and
  parallelising the outer race loop with `joblib`. Predictions stay
  warm during and after the backtest, so you can flip straight to
  Visualization or Replays.
- **2026 mid-season build** — driver lineup, team roster, and the new
  Cadillac and Audi power units are all wired in. The training pipeline
  automatically re-fetches if `data.csv` is missing rounds from the
  latest completed weekend.
- **Race-day GUI** (`app.py`) — Tk + PIL interface with five tabs —
  *Predict Next Race*, *Backtest All Races*, *Race Visualization*,
  *Team Radio*, *Race Replays* — plus a one-click *Refresh* to retrain
  on the latest data.
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
