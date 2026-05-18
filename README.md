# ApexAI — F1 Race Predictor

ApexAI is an end-to-end Formula 1 race prediction suite. It ingests timing
data with [FastF1](https://github.com/theOehrly/Fast-F1), trains a Gradient
Boosting Classifier on five seasons of race history (2022 – 2026
mid-season), and presents calibrated win probabilities for the next race
inside a custom Tk + PIL desktop app — complete with a procedural gold
winner's-trophy podium, per-circuit ambient theming, and live FIA team-radio
playback.

![ApexAI GUI](screenshot.png)

## Highlights

- **Calibrated win probabilities** — Gradient Boosting Classifier
  (`prediction.py`) tuned with `RandomizedSearchCV` over a
  `TimeSeriesSplit` (5-fold, ROC-AUC scoring) so the model never trains on
  races run after the validation set. Raw logits are softmax-calibrated
  with a temperature term so the grid sums to 100 % and no driver collapses
  to a 0 % outlier.
- **Rich feature engineering** — cumulative championship points, rolling
  win/podium rates, average finish, head-to-head vs teammate, DNF rate,
  driver experience, and team / power-unit form, all derived on the fly
  from FastF1 results.
- **2026 mid-season build** — driver lineup, team roster, and the new
  Cadillac and Audi power units are all wired in. The training pipeline
  automatically re-fetches if `data.csv` is missing rounds from the latest
  completed weekend.
- **Race-day GUI** (`app.py`) — Tk + PIL interface with four tabs —
  *Predict Next Race*, *Backtest All Races*, *Race Visualization*, and
  *Team Radio* — plus a one-click *Refresh* to retrain on the latest
  data.
  - 30 fps animated track-map visualisation with per-driver dots, MOM
    zones, and a podium card featuring a procedural gold trophy.
  - Per-race ambient theming: sakura petals, paper lanterns and bonsai
    corners at Suzuka; a Mediterranean sun over Monaco; carnival confetti
    at Interlagos and Mexico City; neon + fireworks on the Vegas Strip;
    rain over Silverstone and Spa; starlit skies over the desert
    circuits, and more.
  - Singleton enforcement — launching `app.py` while another instance is
    open replaces the older window.
  - Prediction + model caching (`model_cache.pkl`, `last_predictions.pkl`)
    is stamped with a `MODEL_VERSION` so a relaunch on a bumped model
    rebuilds automatically and a relaunch on the same version is
    instantaneous.
- **Full-race team radio** (`radio_fia.py`) — radio clips are fetched
  directly from the FIA livetiming archive (with OpenF1 as a fallback) and
  each clip is mapped to its lap number by matching its capture timestamp
  against the race event log. Plays back through `playsound3`.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): FastF1, pandas, scikit-learn, numpy,
matplotlib, Pillow, `f1radio[playback]`, `playsound3`.

The first launch downloads timing data through the official F1 API and
caches it in `cache/`. If `data.csv` is missing or stale, race results from
2022 through the most recent completed 2026 round are pulled to rebuild the
training dataset (cold-start can take a few minutes the first time;
subsequent launches hydrate from the cache and are near-instant).

## Usage

### GUI (recommended)

```bash
python app.py
```

Click **Predict Next Race** to fetch data, train the model, and view the
podium card + win probabilities for the next round. Switch to **Race
Visualization** to see the animated circuit map with the per-race ambient
scene, or to **Team Radio** to play back the full race radio for any
driver.

### Command line

```bash
python predict_winner.py
```

Prints the predicted winner of the most recent race, the upcoming round,
and the model's overall validation accuracy.

## Team logos

To use the official team logos instead of coloured initials badges:

```bash
python fetch_logos.py
```

This downloads up-to-date PNGs (Wikimedia thumbnails, 500 px wide) to
`logos/`. `--force` re-downloads existing files. Logos are auto-cropped and
aspect-preservingly resized at runtime, with icon-only crops used at small
sizes (grid rows, podium) for legibility.

## Project layout

```
app.py            Tk + PIL desktop app (GUI, viz, radio deck)
prediction.py     Data ingest, feature engineering, model training,
                  caching, and inference
predict_winner.py Headless CLI entry point
radio_fia.py      FIA livetiming archive client for full-race radio
team_colors.py    Official team-colour palette
team_logos.py     Logo loading, icon cropping, alpha-aware resizing
track_layouts.py  Hand-curated layouts for every 2026 circuit
fetch_logos.py    Wikimedia logo downloader
```

## Building a Mac executable (optional)

```bash
pip install pyinstaller
./build_mac.sh
```

Produces `dist/F1 Winner Predictor.app`. Data and cache live in
`~/Library/Application Support/F1 Winner Predictor/`.

## Data source

Race and timing data is retrieved via `fastf1`, which talks to the
official F1 live timing API. Team radio is sourced from the FIA
livetiming archive (`TeamRadio.json` + `TeamRadio.jsonStream`) with
OpenF1 as a fallback.
