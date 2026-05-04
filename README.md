# NBA Achilles Tendon Rupture — Survival Analysis

Predicts the time-to-Achilles-rupture risk for NBA players using **DeepHit competing-risks survival analysis**, combining biomechanical workload features, play-style embeddings, and NLP-derived prodromal signals from injury reports and press notes.

---

## Research question

> Given a player's cumulative workload, biometric profile, and recent injury language, what is the probability that they suffer an Achilles tendon rupture within the next 1 / 2 / 3 seasons?

---

## Architecture

```
Raw sources                Feature engineering          Model
──────────────             ───────────────────          ─────
Basketball-Reference  ──►  ACWR (EWMA, 3 scales)  ──►
NBA Stats API         ──►  Play-style embedding   ──►  DeepHit
ProSportsTransactions ──►  Bio / load features    ──►  (competing risks)
Injury report PDFs    ──►  BioBERT prodromal score ──►
```

### DeepHit

A deep learning survival model that handles **competing risks** (Achilles rupture vs. other career-ending event).

- Shared MLP sub-network over covariate vector **x**
- Two cause-specific output heads, each producing a PMF over T discretised time bins
- Loss = α · negative log-likelihood + (1−α) · pairwise ranking loss
- Trained with Optuna HPO (50 trials, pruned with MedianPruner)

### ACWR (Acute:Chronic Workload Ratio)

Computed via **exponentially weighted moving averages** at three scales:

| Scale  | Acute window | Chronic window |
|--------|-------------|----------------|
| Short  | 7 days      | 28 days        |
| Medium | 14 days     | 42 days        |
| Long   | 28 days     | 84 days        |

### NLP pipeline

1. **Snorkel** weak supervision: 8 labeling functions over injury text snippets produce probabilistic `HEALTHY / PRODROMAL / RUPTURE` labels without manual annotation.
2. **BioBERT** (`dmis-lab/biobert-base-cased-v1.2`) fine-tuned on the Snorkel labels; softmax probability of class PRODROMAL is used as a continuous risk feature.

---

## Data sources

| Source | What we pull | Script |
|--------|-------------|--------|
| Basketball-Reference | Game logs 1990–present, player bio, injury designations | `data/scraping/scrape_bball_reference.py` |
| NBA Stats API | Season stats, play-type breakdowns, player registry | `data/scraping/scrape_nba_api.py` |
| ProSportsTransactions | Ground-truth IL placements with "achilles" in reason field | `data/scraping/scrape_prosports.py` |
| NBA official injury reports | PDF reports (2015–present) | `data/scraping/scrape_injury_reports.py` |

All scrapers are **resume-safe** (check disk cache before requesting) and implement the IP-safety pattern:
- Random delay between requests (3–6 s base)
- Per-domain rate limit (≤ 10 req/min)
- Exponential back-off on HTTP 429
- Rotating User-Agent pool
- Session cooldown after 120–150 requests

---

## Project layout

```
nba-achilles-survival/
├── data/
│   ├── raw/                  # cached API responses, never modified
│   ├── processed/            # cleaned, feature-engineered outputs
│   └── scraping/
│       ├── scrape_bball_reference.py
│       ├── scrape_nba_api.py
│       ├── scrape_prosports.py
│       └── scrape_injury_reports.py
├── features/
│   ├── acwr.py               # EWMA-ACWR at 3 window scales
│   ├── feature_store.py      # point-in-time joins, no leakage
│   └── play_style.py         # play-by-play PCA embedding
├── nlp/
│   ├── snorkel_pipeline.py   # weak supervision labeling functions
│   └── biobert_finetune.py   # BioBERT fine-tuning
├── models/
│   ├── deephit.py            # DeepHit architecture + loss
│   ├── train.py              # training loop + Optuna HPO
│   └── evaluate.py           # C-index, D-calibration, Brier Score
├── notebooks/
│   └── eda.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect data

```bash
# Ground-truth Achilles IL placements (ground truth labels)
python data/scraping/scrape_prosports.py

# Player profiles + game logs from Basketball-Reference
python data/scraping/scrape_bball_reference.py --min-year 1990

# NBA Stats API season data (start from 1996 — earlier data is sparse)
python data/scraping/scrape_nba_api.py --min-season 1996

# Official NBA injury report PDFs (2015-present)
python data/scraping/scrape_injury_reports.py --start 2015-10-01
```

> All scrapers skip files already on disk, so you can re-run safely after interruption.

### 3. Build features

```python
from features.acwr import compute_acwr
from features.feature_store import build_feature_matrix

# Compute ACWR from game logs
acwr_df = compute_acwr(gamelogs_df)
acwr_df.to_csv("data/processed/acwr_features.csv", index=False)

# Assemble final feature matrix (point-in-time, no leakage)
build_feature_matrix()
```

### 4. NLP labeling

```bash
# Run Snorkel labeling functions on injury text snippets
python nlp/snorkel_pipeline.py

# Fine-tune BioBERT on Snorkel labels
python nlp/biobert_finetune.py --train

# Score all snippets
python nlp/biobert_finetune.py --infer
```

### 5. Train DeepHit

```bash
# With Optuna HPO (recommended)
python models/train.py --hpo-trials 50

# Quick run with fixed hyperparameters
python models/train.py --no-hpo --epochs 100
```

### 6. Evaluate

```python
from models.evaluate import full_evaluation_report
import numpy as np

# H, t, e from model inference on test set
report = full_evaluation_report(H, t, e, cause=1)
# Prints C-index, IBS, D-calibration p-value
```

---

## Evaluation metrics

| Metric | What it measures |
|--------|-----------------|
| **C-index** | Discrimination: does the model rank higher-risk players correctly? |
| **Brier Score / IBS** | Calibration + discrimination: mean squared error of predicted CIF |
| **D-calibration** | Distributional calibration: χ² test across predicted-risk deciles |

---

## Notes & caveats

- ProSportsTransactions is the primary ground-truth source. Cross-validate against NBA official injury reports and Basketball-Reference transaction data.
- The study period is **1990–2024**. Pre-2000 injury data is sparser; consider a sensitivity analysis restricted to 2000+.
- ACWR thresholds (>1.5 = "spike") follow the sports science literature but may need position-specific tuning for NBA players.
- BioBERT prodromal scoring depends on having scraped sufficient free-text data (beat reporter notes, injury report PDFs). The snorkel pipeline can be run in "LF-only" mode if BioBERT training data is insufficient.
