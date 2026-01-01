# NFL Predictor

A production-grade machine learning system for predicting NFL playoff brackets using ensemble Random Forest models with interactive Streamlit visualization. The system automatically scrapes NFL game data, engineers 15+ advanced features including Elo ratings and turnover margins, and deploys two specialized models: a playoff qualifier that identifies the top 7 teams per conference, and a bracket predictor that simulates head-to-head matchups through Wild Card, Divisional, Conference Championships, and the Super Bowl. Features include clickable bracket visualization with real-time win probabilities, feature importance analysis, and support for both AutoML predictions and manual team selection modes.

## Check out the website [here](https://nfl-predictor28.streamlit.app/)!

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [Repository Layout](#repository-layout)
4. [Data Pipeline & Scraping](#data-pipeline--scraping)
5. [Feature Engineering](#feature-engineering)
6. [Models](#models)
7. [Model Export Format & Bundles](#model-export-format--bundles)
8. [Training a New Model](#training-a-new-model)
9. [Running the Streamlit App](#running-the-streamlit-app)
10. [How Predictions Work](#how-predictions-work)
11. [Interpreting Results](#interpreting-results)
12. [Limitations & Bias](#limitations--bias)
13. [Appendix](#appendix)

---

## Project Overview

### What It Does

NFL Predictor is an end-to-end machine learning system that:

- **Scrapes** NFL game data from [nflverse](https://github.com/nflverse) (via `nflreadpy`)
- **Engineers features** from raw per-game statistics (aggregations, rolling windows, Elo ratings, differential features)
- **Trains two models**:
  - **Model 1 (Playoff Qualifier)**: Binary classifier predicting which teams make playoffs
  - **Model 2 (Bracket Predictor)**: Binary classifier predicting winners of head-to-head playoff matchups
- **Simulates playoff brackets** deterministically using Model 2 to predict each round (Wild Card → Divisional → Conference → Super Bowl)
- **Visualizes results** in an interactive Streamlit UI with clickable bracket buttons showing game probabilities, feature importance, and season stats

### Architecture Overview

```
┌─────────────────┐
│  Data Sources   │  nflverse CSV files (per-game team rows)
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Scraper       │  Scraping_Data.py → data_files/*.csv
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Feature Eng.    │  Aggregate stats, rolling windows, Elo, diff features
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Training       │  training_model.py → Model1 + Model2
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Model Bundle   │  models/model_YYYY-YYYY/*.pkl + metadata.json
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Inference     │  bracket_predictor.py (predict_playoff_teams, simulate_game)
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Streamlit UI  │  app.py (interactive bracket, modals, controls)
└─────────────────┘
```

### Problems Solved

1. **Playoff Prediction**: Automates prediction of which teams qualify for playoffs based on regular-season performance
2. **Bracket Simulation**: Simulates playoff matchups round-by-round using learned head-to-head win probabilities
3. **Interpretability**: Provides feature importance, probability distributions, and season stats for each matchup
4. **Deployment-Ready**: Bundles models with precomputed features for inference without CSV dependencies

---

## Setup

### Prerequisites

- **Python 3.10+** (recommended: 3.10.x or 3.11.x)
- **pip** (package manager)
- **Git** (for cloning)

### macOS / Linux

```bash
# Clone repository
git clone https://github.com/yourusername/NFL-Predictor.git
cd NFL-Predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the app (assumes you have a trained model bundle in models/)
streamlit run app.py
```

### Windows (PowerShell)

```powershell
# Clone repository
git clone https://github.com/yourusername/NFL-Predictor.git
cd NFL-Predictor

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the app (assumes you have a trained model bundle in models/)
streamlit run app.py
```

### Expected Output

After running `streamlit run app.py`, your browser should open to `http://localhost:8501` showing:

- A landing page with an NFL logo
- "Choose a Model" dropdown (if you have trained models in `models/`)
- "Predict Bracket" button (AutoML mode)
- "Choose Teams" mode for manual team selection

---

## Repository Layout

### Core Files

#### `app.py`

**Streamlit frontend application.** Handles:

- UI rendering (bracket visualization, dropdowns, buttons)
- Session state management (bracket results, models, team stats)
- Mode switching (AutoML vs. Manual team selection)
- Interactive bracket buttons with modals (game details, probabilities, feature importance)
- Model loading and inference orchestration
- Training tab (initiates scraping + model training)

Key functions:

- `render_bracket_button()`: Renders clickable buttons for bracket slots
- `show_game_modal()`: `@st.dialog` decorator for modal popups
- `get_matchup_context()`: Extracts matchup data, computes feature importance
- `determine_button_enabled()`: Logic to enable only winning team buttons

#### `training_model.py`

**Model training and export utilities.** Contains:

- `train_models()`: Main training pipeline (loads CSVs, engineers features, trains Model1 and Model2, exports bundles)
- `export_model_bundle()`: Exports models + precomputed features + metadata to `models/model_YYYY-YYYY/`
- Hyperparameters: `RandomForestClassifier(n_estimators=200/300, random_state=42, class_weight='balanced')`
- Train/test splits (80/20), no data leakage
- Feature exclusions (`team`, `season`, `made_playoffs`, `won_game`)

Exports:

- `model1_playoff_qualifier.pkl`
- `model2_bracket.pkl`
- `precomputed_team_stats.pkl`
- `precomputed_elo_ratings.pkl`
- `metadata.json`

#### `bracket_predictor.py`

**Bracket simulation and inference logic.** Core functions:

- `predict_playoff_teams(model1, team_stats, teams_per_conference)`: Predicts top N teams per conference with probabilities
- `seed_conference(teams_ranked)`: Assigns playoff seeds (1-7)
- `simulate_game(model2, team_stats, elo_dict, team_a, team_b)`: Predicts matchup winner using Model2
- `predict_matchup(model2, team_stats, elo_dict, team_a, team_b)`: Returns probability and features for a matchup
- `make_matchup_features(team_stats, team_a, team_b, elo_dict)`: Creates diff features (team - opponent)
- `compute_team_stats_for_season(df, season)`: Aggregates per-game rows into team-level stats
- `compute_elo_ratings(df, season)`: Computes Elo ratings from regular-season games
- `run_automl_bracket(model1, model2, df, season)`: End-to-end AutoML bracket prediction (playoff qualification + simulation)
- `run_automl_bracket_inference(model1, model2, precomputed_team_stats, precomputed_elo_ratings, season)`: Inference-only version (no CSV dependency)
- `run_manual_bracket(model2, df, season, afc_seeds, nfc_seeds)`: Manual team selection mode
- `run_manual_bracket_inference(model2, precomputed_team_stats, precomputed_elo_ratings, season, afc_seeds, nfc_seeds)`: Inference-only manual mode

Data structures:

- `TEAM_TO_CONFERENCE`: Dict mapping team full names to 'AFC' or 'NFC'
- `TEAM_ABBREV_TO_FULL`: Dict mapping abbreviations to full names

#### `Scraping_Data.py`

**Data scraper** (replace with actual filename if different). Fetches NFL game data from nflverse using `nflreadpy`:

- Downloads per-game team rows for specified seasons (e.g., 2005-2024)
- Saves to `data_files/nflverse_stats_{year}.csv` (one file per season)
- Columns include: `season`, `week`, `season_type`, `team`, `opponent_team`, `points`, `passing_yards`, `rushing_yards`, `turnovers`, `def_takeaways`, etc.

Run with:

```bash
python Scraping_Data.py
```

#### `utils/model_io.py`

**Model bundle I/O helpers.** Functions:

- `load_model_and_optional_features(model_folder)`: Loads models, precomputed features, and metadata from a bundle
- `load_csvs_into_df(data_folder)`: Fallback CSV loading if precomputed features not found
- `md5_of_file(filepath)`: Computes MD5 hash for model versioning

Used by `app.py` to load models and handle feature fallback logic.

### Folders

#### `data_files/`

Stores scraped CSV files:

```
data_files/
├── nflverse_stats_2005.csv
├── nflverse_stats_2006.csv
├── ...
└── nflverse_stats_2024.csv
```

Each CSV contains per-game team rows with columns like `season`, `week`, `team`, `opponent_team`, `points`, `passing_yards`, etc.

#### `models/`

Contains trained model bundles:

```
models/
├── model_2005-2014/
│   ├── model1_playoff_qualifier.pkl
│   ├── model2_bracket.pkl
│   ├── precomputed_team_stats.pkl
│   ├── precomputed_elo_ratings.pkl
│   └── metadata.json
├── model_2010-2019/
│   └── ...
└── model_2015-2024/
    └── ...
```

Each folder represents a training window (e.g., `model_2015-2024` trained on seasons 2015-2024).

#### `static/`

Static assets (images, icons):

```
static/
└── nfl.png  # Logo displayed on landing page
```

---

## Data Pipeline & Scraping

### Data Source

Data is scraped from **nflverse** (https://github.com/nflverse) using `nflreadpy` library:

- `nflreadpy.import_weekly_data(years=[...])` fetches per-game team rows
- Each row represents one team's performance in one game
- Columns include: `season`, `week`, `season_type` ('REG' or 'POST'), `team`, `opponent_team`, `points`, `passing_yards`, `rushing_yards`, `turnovers`, `def_takeaways`, etc.

### Scraper Behavior

The scraper (`Scraping_Data.py` or similar):

1. Defines a year range (e.g., `range(2005, 2025)` for seasons 2005-2024)
2. Calls `nflreadpy.import_weekly_data(years=year_list)`
3. Saves each season to `data_files/nflverse_stats_{year}.csv`
4. Logs progress and handles errors (missing seasons, API timeouts)

**Example command:**

```bash
python Scraping_Data.py
# Output:
# Scraping season 2005...
# Saved to data_files/nflverse_stats_2005.csv
# Scraping season 2006...
# ...
```

### CSV Schema

Each CSV has the following columns (may vary by nflverse version):

- `season` (int): Year (e.g., 2024)
- `week` (int): Week number (1-18 regular season, 19-22 playoffs)
- `season_type` (str): 'REG' or 'POST'
- `team` (str): Team abbreviation (e.g., 'KC', 'SF')
- `opponent_team` (str): Opponent abbreviation
- `points` (float): Points scored (computed if missing)
- `passing_yards`, `rushing_yards`, `receiving_yards` (float)
- `turnovers`, `def_takeaways` (int): Turnovers committed, takeaways by defense
- `passing_tds`, `rushing_tds`, `receiving_tds`, `field_goals_made`, `extra_points_made` (int)
- And many more...

---

## Feature Engineering

### Overview

Feature engineering transforms raw per-game rows into season-level team statistics suitable for ML models. The pipeline is in `compute_team_stats_for_season()` in `bracket_predictor.py`.

### Aggregation Logic

For each team in a season, we aggregate regular-season games only (`season_type == 'REG'`):

- **Sum aggregations**: Total passing yards, rushing yards, turnovers, takeaways, etc.
- **Mean aggregations**: Average points per game, average yards, etc.
- **Derived features**: Computed from sums/means

**Key columns aggregated:**

- `points`: Mean and sum
- `passing_yards`, `rushing_yards`, `receiving_yards`: Sum
- `turnovers`, `def_takeaways`: Sum
- `passing_tds`, `rushing_tds`, `receiving_tds`, `field_goals_made`: Sum

**Example aggregation:**

```python
team_stats = df_reg.groupby('team_full').agg({
    'points': ['mean', 'sum'],
    'passing_yards': 'sum',
    'turnovers': 'sum',
    'def_takeaways': 'sum',
    # ... more columns
})
```

### Derived Features

#### 1. Points Calculation

If `points` column is missing:

```python
points = (passing_tds + rushing_tds + receiving_tds) * 6 + field_goals_made * 3 + extra_points_made
```

#### 2. Point Differential

```python
point_diff = points_sum - (opponent points summed across all games)
```

Measures offensive output minus defensive points allowed.

#### 3. Turnover Margin

```python
turnover_margin = def_takeaways_sum - turnovers_sum
```

Positive values indicate more takeaways than turnovers (good).

#### 4. Recent Average Points (Rolling Window)

For playoff prediction, we compute a rolling 4-week average of points:

```python
recent_avg_points = df.sort_values('week').tail(4)['points'].mean()
```

Captures late-season momentum.

#### 5. Differential Features (Model 2 Only)

For bracket prediction, we create **diff features** by subtracting opponent stats from team stats:

```python
diff_points_mean = team_a_points_mean - team_b_points_mean
diff_point_diff = team_a_point_diff - team_b_point_diff
diff_elo = elo_dict[team_a] - elo_dict[team_b]
diff_recent_avg_points = team_a_recent - team_b_recent
# ... all features
```

This creates a **symmetric representation** where positive values favor team_a.

#### 6. Elo Ratings

Elo ratings are computed using a simple algorithm in `compute_elo_ratings()`:

- Initialize all teams at 1500
- For each game, update Elo based on win/loss and expected probability
- Formula: `Elo_new = Elo_old + K * (actual_score - expected_score)`
- `K = 20` (learning rate)
- Expected score: `1 / (1 + 10^((Elo_opponent - Elo_team) / 400))`

**Why Elo?** Captures strength-of-schedule and cumulative performance better than raw win-loss records.

### Feature List (Model 1)

Model 1 uses aggregated team stats:

```python
feature_cols = [
    'points_mean', 'points_sum', 'passing_yards_sum', 'rushing_yards_sum',
    'turnovers_sum', 'def_takeaways_sum', 'turnover_margin', 'point_diff',
    'recent_avg_points', 'elo', 'passing_tds_sum', 'rushing_tds_sum',
    'third_down_pct', 'red_zone_pct', ...
]
```

### Feature List (Model 2)

Model 2 uses **diff features**:

```python
matchup_feature_cols = [
    'diff_points_mean', 'diff_point_diff', 'diff_elo', 'diff_turnovers_sum',
    'diff_turnover_margin', 'diff_recent_avg_points', 'diff_passing_yards_sum',
    'diff_rushing_yards_sum', ...
]
```

### Reproducibility

All random operations (train/test splits, model training) use `random_state=42`. For global reproducibility:

```python
import numpy as np
np.random.seed(42)
```

---

## Models

### Model 1: Playoff Qualifier

**Task**: Binary classification predicting whether a team makes the playoffs (`made_playoffs = 1`) or not (`made_playoffs = 0`).

**Algorithm**:

```python
from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',  # Handle class imbalance (only 14/32 teams make playoffs)
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)
```

**Training Data**:

- Input: Regular-season aggregated team stats (see Feature Engineering)
- Label: `made_playoffs` (1 if team appears in any `season_type == 'POST'` row, else 0)
- Train/test split: 80/20 stratified by `made_playoffs`

**Training Pipeline**:

1. Load CSVs from `data_files/`
2. For each season, compute team stats and label teams as `made_playoffs = 1` if they have postseason games
3. Aggregate across all seasons into one DataFrame
4. Exclude columns: `team`, `season`, `made_playoffs`, `conference`, etc.
5. Train/test split
6. Train model on training set
7. Evaluate on test set (accuracy, precision, recall, F1, ROC-AUC)

**Outputs**:

- Playoff probability for each team (via `predict_proba()`)
- Top 7 teams per conference (via `predict_playoff_teams()`)

**Deterministic Behavior**:
Setting `random_state=42` ensures identical results across runs.

### Model 2: Bracket Predictor

**Task**: Binary classification predicting which team wins a playoff matchup (`won_game = 1` for team_a, `won_game = 0` for team_b).

**Algorithm**:

```python
model2 = RandomForestClassifier(
    n_estimators=300,  # More trees for better generalization
    random_state=42,
    class_weight='balanced',  # Roughly 50/50 wins in playoff games
    max_depth=None
)
```

**Training Data**:

- Input: **Diff features** (`diff_points_mean`, `diff_elo`, etc.) for each playoff matchup
- Label: `won_game` (1 if team_a won, 0 if team_b won)
- Constructed from playoff games (`season_type == 'POST'`)

**Constructing Training Examples**:
For each playoff game:

1. Identify team_a (home or higher seed) and team_b (away or lower seed)
2. Compute team_a and team_b stats for that season
3. Create diff features: `diff_* = team_a_* - team_b_*`
4. Label: `won_game = 1` if team_a won (more points), else `won_game = 0`

**Training Pipeline**:

1. Extract all playoff games from CSVs
2. Pair each game into a matchup (team_a vs team_b)
3. Compute diff features using `make_matchup_features()`
4. Train/test split
5. Train model on training set
6. Evaluate on test set

**Inference**:

- `predict_proba(X)[0, 1]` returns probability team_a wins
- `simulate_game(model2, team_stats, elo_dict, team_a, team_b)` uses this probability to determine winner

**Feature Alignment**:
Model 2 expects diff features in a specific order (determined during training). The function `make_matchup_features()` ensures inference features match training features:

```python
def make_matchup_features(team_stats, team_a, team_b, elo_dict):
    # Extract team_a and team_b stats
    # Compute diff features
    # Return DataFrame with columns matching model2.feature_names_in_
    return matchup_df
```

---

## Model Export Format & Bundles

### Bundle Structure

A model bundle is a folder: `models/model_YYYY-YYYY/` containing:

```
model_2015-2024/
├── model1_playoff_qualifier.pkl      # Model 1 (joblib)
├── model2_bracket.pkl                 # Model 2 (joblib)
├── precomputed_team_stats.pkl         # DataFrame of team stats for latest season
├── precomputed_elo_ratings.pkl        # Dict of Elo ratings for latest season
└── metadata.json                      # Bundle metadata
```

### `metadata.json` Format

```json
{
  "start_year": 2015,
  "end_year": 2024,
  "latest_season": 2024,
  "exported_at": "2025-01-01T12:00:00Z",
  "model1_md5": "abc123...",
  "model2_md5": "def456...",
  "feature_names": ["points_mean", "points_sum", "elo", ...],
  "model2_features": ["diff_points_mean", "diff_elo", ...],
  "has_inference_data": true,
  "notes": "Trained on seasons 2015-2024, exported for inference"
}
```

**Key Fields**:

- `start_year`, `end_year`: Training window
- `latest_season`: Season for which precomputed features were generated (used for inference)
- `model1_md5`, `model2_md5`: Model file checksums (for versioning)
- `feature_names`: Feature columns used by Model 1
- `model2_features`: Feature columns used by Model 2
- `has_inference_data`: `true` if precomputed features are included

### Why Precomputed Features?

Precomputed features allow **inference without CSV dependencies**:

- App loads `precomputed_team_stats.pkl` instead of reading/parsing CSVs
- Faster startup time
- No need to ship 20 years of CSVs with the app
- Ensures feature engineering is identical between training and inference

### `export_model_bundle()` Function

Located in `training_model.py`:

```python
def export_model_bundle(
    model1,
    model2,
    team_stats,        # DataFrame from latest season
    elo_dict,          # Elo ratings from latest season
    start_year,
    end_year,
    output_folder      # e.g., 'models/model_2015-2024'
):
    """
    Export trained models and precomputed features to a bundle folder.
    """
    import joblib
    import json
    from pathlib import Path
    from datetime import datetime

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(model1, f'{output_folder}/model1_playoff_qualifier.pkl')
    joblib.dump(model2, f'{output_folder}/model2_bracket.pkl')

    # Save precomputed features
    joblib.dump(team_stats, f'{output_folder}/precomputed_team_stats.pkl')
    joblib.dump(elo_dict, f'{output_folder}/precomputed_elo_ratings.pkl')

    # Compute MD5 hashes
    model1_md5 = md5_of_file(f'{output_folder}/model1_playoff_qualifier.pkl')
    model2_md5 = md5_of_file(f'{output_folder}/model2_bracket.pkl')

    # Export metadata
    metadata = {
        'start_year': start_year,
        'end_year': end_year,
        'latest_season': end_year,
        'exported_at': datetime.utcnow().isoformat() + 'Z',
        'model1_md5': model1_md5,
        'model2_md5': model2_md5,
        'feature_names': list(model1.feature_names_in_),
        'model2_features': list(model2.feature_names_in_),
        'has_inference_data': True
    }

    with open(f'{output_folder}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Bundle exported to {output_folder}/")
```

---

## Training a New Model

### Step 1: Scrape Data

Ensure you have CSV files in `data_files/`:

```bash
python Scraping_Data.py
# This downloads nflverse data for seasons 2005-2024 (or your configured range)
# Output: data_files/nflverse_stats_{year}.csv
```

### Step 2: Train Models

Run the training script:

```bash
python training_model.py
```

**What happens:**

1. Loads all CSVs from `data_files/`
2. Computes team stats for each season
3. Labels teams as `made_playoffs = 1/0`
4. Trains Model 1 (playoff qualifier)
5. Constructs playoff matchup training data
6. Trains Model 2 (bracket predictor)
7. Evaluates both models on test sets (prints accuracy, F1, ROC-AUC)
8. **Does NOT auto-export** — you must call `export_model_bundle()` manually

### Step 3: Export Bundle

After training, export the bundle with precomputed features:

```python
from training_model import export_model_bundle, train_models
from bracket_predictor import compute_team_stats_for_season, compute_elo_ratings
import pandas as pd

# Assume you have trained models and a DataFrame df
start_year = 2015
end_year = 2024

# Train models (returns model1, model2, and evaluation metrics)
model1, model2, metrics = train_models(df, start_year, end_year)

# Compute precomputed features for latest season (2024)
latest_df = df[df['season'] == end_year]
team_stats = compute_team_stats_for_season(latest_df, end_year)
elo_dict = compute_elo_ratings(df, end_year)

# Export bundle
export_model_bundle(
    model1=model1,
    model2=model2,
    team_stats=team_stats,
    elo_dict=elo_dict,
    start_year=start_year,
    end_year=end_year,
    output_folder=f'models/model_{start_year}-{end_year}'
)
```

**Output:**

```
Bundle exported to models/model_2015-2024/
├── model1_playoff_qualifier.pkl
├── model2_bracket.pkl
├── precomputed_team_stats.pkl
├── precomputed_elo_ratings.pkl
└── metadata.json
```

### Step 4: Verify Bundle

Run validation checks:

```bash
python run_checks.py --bundle models/model_2015-2024
```

---

## Running the Streamlit App

### Start the App

```bash
streamlit run app.py
```

**Expected Output:**

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

### UI Flow

1. **Landing Page**: Click "Get Started" button
2. **Choose Model**: Dropdown to select a model bundle (e.g., `model_2015-2024`)
3. **Prediction Mode**:
   - **AutoML**: Click "Predict Bracket" to run full prediction (Model1 → Model2 → bracket)
   - **Choose Teams**: Manually select 14 playoff teams (7 per conference), then click "Predict Bracket"
4. **Bracket Visualization**: 7-column layout (AFC Wild, AFC Div, AFC Conf, Super Bowl, NFC Conf, NFC Div, NFC Wild)
5. **Interactive Buttons**: Click any winning team button to open a modal showing:
   - **Win Probability (Model 2)**: Bar chart and percentages for both teams
   - **Playoff Probability (Model 1)**: % chance each team makes playoffs
   - **Top Contributing Factors**: Top 5 features by `abs(diff_value) * importance`
   - **Season Stats**: Table of aggregated stats for both teams
6. **Champion Display**: Shows predicted Super Bowl champion at the bottom
7. **Reset/Re-run**: Click "Reset Bracket" (manual mode) or switch to AutoML to re-predict

### Debug Panel

If you enable debug mode in `app.py`, a sidebar panel shows:

- Model MD5 hashes
- Metadata season
- Whether precomputed features were loaded
- Feature name counts
- Session state inspection

---

## How Predictions Work

### End-to-End Flow (AutoML Mode)

1. **User clicks "Predict Bracket"**
2. **App loads model bundle**:
   ```python
   models = load_model_version('model_2015-2024')
   # Returns: {'model1': ..., 'model2': ..., 'precomputed_team_stats': ..., 'precomputed_elo_ratings': ...}
   ```
3. **Check for precomputed features**:
   - If `models['precomputed_team_stats']` exists → use inference mode
   - Else → load CSVs from `data_files/`, compute features on the fly
4. **Predict playoff teams** (Model 1):
   ```python
   playoff_teams = predict_playoff_teams(model1, team_stats, teams_per_conference=7)
   # Returns: {'AFC': [(team, prob), ...], 'NFC': [(team, prob), ...]}
   ```
5. **Seed conferences**:
   ```python
   afc_seeds = seed_conference(playoff_teams['AFC'])  # {1: 'Kansas City Chiefs', 2: 'Buffalo Bills', ...}
   nfc_seeds = seed_conference(playoff_teams['NFC'])
   ```
6. **Fill bracket slots**:
   - Wild Card matchups: #2 vs #7, #3 vs #6, #4 vs #5
   - #1 seed gets bye
7. **Simulate Wild Card round**:
   ```python
   for matchup in wild_card_matchups:
       winner = simulate_game(model2, team_stats, elo_dict, team_a, team_b)
       wc_results.append({'matchup': f'{team_a} vs {team_b}', 'winner': winner, 'probability': prob})
   ```
8. **Simulate Divisional round**:
   - #1 seed plays lowest remaining seed
   - Other winners play each other
9. **Simulate Conference Championship**: AFC winner vs NFC winner
10. **Simulate Super Bowl**: AFC champion vs NFC champion
11. **Store results in session state**:
    ```python
    st.session_state.bracket_result = {
        'wild_card_results': [...],
        'divisional_results': [...],
        'conference_results': {'AFC': {...}, 'NFC': {...}},
        'super_bowl_result': {...},
        'champion': 'Kansas City Chiefs',
        'bracket_slots': {...}
    }
    ```
12. **Render bracket UI**: Update buttons with team names, enable winning teams

### `simulate_game()` Details

```python
def simulate_game(model2, team_stats, elo_dict, team_a, team_b):
    """
    Predict winner of team_a vs team_b using Model 2.

    Returns:
        winner (str): Team name of winner
        probability (float): Probability winner wins (0.5-1.0)
    """
    matchup_features = make_matchup_features(team_stats, team_a, team_b, elo_dict)
    prob_a_wins = model2.predict_proba(matchup_features)[0, 1]

    # Deterministic: choose team with higher probability
    if prob_a_wins >= 0.5:
        return team_a, prob_a_wins
    else:
        return team_b, 1 - prob_a_wins
```

### Session State Management

Streamlit uses `st.session_state` to persist data across reruns:

- `st.session_state.bracket_result`: Full bracket prediction results
- `st.session_state.bracket_filled`: Boolean flag (True if prediction complete)
- `st.session_state.bracket_models`: Dict of loaded models
- `st.session_state.bracket_team_stats`: Precomputed team stats
- `st.session_state.bracket_elo_dict`: Precomputed Elo ratings
- `st.session_state.playoff_probs_dict`: Dict of playoff probabilities for all teams (for modal display)
- `st.session_state.manual_bracket_predicted`: Boolean (True if manual mode predicted, shows buttons instead of dropdowns)

### Button Enable/Disable Logic

Function `determine_button_enabled()`:

```python
def determine_button_enabled(slot_key, result):
    """
    Returns True if button should be enabled (clickable).

    Rules:
    - Bye teams: disabled (no opponent, no stats)
    - Losing teams: disabled (no reason to show stats)
    - Winning teams: enabled (show matchup details)
    """
    if '_bye' in slot_key:
        return False  # Bye teams have no opponent

    # Check if this slot's team is a winner in any game
    team = result['bracket_slots'].get(slot_key)
    if not team:
        return False

    # Check all game results to see if this team won
    for game in result.get('wild_card_results', []) + result.get('divisional_results', []):
        if game['winner'] == team:
            return True

    # Check conference and Super Bowl results
    for conf_result in result.get('conference_results', {}).values():
        if conf_result.get('winner') == team:
            return True

    sb_result = result.get('super_bowl_result', {})
    if sb_result.get('winner') == team:
        return True

    return False
```

---

## Testing & Validation

To validate your model predictions:

1. **Run predictions in the app**: Use both AutoML and Manual modes to ensure consistency
2. **Compare multiple model bundles**: Train models on different year ranges (e.g., 2005-2014, 2010-2019, 2015-2024) and compare champion predictions
3. **Inspect feature importance**: Check which features contribute most to predictions via the interactive modals
4. **Verify bracket integrity**: Ensure no duplicate teams appear in bracket slots and exactly 7 seeds per conference
5. **Check metadata**: Verify `metadata.json` contains correct season range and feature names

**Manual Validation Checklist:**

```python
# After running a prediction, verify:
import joblib
import json

# 1. Check model files exist
model1 = joblib.load('models/model_2015-2024/model1_playoff_qualifier.pkl')
model2 = joblib.load('models/model_2015-2024/model2_bracket.pkl')

# 2. Check metadata
with open('models/model_2015-2024/metadata.json') as f:
    metadata = json.load(f)
    print(f"Training range: {metadata['start_year']}-{metadata['end_year']}")
    print(f"Features: {len(metadata['feature_names'])}")

# 3. Verify precomputed features
team_stats = joblib.load('models/model_2015-2024/precomputed_team_stats.pkl')
print(f"Teams in dataset: {len(team_stats)}")
```

---

## Interpreting Results

### Reading Probabilities

**Model 1 (Playoff Qualifier)**:

- Output: Probability each team makes playoffs (0.0 to 1.0)
- Example: `Kansas City Chiefs: 0.92` → 92% chance of making playoffs
- Top 7 teams per conference are selected as playoff seeds

**Model 2 (Bracket Predictor)**:

- Output: Probability team_a beats team_b (0.5 to 1.0)
- Example: `Chiefs vs Bills: 0.68` → Chiefs have 68% chance of winning
- Deterministic: always chooses team with higher probability

### Feature Importance

The modal displays **Top 5 Contributing Factors** computed as:

```python
contribution = abs(diff_value) * feature_importance
```

**Example:**

```
1. diff_elo: +125.3 (Importance: 0.18) → Chiefs have 125 Elo advantage
2. diff_point_diff: +87.2 (Importance: 0.15) → Chiefs scored 87 more net points
3. diff_recent_avg_points: +4.8 (Importance: 0.12) → Chiefs averaging 4.8 more points recently
4. diff_turnover_margin: +8 (Importance: 0.10) → Chiefs have +8 better turnover margin
5. diff_passing_yards_sum: +420 (Importance: 0.09) → Chiefs have 420 more passing yards
```

**Positive diff values favor team_a (listed first in matchup).**

### Model Agreement

If multiple model bundles (trained on different years) predict the same champion, this indicates:

- High confidence in prediction
- Champion team has consistently strong features across training windows

**Why might older models predict the same champion?**
Models evaluate **current season features** (e.g., 2024 team stats), not training data. Even a model trained on 2005-2014 uses 2024 Elo ratings and stats for inference.

### Uncertainty

Sources of uncertainty:

1. **Model stochasticity**: Random forest hyperparameters (e.g., `n_estimators`)
2. **Training data**: Different training windows produce different decision trees
3. **Feature noise**: Small changes in features (e.g., 1 turnover) can flip predictions
4. **Bracket simulation**: Deterministic (always argmax), but probabilities are continuous

**Recommendations:**

- Run Monte Carlo simulations with `--stochastic` to sample from probabilities
- Compare predictions from multiple model bundles (2005-2014, 2010-2019, 2015-2024)
- Check sensitivity analysis results

---

## Limitations & Bias

### Model Limitations

1. **Feature Simplicity**: Models use only aggregated stats (no play-by-play, no player-level data)
2. **Algorithm**: Random forests are interpretable but less powerful than gradient boosting or neural networks
3. **Training Data**: Limited to ~20 years of data; playoff formats changed over time (e.g., 6 teams → 7 teams in 2020)
4. **Deterministic Simulation**: Always chooses argmax probability (no stochastic sampling by default)

### Data Limitations

1. **Missing Context**:
   - Injuries (e.g., Patrick Mahomes injured → Chiefs win probability drops)
   - Trades/roster changes mid-season
   - Weather conditions (snow, wind)
   - Home-field advantage beyond Elo
2. **Regular Season Only**: Playoff performance is qualitatively different (higher stakes, better defenses)
3. **Incomplete Features**: No special teams stats, no coaching metrics, no advanced analytics (EPA, DVOA)

### Bias Sources

1. **Historical Bias**: Models favor teams with historically strong performance (e.g., Patriots 2000-2019)
2. **Elo Initialization**: All teams start at 1500; true strength may differ
3. **Class Imbalance**: Only 14/32 teams make playoffs → Model 1 may underpredict underdogs
4. **Playoff Sample Size**: Fewer playoff games per season → Model 2 has less training data

### Suggestions for Improvement

1. **Add player-level features**:
   - QB rating, pass rush stats, injury reports
   - Roster turnover, salary cap data
2. **Incorporate betting lines**:
   - Vegas odds as a feature (wisdom of crowds)
3. **Use gradient boosting**:
   - `XGBoost` or `LightGBM` for better performance
4. **Ensemble methods**:
   - Combine Random Forest, Logistic Regression, Neural Network predictions
5. **Stochastic simulation**:
   - Sample from Model 2 probabilities instead of argmax
   - Run 1000 simulations and report distribution
6. **Advanced metrics**:
   - EPA (Expected Points Added), DVOA (Defense-adjusted Value Over Average)
   - Play-level features (e.g., 3rd down conversion rate, red zone efficiency)

---

## Appendix

### A. Example `metadata.json`

```json
{
  "start_year": 2015,
  "end_year": 2024,
  "latest_season": 2024,
  "exported_at": "2025-01-01T18:30:00Z",
  "model1_md5": "a3f7b2c1d4e5f6g7h8i9j0k1l2m3n4o5",
  "model2_md5": "b4g8c2d5e6f7g8h9i0j1k2l3m4n5o6p7",
  "feature_names": [
    "points_mean",
    "points_sum",
    "passing_yards_sum",
    "rushing_yards_sum",
    "turnovers_sum",
    "def_takeaways_sum",
    "turnover_margin",
    "point_diff",
    "recent_avg_points",
    "elo",
    "passing_tds_sum",
    "rushing_tds_sum",
    "field_goals_made_sum",
    "third_down_pct",
    "red_zone_pct"
  ],
  "model2_features": [
    "diff_points_mean",
    "diff_point_diff",
    "diff_elo",
    "diff_turnovers_sum",
    "diff_turnover_margin",
    "diff_recent_avg_points",
    "diff_passing_yards_sum",
    "diff_rushing_yards_sum",
    "diff_passing_tds_sum",
    "diff_rushing_tds_sum"
  ],
  "has_inference_data": true,
  "notes": "Trained on seasons 2015-2024 using RandomForestClassifier. Model1: n_estimators=200, Model2: n_estimators=300. Precomputed features for season 2024 included."
}
```

### B. Example `run_automl_bracket()` Output

```python
result = {
    'afc_seeds': {
        1: 'Kansas City Chiefs',
        2: 'Buffalo Bills',
        3: 'Cincinnati Bengals',
        4: 'Jacksonville Jaguars',
        5: 'Los Angeles Chargers',
        6: 'Miami Dolphins',
        7: 'Pittsburgh Steelers'
    },
    'nfc_seeds': {
        1: 'San Francisco 49ers',
        2: 'Philadelphia Eagles',
        3: 'Detroit Lions',
        4: 'Tampa Bay Buccaneers',
        5: 'Dallas Cowboys',
        6: 'Los Angeles Rams',
        7: 'Green Bay Packers'
    },
    'wild_card_results': [
        {'matchup': 'Buffalo Bills vs Pittsburgh Steelers', 'winner': 'Buffalo Bills', 'probability': 0.82},
        {'matchup': 'Cincinnati Bengals vs Miami Dolphins', 'winner': 'Cincinnati Bengals', 'probability': 0.67},
        {'matchup': 'Jacksonville Jaguars vs Los Angeles Chargers', 'winner': 'Jacksonville Jaguars', 'probability': 0.54},
        # NFC...
    ],
    'divisional_results': [
        {'matchup': 'Kansas City Chiefs vs Jacksonville Jaguars', 'winner': 'Kansas City Chiefs', 'probability': 0.78},
        {'matchup': 'Buffalo Bills vs Cincinnati Bengals', 'winner': 'Buffalo Bills', 'probability': 0.63},
        # NFC...
    ],
    'conference_results': {
        'AFC': {'matchup': 'Kansas City Chiefs vs Buffalo Bills', 'winner': 'Kansas City Chiefs', 'probability': 0.71},
        'NFC': {'matchup': 'San Francisco 49ers vs Philadelphia Eagles', 'winner': 'San Francisco 49ers', 'probability': 0.68}
    },
    'super_bowl_result': {
        'matchup': 'Kansas City Chiefs vs San Francisco 49ers',
        'winner': 'Kansas City Chiefs',
        'probability': 0.65
    },
    'champion': 'Kansas City Chiefs',
    'bracket_slots': {
        'afc_bye': 'Kansas City Chiefs',
        'afc_wild1': 'Buffalo Bills',
        'afc_wild2': 'Pittsburgh Steelers',
        'afc_wild3': 'Cincinnati Bengals',
        'afc_wild4': 'Miami Dolphins',
        'afc_wild5': 'Jacksonville Jaguars',
        'afc_wild6': 'Los Angeles Chargers',
        'afc_div_win_1': 'Kansas City Chiefs',
        'afc_div_win_2': 'Jacksonville Jaguars',
        'afc_div_win_3': 'Buffalo Bills',
        'afc_div_win_4': 'Cincinnati Bengals',
        'afc_conf_win_1': 'Kansas City Chiefs',
        'afc_conf_win_2': 'Buffalo Bills',
        'sb_team_a': 'Kansas City Chiefs',
        'sb_team_b': 'San Francisco 49ers',
        # NFC slots...
    }
}
```

### C. Example Modal Content (Format)

When user clicks a bracket button (e.g., "Kansas City Chiefs" in Divisional round):

```
Modal Title: "Divisional Round"
Subtitle: "Kansas City Chiefs vs Buffalo Bills"

[Left Column]
Win Probability (Model 2):
  Bar Chart:
    Kansas City Chiefs: ████████████████████ 71.2%
    Buffalo Bills:      █████████ 28.8%

  Metrics:
    Kansas City Chiefs: 71.2%
    Buffalo Bills: 28.8%

Playoff Probability (Model 1):
  Chance Kansas City Chiefs makes playoffs: 94.3%
  Chance Buffalo Bills makes playoffs: 87.6%

[Right Column]
Top Contributing Factors:
  1. diff_elo: +127.8 (Importance: 0.182)
     → Chiefs have 128 point Elo advantage

  2. diff_point_diff: +89.5 (Importance: 0.154)
     → Chiefs scored 90 more net points this season

  3. diff_recent_avg_points: +5.2 (Importance: 0.121)
     → Chiefs averaging 5.2 more points in last 4 games

  4. diff_turnover_margin: +9 (Importance: 0.105)
     → Chiefs have +9 better turnover margin

  5. diff_passing_yards_sum: +457 (Importance: 0.093)
     → Chiefs have 457 more passing yards this season

Season Stats:
  | Stat                | Kansas City Chiefs | Buffalo Bills |
  |---------------------|--------------------|---------------|
  | Points (mean)       | 28.7               | 26.3          |
  | Points (sum)        | 487                | 447           |
  | Point Diff          | +142               | +89           |
  | Turnover Margin     | +12                | +3            |
  | Elo Rating          | 1627               | 1499          |
  | Passing Yards (sum) | 4523               | 4066          |
  | Rushing Yards (sum) | 2134               | 1987          |
```
