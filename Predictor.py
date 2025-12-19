import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===== Merge all CSVs =====
def merge_data(folder_path="data_files"):
    files = glob.glob(f"{folder_path}/*.csv")
    dfs = []
    for f in files:
        year = os.path.basename(f).split('.')[0]
        temp_df = pd.read_csv(f)
        temp_df['season'] = int(''.join(filter(str.isdigit, year)))
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    return df

df_full = merge_data()

# ===== Create points column if not exist =====
if 'points' not in df_full.columns:
    df_full['points'] = (
        df_full.get("passing_tds", 0)*6 +
        df_full.get("rushing_tds", 0)*6 +
        df_full.get("receiving_tds", 0)*6 +
        df_full.get("fg_made", 0)*3 +
        df_full.get("pat_made", 0)
    )

# =========================
# Model 1: Predict Playoff Teams
# =========================
df_reg = df_full[df_full['season_type'].str.lower() == 'reg'].copy()

agg_cols = [
    'passing_yards','rushing_yards','receiving_yards',
    'passing_tds','rushing_tds','receiving_tds',
    'passing_interceptions','rushing_fumbles','receiving_fumbles',
    'sacks_suffered','def_sacks','def_interceptions','def_fumbles_forced',
    'points'
]

team_stats = df_reg.groupby(['season','team'])[agg_cols].agg(['sum','mean']).reset_index()
team_stats.columns = ['_'.join(filter(None, col)).strip('_') for col in team_stats.columns.values]

# Extra metrics
team_stats['point_diff'] = team_stats['points_sum'] - team_stats['points_mean']
team_stats['turnovers_sum'] = team_stats['passing_interceptions_sum'] + team_stats['rushing_fumbles_sum'] + team_stats['receiving_fumbles_sum']
team_stats['def_takeaways_sum'] = team_stats['def_interceptions_sum'] + team_stats['def_fumbles_forced_sum']
team_stats['turnover_margin'] = team_stats['def_takeaways_sum'] - team_stats['turnovers_sum']

# Playoff label
playoff_teams = (
    df_full[df_full['season_type'].str.lower() == 'post']
    .groupby('season')['team']
    .unique()
)

team_stats['made_playoffs'] = team_stats.apply(
    lambda row: 1 if row['team'] in playoff_teams.get(row['season'], []) else 0,
    axis=1
)

# Train/Test split
X1 = team_stats.drop(columns=['team','season','made_playoffs'])
y1 = team_stats['made_playoffs']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Train Random Forest
model1 = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model1.fit(X_train1, y_train1)

# Evaluate
y_pred1 = model1.predict(X_test1)
print("Model 1: Predict Playoff Teams")
print("Accuracy:", accuracy_score(y_test1, y_pred1))
print(classification_report(y_test1, y_pred1))

# =========================
# Model 2: Predict Playoff Game Winners (Improved)
# =========================

playoff_df = df_full[df_full['season_type'].str.lower() == 'post'].copy()

if playoff_df.empty:
    raise ValueError("No playoff games found in the dataset.")

# Ensure points column exists
if 'points' not in playoff_df.columns:
    playoff_df['points'] = (
        playoff_df.get("passing_tds", 0)*6 +
        playoff_df.get("rushing_tds", 0)*6 +
        playoff_df.get("receiving_tds", 0)*6
    )

# Create game_id safely
teams_min = playoff_df[['team', 'opponent_team']].min(axis=1)
teams_max = playoff_df[['team', 'opponent_team']].max(axis=1)
playoff_df['game_id'] = (
    playoff_df['season'].astype(str) + "_" +
    playoff_df['week'].astype(str) + "_" +
    teams_min + "_" + teams_max
)

# Label: did team win this playoff game?
playoff_df['won_game'] = (
    playoff_df['points'] == playoff_df.groupby('game_id')['points'].transform('max')
).astype(int)

# Merge regular-season stats (diff features)
team_stats_renamed = team_stats.rename(
    columns=lambda c: f"team_{c}" if c not in ['season', 'team'] else c
)

playoff_df = playoff_df.merge(
    team_stats_renamed,
    on=['season', 'team'],
    how='left'
)

opp_stats = team_stats_renamed.rename(
    columns={
        'team': 'opponent_team',
        **{c: c.replace('team_', 'opp_') for c in team_stats_renamed.columns if c.startswith('team_')}
    }
)

playoff_df = playoff_df.merge(
    opp_stats,
    on=['season', 'opponent_team'],
    how='left'
)

# Create diff features
feature_cols = [c for c in playoff_df.columns if c.startswith('team_') and c.replace('team_', 'opp_') in playoff_df.columns]
for col in feature_cols:
    playoff_df[col.replace('team_', 'diff_')] = playoff_df[col] - playoff_df[col.replace('team_', 'opp_')]

# ---------- Add recent form features ----------
# Compute last 3 game average points and point_diff per team
recent_games = 3
df_full_sorted = df_full[df_full['season_type'].str.lower() == 'reg'].sort_values(['team','week'])
team_last_games = df_full_sorted.groupby('team').rolling(recent_games, on='week')[['points']].mean().reset_index()
team_last_games.rename(columns={'points':'recent_avg_points'}, inplace=True)

playoff_df = playoff_df.merge(
    team_last_games[['team','week','recent_avg_points']],
    on=['team','week'],
    how='left'
)

opp_last_games = df_full_sorted.groupby('team').rolling(recent_games, on='week')[['points']].mean().reset_index()
opp_last_games.rename(columns={'team':'opponent_team','points':'opp_recent_avg_points','week':'week'}, inplace=True)

playoff_df = playoff_df.merge(
    opp_last_games[['opponent_team','week','opp_recent_avg_points']],
    on=['opponent_team','week'],
    how='left'
)

playoff_df['diff_recent_avg_points'] = playoff_df['recent_avg_points'] - playoff_df['opp_recent_avg_points']

# ---------- Simple Elo rating ----------
# Initialize Elo 1500 for every team per season
elo = {}
for season in playoff_df['season'].unique():
    teams = df_full[df_full['season']==season]['team'].unique()
    for t in teams:
        elo[(season,t)] = 1500

# Update Elo after each game in df_full
k = 20
df_sorted = df_full.sort_values(['season','week'])
for _, row in df_sorted.iterrows():
    s = row['season']
    t = row['team']
    opp = row['opponent_team']
    pts_team = row.get('points',0)
    pts_opp = row.get('points',0)  # if opponent points are missing, set 0
    expected = 1/(1+10**((elo.get((s,opp),1500)-elo[(s,t)])/400))
    score = 1 if pts_team>pts_opp else 0
    elo[(s,t)] += k*(score-expected)

# Merge Elo into playoff_df
playoff_df['elo'] = playoff_df.apply(lambda r: elo.get((r['season'],r['team']),1500),axis=1)
playoff_df['opp_elo'] = playoff_df.apply(lambda r: elo.get((r['season'],r['opponent_team']),1500),axis=1)
playoff_df['diff_elo'] = playoff_df['elo'] - playoff_df['opp_elo']

# ---------- Build final X / y ----------
X2 = playoff_df[[c for c in playoff_df.columns if c.startswith('diff_')]]
y2 = playoff_df['won_game']

# Clean X
X2 = X2.replace([np.inf,-np.inf],np.nan).fillna(0)

# Train/test split
seasons = sorted(playoff_df['season'].unique())
if len(seasons) > 1:
    train_seasons = seasons[:-1]
    test_seasons = [seasons[-1]]
    train_idx = playoff_df['season'].isin(train_seasons)
    test_idx = playoff_df['season'].isin(test_seasons)
    X_train2 = X2.loc[train_idx]
    X_test2 = X2.loc[test_idx]
    y_train2 = y2.loc[train_idx]
    y_test2 = y2.loc[test_idx]
else:
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Train Random Forest
model2 = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
model2.fit(X_train2, y_train2)

# Evaluate
y_pred2 = model2.predict(X_test2)
print("Improved Playoff Game Winner Model")
print("Accuracy:", accuracy_score(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))
