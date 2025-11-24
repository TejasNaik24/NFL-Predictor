import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Merge all CSVs
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

df = merge_data()

# Get playoff teams per season
playoff_teams = (
    df[df['season_type'].str.lower() == 'post']
    .groupby('season')['team']
    .unique()
)

df = df[df['season_type'].str.lower() == 'reg'].copy()

# Aggregate per season per team
agg_cols = [
    'passing_yards','rushing_yards','receiving_yards',
    'passing_tds','rushing_tds','receiving_tds',
    'passing_interceptions','rushing_fumbles','receiving_fumbles',
    'sacks_suffered','def_sacks','def_interceptions','def_fumbles_forced',
    'points'
]

# create points if not exist
if 'points' not in df.columns:
    df['points'] = (
        df.get("passing_tds", 0)*6 +
        df.get("rushing_tds", 0)*6 +
        df.get("receiving_tds", 0)*6 +
        df.get("fg_made", 0)*3 +
        df.get("pat_made", 0)
    )

team_stats = df.groupby(['season','team'])[agg_cols].agg(['sum','mean']).reset_index()
team_stats.columns = ['_'.join(filter(None, col)).strip('_') for col in team_stats.columns.values]

# Point differential per team
team_stats['point_diff'] = team_stats['points_sum'] - team_stats['points_mean']

# Turnover margin
team_stats['turnovers_sum'] = team_stats['passing_interceptions_sum'] + \
                              team_stats['rushing_fumbles_sum'] + \
                              team_stats['receiving_fumbles_sum']
team_stats['def_takeaways_sum'] = team_stats['def_interceptions_sum'] + \
                                  team_stats['def_fumbles_forced_sum']
team_stats['turnover_margin'] = team_stats['def_takeaways_sum'] - team_stats['turnovers_sum']

# Playoff label
team_stats['made_playoffs'] = team_stats.apply(
    lambda row: 1 if row['team'] in playoff_teams.get(row['season'], []) else 0,
    axis=1
)

# 5️⃣ Train/Test Split
X = team_stats.drop(columns=['team','season','made_playoffs'])
y = team_stats['made_playoffs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train Random Forest with balanced class weights
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 7️⃣ Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
