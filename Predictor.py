import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

playoff_teams = (
    df[df['season_type'].str.lower() == 'post']
    .groupby('season')['team']
    .unique()
)

df = df[df['season_type'].str.lower() == 'reg'].copy()

df["points"] = (
    df.get("passing_tds", 0) * 6 +
    df.get("rushing_tds", 0) * 6 +
    df.get("receiving_tds", 0) * 6 +
    df.get("fg_made", 0) * 3 +
    df.get("pat_made", 0)
)

merged = df.merge(
    df[["season", "week", "team", "points"]],
    left_on=["season", "week", "opponent_team"],
    right_on=["season", "week", "team"],
    suffixes=("", "_opp")
)

merged["team_win"] = (merged["points"] > merged["points_opp"]).astype(int)

merged["made_playoffs"] = merged.apply(
    lambda row: 1 if row["team"] in playoff_teams.get(row["season"], []) else 0,
    axis=1
)

keep_cols = [
    "season", "week", "team", "opponent_team",
    "team_win", "passing_yards", "rushing_yards",
    "passing_epa", "rushing_epa", "points",
    "points_opp", "made_playoffs"
]
final_df = merged[keep_cols].dropna()

X = final_df.drop(columns=["team", "opponent_team", "season", "week", "made_playoffs"], errors="ignore")
y = final_df["made_playoffs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
