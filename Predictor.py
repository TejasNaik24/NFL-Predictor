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

if 'opponent_team' in df.columns:
    df = pd.get_dummies(df, columns=['opponent_team'], drop_first=True)

df['made_playoffs'] = df.apply(
    lambda row: 1 if row['team'] in playoff_teams.get(row['season'], []) else 0,
    axis=1
)

X = df.drop(columns=['made_playoffs'])
y = df['made_playoffs']

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))