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
        temp_df['year'] = year
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    df['year'] = df['year'].str.extract('(\d+)').astype(int)
    return df

df = merge_data()

# --- Binary result for Home win ---
df['ResultBinary'] = df['Result'].apply(lambda x: 1 if x == 'Win' else 0)

# --- Initialize cumulative stats dictionary ---
teams = df['Winner/tie'].unique().tolist() + df['Loser/tie'].unique().tolist()
teams = list(set(teams))
cum_stats = {team: {'games': 0, 'pts': 0, 'yds': 0, 'to': 0, 'results': []} for team in teams}

# --- Prepare pre-game features ---
home_pts_avg = []
away_pts_avg = []
home_yds_avg = []
away_yds_avg = []
home_to_avg = []
away_to_avg = []

home_pts_last3 = []
away_pts_last3 = []
home_win_streak = []
away_win_streak = []

for idx, row in df.iterrows():
    home_team = row['Winner/tie'] if row['Home/Away']=='Home' else row['Loser/tie']
    away_team = row['Loser/tie'] if row['Home/Away']=='Home' else row['Winner/tie']
    # Pre-game averages
    h_stats = cum_stats[home_team]
    a_stats = cum_stats[away_team]

    # Season averages
    home_pts_avg.append(h_stats['pts']/h_stats['games'] if h_stats['games']>0 else 0)
    home_yds_avg.append(h_stats['yds']/h_stats['games'] if h_stats['games']>0 else 0)
    home_to_avg.append(h_stats['to']/h_stats['games'] if h_stats['games']>0 else 0)

    away_pts_avg.append(a_stats['pts']/a_stats['games'] if a_stats['games']>0 else 0)
    away_yds_avg.append(a_stats['yds']/a_stats['games'] if a_stats['games']>0 else 0)
    away_to_avg.append(a_stats['to']/a_stats['games'] if a_stats['games']>0 else 0)

    # Last-3-games averages
    home_pts_last3.append(np.mean(h_stats['results'][-3:]) if len(h_stats['results'])>0 else 0)
    away_pts_last3.append(np.mean(a_stats['results'][-3:]) if len(a_stats['results'])>0 else 0)

    # Win streak
    def win_streak(results):
        streak = 0
        for r in reversed(results):
            if r == 1:
                streak += 1
            else:
                break
        return streak

    home_win_streak.append(win_streak(h_stats['results']))
    away_win_streak.append(win_streak(a_stats['results']))

    # --- Update cumulative stats AFTER using them ---
    if row['Home/Away']=='Home':
        cum_stats[home_team]['games'] += 1
        cum_stats[home_team]['pts'] += row['PtsW']
        cum_stats[home_team]['yds'] += row['YdsW']
        cum_stats[home_team]['to'] += row['TOW']
        cum_stats[home_team]['results'].append(1 if row['Result']=='Win' else 0)

        cum_stats[away_team]['games'] += 1
        cum_stats[away_team]['pts'] += row['PtsL']
        cum_stats[away_team]['yds'] += row['YdsL']
        cum_stats[away_team]['to'] += row['TOL']
        cum_stats[away_team]['results'].append(1 if row['Result']=='Loss' else 0)
    else:
        cum_stats[home_team]['games'] += 1
        cum_stats[home_team]['pts'] += row['PtsL']
        cum_stats[home_team]['yds'] += row['YdsL']
        cum_stats[home_team]['to'] += row['TOL']
        cum_stats[home_team]['results'].append(1 if row['Result']=='Loss' else 0)

        cum_stats[away_team]['games'] += 1
        cum_stats[away_team]['pts'] += row['PtsW']
        cum_stats[away_team]['yds'] += row['YdsW']
        cum_stats[away_team]['to'] += row['TOW']
        cum_stats[away_team]['results'].append(1 if row['Result']=='Win' else 0)

# --- Add pre-game features to df ---
df['HomePtsAvg'] = home_pts_avg
df['AwayPtsAvg'] = away_pts_avg
df['HomeYdsAvg'] = home_yds_avg
df['AwayYdsAvg'] = away_yds_avg
df['HomeTOAvg'] = home_to_avg
df['AwayTOAvg'] = away_to_avg
df['HomeLast3Wins'] = home_pts_last3
df['AwayLast3Wins'] = away_pts_last3
df['HomeWinStreak'] = home_win_streak
df['AwayWinStreak'] = away_win_streak

# --- Feature engineering: differences ---
df['PtsDiff'] = df['HomePtsAvg'] - df['AwayPtsAvg']
df['YdsDiff'] = df['HomeYdsAvg'] - df['AwayYdsAvg']
df['TODiff'] = df['HomeTOAvg'] - df['AwayTOAvg']
df['Last3WinDiff'] = df['HomeLast3Wins'] - df['AwayLast3Wins']
df['WinStreakDiff'] = df['HomeWinStreak'] - df['AwayWinStreak']

# --- Define X and y ---
X = df[['HomePtsAvg', 'AwayPtsAvg', 'HomeYdsAvg', 'AwayYdsAvg', 'HomeTOAvg', 'AwayTOAvg',
        'PtsDiff', 'YdsDiff', 'TODiff', 'HomeLast3Wins', 'AwayLast3Wins', 'Last3WinDiff',
        'HomeWinStreak', 'AwayWinStreak', 'WinStreakDiff']]
y = df['ResultBinary']

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Random Forest ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Feature importance ---
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:\n", importances)
