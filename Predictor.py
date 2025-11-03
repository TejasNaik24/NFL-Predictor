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

playoff_seasons = df.loc[df['season_type'].str.lower() == 'post', 'season'].unique()

df['made_playoffs'] = df['season'].apply(lambda s: 1 if s in playoff_seasons else 0)

X = df.drop(columns=['made_playoffs'])
y = df['made_playoffs']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)