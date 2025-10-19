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
