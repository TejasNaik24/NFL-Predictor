import os
import glob
import joblib
import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Utilities
# -------------------------
def merge_data(folder_path="data_files"):
    files = glob.glob(f"{folder_path}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{folder_path}'. Run the scraper first.")
    dfs = []
    for f in sorted(files):
        name = os.path.basename(f)
        temp_df = pd.read_csv(f)
        # ensure season exists; fallback to digits in filename
        if 'season' not in temp_df.columns or temp_df['season'].isnull().all():
            try:
                yr = int(''.join(filter(str.isdigit, name)))
                temp_df['season'] = yr
            except Exception:
                temp_df['season'] = np.nan
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    return df

def ensure_points(df):
    if 'points' not in df.columns:
        df['points'] = (
            df.get("passing_tds", 0).fillna(0)*6 +
            df.get("rushing_tds", 0).fillna(0)*6 +
            df.get("receiving_tds", 0).fillna(0)*6 +
            df.get("fg_made", 0).fillna(0)*3 +
            df.get("pat_made", 0).fillna(0)
        )
    return df

def make_game_id(df):
    # safe sorted team pair per row so both sides of a game share same id
    teams_min = df[['team','opponent_team']].min(axis=1)
    teams_max = df[['team','opponent_team']].max(axis=1)
    df['game_id'] = df['season'].astype(str) + "_" + df['week'].astype(str) + "_" + teams_min + "_" + teams_max
    return df

# -------------------------
# Main Training Function
# -------------------------
def train_models(
    progress_callback: Optional[Callable[[str, str], None]] = None,
    data_folder: str = "data_files"
) -> Dict[str, Any]:
    """
    Train both NFL playoff prediction models.
    
    Args:
        progress_callback: Optional callback function(message: str, type: str) for progress updates
                          type can be: "info", "success", "error"
        data_folder: Folder containing scraped CSV files (default: "data_files")
    
    Returns:
        Dictionary with:
            - model1: dict with accuracy, classification_report, model_object
            - model2: dict with accuracy, classification_report, model_object (or None)
            - version_folder: str (e.g., "model_2015-2024")
            - success: bool
            - error: str or None
    """
    def log(message: str, msg_type: str = "info"):
        """Helper to log messages via callback or print"""
        if progress_callback:
            progress_callback(message, msg_type)
        else:
            print(message)
    
    try:
        # -------------------------
        # Load data (silent - user doesn't need details)
        # -------------------------
        df_full = merge_data(data_folder)
        df_full = ensure_points(df_full)
        seasons_found = sorted(df_full['season'].dropna().unique())
        
        # -------------------------
        # Model 1: Playoff Qualifier (season aggregates)
        # -------------------------
        log("Training Model 1: Playoff Qualifier", "info")
        df_reg = df_full[df_full['season_type'].str.lower() == 'reg'].copy()
        if df_reg.empty:
            raise RuntimeError("No regular-season rows found (season_type == 'reg') — can't train Model 1.")
        
        # Aggregation columns we expect (use only those present)
        expected_agg_cols = [
            'passing_yards','rushing_yards','receiving_yards',
            'passing_tds','rushing_tds','receiving_tds',
            'passing_interceptions','rushing_fumbles','receiving_fumbles',
            'sacks_suffered','def_sacks','def_interceptions','def_fumbles_forced',
            'points'
        ]
        agg_cols = [c for c in expected_agg_cols if c in df_reg.columns]
        if not agg_cols:
            raise RuntimeError("None of the expected aggregate columns exist in your regular-season CSVs.")
        
        # group and create sum/mean features
        team_stats = df_reg.groupby(['season','team'])[agg_cols].agg(['sum','mean']).reset_index()
        team_stats.columns = ['_'.join(filter(None, col)).strip('_') for col in team_stats.columns.values]
        
        # extra engineered metrics (robust to missing cols)
        if 'points_sum' in team_stats.columns and 'points_mean' in team_stats.columns:
            team_stats['point_diff'] = team_stats['points_sum'] - team_stats['points_mean']
        else:
            team_stats['point_diff'] = 0
        
        # turnovers & turnover margin if parts exist
        team_stats['turnovers_sum'] = 0
        for c in ['passing_interceptions_sum','rushing_fumbles_sum','receiving_fumbles_sum']:
            if c in team_stats.columns:
                team_stats['turnovers_sum'] += team_stats[c]
        team_stats['def_takeaways_sum'] = 0
        for c in ['def_interceptions_sum','def_fumbles_forced_sum']:
            if c in team_stats.columns:
                team_stats['def_takeaways_sum'] += team_stats[c]
        team_stats['turnover_margin'] = team_stats['def_takeaways_sum'] - team_stats['turnovers_sum']
        
        # playoff labels from df_full post-season listing
        playoff_teams_map = (df_full[df_full['season_type'].str.lower() == 'post']
                             .groupby('season')['team'].unique().to_dict())
        
        team_stats['made_playoffs'] = team_stats.apply(lambda r: 1 if r['team'] in playoff_teams_map.get(r['season'], []) else 0, axis=1)
        
        # prepare X/y (drop identifiers)
        drop_cols = [c for c in ['team','season','made_playoffs'] if c in team_stats.columns]
        X1 = team_stats.drop(columns=drop_cols, errors='ignore')
        y1 = team_stats['made_playoffs']
        
        # train/test split & model
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
        model1 = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        model1.fit(X_train1, y_train1)
        
        y_pred1 = model1.predict(X_test1)
        model1_accuracy = accuracy_score(y_test1, y_pred1)
        model1_report = classification_report(y_test1, y_pred1)
        
        log("Model 1 training complete", "success")
        
        # -------------------------
        # Model 2: Playoff Game Winner (diff features + recent-form + Elo)
        # -------------------------
        log("Training Model 2: Bracket Predictor", "info")
        playoff_df = df_full[df_full['season_type'].str.lower() == 'post'].copy()
        if playoff_df.empty:
            log("Warning: no playoff rows found (season_type == 'post'). Model 2 will be skipped.", "info")
            model2 = None
            model2_accuracy = None
            model2_report = None
        else:
            # ensure points present
            playoff_df = ensure_points(playoff_df)
            
            # make game ids across full data and playoff subset
            df_full = make_game_id(df_full)
            playoff_df = make_game_id(playoff_df)
            
            # label winners
            playoff_df['won_game'] = (playoff_df['points'] == playoff_df.groupby('game_id')['points'].transform('max')).astype(int)
            
            # merge season-level team_stats for team and opponent
            ts = team_stats.copy()
            # rename team_stats columns to team_* for merging
            ts_team = ts.rename(columns={c: (f"team_{c}" if c not in ['season','team'] else c) for c in ts.columns})
            playoff_df = playoff_df.merge(ts_team, on=['season','team'], how='left')
            
            # opponent stats: rename and merge
            ts_opp = ts_team.rename(columns={'team': 'opponent_team', **{c: c.replace('team_','opp_') for c in ts_team.columns if c.startswith('team_')}})
            playoff_df = playoff_df.merge(ts_opp, on=['season','opponent_team'], how='left')
            
            # create diff_ columns where both team_ and opp_ exist
            diff_cols = []
            for col in playoff_df.columns:
                if col.startswith('team_'):
                    opp_col = col.replace('team_','opp_')
                    if opp_col in playoff_df.columns:
                        diff_col = col.replace('team_','diff_')
                        playoff_df[diff_col] = playoff_df[col].fillna(0) - playoff_df[opp_col].fillna(0)
                        diff_cols.append(diff_col)
            
            # ===== recent-form: last-3 regular-season games avg points per (season,team) =====
            df_reg_full = df_full[df_full['season_type'].str.lower() == 'reg'].copy()
            if df_reg_full.empty:
                log("Warning: no regular-season rows available to compute recent-form. Filling zeros.", "info")
                playoff_df['recent_avg_points'] = 0
                playoff_df['opp_recent_avg_points'] = 0
            else:
                # compute rolling mean of points per (season,team) ordered by week
                df_reg_full = df_reg_full.sort_values(['season','team','week'])
                recent_series = []
                for (season, team), grp in df_reg_full.groupby(['season','team']):
                    grp_sorted = grp.sort_values('week')
                    recent_mean = grp_sorted['points'].rolling(window=3, min_periods=1).mean()
                    tmp = pd.DataFrame({
                        'season': grp_sorted['season'].values,
                        'team': grp_sorted['team'].values,
                        'week': grp_sorted['week'].values,
                        'recent_avg_points': recent_mean.values
                    })
                    recent_series.append(tmp)
                recent_df = pd.concat(recent_series, ignore_index=True) if recent_series else pd.DataFrame(columns=['season','team','week','recent_avg_points'])
                
                # merge team recent into playoff_df on season,team,week
                playoff_df = playoff_df.merge(recent_df, on=['season','team','week'], how='left')
                # opponent recent: rename then merge using opponent_team
                recent_opp = recent_df.rename(columns={'team':'opponent_team','recent_avg_points':'opp_recent_avg_points'})
                playoff_df = playoff_df.merge(recent_opp[['season','opponent_team','week','opp_recent_avg_points']], on=['season','opponent_team','week'], how='left')
                
                playoff_df['recent_avg_points'] = playoff_df['recent_avg_points'].fillna(0)
                playoff_df['opp_recent_avg_points'] = playoff_df['opp_recent_avg_points'].fillna(0)
            
            # diff recent
            playoff_df['diff_recent_avg_points'] = playoff_df['recent_avg_points'] - playoff_df['opp_recent_avg_points']
            
            # ===== Elo rating per season using complete games from df_full =====
            # build games_df with pairs having two rows
            games_rows = []
            for gid, g in df_full.groupby('game_id'):
                if len(g) != 2:
                    continue
                g0 = g.iloc[0]
                g1 = g.iloc[1]
                games_rows.append({
                    'season': g0['season'],
                    'week': g0['week'],
                    'teamA': g0['team'],
                    'teamB': g1['team'],
                    'ptsA': g0['points'],
                    'ptsB': g1['points']
                })
            games_df = pd.DataFrame(games_rows).sort_values(['season','week']).reset_index(drop=True)
            # initialize elos
            elo = {}
            seasons_present = sorted(playoff_df['season'].dropna().unique())
            for s in seasons_present:
                teams_in_season = df_full[df_full['season']==s]['team'].unique()
                for t in teams_in_season:
                    elo[(s,t)] = 1500
            # update elo by iterating games_df
            K = 20
            for _, row in games_df.iterrows():
                s = row['season']
                A = row['teamA']
                B = row['teamB']
                ptsA = row['ptsA'] if not pd.isna(row['ptsA']) else 0
                ptsB = row['ptsB'] if not pd.isna(row['ptsB']) else 0
                eA = elo.get((s,A),1500)
                eB = elo.get((s,B),1500)
                expA = 1/(1+10**((eB-eA)/400))
                expB = 1/(1+10**((eA-eB)/400))
                scoreA = 1 if ptsA > ptsB else 0
                scoreB = 1 - scoreA
                elo[(s,A)] = eA + K*(scoreA - expA)
                elo[(s,B)] = eB + K*(scoreB - expB)
            
            # merge elo values into playoff_df
            playoff_df['elo'] = playoff_df.apply(lambda r: elo.get((r['season'], r['team']), 1500), axis=1)
            playoff_df['opp_elo'] = playoff_df.apply(lambda r: elo.get((r['season'], r['opponent_team']), 1500), axis=1)
            playoff_df['diff_elo'] = playoff_df['elo'] - playoff_df['opp_elo']
            
            # final selected features = diff_* + diff_recent_avg_points + diff_elo if available
            final_feature_cols = [c for c in playoff_df.columns if c.startswith('diff_')]
            if 'diff_recent_avg_points' in playoff_df.columns and 'diff_recent_avg_points' not in final_feature_cols:
                final_feature_cols.append('diff_recent_avg_points')
            if 'diff_elo' in playoff_df.columns and 'diff_elo' not in final_feature_cols:
                final_feature_cols.append('diff_elo')
            
            # build X2,y2
            X2 = playoff_df[final_feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
            y2 = playoff_df['won_game']
            
            if X2.shape[0] == 0 or X2.shape[1] == 0:
                raise RuntimeError("No valid features for Model 2 after preprocessing (X2 empty).")
            
            # Train/test split (season-based if possible)
            seasons_for_split = sorted(playoff_df['season'].dropna().unique())
            if len(seasons_for_split) > 1:
                train_seasons = seasons_for_split[:-1]
                test_seasons = [seasons_for_split[-1]]
                train_idx = playoff_df['season'].isin(train_seasons)
                test_idx = playoff_df['season'].isin(test_seasons)
                X_train2 = X2.loc[train_idx]
                X_test2 = X2.loc[test_idx]
                y_train2 = y2.loc[train_idx]
                y_test2 = y2.loc[test_idx]
            else:
                X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
            
            # Train model 2 (silent)
            model2 = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
            model2.fit(X_train2, y_train2)
            
            y_pred2 = model2.predict(X_test2)
            model2_accuracy = accuracy_score(y_test2, y_pred2)
            model2_report = classification_report(y_test2, y_pred2)
            
            log("Model 2 training complete", "success")
        
        # -------------------------
        # Save models into models/model_START-END (silent)
        # -------------------------
        base_dir = "models"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        
        # determine season range from team_stats (Model 1 training data)
        try:
            seasons_used = sorted(team_stats['season'].dropna().unique())
            if seasons_used:
                model_year_range = f"{int(min(seasons_used))}-{int(max(seasons_used))}"
            else:
                model_year_range = "unknown"
        except Exception:
            model_year_range = "unknown"
        
        version_folder = f"model_{model_year_range}"
        version_path = os.path.join(base_dir, version_folder)
        if not os.path.exists(version_path):
            os.makedirs(version_path, exist_ok=True)
        
        # save Model 1 (silent)
        m1_file = os.path.join(version_path, "model1_playoff_qualifier.pkl")
        joblib.dump(model1, m1_file)
        
        # save Model 2 if it exists (silent)
        if model2 is not None:
            m2_file = os.path.join(version_path, "model2_bracket.pkl")
            joblib.dump(model2, m2_file)
        
        # Return results
        return {
            "model1": {
                "accuracy": model1_accuracy,
                "classification_report": model1_report,
                "model_object": model1
            },
            "model2": {
                "accuracy": model2_accuracy,
                "classification_report": model2_report,
                "model_object": model2
            } if model2 is not None else None,
            "version_folder": version_folder,
            "success": True,
            "error": None
        }
    
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        log(error_msg, "error")
        return {
            "model1": None,
            "model2": None,
            "version_folder": None,
            "success": False,
            "error": error_msg
        }


# -------------------------
# Main execution (backward compatibility)
# -------------------------
if __name__ == "__main__":
    result = train_models()
    
    if not result["success"]:
        print(f"\nError: {result['error']}")

