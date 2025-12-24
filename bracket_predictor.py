"""
bracket_predictor.py - AutoML Bracket Prediction Module

This module provides helper functions for predicting NFL playoff brackets
using trained machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# -------------------------
# Team-to-Conference Mapping
# -------------------------
TEAM_TO_CONFERENCE: Dict[str, str] = {
    # AFC Teams
    "Baltimore Ravens": "AFC",
    "Buffalo Bills": "AFC",
    "Cincinnati Bengals": "AFC",
    "Cleveland Browns": "AFC",
    "Denver Broncos": "AFC",
    "Houston Texans": "AFC",
    "Indianapolis Colts": "AFC",
    "Jacksonville Jaguars": "AFC",
    "Kansas City Chiefs": "AFC",
    "Las Vegas Raiders": "AFC",
    "Los Angeles Chargers": "AFC",
    "Miami Dolphins": "AFC",
    "New England Patriots": "AFC",
    "New York Jets": "AFC",
    "Pittsburgh Steelers": "AFC",
    "Tennessee Titans": "AFC",
    # NFC Teams
    "Arizona Cardinals": "NFC",
    "Atlanta Falcons": "NFC",
    "Carolina Panthers": "NFC",
    "Chicago Bears": "NFC",
    "Dallas Cowboys": "NFC",
    "Detroit Lions": "NFC",
    "Green Bay Packers": "NFC",
    "Los Angeles Rams": "NFC",
    "Minnesota Vikings": "NFC",
    "New Orleans Saints": "NFC",
    "New York Giants": "NFC",
    "Philadelphia Eagles": "NFC",
    "San Francisco 49ers": "NFC",
    "Seattle Seahawks": "NFC",
    "Tampa Bay Buccaneers": "NFC",
    "Washington Commanders": "NFC",
}

# Reverse mapping: abbreviation -> full name (for data matching)
TEAM_ABBREV_TO_FULL: Dict[str, str] = {
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DEN": "Denver Broncos",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "MIA": "Miami Dolphins",
    "NE": "New England Patriots",
    "NYJ": "New York Jets",
    "PIT": "Pittsburgh Steelers",
    "TEN": "Tennessee Titans",
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "DAL": "Dallas Cowboys",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "LA": "Los Angeles Rams",
    "LAR": "Los Angeles Rams",
    "MIN": "Minnesota Vikings",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "PHI": "Philadelphia Eagles",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "WAS": "Washington Commanders",
    # Historical abbreviations
    "OAK": "Las Vegas Raiders",
    "SD": "Los Angeles Chargers",
    "STL": "Los Angeles Rams",
}


def ensure_points(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure points column exists by computing from component stats."""
    if 'points' not in df.columns:
        df = df.copy()
        df['points'] = (
            df.get("passing_tds", pd.Series(0, index=df.index)).fillna(0) * 6 +
            df.get("rushing_tds", pd.Series(0, index=df.index)).fillna(0) * 6 +
            df.get("receiving_tds", pd.Series(0, index=df.index)).fillna(0) * 6 +
            df.get("fg_made", pd.Series(0, index=df.index)).fillna(0) * 3 +
            df.get("pat_made", pd.Series(0, index=df.index)).fillna(0)
        )
    return df


def compute_team_stats_for_season(df_full: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Compute team-level aggregated stats for a given season.
    Matches the feature engineering used in Model 1 training.
    
    Args:
        df_full: Full DataFrame with all game data
        season: Season year to compute stats for
        
    Returns:
        DataFrame with one row per team, containing aggregated features
    """
    df_full = ensure_points(df_full)
    
    # Filter for regular season of the specified year
    df_reg = df_full[
        (df_full['season'] == season) & 
        (df_full['season_type'].str.lower() == 'reg')
    ].copy()
    
    if df_reg.empty:
        raise ValueError(f"No regular season data found for {season}")
    
    # Same aggregation columns as Model 1 training
    expected_agg_cols = [
        'passing_yards', 'rushing_yards', 'receiving_yards',
        'passing_tds', 'rushing_tds', 'receiving_tds',
        'passing_interceptions', 'rushing_fumbles', 'receiving_fumbles',
        'sacks_suffered', 'def_sacks', 'def_interceptions', 'def_fumbles_forced',
        'points'
    ]
    agg_cols = [c for c in expected_agg_cols if c in df_reg.columns]
    
    if not agg_cols:
        raise ValueError("None of the expected aggregate columns exist in the data.")
    
    # Group by team and aggregate
    team_stats = df_reg.groupby('team')[agg_cols].agg(['sum', 'mean']).reset_index()
    team_stats.columns = ['_'.join(filter(None, col)).strip('_') for col in team_stats.columns.values]
    
    # Extra engineered metrics
    if 'points_sum' in team_stats.columns and 'points_mean' in team_stats.columns:
        team_stats['point_diff'] = team_stats['points_sum'] - team_stats['points_mean']
    else:
        team_stats['point_diff'] = 0
    
    # Turnovers
    team_stats['turnovers_sum'] = 0
    for c in ['passing_interceptions_sum', 'rushing_fumbles_sum', 'receiving_fumbles_sum']:
        if c in team_stats.columns:
            team_stats['turnovers_sum'] += team_stats[c]
    
    team_stats['def_takeaways_sum'] = 0
    for c in ['def_interceptions_sum', 'def_fumbles_forced_sum']:
        if c in team_stats.columns:
            team_stats['def_takeaways_sum'] += team_stats[c]
    
    team_stats['turnover_margin'] = team_stats['def_takeaways_sum'] - team_stats['turnovers_sum']
    
    # Add full team name mapping
    team_stats['team_full'] = team_stats['team'].map(
        lambda x: TEAM_ABBREV_TO_FULL.get(x, x)
    )
    
    # Add conference
    team_stats['conference'] = team_stats['team_full'].map(
        lambda x: TEAM_TO_CONFERENCE.get(x, "Unknown")
    )
    
    return team_stats


def get_feature_columns(team_stats: pd.DataFrame) -> List[str]:
    """Get the feature columns used for Model 1 prediction (exclude identifiers)."""
    exclude_cols = ['team', 'team_full', 'conference', 'season', 'made_playoffs']
    return [c for c in team_stats.columns if c not in exclude_cols]


def predict_playoff_teams(
    model1,
    team_stats: pd.DataFrame,
    teams_per_conference: int = 7
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Use Model 1 to predict which teams make the playoffs.
    
    Args:
        model1: Trained playoff qualifier model
        team_stats: DataFrame with team stats for current season
        teams_per_conference: Number of playoff teams per conference (default 7)
        
    Returns:
        Dict with 'AFC' and 'NFC' keys, each containing list of (team_name, probability) tuples
        ordered by probability descending
    """
    # Get feature columns
    feature_cols = get_feature_columns(team_stats)
    X = team_stats[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Get playoff probabilities
    probs = model1.predict_proba(X)[:, 1]  # Probability of making playoffs
    team_stats = team_stats.copy()
    team_stats['playoff_prob'] = probs
    
    # Split by conference and get top teams
    result = {}
    for conf in ['AFC', 'NFC']:
        conf_teams = team_stats[team_stats['conference'] == conf].copy()
        conf_teams = conf_teams.sort_values('playoff_prob', ascending=False)
        
        # Take top N teams
        top_teams = conf_teams.head(teams_per_conference)
        result[conf] = [
            (row['team_full'], row['playoff_prob']) 
            for _, row in top_teams.iterrows()
        ]
    
    return result


def seed_conference(teams_ranked: List[Tuple[str, float]]) -> Dict[int, str]:
    """
    Assign playoff seedings based on probability ranking.
    
    Args:
        teams_ranked: List of (team_name, probability) tuples, ordered by probability
        
    Returns:
        Dict mapping seed number (1-7) to team name
    """
    return {i + 1: team for i, (team, _) in enumerate(teams_ranked)}


def compute_elo_ratings(df_full: pd.DataFrame, season: int) -> Dict[str, float]:
    """
    Compute Elo ratings for teams based on regular season games.
    
    Args:
        df_full: Full DataFrame with all game data
        season: Season year
        
    Returns:
        Dict mapping team abbreviation to Elo rating
    """
    df_full = ensure_points(df_full)
    df_reg = df_full[
        (df_full['season'] == season) & 
        (df_full['season_type'].str.lower() == 'reg')
    ].copy()
    
    if df_reg.empty:
        return {}
    
    # Create game IDs
    teams_min = df_reg[['team', 'opponent_team']].min(axis=1)
    teams_max = df_reg[['team', 'opponent_team']].max(axis=1)
    df_reg['game_id'] = df_reg['season'].astype(str) + "_" + df_reg['week'].astype(str) + "_" + teams_min + "_" + teams_max
    
    # Build games for Elo calculation
    games_rows = []
    for gid, g in df_reg.groupby('game_id'):
        if len(g) != 2:
            continue
        g0 = g.iloc[0]
        g1 = g.iloc[1]
        games_rows.append({
            'week': g0['week'],
            'teamA': g0['team'],
            'teamB': g1['team'],
            'ptsA': g0.get('points', 0),
            'ptsB': g1.get('points', 0)
        })
    
    if not games_rows:
        return {}
    
    games_df = pd.DataFrame(games_rows).sort_values('week').reset_index(drop=True)
    
    # Initialize Elo ratings
    teams = set(df_reg['team'].unique())
    elo = {t: 1500.0 for t in teams}
    
    # Update Elo
    K = 20
    for _, row in games_df.iterrows():
        A, B = row['teamA'], row['teamB']
        ptsA = row['ptsA'] if not pd.isna(row['ptsA']) else 0
        ptsB = row['ptsB'] if not pd.isna(row['ptsB']) else 0
        
        eA = elo.get(A, 1500)
        eB = elo.get(B, 1500)
        
        expA = 1 / (1 + 10 ** ((eB - eA) / 400))
        expB = 1 / (1 + 10 ** ((eA - eB) / 400))
        
        scoreA = 1 if ptsA > ptsB else 0
        scoreB = 1 - scoreA
        
        elo[A] = eA + K * (scoreA - expA)
        elo[B] = eB + K * (scoreB - expB)
    
    return elo


def make_matchup_features(
    team_stats: pd.DataFrame,
    team_a: str,
    team_b: str,
    elo_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Create feature DataFrame for a single matchup between two teams.
    Matches the diff_* feature engineering used in Model 2 training.
    
    Args:
        team_stats: DataFrame with team stats
        team_a: Full name of team A
        team_b: Full name of team B
        elo_dict: Dict mapping team abbreviation to Elo rating
        
    Returns:
        Single-row DataFrame with diff_* features for Model 2
    """
    # Get team abbreviations
    team_a_abbrev = None
    team_b_abbrev = None
    for abbrev, full in TEAM_ABBREV_TO_FULL.items():
        if full == team_a:
            team_a_abbrev = abbrev
        if full == team_b:
            team_b_abbrev = abbrev
    
    # Get stats for both teams
    stats_a = team_stats[team_stats['team_full'] == team_a]
    stats_b = team_stats[team_stats['team_full'] == team_b]
    
    if stats_a.empty or stats_b.empty:
        # Fallback: try matching by abbreviation
        stats_a = team_stats[team_stats['team'] == team_a_abbrev] if stats_a.empty else stats_a
        stats_b = team_stats[team_stats['team'] == team_b_abbrev] if stats_b.empty else stats_b
    
    if stats_a.empty or stats_b.empty:
        raise ValueError(f"Could not find stats for matchup: {team_a} vs {team_b}")
    
    # Compute diff features
    diff_features = {}
    feature_cols = get_feature_columns(team_stats)
    
    for col in feature_cols:
        val_a = stats_a[col].values[0] if col in stats_a.columns else 0
        val_b = stats_b[col].values[0] if col in stats_b.columns else 0
        diff_features[f'diff_{col}'] = float(val_a) - float(val_b)
    
    # Add Elo diff
    elo_a = elo_dict.get(team_a_abbrev, 1500) if team_a_abbrev else 1500
    elo_b = elo_dict.get(team_b_abbrev, 1500) if team_b_abbrev else 1500
    diff_features['diff_elo'] = elo_a - elo_b
    
    # Add diff_recent_avg_points placeholder (would need game-by-game data)
    diff_features['diff_recent_avg_points'] = diff_features.get('diff_points_mean', 0)
    
    return pd.DataFrame([diff_features])


def predict_matchup(model2, X_matchup: pd.DataFrame) -> float:
    """
    Predict the probability that team A wins the matchup.
    
    Args:
        model2: Trained bracket predictor model
        X_matchup: Single-row DataFrame with matchup features
        
    Returns:
        Probability that team A wins (0.0 to 1.0)
    """
    # Get the feature columns the model expects
    model_features = model2.feature_names_in_ if hasattr(model2, 'feature_names_in_') else X_matchup.columns
    
    # Align features
    X_aligned = pd.DataFrame(columns=model_features)
    for col in model_features:
        if col in X_matchup.columns:
            X_aligned[col] = X_matchup[col].values
        else:
            X_aligned[col] = 0
    
    X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Predict
    prob = model2.predict_proba(X_aligned)[:, 1][0]
    return prob


def simulate_game(
    model2,
    team_a: str,
    team_b: str,
    team_stats: pd.DataFrame,
    elo_dict: Dict[str, float]
) -> Tuple[str, float]:
    """
    Simulate a single game and return the winner.
    
    Args:
        model2: Trained bracket predictor model
        team_a: Full name of team A
        team_b: Full name of team B
        team_stats: DataFrame with team stats
        elo_dict: Elo ratings dict
        
    Returns:
        Tuple of (winner_name, win_probability)
    """
    X = make_matchup_features(team_stats, team_a, team_b, elo_dict)
    prob_a = predict_matchup(model2, X)
    
    if prob_a >= 0.5:
        return team_a, prob_a
    else:
        return team_b, 1 - prob_a


def run_automl_bracket(
    models: Dict[str, Any],
    df_full: pd.DataFrame,
    current_season: int,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run the full AutoML bracket prediction.
    
    Args:
        models: Dict with 'model1' and 'model2' keys
        df_full: Full DataFrame with all game data
        current_season: Season year to predict
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict with bracket results including:
        - afc_seeds: Dict[int, str] mapping seed to team
        - nfc_seeds: Dict[int, str] mapping seed to team
        - wild_card_winners: List of winners
        - divisional_winners: List of winners
        - conference_winners: Dict with AFC and NFC winners
        - champion: Super Bowl winner
        - bracket_slots: Dict mapping UI slot keys to team names
    """
    model1 = models.get('model1')
    model2 = models.get('model2')
    
    if model1 is None:
        raise ValueError("Model 1 (Playoff Qualifier) is required")
    if model2 is None:
        raise ValueError("Model 2 (Bracket Predictor) is required")
    
    result = {
        'afc_seeds': {},
        'nfc_seeds': {},
        'wild_card_results': [],
        'divisional_results': [],
        'conference_results': {},
        'super_bowl_result': {},
        'champion': None,
        'bracket_slots': {},
        'probabilities': {}
    }
    
    # Step 1: Compute team stats for current season
    if progress_callback:
        progress_callback("Computing team statistics...", "info")
    
    team_stats = compute_team_stats_for_season(df_full, current_season)
    elo_dict = compute_elo_ratings(df_full, current_season)
    
    # Step 2: Predict playoff teams
    if progress_callback:
        progress_callback("Predicting playoff teams...", "info")
    
    playoff_teams = predict_playoff_teams(model1, team_stats)
    
    # Step 3: Seed conferences
    afc_seeds = seed_conference(playoff_teams['AFC'])
    nfc_seeds = seed_conference(playoff_teams['NFC'])
    result['afc_seeds'] = afc_seeds
    result['nfc_seeds'] = nfc_seeds
    
    # Fill bracket slots for initial seeding
    result['bracket_slots']['afc_bye'] = afc_seeds[1]
    result['bracket_slots']['nfc_bye'] = nfc_seeds[1]
    
    # Wild card matchups: 2v7, 3v6, 4v5
    for i, (high, low) in enumerate([(2, 7), (3, 6), (4, 5)], 1):
        result['bracket_slots'][f'afc_wild{i*2-1}'] = afc_seeds[high]
        result['bracket_slots'][f'afc_wild{i*2}'] = afc_seeds[low]
        result['bracket_slots'][f'nfc_wild{i*2-1}'] = nfc_seeds[high]
        result['bracket_slots'][f'nfc_wild{i*2}'] = nfc_seeds[low]
    
    # Step 4: Simulate Wild Card Round
    if progress_callback:
        progress_callback("Simulating Wild Card round...", "info")
    
    afc_wc_winners = []
    nfc_wc_winners = []
    
    # AFC Wild Card: 2v7, 3v6, 4v5
    for high, low in [(2, 7), (3, 6), (4, 5)]:
        winner, prob = simulate_game(model2, afc_seeds[high], afc_seeds[low], team_stats, elo_dict)
        afc_wc_winners.append(winner)
        result['wild_card_results'].append({
            'matchup': f"{afc_seeds[high]} vs {afc_seeds[low]}",
            'winner': winner,
            'probability': prob
        })
    
    # NFC Wild Card: 2v7, 3v6, 4v5
    for high, low in [(2, 7), (3, 6), (4, 5)]:
        winner, prob = simulate_game(model2, nfc_seeds[high], nfc_seeds[low], team_stats, elo_dict)
        nfc_wc_winners.append(winner)
        result['wild_card_results'].append({
            'matchup': f"{nfc_seeds[high]} vs {nfc_seeds[low]}",
            'winner': winner,
            'probability': prob
        })
    
    # Step 5: Simulate Divisional Round
    if progress_callback:
        progress_callback("Simulating Divisional round...", "info")
    
    # AFC Divisional: 1 seed vs lowest remaining, 2nd highest vs 3rd highest
    afc_div_teams = [afc_seeds[1]] + afc_wc_winners
    # Sort WC winners by original seed
    wc_seed_map = {afc_seeds[s]: s for s in [2, 3, 4, 5, 6, 7]}
    afc_wc_sorted = sorted(afc_wc_winners, key=lambda t: wc_seed_map.get(t, 99))
    
    # 1 seed plays lowest remaining seed
    afc_div1_winner, prob1 = simulate_game(model2, afc_seeds[1], afc_wc_sorted[-1], team_stats, elo_dict)
    # Other two WC winners play each other
    afc_div2_winner, prob2 = simulate_game(model2, afc_wc_sorted[0], afc_wc_sorted[1], team_stats, elo_dict)
    
    result['bracket_slots']['afc_div_win_1'] = afc_div1_winner
    result['bracket_slots']['afc_div_win_2'] = afc_wc_sorted[-1]
    result['bracket_slots']['afc_div_win_3'] = afc_div2_winner
    result['bracket_slots']['afc_div_win_4'] = afc_wc_sorted[0] if afc_wc_sorted[0] != afc_div2_winner else afc_wc_sorted[1]
    
    result['divisional_results'].append({'matchup': f"{afc_seeds[1]} vs {afc_wc_sorted[-1]}", 'winner': afc_div1_winner, 'probability': prob1})
    result['divisional_results'].append({'matchup': f"{afc_wc_sorted[0]} vs {afc_wc_sorted[1]}", 'winner': afc_div2_winner, 'probability': prob2})
    
    # NFC Divisional
    nfc_wc_seed_map = {nfc_seeds[s]: s for s in [2, 3, 4, 5, 6, 7]}
    nfc_wc_sorted = sorted(nfc_wc_winners, key=lambda t: nfc_wc_seed_map.get(t, 99))
    
    nfc_div1_winner, prob3 = simulate_game(model2, nfc_seeds[1], nfc_wc_sorted[-1], team_stats, elo_dict)
    nfc_div2_winner, prob4 = simulate_game(model2, nfc_wc_sorted[0], nfc_wc_sorted[1], team_stats, elo_dict)
    
    result['bracket_slots']['nfc_div_win_1'] = nfc_div1_winner
    result['bracket_slots']['nfc_div_win_2'] = nfc_wc_sorted[-1]
    result['bracket_slots']['nfc_div_win_3'] = nfc_div2_winner
    result['bracket_slots']['nfc_div_win_4'] = nfc_wc_sorted[0] if nfc_wc_sorted[0] != nfc_div2_winner else nfc_wc_sorted[1]
    
    result['divisional_results'].append({'matchup': f"{nfc_seeds[1]} vs {nfc_wc_sorted[-1]}", 'winner': nfc_div1_winner, 'probability': prob3})
    result['divisional_results'].append({'matchup': f"{nfc_wc_sorted[0]} vs {nfc_wc_sorted[1]}", 'winner': nfc_div2_winner, 'probability': prob4})
    
    # Step 6: Simulate Conference Championships
    if progress_callback:
        progress_callback("Simulating Conference Championships...", "info")
    
    afc_champ, afc_prob = simulate_game(model2, afc_div1_winner, afc_div2_winner, team_stats, elo_dict)
    nfc_champ, nfc_prob = simulate_game(model2, nfc_div1_winner, nfc_div2_winner, team_stats, elo_dict)
    
    result['bracket_slots']['afc_conf_win_1'] = afc_champ
    result['bracket_slots']['afc_conf_win_2'] = afc_div1_winner if afc_div1_winner != afc_champ else afc_div2_winner
    result['bracket_slots']['nfc_conf_win_1'] = nfc_champ
    result['bracket_slots']['nfc_conf_win_2'] = nfc_div1_winner if nfc_div1_winner != nfc_champ else nfc_div2_winner
    
    result['conference_results']['AFC'] = {'winner': afc_champ, 'probability': afc_prob}
    result['conference_results']['NFC'] = {'winner': nfc_champ, 'probability': nfc_prob}
    
    # Step 7: Simulate Super Bowl
    if progress_callback:
        progress_callback("Simulating Super Bowl...", "info")
    
    champion, sb_prob = simulate_game(model2, afc_champ, nfc_champ, team_stats, elo_dict)
    
    result['bracket_slots']['sb_team_a'] = afc_champ
    result['bracket_slots']['sb_team_b'] = nfc_champ
    result['super_bowl_result'] = {
        'matchup': f"{afc_champ} vs {nfc_champ}",
        'winner': champion,
        'probability': sb_prob
    }
    result['champion'] = champion
    
    if progress_callback:
        progress_callback(f"Prediction complete! Champion: {champion}", "success")
    
    return result
