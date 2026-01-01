import streamlit as st
import os
import re
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ---- config ----
icon_path = os.path.join("static", "nfl.png")
st.set_page_config(page_title="NFL Predictor", page_icon=icon_path, layout="wide")

# ===================================================================
# INTERACTIVE BRACKET HELPERS
# ===================================================================
# Usage: After AutoML prediction, buttons become clickable.
# Click any enabled button to see game details modal with:
# - Matchup win probabilities (Model 2)
# - Playoff probabilities (Model 1)
# - Top 5 feature contributions
# - Visual charts and explanations
# ===================================================================

# ---- helpers ----
def vspace(n: int):
    for _ in range(n):
        st.write("")


def determine_button_enabled(slot_key: str, result: Dict) -> Tuple[bool, Optional[str]]:
    """
    Determine if a bracket button should be enabled and what team to display.
    
    Only enable buttons for teams that WON their game (have game stats to show).
    Disable: bye teams (no game), losing teams (eliminated).
    
    Args:
        slot_key: UI slot key (e.g., 'afc_wild1', 'afc_div_win_1')
        result: Bracket prediction result dict
        
    Returns:
        Tuple of (is_enabled, team_name)
    """
    if not result or 'bracket_slots' not in result:
        return (False, None)
    
    bracket_slots = result['bracket_slots']
    team = bracket_slots.get(slot_key)
    
    if not team:
        return (False, None)
    
    # Bye slots: DISABLED - no game to show
    if 'bye' in slot_key:
        return (False, team)
    
    # Wild card slots: check if this team WON their wild card game
    if 'wild' in slot_key:
        for wc_result in result.get('wild_card_results', []):
            winner = wc_result.get('winner')
            if team == winner:
                return (True, team)  # This team won - enable button
            elif team in wc_result.get('matchup', ''):
                return (False, team)  # This team played but lost - disable
        return (False, team)
    
    # Divisional slots: check if this team WON their divisional game
    if 'div' in slot_key:
        for div_result in result.get('divisional_results', []):
            winner = div_result.get('winner')
            if team == winner:
                return (True, team)  # This team won - enable button
            elif team in div_result.get('matchup', ''):
                return (False, team)  # This team played but lost - disable
        return (False, team)
    
    # Conference slots: check if this team WON their conference championship
    if 'conf' in slot_key:
        conf = 'AFC' if 'afc' in slot_key else 'NFC'
        conf_result = result.get('conference_results', {}).get(conf, {})
        winner = conf_result.get('winner')
        if team == winner:
            return (True, team)  # Won conference - enable
        else:
            return (False, team)  # Lost conference - disable
    
    # Super Bowl slots: only the WINNER is enabled
    if 'sb' in slot_key:
        sb_result = result.get('super_bowl_result', {})
        winner = sb_result.get('winner')
        if team == winner:
            return (True, team)  # Won Super Bowl - enable
        else:
            return (False, team)  # Lost Super Bowl - disable
    
    return (False, team)


def get_matchup_context(slot_key: str, team_clicked: str, result: Dict, models: Dict, 
                       team_stats: pd.DataFrame, elo_dict: Dict) -> Dict[str, Any]:
    """
    Get all matchup information for displaying in modal.
    
    Returns dict with:
        - round_name: str
        - team_a, team_b: str
        - matchup_prob_a, matchup_prob_b: float
        - playoff_prob_a, playoff_prob_b: Optional[float]
        - top_features: List[Dict]
        - team_a_stats, team_b_stats: Dict
    """
    from bracket_predictor import make_matchup_features, predict_matchup, predict_playoff_teams
    
    # Determine round and matchup
    if 'wild' in slot_key or 'bye' in slot_key:
        round_name = "Wild Card Round"
    elif 'div' in slot_key:
        round_name = "Divisional Round"
    elif 'conf' in slot_key:
        round_name = "Conference Championship"
    elif 'sb' in slot_key:
        round_name = "Super Bowl"
    else:
        round_name = "Playoff Game"
    
    # Find the matchup involving this team
    team_a = team_clicked
    team_b = None
    
    # Search through results to find opponent
    if 'wild' in slot_key:
        # Find wild card matchup
        for wc_result in result.get('wild_card_results', []):
            if team_clicked in wc_result['matchup']:
                parts = wc_result['matchup'].split(' vs ')
                team_a = parts[0]
                team_b = parts[1]
                matchup_prob_a = wc_result['probability'] if wc_result['winner'] == team_a else (1 - wc_result['probability'])
                matchup_prob_b = 1 - matchup_prob_a
                break
    elif 'div' in slot_key:
        # Find divisional matchup
        for div_result in result.get('divisional_results', []):
            if team_clicked in div_result['matchup']:
                parts = div_result['matchup'].split(' vs ')
                team_a = parts[0]
                team_b = parts[1]
                matchup_prob_a = div_result['probability'] if div_result['winner'] == team_a else (1 - div_result['probability'])
                matchup_prob_b = 1 - matchup_prob_a
                break
    elif 'conf' in slot_key:
        # Find conference championship
        conf = 'AFC' if 'afc' in slot_key else 'NFC'
        conf_result = result.get('conference_results', {}).get(conf, {})
        # Get both teams from bracket slots
        if 'afc' in slot_key:
            teams = [result['bracket_slots'].get('afc_conf_win_1'), result['bracket_slots'].get('afc_conf_win_2')]
        else:
            teams = [result['bracket_slots'].get('nfc_conf_win_1'), result['bracket_slots'].get('nfc_conf_win_2')]
        team_a = teams[0] if teams[0] else team_clicked
        team_b = teams[1] if teams[1] else None
        if team_b:
            winner = conf_result.get('winner')
            matchup_prob_a = conf_result.get('probability', 0.5) if winner == team_a else (1 - conf_result.get('probability', 0.5))
            matchup_prob_b = 1 - matchup_prob_a
    elif 'sb' in slot_key:
        sb_result = result.get('super_bowl_result', {})
        parts = sb_result.get('matchup', '').split(' vs ')
        if len(parts) == 2:
            team_a = parts[0]
            team_b = parts[1]
            winner = sb_result.get('winner')
            matchup_prob_a = sb_result.get('probability', 0.5) if winner == team_a else (1 - sb_result.get('probability', 0.5))
            matchup_prob_b = 1 - matchup_prob_a
    
    # Get playoff probabilities (Model 1) if available
    playoff_prob_a = None
    playoff_prob_b = None
    
    # First check stored playoff probabilities (for manual mode or cached results)
    if st.session_state.get('playoff_probs_dict'):
        playoff_prob_a = st.session_state.playoff_probs_dict.get(team_a)
        if team_b:
            playoff_prob_b = st.session_state.playoff_probs_dict.get(team_b)
    
    # If not found in stored dict, try computing from model
    if (playoff_prob_a is None or (team_b and playoff_prob_b is None)):
        try:
            if models.get('model1') and team_stats is not None:
                playoff_teams = predict_playoff_teams(models['model1'], team_stats)
                for conf in ['AFC', 'NFC']:
                    for team, prob in playoff_teams.get(conf, []):
                        if team == team_a and playoff_prob_a is None:
                            playoff_prob_a = prob
                        if team_b and team == team_b and playoff_prob_b is None:
                            playoff_prob_b = prob
        except:
            pass
    
    # Get feature importance if available
    top_features = []
    if team_b and models.get('model2') and team_stats is not None and elo_dict:
        try:
            X_matchup = make_matchup_features(team_stats, team_a, team_b, elo_dict)
            
            if hasattr(models['model2'], 'feature_importances_') and hasattr(models['model2'], 'feature_names_in_'):
                importances = models['model2'].feature_importances_
                feature_names = models['model2'].feature_names_in_
                
                # Get diff values
                feature_contribs = []
                for fname, importance in zip(feature_names, importances):
                    if fname in X_matchup.columns:
                        diff_val = X_matchup[fname].values[0]
                        contrib = abs(diff_val) * importance
                        feature_contribs.append({
                            'feature': fname,
                            'diff_value': diff_val,
                            'importance': importance,
                            'contribution': contrib
                        })
                
                # Sort by contribution
                feature_contribs.sort(key=lambda x: x['contribution'], reverse=True)
                top_features = feature_contribs[:5]
        except Exception as e:
            st.warning(f"Could not compute feature importance: {str(e)}")
    
    # Get team stats
    team_a_stats = {}
    team_b_stats = {}
    if team_stats is not None:
        try:
            a_row = team_stats[team_stats['team_full'] == team_a]
            if not a_row.empty:
                team_a_stats = {
                    'points_mean': a_row['points_mean'].values[0] if 'points_mean' in a_row else None,
                    'points_sum': a_row['points_sum'].values[0] if 'points_sum' in a_row else None,
                    'turnover_margin': a_row['turnover_margin'].values[0] if 'turnover_margin' in a_row else None,
                    'point_diff': a_row['point_diff'].values[0] if 'point_diff' in a_row else None,
                }
            
            if team_b:
                b_row = team_stats[team_stats['team_full'] == team_b]
                if not b_row.empty:
                    team_b_stats = {
                        'points_mean': b_row['points_mean'].values[0] if 'points_mean' in b_row else None,
                        'points_sum': b_row['points_sum'].values[0] if 'points_sum' in b_row else None,
                        'turnover_margin': b_row['turnover_margin'].values[0] if 'turnover_margin' in b_row else None,
                        'point_diff': b_row['point_diff'].values[0] if 'point_diff' in b_row else None,
                    }
        except:
            pass
    
    # Add ELO
    if elo_dict:
        from bracket_predictor import TEAM_ABBREV_TO_FULL
        abbrev_to_full_inv = {v: k for k, v in TEAM_ABBREV_TO_FULL.items()}
        team_a_abbrev = abbrev_to_full_inv.get(team_a)
        team_b_abbrev = abbrev_to_full_inv.get(team_b) if team_b else None
        
        if team_a_abbrev:
            team_a_stats['elo'] = elo_dict.get(team_a_abbrev, 1500)
        if team_b_abbrev:
            team_b_stats['elo'] = elo_dict.get(team_b_abbrev, 1500)
    
    return {
        'round_name': round_name,
        'team_a': team_a,
        'team_b': team_b,
        'matchup_prob_a': matchup_prob_a if team_b else 1.0,
        'matchup_prob_b': matchup_prob_b if team_b else 0.0,
        'playoff_prob_a': playoff_prob_a,
        'playoff_prob_b': playoff_prob_b,
        'top_features': top_features,
        'team_a_stats': team_a_stats,
        'team_b_stats': team_b_stats
    }


@st.dialog("Game Details", width="large")
def show_game_modal(context: Dict[str, Any]):
    """Display game details modal with matchup info, probabilities, and feature explanations."""
    
    st.subheader(f"{context['round_name']}")
    
    if context['team_b']:
        st.markdown(f"### {context['team_a']} vs {context['team_b']}")
    else:
        st.markdown(f"### {context['team_a']}")
        st.info("This team advanced via bye week or has no opponent data available.")
        return
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Win Probability (Model 2)")
        
        # Bar chart
        prob_data = pd.DataFrame({
            'Team': [context['team_a'], context['team_b']],
            'Probability': [context['matchup_prob_a'] * 100, context['matchup_prob_b'] * 100]
        })
        st.bar_chart(prob_data.set_index('Team'), height=200)
        
        # Numeric values
        st.metric(context['team_a'], f"{context['matchup_prob_a']*100:.1f}%")
        st.metric(context['team_b'], f"{context['matchup_prob_b']*100:.1f}%")
        
        # Playoff probabilities - always show for both teams
        st.markdown("#### Playoff Probability (Model 1)")
        if context['playoff_prob_a'] is not None:
            st.metric(f"Chance {context['team_a']} makes playoffs", f"{context['playoff_prob_a']*100:.1f}%")
        else:
            st.metric(f"Chance {context['team_a']} makes playoffs", "N/A")
        
        if context['playoff_prob_b'] is not None:
            st.metric(f"Chance {context['team_b']} makes playoffs", f"{context['playoff_prob_b']*100:.1f}%")
        else:
            st.metric(f"Chance {context['team_b']} makes playoffs", "N/A")
    
    with col2:
        st.markdown("#### Top Contributing Factors")
        
        if context['top_features']:
            st.markdown("**Top 5 Features (by importance × magnitude)**")
            
            for i, feat in enumerate(context['top_features'], 1):
                feat_name = feat['feature'].replace('diff_', '').replace('_', ' ').title()
                diff_val = feat['diff_value']
                sign = "+" if diff_val > 0 else ""
                
                st.markdown(f"{i}. **{feat_name}**: {sign}{diff_val:.2f}")
                st.caption(f"   Importance: {feat['importance']:.4f} | Contribution: {feat['contribution']:.4f}")
            
            # Generate explanation
            st.markdown("#### Model Explanation")
            winner = context['team_a'] if context['matchup_prob_a'] > context['matchup_prob_b'] else context['team_b']
            win_pct = max(context['matchup_prob_a'], context['matchup_prob_b']) * 100
            
            top_3 = context['top_features'][:3]
            factors = []
            for feat in top_3:
                fname = feat['feature'].replace('diff_', '').replace('_', ' ')
                diff_val = feat['diff_value']
                sign_word = "advantage" if diff_val > 0 else "disadvantage"
                factors.append(f"{fname} ({sign_word})")
            
            explanation = f"The model predicts **{winner}** wins with **{win_pct:.1f}%** probability. "
            explanation += f"Key factors: {', '.join(factors)}."
            
            st.info(explanation)
        else:
            st.info("Feature importance not available for this model.")
        
        # Team stats
        st.markdown("#### Season Stats Summary")
        
        stats_df = pd.DataFrame({
            context['team_a']: [
                f"{context['team_a_stats'].get('points_mean', 0):.1f}" if context['team_a_stats'].get('points_mean') is not None else "N/A",
                f"{context['team_a_stats'].get('turnover_margin', 0):.1f}" if context['team_a_stats'].get('turnover_margin') is not None else "N/A",
                f"{context['team_a_stats'].get('elo', 1500):.0f}" if context['team_a_stats'].get('elo') is not None else "N/A",
            ],
            context['team_b']: [
                f"{context['team_b_stats'].get('points_mean', 0):.1f}" if context['team_b_stats'].get('points_mean') is not None else "N/A",
                f"{context['team_b_stats'].get('turnover_margin', 0):.1f}" if context['team_b_stats'].get('turnover_margin') is not None else "N/A",
                f"{context['team_b_stats'].get('elo', 1500):.0f}" if context['team_b_stats'].get('elo') is not None else "N/A",
            ]
        }, index=['Avg Points/Game', 'Turnover Margin', 'ELO Rating'])
        
        st.dataframe(stats_df, use_container_width=True)


def render_bracket_button(slot_key: str, default_label: str, result: Dict, models: Dict, 
                          team_stats: pd.DataFrame, elo_dict: Dict, key_suffix: str = ""):
    """
    Render a bracket button with click handling for game details modal.
    
    Args:
        slot_key: Bracket slot key (e.g., 'afc_wild1')
        default_label: Default label if no team assigned
        result: Bracket result dict
        models: Models dict
        team_stats: Team stats DataFrame
        elo_dict: ELO ratings dict
        key_suffix: Optional suffix for button key uniqueness
    """
    enabled, team = determine_button_enabled(slot_key, result)
    label = team if team else default_label
    button_key = f"btn_{slot_key}{key_suffix}"
    
    if st.button(label, key=button_key, disabled=not enabled):
        if team and models and team_stats is not None and elo_dict:
            try:
                context = get_matchup_context(slot_key, team, result, models, team_stats, elo_dict)
                show_game_modal(context)
            except Exception as e:
                st.error(f"Could not load game details: {str(e)}")


def list_model_versions(models_dir: str = "models") -> List[str]:
    """
    Discover and list all model version folders in the models/ directory.
    
    Args:
        models_dir: Path to the models directory (default: "models")
    
    Returns:
        A sorted list of model version folder names (e.g., ["model_2015-2024", "model_2014-2023"]).
        Returns empty list if models_dir doesn't exist or has no matching folders.
    
    Sorting:
        Sorts by end year (descending) if folder name matches pattern "model_*-YYYY",
        otherwise sorts alphabetically (descending).
    """
    models_root = Path(models_dir)
    
    # If models directory doesn't exist, return empty list
    if not models_root.exists() or not models_root.is_dir():
        return []
    
    # Find all subdirectories that start with "model_"
    versions = [
        p.name for p in models_root.iterdir() 
        if p.is_dir() and p.name.startswith("model_") and not p.name.startswith(".")
    ]
    
    # Helper function to extract end year from folder name (e.g., "model_2015-2024" -> 2024)
    def end_year(folder_name: str) -> int:
        match = re.search(r"-(\d{4})$", folder_name)
        return int(match.group(1)) if match else 0
    
    # Sort by end year (descending), then by name (descending) as fallback
    versions.sort(key=lambda x: (end_year(x), x), reverse=True)
    
    return versions


@st.cache_resource
def load_model_version(version_folder: str, models_dir: str = "models") -> Dict[str, Optional[object]]:
    """
    Load the ML models from a specific version folder using joblib.
    
    Args:
        version_folder: Name of the version folder (e.g., "model_2015-2024")
        models_dir: Path to the models directory (default: "models")
    
    Returns:
        A dictionary with keys:
            - "model1": The playoff_qualifier model object (or None if not found)
            - "model2": The bracket model object (or None if not found)
            - "path": Absolute path to the version folder (str)
    
    Example usage:
        models = load_model_version("model_2015-2024")
        if models["model1"]:
            predictions = models["model1"].predict(data)
    
    Note:
        This function is cached with @st.cache_resource to avoid reloading
        models on every Streamlit rerun. The cache persists across user sessions.
    """
    version_path = Path(models_dir) / version_folder
    
    # Expected filenames
    model1_file = version_path / "model1_playoff_qualifier.pkl"
    model2_file = version_path / "model2_bracket.pkl"
    
    # Load model1 if it exists
    model1 = None
    if model1_file.exists():
        try:
            model1 = joblib.load(model1_file)
        except Exception as e:
            st.warning(f"Failed to load model1 from {model1_file}: {e}")
    
    # Load model2 if it exists
    model2 = None
    if model2_file.exists():
        try:
            model2 = joblib.load(model2_file)
        except Exception as e:
            st.warning(f"Failed to load model2 from {model2_file}: {e}")
    
    # Load precomputed inference features if they exist
    precomputed_team_stats = None
    precomputed_elo = None
    metadata = None
    
    team_stats_file = version_path / "precomputed_team_stats.pkl"
    elo_file = version_path / "precomputed_elo_ratings.pkl"
    metadata_file = version_path / "metadata.json"
    
    if team_stats_file.exists():
        try:
            precomputed_team_stats = joblib.load(team_stats_file)
        except Exception as e:
            st.warning(f"Failed to load precomputed team stats: {e}")
    
    if elo_file.exists():
        try:
            precomputed_elo = joblib.load(elo_file)
        except Exception as e:
            st.warning(f"Failed to load precomputed ELO ratings: {e}")
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            st.warning(f"Failed to load metadata: {e}")
    
    return {
        "model1": model1,
        "model2": model2,
        "path": str(version_path.absolute()),
        "precomputed_team_stats": precomputed_team_stats,
        "precomputed_elo_ratings": precomputed_elo,
        "metadata": metadata,
        "has_inference_data": precomputed_team_stats is not None and precomputed_elo is not None
    }

# ---- Session state ----
if "clicked" not in st.session_state:
    st.session_state.clicked = False
if "flow" not in st.session_state:
    st.session_state.flow = None  # None | "predicting" | "training"
if "control_mode" not in st.session_state:
    st.session_state.control_mode = "AutoML"  # Default to AutoML mode
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None  # Track which model was selected
if "bracket_result" not in st.session_state:
    st.session_state.bracket_result = None  # AutoML bracket prediction results
if "bracket_filled" not in st.session_state:
    st.session_state.bracket_filled = False  # Whether bracket has been filled by AutoML
if "automl_predicting" not in st.session_state:
    st.session_state.automl_predicting = False
if "manual_predicting" not in st.session_state:
    st.session_state.manual_predicting = False
if "manual_bracket_predicted" not in st.session_state:
    st.session_state.manual_bracket_predicted = False  # Whether manual bracket has been predicted and should show buttons
# Interactive bracket state
if "bracket_models" not in st.session_state:
    st.session_state.bracket_models = None
if "bracket_team_stats" not in st.session_state:
    st.session_state.bracket_team_stats = None
if "bracket_elo_dict" not in st.session_state:
    st.session_state.bracket_elo_dict = None
if "playoff_probs_dict" not in st.session_state:
    st.session_state.playoff_probs_dict = {}


# ---- Centered Title (ALWAYS VISIBLE) ----
col1, col2, col3 = st.columns([1, 1, 0.5])
with col2:
    st.title("NFL Predictor")

# ---- image placeholder (landing) ----
image_path = Path("static/nfl.png")
if not st.session_state.clicked:
    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col2:
        st.image(str(image_path), caption="NFL Logo")
        col_l, col_c, col_r = st.columns([1, 3, 3])
        if col_c.button("Predict", key="predict_button"):
            st.session_state.clicked = True
            st.rerun()

# ---- MAIN UI (ONLY if clicked AND no flow chosen yet) ----
if st.session_state.clicked and st.session_state.flow is None:

    today = datetime.now()
    latest_year = today.year - 1
    start_year = latest_year - 9

    outer_l, outer_c, outer_r = st.columns([1.2, 2, 1.2])
    with outer_c:
        st.subheader("Model Selection & Training")

        tab_use, tab_train = st.tabs(
            ["Use Pretrained Model", "Train New Model"]
        )

        with tab_use:
            st.markdown("### Choose Model")
            
            # Discover available model versions from models/ directory
            available_versions = list_model_versions()
            
            # Build selectbox options: always include placeholder, then add discovered versions
            selectbox_options = ["-- Choose a model --"] + available_versions
            
            # Show info message if no models found
            if not available_versions:
                st.info(
                    "No saved model versions found in `models/` directory. "
                    "Please train a new model or copy model folders into `models/`."
                )
            
            # Display selectbox with discovered model versions
            model_choice = st.selectbox(
                "Available models",
                selectbox_options,
                index=0
            )

            # Show "Predict Selected Model" button only if a real model is selected
            if model_choice != "-- Choose a model --":
                # Optional: Show which .pkl files exist in this version
                version_path = Path("models") / model_choice
                has_model1 = (version_path / "model1_playoff_qualifier.pkl").exists()
                has_model2 = (version_path / "model2_bracket.pkl").exists()
                
                # Display available models info
                info_parts = []
                if has_model1:
                    info_parts.append("✓ Playoff Qualifier (model1)")
                if has_model2:
                    info_parts.append("✓ Bracket Simulator (model2)")
                
                if info_parts:
                    st.caption(" | ".join(info_parts))
                else:
                    st.warning(
                        f"⚠️ No valid model files found in `{model_choice}/`. "
                        f"Expected `model1_playoff_qualifier.pkl` and/or `model2_bracket.pkl`."
                    )
                
                # Button to proceed to prediction
                b_l, b_c, b_r = st.columns([1, 1, 1])
                with b_c:
                    if st.button("Predict Selected Model"):
                        # Set session state to the selected version folder name
                        st.session_state.selected_model = model_choice
                        # Reset bracket results when a new model is selected
                        st.session_state.bracket_result = None
                        st.session_state.bracket_filled = False
                        st.session_state.flow = "predicting"
                        st.rerun()
                        
                        # Example: To load the models later (in prediction flow), use:
                        # models = load_model_version(st.session_state.selected_model)
                        # if models["model1"]:
                        #     playoff_predictions = models["model1"].predict(team_data)
                        # if models["model2"]:
                        #     bracket_winner = models["model2"].predict(matchup_data)

        with tab_train:
            st.markdown("### Train New Model")
            st.write("This will train a new model using the **most recent completed NFL seasons**.")

            s0, s1 = st.columns(2)
            s0.markdown(f"**Start:** {start_year}")
            s1.markdown(f"**End:** {latest_year}")

            t_l, t_c, t_r = st.columns([1, 1, 1])
            with t_c:
                if st.button("Train Model Now"):
                    st.session_state.selected_model = "Newly Trained Model"
                    st.session_state.flow = "training"
                    st.rerun()

# ---- PREDICTION / BRACKET UI ----
if st.session_state.flow == "predicting":

    # NFL Teams
    afc_teams_full = [
        "-- Choose a team --",
        "Baltimore Ravens", "Buffalo Bills", "Cincinnati Bengals", "Cleveland Browns",
        "Denver Broncos", "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
        "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers", "Miami Dolphins",
        "New England Patriots", "New York Jets", "Pittsburgh Steelers", "Tennessee Titans"
    ]

    nfc_teams_full = [
        "-- Choose a team --",
        "Arizona Cardinals", "Atlanta Falcons", "Carolina Panthers", "Chicago Bears",
        "Dallas Cowboys", "Detroit Lions", "Green Bay Packers", "Los Angeles Rams",
        "Minnesota Vikings", "New Orleans Saints", "New York Giants", "Philadelphia Eagles",
        "San Francisco 49ers", "Seattle Seahawks", "Tampa Bay Buccaneers", "Washington Commanders"
    ]

    # Top bracket row
    cols = st.columns([1, 1, 1, 0.9, 1, 1, 1])

    # Determine if we're in manual mode (dropdowns) or AutoML mode (buttons)
    manual_mode = (st.session_state.control_mode == "Choose Teams")

    # Helper function to get available teams (excluding already selected ones)
    def get_available_teams(full_list, selected_list, current_key):
        current_value = st.session_state.get(current_key, "-- Choose a team --")
        return [t for t in full_list if t == "-- Choose a team --" or t not in selected_list or t == current_value]

    # Determine if manual mode is active (and not yet predicted)
    # If manual bracket is predicted, show buttons instead of dropdowns
    manual_show_dropdowns = manual_mode and not st.session_state.manual_bracket_predicted
    
    # Get currently selected AFC teams
    afc_selected = []
    afc_team_seeds = {}  # For manual bracket: seed -> team name
    if manual_show_dropdowns:
        # Map UI keys to actual playoff seeds
        afc_key_to_seed = {
            "afc_bye": 1,    # #1 seed gets bye
            "afc_wild1": 2,  # Plays #7
            "afc_wild2": 7,  # Plays #2
            "afc_wild3": 3,  # Plays #6
            "afc_wild4": 6,  # Plays #3
            "afc_wild5": 4,  # Plays #5
            "afc_wild6": 5   # Plays #4
        }
        for key, seed in afc_key_to_seed.items():
            val = st.session_state.get(key, "-- Choose a team --")
            if val and val != "-- Choose a team --":
                afc_selected.append(val)
                afc_team_seeds[seed] = val

    # Get currently selected NFC teams
    nfc_selected = []
    nfc_team_seeds = {}  # For manual bracket: seed -> team name
    if manual_show_dropdowns:
        # Map UI keys to actual playoff seeds
        nfc_key_to_seed = {
            "nfc_bye": 1,    # #1 seed gets bye
            "nfc_wild1": 2,  # Plays #7
            "nfc_wild2": 7,  # Plays #2
            "nfc_wild3": 3,  # Plays #6
            "nfc_wild4": 6,  # Plays #3
            "nfc_wild5": 4,  # Plays #5
            "nfc_wild6": 5   # Plays #4
        }
        for key, seed in nfc_key_to_seed.items():
            val = st.session_state.get(key, "-- Choose a team --")
            if val and val != "-- Choose a team --":
                nfc_selected.append(val)
                nfc_team_seeds[seed] = val

    # Check if all teams are selected for manual mode
    all_afc_selected = len(afc_selected) == 7
    all_nfc_selected = len(nfc_selected) == 7
    manual_teams_complete = all_afc_selected and all_nfc_selected

    # ALL BUTTONS DISABLED UNLESS BRACKET IS FILLED - now handled per button
    # Interactive mode: buttons become clickable to show game details
    
    # Get shared state for rendering interactive buttons
    result = st.session_state.bracket_result
    models = st.session_state.bracket_models
    team_stats = st.session_state.bracket_team_stats
    elo_dict = st.session_state.bracket_elo_dict
    
    # Helper to get bracket slot label dynamically
    def get_bracket_label(slot_key: str, default_label: str) -> str:
        """Get label from bracket result or return default."""
        if st.session_state.bracket_filled and st.session_state.bracket_result:
            slots = st.session_state.bracket_result.get("bracket_slots", {})
            return slots.get(slot_key, default_label)
        return default_label

    # ---- AFC Wild / Bye ----
    with cols[0]:
        st.markdown("**AFC Wild**")
        if manual_show_dropdowns:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_bye")
            current = st.session_state.get("afc_bye", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("AFC Bye", available, index=idx, key="afc_bye", label_visibility="collapsed")
        else:
            render_bracket_button("afc_bye", "AFC Bye", result, models, team_stats, elo_dict, "_bye")
        st.markdown("Bye Week")
        vspace(1)
        if manual_show_dropdowns:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild1")
            current = st.session_state.get("afc_wild1", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 1", available, index=idx, key="afc_wild1", label_visibility="collapsed")
            
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild2")
            current = st.session_state.get("afc_wild2", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 2", available, index=idx, key="afc_wild2", label_visibility="collapsed")
        else:
            render_bracket_button("afc_wild1", "AFC Wild 1", result, models, team_stats, elo_dict, "_w1")
            render_bracket_button("afc_wild2", "AFC Wild 2", result, models, team_stats, elo_dict, "_w2")
        vspace(1)
        if manual_show_dropdowns:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild3")
            current = st.session_state.get("afc_wild3", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 3", available, index=idx, key="afc_wild3", label_visibility="collapsed")
            
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild4")
            current = st.session_state.get("afc_wild4", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 4", available, index=idx, key="afc_wild4", label_visibility="collapsed")
        else:
            render_bracket_button("afc_wild3", "AFC Wild 3", result, models, team_stats, elo_dict, "_w3")
            render_bracket_button("afc_wild4", "AFC Wild 4", result, models, team_stats, elo_dict, "_w4")
        vspace(1)
        if manual_show_dropdowns:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild5")
            current = st.session_state.get("afc_wild5", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 5", available, index=idx, key="afc_wild5", label_visibility="collapsed")
            
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild6")
            current = st.session_state.get("afc_wild6", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 6", available, index=idx, key="afc_wild6", label_visibility="collapsed")
        else:
            render_bracket_button("afc_wild5", "AFC Wild 5", result, models, team_stats, elo_dict, "_w5")
            render_bracket_button("afc_wild6", "AFC Wild 6", result, models, team_stats, elo_dict, "_w6")

    # ---- AFC Divisional ----
    with cols[1]:
        st.markdown("**AFC Divisional**")
        vspace(3)
        render_bracket_button("afc_div_win_1", "AFC Div Win 1", result, models, team_stats, elo_dict, "_top1")
        render_bracket_button("afc_div_win_2", "AFC Div Win 2", result, models, team_stats, elo_dict, "_top2")
        vspace(9)
        render_bracket_button("afc_div_win_3", "AFC Div Win 3", result, models, team_stats, elo_dict, "_bot1")
        render_bracket_button("afc_div_win_4", "AFC Div Win 4", result, models, team_stats, elo_dict, "_bot2")

    # ---- AFC Conference ----
    with cols[2]:
        st.markdown("**AFC Conference**")
        vspace(11)
        render_bracket_button("afc_conf_win_1", "AFC Conf Win 1", result, models, team_stats, elo_dict, "_1")
        render_bracket_button("afc_conf_win_2", "AFC Conf Win 2", result, models, team_stats, elo_dict, "_2")

    # ---- SUPER BOWL (centered vertically) ----
    with cols[3]:
        st.markdown("**Super Bowl**")
        vspace(11)
        render_bracket_button("sb_team_a", "SB Team A", result, models, team_stats, elo_dict, "_a")
        render_bracket_button("sb_team_b", "SB Team B", result, models, team_stats, elo_dict, "_b")

    # ---- NFC Conference ----
    with cols[4]:
        st.markdown("**NFC Conference**")
        vspace(11)
        render_bracket_button("nfc_conf_win_1", "NFC Conf Win 1", result, models, team_stats, elo_dict, "_1")
        render_bracket_button("nfc_conf_win_2", "NFC Conf Win 2", result, models, team_stats, elo_dict, "_2")

    # ---- NFC Divisional ----
    with cols[5]:
        st.markdown("**NFC Divisional**")
        vspace(3)
        render_bracket_button("nfc_div_win_1", "NFC Div Win 1", result, models, team_stats, elo_dict, "_top1")
        render_bracket_button("nfc_div_win_2", "NFC Div Win 2", result, models, team_stats, elo_dict, "_top2")
        vspace(9)
        render_bracket_button("nfc_div_win_3", "NFC Div Win 3", result, models, team_stats, elo_dict, "_bot1")
        render_bracket_button("nfc_div_win_4", "NFC Div Win 4", result, models, team_stats, elo_dict, "_bot2")

    # ---- NFC Wild / Bye ----
    with cols[6]:
        st.markdown("**NFC Wild**")
        if manual_show_dropdowns:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_bye")
            current = st.session_state.get("nfc_bye", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("NFC Bye", available, index=idx, key="nfc_bye", label_visibility="collapsed")
        else:
            render_bracket_button("nfc_bye", "NFC Bye", result, models, team_stats, elo_dict, "_bye")
        st.markdown("Bye Week")
        vspace(1)
        if manual_show_dropdowns:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild1")
            current = st.session_state.get("nfc_wild1", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 1", available, index=idx, key="nfc_wild1", label_visibility="collapsed")
            
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild2")
            current = st.session_state.get("nfc_wild2", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 2", available, index=idx, key="nfc_wild2", label_visibility="collapsed")
        else:
            render_bracket_button("nfc_wild1", "NFC Wild 1", result, models, team_stats, elo_dict, "_w1")
            render_bracket_button("nfc_wild2", "NFC Wild 2", result, models, team_stats, elo_dict, "_w2")
        vspace(1)
        if manual_show_dropdowns:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild3")
            current = st.session_state.get("nfc_wild3", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 3", available, index=idx, key="nfc_wild3", label_visibility="collapsed")
            
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild4")
            current = st.session_state.get("nfc_wild4", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 4", available, index=idx, key="nfc_wild4", label_visibility="collapsed")
        else:
            render_bracket_button("nfc_wild3", "NFC Wild 3", result, models, team_stats, elo_dict, "_w3")
            render_bracket_button("nfc_wild4", "NFC Wild 4", result, models, team_stats, elo_dict, "_w4")
        vspace(1)
        if manual_show_dropdowns:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild5")
            current = st.session_state.get("nfc_wild5", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 5", available, index=idx, key="nfc_wild5", label_visibility="collapsed")
            
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild6")
            current = st.session_state.get("nfc_wild6", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 6", available, index=idx, key="nfc_wild6", label_visibility="collapsed")
        else:
            render_bracket_button("nfc_wild5", "NFC Wild 5", result, models, team_stats, elo_dict, "_w5")
            render_bracket_button("nfc_wild6", "NFC Wild 6", result, models, team_stats, elo_dict, "_w6")

    # ---- CONTROLS SECTION (wider, centered below) ----
    control_cols = st.columns([2.8, 5, 3.3])
    
    with control_cols[1]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1.3, 1, 1])
            with c2:
                st.markdown("### Controls")
            
            # Simple toggle between modes - centered with columns
            r1, r2, r3 = st.columns([1.4, 2.4, 0.8])
            with r2:
                mode = st.radio(
                    "Control Mode",
                    ["AutoML", "Choose Teams"],
                    index=0 if st.session_state.control_mode == "AutoML" else 1,
                    horizontal=True,
                    label_visibility="collapsed"
                )
            
            # Update session state and rerun if mode changed
            if mode != st.session_state.control_mode:
                st.session_state.control_mode = mode
                # Reset bracket results when switching modes
                st.session_state.bracket_result = None
                st.session_state.bracket_filled = False
                st.session_state.manual_bracket_predicted = False
                st.rerun()
            
            st.markdown("---")
            
            # Model info section - compact row
            if st.session_state.selected_model:
                model_row = st.columns([1.3, 1])
                with model_row[0]:
                    st.markdown(f"**Model:** {st.session_state.selected_model}")
                with model_row[1]:
                    if st.button("Choose a different model", key="change_model"):
                        st.session_state.flow = None
                        st.session_state.clicked = True
                        st.session_state.bracket_result = None
                        st.session_state.bracket_filled = False
                        st.session_state.manual_bracket_predicted = False
                        st.rerun()
            
            st.markdown("---")
            
            if mode == "AutoML":
                st.markdown("#### AutoML")
                st.write(
                "Fully autonomously predict the entire playoff bracket using two models: "
                "one to determine which teams qualify for the postseason, and another to simulate each round "
                "and predict the Super Bowl champion.")
                vspace(1)
                # Define button state based on processing status
                automl_btn_label = "Predicting..." if st.session_state.automl_predicting else "Predict Bracket"
                automl_btn_disabled = st.session_state.automl_predicting
                
                if st.button(automl_btn_label, use_container_width=True, key="predict_automl", disabled=automl_btn_disabled):
                    if not st.session_state.selected_model:
                        st.warning("Choose a saved model version first.")
                    else:
                        st.session_state.automl_predicting = True
                        st.rerun()
                
                # Logic runs if state is True (after rerun)
                if st.session_state.automl_predicting:
                    if not st.session_state.selected_model:
                         # Should be caught above, but safety check
                         st.session_state.automl_predicting = False
                         st.rerun()
                    else:
                        with st.spinner("Running AutoML bracket prediction..."):
                            try:
                                import time
                                # Import bracket predictor
                                from bracket_predictor import run_automl_bracket_inference
                                
                                # Load models
                                models = load_model_version(st.session_state.selected_model)
                                if not models.get("model1") or not models.get("model2"):
                                    st.error("Missing model files. Please train or select a valid model with both model1 and model2.")
                                    st.session_state.automl_predicting = False
                                else:
                                    # Check if precomputed features exist (inference-only mode)
                                    if models.get("has_inference_data"):
                                        # INFERENCE MODE: Use precomputed features (no CSV dependency)
                                        metadata = models.get("metadata", {})
                                        current_season = metadata.get("latest_season", 2024)
                                        
                                        # Run inference-only bracket prediction
                                        result = run_automl_bracket_inference(
                                            models["model1"],
                                            models["model2"],
                                            models["precomputed_team_stats"],
                                            models["precomputed_elo_ratings"],
                                            current_season
                                        )
                                        
                                        # Add delay to show the spinner animation longer
                                        time.sleep(2)
                                        
                                        # Store results (including models and stats for interactive buttons)
                                        st.session_state.bracket_result = result
                                        st.session_state.bracket_filled = True
                                        st.session_state.bracket_models = {
                                            'model1': models["model1"],
                                            'model2': models["model2"]
                                        }
                                        st.session_state.bracket_team_stats = models["precomputed_team_stats"]
                                        st.session_state.bracket_elo_dict = models["precomputed_elo_ratings"]
                                        
                                        # Compute and store playoff probabilities for display in modals
                                        # Request ALL teams (teams_per_conference=32) so users can see probabilities for any team they select
                                        playoff_probs_dict = {}
                                        try:
                                            from bracket_predictor import predict_playoff_teams
                                            playoff_teams = predict_playoff_teams(models["model1"], models["precomputed_team_stats"], teams_per_conference=32)
                                            for conf in ['AFC', 'NFC']:
                                                for team, prob in playoff_teams.get(conf, []):
                                                    playoff_probs_dict[team] = prob
                                        except Exception as e:
                                            pass
                                        st.session_state.playoff_probs_dict = playoff_probs_dict
                                        
                                        st.session_state.automl_predicting = False
                                        st.rerun()
                                    else:
                                        # FAIL FAST: Model lacks inference data
                                        st.error(
                                            "⚠️ **This model lacks inference data.**\n\n"
                                            "This model was exported without precomputed features and cannot run predictions without CSV files.\n\n"
                                            "**Options:**\n"
                                            "1. Train a new model (it will include inference data)\n"
                                            "2. Place CSV files in `data_files/` folder and re-select this model\n"
                                            "3. Delete this model folder and use a newer model"
                                        )
                                        st.session_state.automl_predicting = False
                            except Exception as e:
                                st.error(f"Bracket prediction failed: {str(e)}")
                                st.session_state.automl_predicting = False
                
                # Show champion if bracket is filled
                if st.session_state.bracket_filled and st.session_state.bracket_result:
                    champion = st.session_state.bracket_result.get("champion")
                    if champion:
                        st.success(f" Predicted Super Bowl Winner: {champion}")
            else:
                st.markdown("#### Manual Selection")
                st.write(
                "Choose the playoff teams manually using the dropdowns. "
                "This mode uses only the playoff bracket model to simulate each round "
                "from the Wild Card through the Super Bowl and determine the champion.")
                vspace(1)
                
                # Check if manual bracket has been predicted
                if st.session_state.manual_bracket_predicted:
                    # Show Reset Bracket button
                    if st.button("Reset Bracket", use_container_width=True, key="reset_manual"):
                        # Reset all manual bracket state
                        st.session_state.manual_bracket_predicted = False
                        st.session_state.bracket_result = None
                        st.session_state.bracket_filled = False
                        st.session_state.bracket_models = None
                        st.session_state.bracket_team_stats = None
                        st.session_state.bracket_elo_dict = None
                        st.session_state.playoff_probs_dict = {}
                        st.rerun()
                else:
                    # Show Predict Bracket button
                    # Button disabled until all 14 teams selected OR if currently predicting
                    predict_manual_disabled = (not manual_teams_complete) or st.session_state.manual_predicting
                    manual_btn_label = "Predicting..." if st.session_state.manual_predicting else "Predict Bracket"
                    
                    if st.button(manual_btn_label, use_container_width=True, key="predict_manual", disabled=predict_manual_disabled):
                        if not st.session_state.selected_model:
                            st.warning("Choose a saved model version first.")
                        else:
                            st.session_state.manual_predicting = True
                            st.rerun()
                
                # Logic runs if state is True (after rerun)
                if st.session_state.manual_predicting:
                    if not st.session_state.selected_model:
                         st.session_state.manual_predicting = False
                         st.rerun()
                    else:
                        with st.spinner("Running bracket simulation..."):
                            try:
                                import time
                                # Import bracket predictor
                                from bracket_predictor import run_manual_bracket_inference
                                
                                # Load models (only need model2)
                                models = load_model_version(st.session_state.selected_model)
                                if not models.get("model2"):
                                    st.error("Missing Model 2 (Bracket Predictor). Please train or select a valid model.")
                                    st.session_state.manual_predicting = False
                                else:
                                    # Check if precomputed features exist (inference-only mode)
                                    if models.get("has_inference_data"):
                                        # INFERENCE MODE: Use precomputed features (no CSV dependency)
                                        metadata = models.get("metadata", {})
                                        current_season = metadata.get("latest_season", 2024)
                                        
                                        # Compute playoff probabilities for ALL teams (for display purposes)
                                        # even though user manually selected playoff teams
                                        # Request ALL teams (teams_per_conference=32) so any manually selected team shows its probability
                                        playoff_probs_dict = {}
                                        if models.get("model1") and models.get("precomputed_team_stats") is not None:
                                            from bracket_predictor import predict_playoff_teams
                                            try:
                                                playoff_teams = predict_playoff_teams(models["model1"], models["precomputed_team_stats"], teams_per_conference=32)
                                                # Convert to dict for easy lookup
                                                for conf in ['AFC', 'NFC']:
                                                    for team, prob in playoff_teams.get(conf, []):
                                                        playoff_probs_dict[team] = prob
                                            except Exception as e:
                                                # If prediction fails, that's okay - we'll just not show playoff probs
                                                pass
                                        
                                        # Run inference-only bracket prediction
                                        result = run_manual_bracket_inference(
                                            models["model2"],
                                            models["precomputed_team_stats"],
                                            models["precomputed_elo_ratings"],
                                            current_season,
                                            afc_team_seeds,
                                            nfc_team_seeds
                                        )
                                        
                                        # Add delay to show the spinner animation longer
                                        time.sleep(2)
                                        
                                        # Store results (including models and stats for interactive buttons)
                                        st.session_state.bracket_result = result
                                        st.session_state.bracket_filled = True
                                        st.session_state.bracket_models = {
                                            'model1': models.get("model1"),
                                            'model2': models["model2"]
                                        }
                                        st.session_state.bracket_team_stats = models["precomputed_team_stats"]
                                        st.session_state.bracket_elo_dict = models["precomputed_elo_ratings"]
                                        # Store playoff probabilities for manual mode display
                                        st.session_state.playoff_probs_dict = playoff_probs_dict
                                        # Set flag to show buttons instead of dropdowns
                                        st.session_state.manual_bracket_predicted = True
                                        st.session_state.manual_predicting = False
                                        st.rerun()
                                    else:
                                        # FAIL FAST: Model lacks inference data
                                        st.error(
                                            "⚠️ **This model lacks inference data.**\n\n"
                                            "This model was exported without precomputed features and cannot run predictions without CSV files.\n\n"
                                            "**Options:**\n"
                                            "1. Train a new model (it will include inference data)\n"
                                            "2. Place CSV files in `data_files/` folder and re-select this model\n"
                                            "3. Delete this model folder and use a newer model"
                                        )
                                        st.session_state.manual_predicting = False
                            except Exception as e:
                                st.error(f"Bracket prediction failed: {str(e)}")
                                st.session_state.manual_predicting = False
                
                
                # Show champion if bracket is filled
                if st.session_state.bracket_filled and st.session_state.bracket_result:
                    champion = st.session_state.bracket_result.get("champion")
                    if champion:
                        st.success(f"🏆 Predicted Champion: {champion}")

# ---- TRAINING state ----
if st.session_state.flow == "training":
    # Initialize session state for training
    if "training_status" not in st.session_state:
        st.session_state.training_status = None  # "scraping" | "training" | "complete"
    if "training_log" not in st.session_state:
        st.session_state.training_log = []
    if "training_complete" not in st.session_state:
        st.session_state.training_complete = False
    if "training_started" not in st.session_state:
        st.session_state.training_started = False
    if "training_results" not in st.session_state:
        st.session_state.training_results = None
    
    # Title
    st.markdown("### Training New Model")
    st.markdown("---")
    
    # Define callback function to capture progress
    def training_progress_callback(message: str, msg_type: str = "info"):
        """Callback for training progress updates"""
        st.session_state.training_log.append({"message": message, "type": msg_type})
    
    # Auto-start training if not started
    if not st.session_state.training_started:
        st.session_state.training_started = True
        st.session_state.training_status = "scraping"
        st.session_state.current_phase = "scraping"  # scraping, model1, model2
        
        # Import the training functions
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from Scraping_Data import scrape_nfl_data
        from training_model import train_models
        
        # Display containers for live updates
        # Initial pause requested by user before showing Scraping header
        import time
        time.sleep(1.5)
        
        scraping_container = st.empty()
        model1_container = st.empty()
        model2_container = st.empty()
        
        # Create persistent containers for each section
        with scraping_container.container():
            st.subheader("Scraping NFL Data From NFLVERSE...")
            scraping_log = st.empty()
            
        with model1_container.container():
            model1_header = st.empty() # Initially empty
            model1_log = st.empty()
            model1_metrics = st.empty()
            
        with model2_container.container():
            model2_header = st.empty() # Initially empty
            model2_log = st.empty()
            model2_metrics = st.empty()

        # Define callback function to capture progress AND render immediately
        def training_progress_callback(message: str, msg_type: str = "info"):
            """Callback for training progress updates - renders immediately"""
            # Special handling for metrics JSON payloads
            if msg_type == "metrics_model1":
                data = json.loads(message)
                with model1_metrics.container():
                    st.metric("Accuracy", f"{data['accuracy']:.4f}")
                    with st.expander("Classification Report"):
                        st.text(data['report'])
                return
            
            if msg_type == "metrics_model2":
                data = json.loads(message)
                with model2_metrics.container():
                    st.metric("Accuracy", f"{data['accuracy']:.4f}")
                    with st.expander("Classification Report"):
                        st.text(data['report'])
                return
            
            # Add to history
            st.session_state.training_log.append({"message": message, "type": msg_type})
            
            # Detect phase change & Reveal Headers
            if "Training Model 2" in message or "Bracket Predictor" in message:
                st.session_state.current_phase = "model2"
                model2_header.subheader("Training Model 2: Bracket Predictor")
            elif "Training Model 1" in message:
                st.session_state.current_phase = "model1"
                model1_header.subheader("Training Model 1: Playoff Qualifier")
                
            # Render logs for the current phase
            target_log = None
            if st.session_state.current_phase == "scraping":
                target_log = scraping_log
            elif st.session_state.current_phase == "model1":
                target_log = model1_log
            elif st.session_state.current_phase == "model2":
                target_log = model2_log
            
            # Filter logs for this phase to re-render the list
            if target_log:
                with target_log.container():
                    current_log_phase = "scraping"
                    for i, log in enumerate(st.session_state.training_log):
                        msg_text = log["message"]
                        if "Training Model 1" in msg_text:
                            current_log_phase = "model1"
                        elif "Training Model 2" in msg_text:
                            current_log_phase = "model2"
                            
                        if current_log_phase == st.session_state.current_phase:
                             # Don't show the header trigger messages in the log
                             if "Training Model 1" in msg_text or "Training Model 2" in msg_text:
                                 continue
                             
                             # Deduplicate consecutive messages to prevent "Processing..." double vision
                             # This handles potential state/hot-reload glitches
                             if i > 0:
                                 prev_msg = st.session_state.training_log[i-1]["message"]
                                 if msg_text == prev_msg:
                                     continue
                             
                             if log["type"] == "success":
                                st.success(log["message"])
                             elif log["type"] == "error":
                                st.error(log["message"])
                             else:
                                msg = log["message"]
                                # User requested "Processing..." and "Training..." be bigger.
                                # Using standard Markdown headers (safe mode).
                                if msg.strip() in ["Processing...", "Training..."]:
                                    st.markdown(f"### {msg}")
                                else:
                                    st.write(msg)

        # Clear log for scraping
        st.session_state.training_log = []
        
        # Run scraping
        st.session_state.current_phase = "scraping"
        scraping_result = scrape_nfl_data(progress_callback=training_progress_callback)
        
        if not scraping_result["success"]:
            st.error(f"Scraping failed: {scraping_result.get('error', 'Unknown error')}")
            if st.button("Restart Training"):
                st.session_state.training_started = False
                st.session_state.training_log = []
                st.rerun()
            st.stop()
        
        # === MODEL TRAINING PHASE ===
        st.session_state.training_status = "training"
        
        # Run training
        training_result = train_models(progress_callback=training_progress_callback)
        
        # Check if training was successful
        if training_result["success"]:
             # Ensure headers are visible at end (safety fallback)
             model1_header.subheader("Training Model 1: Playoff Qualifier")
             model2_header.subheader("Training Model 2: Bracket Predictor")
             
             st.session_state.training_complete = True
             st.session_state.training_results = training_result
             st.session_state.selected_model = training_result["version_folder"]
             st.success("Training complete! Your model is ready to use.")
        else:
            st.error(f"Training failed: {training_result.get('error', 'Unknown error')}")
            if st.button("Restart Training"):
                st.session_state.training_started = False
                st.session_state.training_log = []
                st.session_state.training_complete = False
                st.rerun()
            st.stop()
    
    # Show continue button if training is complete
    if st.session_state.training_complete:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Continue to Predictions", use_container_width=True, type="primary"):
                st.session_state.flow = "predicting"
                st.session_state.training_started = False
                st.session_state.training_log = []
                st.session_state.training_complete = False
                st.rerun()