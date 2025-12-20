import streamlit as st
import os
import re
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---- config ----
icon_path = os.path.join("static", "nfl.png")
st.set_page_config(page_title="NFL Predictor", page_icon=icon_path, layout="wide")

# ---- helpers ----
def vspace(n: int):
    for _ in range(n):
        st.write("")


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
    
    return {
        "model1": model1,
        "model2": model2,
        "path": str(version_path.absolute())
    }

# ---- Session state ----
if "clicked" not in st.session_state:
    st.session_state.clicked = False
if "flow" not in st.session_state:
    st.session_state.flow = None  # None | "predicting" | "training"
if "control_mode" not in st.session_state:
    st.session_state.control_mode = "Choose Teams"  # Default to Choose Team so dropdowns are enabled
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None  # Track which model was selected

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

    # Get currently selected AFC teams
    afc_selected = []
    if manual_mode:
        afc_keys = ["afc_bye", "afc_wild1", "afc_wild2", "afc_wild3", "afc_wild4", "afc_wild5", "afc_wild6"]
        for key in afc_keys:
            val = st.session_state.get(key, "-- Choose a team --")
            if val and val != "-- Choose a team --":
                afc_selected.append(val)

    # Get currently selected NFC teams
    nfc_selected = []
    if manual_mode:
        nfc_keys = ["nfc_bye", "nfc_wild1", "nfc_wild2", "nfc_wild3", "nfc_wild4", "nfc_wild5", "nfc_wild6"]
        for key in nfc_keys:
            val = st.session_state.get(key, "-- Choose a team --")
            if val and val != "-- Choose a team --":
                nfc_selected.append(val)

    # ALL BUTTONS DISABLED FOR NOW
    buttons_disabled = True

    # ---- AFC Wild / Bye ----
    with cols[0]:
        st.markdown("**AFC Wild**")
        if manual_mode:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_bye")
            current = st.session_state.get("afc_bye", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("AFC Bye", available, index=idx, key="afc_bye", label_visibility="collapsed")
        else:
            st.button("AFC Bye", key="afc_bye_btn", disabled=buttons_disabled)
        st.markdown("Bye Week")
        vspace(1)
        if manual_mode:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild1")
            current = st.session_state.get("afc_wild1", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 1", available, index=idx, key="afc_wild1", label_visibility="collapsed")
            
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild2")
            current = st.session_state.get("afc_wild2", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 2", available, index=idx, key="afc_wild2", label_visibility="collapsed")
        else:
            st.button("AFC_Wild1", key="afc_wild1_btn", disabled=buttons_disabled)
            st.button("AFC_Wild2", key="afc_wild2_btn", disabled=buttons_disabled)
        vspace(1)
        if manual_mode:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild3")
            current = st.session_state.get("afc_wild3", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 3", available, index=idx, key="afc_wild3", label_visibility="collapsed")
            
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild4")
            current = st.session_state.get("afc_wild4", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 4", available, index=idx, key="afc_wild4", label_visibility="collapsed")
        else:
            st.button("AFC_Div1", key="afc_div1_btn", disabled=buttons_disabled)
            st.button("AFC_Div2", key="afc_div2_btn", disabled=buttons_disabled)
        vspace(1)
        if manual_mode:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild5")
            current = st.session_state.get("afc_wild5", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 5", available, index=idx, key="afc_wild5", label_visibility="collapsed")
            
            available = get_available_teams(afc_teams_full, afc_selected, "afc_wild6")
            current = st.session_state.get("afc_wild6", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 6", available, index=idx, key="afc_wild6", label_visibility="collapsed")
        else:
            st.button("AFC_Conf1", key="afc_conf1_btn", disabled=buttons_disabled)
            st.button("AFC_Conf2", key="afc_conf2_btn", disabled=buttons_disabled)

    # ---- AFC Divisional ----
    with cols[1]:
        st.markdown("**AFC Divisional**")
        vspace(3)
        st.button("AFC Div Win 1", key="afc_div_win_1", disabled=buttons_disabled)
        st.button("AFC Div Win 2", key="afc_div_win_2", disabled=buttons_disabled)
        vspace(9)
        st.button("AFC Div Win 3", key="afc_div_win_3", disabled=buttons_disabled)
        st.button("AFC Div Win 4", key="afc_div_win_4", disabled=buttons_disabled)

    # ---- AFC Conference ----
    with cols[2]:
        st.markdown("**AFC Conference**")
        vspace(11)
        st.button("AFC Conf Win 1", key="afc_conf_win_1", disabled=buttons_disabled)
        st.button("AFC Conf Win 2", key="afc_conf_win_2", disabled=buttons_disabled)

    # ---- SUPER BOWL (centered vertically) ----
    with cols[3]:
        st.markdown("**Super Bowl**")
        vspace(11)
        st.button("SB Team A", key="sb_team_a", disabled=buttons_disabled)
        st.button("SB Team B", key="sb_team_b", disabled=buttons_disabled)

    # ---- NFC Conference ----
    with cols[4]:
        st.markdown("**NFC Conference**")
        vspace(11)
        st.button("NFC Conf Win 1", key="nfc_conf_win_1", disabled=buttons_disabled)
        st.button("NFC Conf Win 2", key="nfc_conf_win_2", disabled=buttons_disabled)

    # ---- NFC Divisional ----
    with cols[5]:
        st.markdown("**NFC Divisional**")
        vspace(3)
        st.button("NFC Div Win 1", key="nfc_div_win_1", disabled=buttons_disabled)
        st.button("NFC Div Win 2", key="nfc_div_win_2", disabled=buttons_disabled)
        vspace(9)
        st.button("NFC Div Win 3", key="nfc_div_win_3", disabled=buttons_disabled)
        st.button("NFC Div Win 4", key="nfc_div_win_4", disabled=buttons_disabled)

    # ---- NFC Wild / Bye ----
    with cols[6]:
        st.markdown("**NFC Wild**")
        if manual_mode:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_bye")
            current = st.session_state.get("nfc_bye", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("NFC Bye", available, index=idx, key="nfc_bye", label_visibility="collapsed")
        else:
            st.button("NFC Bye", key="nfc_bye_btn", disabled=buttons_disabled)
        st.markdown("Bye Week")
        vspace(1)
        if manual_mode:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild1")
            current = st.session_state.get("nfc_wild1", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 1", available, index=idx, key="nfc_wild1", label_visibility="collapsed")
            
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild2")
            current = st.session_state.get("nfc_wild2", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 2", available, index=idx, key="nfc_wild2", label_visibility="collapsed")
        else:
            st.button("NFC_Wild1", key="nfc_wild1_btn", disabled=buttons_disabled)
            st.button("NFC_Wild2", key="nfc_wild2_btn", disabled=buttons_disabled)
        vspace(1)
        if manual_mode:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild3")
            current = st.session_state.get("nfc_wild3", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 3", available, index=idx, key="nfc_wild3", label_visibility="collapsed")
            
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild4")
            current = st.session_state.get("nfc_wild4", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 4", available, index=idx, key="nfc_wild4", label_visibility="collapsed")
        else:
            st.button("NFC_Div1", key="nfc_div1_btn", disabled=buttons_disabled)
            st.button("NFC_Div2", key="nfc_div2_btn", disabled=buttons_disabled)
        vspace(1)
        if manual_mode:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild5")
            current = st.session_state.get("nfc_wild5", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 5", available, index=idx, key="nfc_wild5", label_visibility="collapsed")
            
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_wild6")
            current = st.session_state.get("nfc_wild6", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("Wild Card 6", available, index=idx, key="nfc_wild6", label_visibility="collapsed")
        else:
            st.button("NFC_Conf1", key="nfc_conf1_btn", disabled=buttons_disabled)
            st.button("NFC_Conf2", key="nfc_conf2_btn", disabled=buttons_disabled)

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
                        st.rerun()
            
            st.markdown("---")
            
            if mode == "AutoML":
                st.markdown("#### AutoML")
                st.write(
                "Fully autonomously predict the entire playoff bracket using two models: "
                "one to determine which teams qualify for the postseason, and another to simulate each round "
                "and predict the Super Bowl champion.")
                vspace(1)
                if st.button("Predict Bracket", use_container_width=True, key="predict_automl"):
                    pass
            else:
                st.markdown("#### Manual Selection")
                st.write(
                "Choose the playoff teams manually using the dropdowns. "
                "This mode uses only the playoff bracket model to simulate each round "
                "from the Wild Card through the Super Bowl and determine the champion.")
                vspace(1)
                if st.button("Predict Bracket", use_container_width=True, key="predict_manual"):
                    pass

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
        from Predictor import train_models
        
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