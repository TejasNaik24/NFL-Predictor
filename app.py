import streamlit as st
import os
from datetime import datetime

# ---- config ----
icon_path = os.path.join("static", "nfl.png")
st.set_page_config(page_title="NFL Predictor", page_icon=icon_path, layout="wide")

# ---- helpers ----
def vspace(n: int):
    for _ in range(n):
        st.write("")

# ---- Session state ----
if "clicked" not in st.session_state:
    st.session_state.clicked = False
if "flow" not in st.session_state:
    st.session_state.flow = None  # None | "predicting" | "training"
if "control_mode" not in st.session_state:
    st.session_state.control_mode = "Choose Team"  # Default to Choose Team so dropdowns are enabled

# ---- Centered Title (ALWAYS VISIBLE) ----
col1, col2, col3 = st.columns([1, 1, 0.5])
with col2:
    st.title("NFL Predictor")

# ---- image placeholder (landing) ----
image_path = os.path.join(os.getcwd(), "static", "nfl.png")
if not st.session_state.clicked:
    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col2:
        st.image(image_path, caption="NFL Logo")
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
            model_choice = st.selectbox(
                "Available models",
                [
                    "-- Choose a model --",
                    f"Model 1 ({start_year}–{latest_year})",
                    f"Model 2 ({start_year}–{latest_year})",
                    f"Model 3 (Legacy {start_year}–{latest_year})"
                ],
                index=0
            )

            if model_choice != "-- Choose a model --":
                b_l, b_c, b_r = st.columns([1, 1, 1])
                with b_c:
                    if st.button("Predict Selected Model"):
                        st.session_state.flow = "predicting"
                        st.rerun()

        with tab_train:
            st.markdown("### Train New Model")
            st.write("This will train a new model using the **most recent completed NFL seasons**.")

            s0, s1 = st.columns(2)
            s0.markdown(f"**Start:** {start_year}")
            s1.markdown(f"**End:** {latest_year}")

            t_l, t_c, t_r = st.columns([1, 1, 1])
            with t_c:
                if st.button("Train Model Now"):
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

    # ---- AFC Wild / Bye ----
    with cols[0]:
        st.markdown("**AFC Wild**")
        if manual_mode:
            available = get_available_teams(afc_teams_full, afc_selected, "afc_bye")
            current = st.session_state.get("afc_bye", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("AFC Bye", available, index=idx, key="afc_bye", label_visibility="collapsed")
        else:
            @st.dialog("AFC Bye - Analysis")
            def show_afc_bye():
                st.write("Placeholder")
            
            if st.button("AFC Bye", key="afc_bye_btn"):
                show_afc_bye()
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
            @st.dialog("AFC Wild 1 - Analysis")
            def show_afc_wild1():
                st.write("Placeholder")
            
            if st.button("AFC_Wild1", key="afc_wild1_btn"):
                show_afc_wild1()
            
            @st.dialog("AFC Wild 2 - Analysis")
            def show_afc_wild2():
                st.write("Placeholder")
            
            if st.button("AFC_Wild2", key="afc_wild2_btn"):
                show_afc_wild2()
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
            @st.dialog("AFC Div 1 - Analysis")
            def show_afc_div1():
                st.write("Placeholder")
            
            if st.button("AFC_Div1", key="afc_div1_btn"):
                show_afc_div1()
            
            @st.dialog("AFC Div 2 - Analysis")
            def show_afc_div2():
                st.write("Placeholder")
            
            if st.button("AFC_Div2", key="afc_div2_btn"):
                show_afc_div2()
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
            @st.dialog("AFC Conf 1 - Analysis")
            def show_afc_conf1():
                st.write("Placeholder")
            
            if st.button("AFC_Conf1", key="afc_conf1_btn"):
                show_afc_conf1()
            
            @st.dialog("AFC Conf 2 - Analysis")
            def show_afc_conf2():
                st.write("Placeholder")
            
            if st.button("AFC_Conf2", key="afc_conf2_btn"):
                show_afc_conf2()

    # ---- AFC Divisional ----
    with cols[1]:
        st.markdown("**AFC Divisional**")
        vspace(3)
        
        @st.dialog("AFC Div Win 1 - Analysis")
        def show_afc_div_win_1():
            st.write("Placeholder")
        
        if st.button("AFC Div Win 1", key="afc_div_win_1"):
            show_afc_div_win_1()
        
        @st.dialog("AFC Div Win 2 - Analysis")
        def show_afc_div_win_2():
            st.write("Placeholder")
        
        if st.button("AFC Div Win 2", key="afc_div_win_2"):
            show_afc_div_win_2()
        
        vspace(9)
        
        @st.dialog("AFC Div Win 3 - Analysis")
        def show_afc_div_win_3():
            st.write("Placeholder")
        
        if st.button("AFC Div Win 3", key="afc_div_win_3"):
            show_afc_div_win_3()
        
        @st.dialog("AFC Div Win 4 - Analysis")
        def show_afc_div_win_4():
            st.write("Placeholder")
        
        if st.button("AFC Div Win 4", key="afc_div_win_4"):
            show_afc_div_win_4()

    # ---- AFC Conference ----
    with cols[2]:
        st.markdown("**AFC Conference**")
        vspace(11)
        
        @st.dialog("AFC Conf Win 1 - Analysis")
        def show_afc_conf_win_1():
            st.write("Placeholder")
        
        if st.button("AFC Conf Win 1", key="afc_conf_win_1"):
            show_afc_conf_win_1()
        
        @st.dialog("AFC Conf Win 2 - Analysis")
        def show_afc_conf_win_2():
            st.write("Placeholder")
        
        if st.button("AFC Conf Win 2", key="afc_conf_win_2"):
            show_afc_conf_win_2()

    # ---- SUPER BOWL (centered vertically) ----
    with cols[3]:
        st.markdown("**Super Bowl**")
        vspace(11)
        
        @st.dialog("SB Team A - Analysis")
        def show_sb_team_a():
            st.write("Placeholder")
        
        if st.button("SB Team A", key="sb_team_a"):
            show_sb_team_a()
        
        @st.dialog("SB Team B - Analysis")
        def show_sb_team_b():
            st.write("Placeholder")
        
        if st.button("SB Team B", key="sb_team_b"):
            show_sb_team_b()

    # ---- NFC Conference ----
    with cols[4]:
        st.markdown("**NFC Conference**")
        vspace(11)
        
        @st.dialog("NFC Conf Win 1 - Analysis")
        def show_nfc_conf_win_1():
            st.write("Placeholder")
        
        if st.button("NFC Conf Win 1", key="nfc_conf_win_1"):
            show_nfc_conf_win_1()
        
        @st.dialog("NFC Conf Win 2 - Analysis")
        def show_nfc_conf_win_2():
            st.write("Placeholder")
        
        if st.button("NFC Conf Win 2", key="nfc_conf_win_2"):
            show_nfc_conf_win_2()

    # ---- NFC Divisional ----
    with cols[5]:
        st.markdown("**NFC Divisional**")
        vspace(3)
        
        @st.dialog("NFC Div Win 1 - Analysis")
        def show_nfc_div_win_1():
            st.write("Placeholder")
        
        if st.button("NFC Div Win 1", key="nfc_div_win_1"):
            show_nfc_div_win_1()
        
        @st.dialog("NFC Div Win 2 - Analysis")
        def show_nfc_div_win_2():
            st.write("Placeholder")
        
        if st.button("NFC Div Win 2", key="nfc_div_win_2"):
            show_nfc_div_win_2()
        
        vspace(9)
        
        @st.dialog("NFC Div Win 3 - Analysis")
        def show_nfc_div_win_3():
            st.write("Placeholder")
        
        if st.button("NFC Div Win 3", key="nfc_div_win_3"):
            show_nfc_div_win_3()
        
        @st.dialog("NFC Div Win 4 - Analysis")
        def show_nfc_div_win_4():
            st.write("Placeholder")
        
        if st.button("NFC Div Win 4", key="nfc_div_win_4"):
            show_nfc_div_win_4()

    # ---- NFC Wild / Bye ----
    with cols[6]:
        st.markdown("**NFC Wild**")
        if manual_mode:
            available = get_available_teams(nfc_teams_full, nfc_selected, "nfc_bye")
            current = st.session_state.get("nfc_bye", "-- Choose a team --")
            idx = available.index(current) if current in available else 0
            st.selectbox("NFC Bye", available, index=idx, key="nfc_bye", label_visibility="collapsed")
        else:
            @st.dialog("NFC Bye - Analysis")
            def show_nfc_bye():
                st.write("Placeholder")
            
            if st.button("NFC Bye", key="nfc_bye_btn"):
                show_nfc_bye()
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
            @st.dialog("NFC Wild 1 - Analysis")
            def show_nfc_wild1():
                st.write("Placeholder")
            
            if st.button("NFC_Wild1", key="nfc_wild1_btn"):
                show_nfc_wild1()
            
            @st.dialog("NFC Wild 2 - Analysis")
            def show_nfc_wild2():
                st.write("Placeholder")
            
            if st.button("NFC_Wild2", key="nfc_wild2_btn"):
                show_nfc_wild2()
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
            @st.dialog("NFC Div 1 - Analysis")
            def show_nfc_div1():
                st.write("Placeholder")
            
            if st.button("NFC_Div1", key="nfc_div1_btn"):
                show_nfc_div1()
            
            @st.dialog("NFC Div 2 - Analysis")
            def show_nfc_div2():
                st.write("Placeholder")
            
            if st.button("NFC_Div2", key="nfc_div2_btn"):
                show_nfc_div2()
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
            @st.dialog("NFC Conf 1 - Analysis")
            def show_nfc_conf1():
                st.write("Placeholder")
            
            if st.button("NFC_Conf1", key="nfc_conf1_btn"):
                show_nfc_conf1()
            
            @st.dialog("NFC Conf 2 - Analysis")
            def show_nfc_conf2():
                st.write("Placeholder")
            
            if st.button("NFC_Conf2", key="nfc_conf2_btn"):
                show_nfc_conf2()

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
            
            if mode == "AutoML":
                st.markdown("#### Manual Prediction")
                st.write("Manually select teams for each round of the playoff bracket using the dropdowns on the left and right.")
                vspace(1)
                if st.button("Predict Bracket", use_container_width=True, key="predict_automl"):
                    pass
            else:
                st.markdown("#### Manual Selection")
                st.write("Choose teams from the dropdowns to manually build your playoff bracket and see how it plays out.")
                vspace(1)
                if st.button("Predict Bracket", use_container_width=True, key="predict_manual"):
                    pass

# ---- TRAINING state ----
if st.session_state.flow == "training":
    pass