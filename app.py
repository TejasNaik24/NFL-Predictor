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
st.markdown("")  # Minimal space at top
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
    afc_teams = [
        "-- Choose a team --",
        "Baltimore Ravens", "Buffalo Bills", "Cincinnati Bengals", "Cleveland Browns",
        "Denver Broncos", "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
        "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers", "Miami Dolphins",
        "New England Patriots", "New York Jets", "Pittsburgh Steelers", "Tennessee Titans"
    ]

    nfc_teams = [
        "-- Choose a team --",
        "Arizona Cardinals", "Atlanta Falcons", "Carolina Panthers", "Chicago Bears",
        "Dallas Cowboys", "Detroit Lions", "Green Bay Packers", "Los Angeles Rams",
        "Minnesota Vikings", "New Orleans Saints", "New York Giants", "Philadelphia Eagles",
        "San Francisco 49ers", "Seattle Seahawks", "Tampa Bay Buccaneers", "Washington Commanders"
    ]

    # Top bracket row
    cols = st.columns([1, 1, 1, 0.9, 1, 1, 1])

    # Determine if dropdowns should be disabled
    dropdowns_disabled = (st.session_state.control_mode == "AutoML")

    # ---- AFC Wild / Bye ----
    with cols[0]:
        st.markdown("**AFC Wild**")
        st.selectbox("AFC Bye", afc_teams, key="afc_bye", label_visibility="collapsed", disabled=dropdowns_disabled)
        st.markdown("Bye Week")
        vspace(1) 
        st.selectbox("Wild Card 1", afc_teams, key="afc_wild1", label_visibility="collapsed", disabled=dropdowns_disabled)
        st.selectbox("Wild Card 2", afc_teams, key="afc_wild2", label_visibility="collapsed", disabled=dropdowns_disabled)
        vspace(1)
        st.selectbox("Wild Card 3", afc_teams, key="afc_wild3", label_visibility="collapsed", disabled=dropdowns_disabled)
        st.selectbox("Wild Card 4", afc_teams, key="afc_wild4", label_visibility="collapsed", disabled=dropdowns_disabled)
        vspace(1)
        st.selectbox("Wild Card 5", afc_teams, key="afc_wild5", label_visibility="collapsed", disabled=dropdowns_disabled)
        st.selectbox("Wild Card 6", afc_teams, key="afc_wild6", label_visibility="collapsed", disabled=dropdowns_disabled)

    # ---- AFC Divisional ----
    with cols[1]:
        st.markdown("**AFC Divisional**")
        vspace(3)
        st.button("AFC Div Win 1")
        st.button("AFC Div Win 2")
        vspace(9)
        st.button("AFC Div Win 3")
        st.button("AFC Div Win 4")

    # ---- AFC Conference ----
    with cols[2]:
        st.markdown("**AFC Conference**")
        vspace(11)
        st.button("AFC Conf Win 1")
        st.button("AFC Conf Win 2")

    # ---- SUPER BOWL (centered vertically) ----
    with cols[3]:
        st.markdown("**Super Bowl**")
        vspace(11)
        st.button("SB Team A")
        st.button("SB Team B")

    # ---- NFC Conference ----
    with cols[4]:
        st.markdown("**NFC Conference**")
        vspace(11)
        st.button("NFC Conf Win 1")
        st.button("NFC Conf Win 2")

    # ---- NFC Divisional ----
    with cols[5]:
        st.markdown("**NFC Divisional**")
        vspace(3)
        st.button("NFC Div Win 1")
        st.button("NFC Div Win 2")
        vspace(9)
        st.button("NFC Div Win 3")
        st.button("NFC Div Win 4")

    # ---- NFC Wild / Bye ----
    with cols[6]:
        st.markdown("**NFC Wild**")
        st.selectbox("NFC Bye", nfc_teams, key="nfc_bye", label_visibility="collapsed", disabled=dropdowns_disabled)
        st.markdown("Bye Week")
        vspace(1)
        st.selectbox("Wild Card 1", nfc_teams, key="nfc_wild1", label_visibility="collapsed", disabled=dropdowns_disabled)
        st.selectbox("Wild Card 2", nfc_teams, key="nfc_wild2", label_visibility="collapsed", disabled=dropdowns_disabled)
        vspace(1)
        st.selectbox("Wild Card 3", nfc_teams, key="nfc_wild3", label_visibility="collapsed", disabled=dropdowns_disabled)
        st.selectbox("Wild Card 4", nfc_teams, key="nfc_wild4", label_visibility="collapsed", disabled=dropdowns_disabled)
        vspace(1)
        st.selectbox("Wild Card 5", nfc_teams, key="nfc_wild5", label_visibility="collapsed", disabled=dropdowns_disabled)
        st.selectbox("Wild Card 6", nfc_teams, key="nfc_wild6", label_visibility="collapsed", disabled=dropdowns_disabled)

    # ---- CONTROLS SECTION (wider, centered below) ----
    control_cols = st.columns([2.3, 2.4, 2.3])
    
    with control_cols[1]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown("### Controls")
            
            # Simple toggle between modes - centered with columns
            r1, r2, r3 = st.columns([0.8, 2.4, 0.8])
            with r2:
                mode = st.radio(
                    "Control Mode",
                    ["AutoML", "Choose Team"],
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
                st.markdown("#### AutoML (placeholder)")
                if st.button("Run AutoML"):
                    pass
            else:
                chosen = st.selectbox(
                    "Pick a team",
                    ["--", "AFC_Wild1", "AFC_Conf1", "NFC_Wild1"]
                )

# ---- TRAINING state ----
if st.session_state.flow == "training":
    pass