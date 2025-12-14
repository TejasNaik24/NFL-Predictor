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

    cols = st.columns([1, 1, 1, 0.9, 1, 1, 1])

    # ---- AFC Wild / Bye ----
    with cols[0]:
        st.markdown("**AFC Wild**")
        st.button("AFC Bye")
        st.markdown("Bye Week")
        vspace(1)
        st.button("AFC_Wild1")
        st.button("AFC_Wild2")
        vspace(1)
        st.button("AFC_Div1")
        st.button("AFC_Div2")
        vspace(1)
        st.button("AFC_Conf1")
        st.button("AFC_Conf2")

    # ---- AFC Divisional ----
    with cols[1]:
        st.markdown("**AFC Divisional**")
        vspace(2)
        st.button("AFC Div Win 1")
        st.button("AFC Div Win 2")
        vspace(8)
        st.button("AFC Div Win 3")
        st.button("AFC Div Win 4")

    # ---- AFC Conference ----
    with cols[2]:
        st.markdown("**AFC Conference**")
        vspace(9)
        st.button("AFC Conf Win 1")
        st.button("AFC Conf Win 2")

    # ---- SUPER BOWL (centered vertically) ----
    with cols[3]:
        st.markdown("**Super Bowl**")
        vspace(9)
        st.button("SB Team A")
        st.button("SB Team B")

    # ---- NFC Conference ----
    with cols[4]:
        st.markdown("**NFC Conference**")
        vspace(9)
        st.button("NFC Conf Win 1")
        st.button("NFC Conf Win 2")

    # ---- NFC Divisional ----
    with cols[5]:
        st.markdown("**NFC Divisional**")
        vspace(2)
        st.button("NFC Div Win 1")
        st.button("NFC Div Win 2")
        vspace(8)
        st.button("NFC Div Win 3")
        st.button("NFC Div Win 4")

    # ---- NFC Wild / Bye ----
    with cols[6]:
        st.markdown("**NFC Wild**")
        st.button("NFC Bye")
        st.markdown("Bye Week")
        vspace(1)
        st.button("NFC_Wild1")
        st.button("NFC_Wild2")
        vspace(1)
        st.button("NFC_Div1")
        st.button("NFC_Div2")
        vspace(1)
        st.button("NFC_Conf1")
        st.button("NFC_Conf2")

    # ---- Bottom control box ----
    outer_l, outer_c, outer_r = st.columns([1.5, 2, 1.5])
    with outer_c:
        st.markdown("---")
        st.markdown("### Controls")
        tabs = st.tabs(["AutoML", "Choose Team"])

        with tabs[0]:
            st.markdown("#### AutoML (placeholder)")
            st.button("Run AutoML")

        with tabs[1]:
            chosen = st.selectbox(
                "Pick a team",
                ["--", "AFC_Wild1", "AFC_Conf1", "NFC_Wild1"]
            )
            st.button("Set Favorite Team")

    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        if st.button("Back to Model Selection"):
            st.session_state.flow = None
            st.rerun()

# ---- TRAINING state ----
if st.session_state.flow == "training":
    pass
