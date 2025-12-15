import streamlit as st
import os
from datetime import datetime
from pathlib import Path  # <-- added

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
    st.session_state.flow = None
if "control_mode" not in st.session_state:
    st.session_state.control_mode = "Choose Teams"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# ---- Centered Title (ALWAYS VISIBLE) ----
col1, col2, col3 = st.columns([1, 1, 0.5])
with col2:
    st.title("NFL Predictor")

# ---- image placeholder (landing) ----
image_path = Path("static/nfl.png")  # <-- ONLY change
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

        tab_use, tab_train = st.tabs(["Use Pretrained Model", "Train New Model"])

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
                        st.session_state.selected_model = model_choice
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
                    st.session_state.selected_model = "Newly Trained Model"
                    st.session_state.flow = "training"
                    st.rerun()

# ---- TRAINING state ----
if st.session_state.flow == "training":
    pass
