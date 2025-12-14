import streamlit as st
import os
from datetime import datetime

# ---- config (wide layout so the UI uses full screen) ----
icon_path = os.path.join("static", "nfl.png")
st.set_page_config(page_title="NFL Predictor", page_icon=icon_path, layout="wide")

# ---- Centered Title ----
col1, col2, col3 = st.columns([1, 1, 0.5])  # larger center column
with col2:
    st.title("NFL Predictor")

# ---- Session state ----
if "clicked" not in st.session_state:
    st.session_state.clicked = False

# ---- image placeholder (NN-style) ----
image_path = os.path.join(os.getcwd(), "static", "nfl.png")
col1, col2, col3 = st.columns([1.5, 1, 1])
with col2:
    image_placeholder = st.empty()
    if not st.session_state.clicked:
        image_placeholder.image(image_path, caption="NFL Logo", use_column_width=False)
        col_l, col_c, col_r = st.columns([1, 3, 3])
        if col_c.button("Predict", key="predict_button"):
            st.session_state.clicked = True
            st.rerun()
    else:
        image_placeholder.empty()

# ---- Main UI (after clicking Predict) ----
if st.session_state.clicked:

    # ---- NFL-season logic: use last COMPLETED season ----
    today = datetime.now()
    latest_year = today.year - 1     # last fully completed NFL season
    start_year = latest_year - 9     # fixed 10-season window (inclusive)

    # ---- Center contents (no boxed container) ----
    outer_l, outer_c, outer_r = st.columns([1.2, 2, 1.2])
    with outer_c:
        # Header (centered)
        st.subheader("Model Selection & Training")

        # ---- Tabs: Use pretrained OR Train new (full-width per tab) ----
        tab_use, tab_train = st.tabs(["Use Pretrained Model", "Train New Model"])

        # ----- TAB 1: Use Pretrained Model -----
        with tab_use:
            st.markdown("### Choose Model")
            model_options = [
                "-- Choose a model --",
                f"Model 1 ({start_year}–{latest_year})",
                f"Model 2 ({start_year}–{latest_year})",
                f"Model 3 (Legacy {start_year}–{latest_year})"
            ]
            model_choice = st.selectbox("Available models", model_options, index=0)
            st.caption("Select a pre-trained model (placeholders).")

            if model_choice != "-- Choose a model --":
                b_l, b_c, b_r = st.columns([1, 1, 1])
                with b_c:
                    if st.button("Predict Selected Model", key="predict_model_button"):
                        st.success(f"Using {model_choice}")
                        # hook prediction logic here

        # ----- TAB 2: Train New Model -----
        with tab_train:
            st.markdown("### Train New Model")
            st.write("This will train a new model using the **most recent completed NFL seasons**.")
            st.markdown("#### Training Window (fixed 10 seasons)")

            # show start and end plainly in one line
            s0, s1 = st.columns(2)
            s0.markdown(f"**Start:** {start_year}")
            s1.markdown(f"**End:** {latest_year}")

            # green summary line (pure Streamlit)
            st.success(f"{start_year} → {latest_year}")

            st.write("")  # small gap
            t_l, t_c, t_r = st.columns([1, 1, 1])
            with t_c:
                if st.button("Train Model Now", key="train_now_button"):
                    st.info(f"(Demo) Would train on seasons {start_year}–{latest_year}")
                    # hook training logic here
