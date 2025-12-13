import streamlit as st
import os

# ---- config ----
icon_path = os.path.join("static", "nfl.png")
st.set_page_config(page_title="NFL Predictor", page_icon=icon_path)

# ---- Centered Title ----
col1, col2, col3 = st.columns([1, 2, 0.5])
with col2:
    st.title("NFL Predictor")

# ---- Session state (predict click) ----
if "clicked" not in st.session_state:
    st.session_state.clicked = False

# ---- image placeholder (NN-style) ----
image_path = os.path.join(os.getcwd(), "static", "nfl.png")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    image_placeholder = st.empty()

    if not st.session_state.clicked:
        image_placeholder.image(image_path, caption="NFL Logo")
        col_l, col_c, col_r = st.columns([1, 2, 1])
        if col_c.button("Predict", key="predict_button"):
            st.session_state.clicked = True
            st.rerun()
    else:
        image_placeholder.empty()

if st.session_state.clicked:
    latest_year = 2024
    earliest_start = 2000

    slider_min = earliest_start
    slider_max = latest_year - 10

    # ---- Boxed section ----
    with st.container(border=True):
        # create three columns and put the header in the center column
        header_left, header_center, header_right = st.columns([1.5, 3, 1])
        with header_center:
            st.subheader("Model Selection & Training")
        st.divider()

        col_left, col_right = st.columns(2)

        # ---- LEFT: Choose Model ----
        with col_left:
            st.markdown("### Choose Model")
            model_options = [
                "-- Choose a model --",
                "Model 1 (2015–2024)",
                "Model 2 (2010–2024)",
                "Model 3 (2000–2024)"
            ]
            model_choice = st.selectbox(
                "Available models",
                model_options,
                index=0  # default to placeholder
            )
            st.caption("Pre-trained models (placeholders for now).")

            # Show Predict button only if a real model is selected
            if model_choice != "-- Choose a model --":
                col_l2, col_c2, col_r2 = st.columns([1, 3, 1])
                with col_c2:
                    if st.button("Predict Selected Model", key="predict_model_button"):
                        st.success(f"You selected: {model_choice}")
                        # Add prediction logic here

        # ---- RIGHT: Train New Model ----
        with col_right:
            st.markdown("### Train New Model")

            start_year = st.slider(
                "Start year",
                min_value=slider_min,
                max_value=slider_max,
                value=latest_year - 5,
                step=1
            )

            end_year = min(start_year + 10, latest_year)

            st.write(f"Will train up to year: **{end_year}**")
            st.write(
                "Select start year (training range will be **START → latest year**):"
            )

            cols = st.columns(2)
            cols[0].markdown(f"**Start:** {start_year}")
            cols[1].markdown(f"**End:** {end_year}")

            st.success(f"Training data range: {start_year} → {end_year}")

            st.button("Train Model", key="train_model_button")
