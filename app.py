import streamlit as st
import os

icon_path = os.path.join("static", "nfl.png")

st.set_page_config(
    page_title="NFL Predictor",
    page_icon=icon_path,
)

# ---- Centered Title ----
col1, col2, col3 = st.columns([1, 2, 0.5])
with col2:
    st.title("NFL Predictor")

# Session state
if "clicked" not in st.session_state:
    st.session_state.clicked = False

image_path = os.path.join(os.getcwd(), "static", "nfl.png")

# ---- Centered Image + Button container ----
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
