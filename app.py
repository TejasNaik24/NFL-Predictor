import streamlit as st
import os

icon_path = os.path.join("static", "nfl.png")

st.set_page_config(
    page_title="NFL Predictor",
    page_icon=icon_path,
)

# ---- Centered Title ----
col1, col2, col3 = st.columns([1, 2, 0.2])
with col2:
    st.title("NFL Predictor")

# ---- Centered Image ----
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("static/nfl.png", width=220)

# ---- Centered Button ----
col1, col2, col3 = st.columns([1, 0.3, 1])
with col2:
    st.button("Predict")
