import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Fraud Detection Dashboard 🕵️")
st.subheader("Instructions")
st.markdown(
    """
1. Go to **Batch Inference** to run predictions on an entire dataset.
2. Go to **Single Inference** to quickly test the model with a single transaction.
"""
)
