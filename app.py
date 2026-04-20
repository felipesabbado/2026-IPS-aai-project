import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

# ── Page configuration ───────────────────────────────────────
st.set_page_config(
    page_title="",
    page_icon="",
    layout="wide"
)

# ── Load model and scaler (cached — runs once per session) ───
@st.cache_resource
def load_model():
    return joblib.load(r'models/best_model_xgb_classification_pipeline.pkl')

pipeline = load_model()
