# app.py - Software-only Pathogen Detector (Safe vs Pathogen Present) with evaluation view

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

st.set_page_config(page_title="Portable Waterborne Pathogen Detector", layout="wide")
st.title("Portable Waterborne Pathogen Detector — Software Version")

# Load model
@st.cache_resource
def load_model(path="model.joblib"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Could not load model.joblib: {e}")
        return None

model = load_model()

# Load dataset automatically
try:
    df = pd.read_csv("pathogen_dataset.csv")
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    st.subheader("Sample Dataset (last 10 rows)")
    st.dataframe(df.tail(10))
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    df = None

# Sidebar inputs
st.sidebar.header("Enter Water Parameters")
ph = st.sidebar.slider("pH level", 5.0, 9.0, 7.0, 0.1)
turbidity_v = st.sidebar.slider("Turbidity (V)", 0.0, 5.0, 2.5, 0.1)
temperature_c = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 25.0, 0.5)

if st.sidebar.button("Check Water Safety"):
    if model is not None:
        # Prepare input
        X = pd.DataFrame([{
            "ph": ph,
            "turbidity_v": turbidity_v,
            "temperature_c": temperature_c,
            "particles": 0.0
        }])

        # Predict
        pred = model.predict(X)[0]
        prob = np.max(model.predict_proba(X)[0])

        if pred == 0:
            st.success(f"✅ Water is SAFE ({prob:.2f} confidence)")
        else:
            st.error(f"⚠️ Pathogen Present ({prob:.2f} confidence)")

# --- Evaluation Section ---
st.subheader("📊 Model Evaluation (on dataset)")

if model is not None and df is not None:
    X_all = df[['ph', 'turbidity_v', 'temperature_c', 'particles']]
    y_true = df['binary_label']
    y_pred = model.predict(X_all)

    acc = accuracy_score(y_true, y_pred)
    st.write(f"**Model Accuracy on dataset:** {acc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "Pathogen"],
                yticklabels=["Safe", "Pathogen"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=["Safe", "Pathogen"], output_dict=True)
    st.write("### Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format(precision=2))
