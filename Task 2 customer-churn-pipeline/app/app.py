# app/app.py
import sys
from pathlib import Path
sys.path.append(str(Path("..").resolve()))

import streamlit as st
import pandas as pd
from src.predict import load_pipeline, predict_churn
from src.config import PIPELINE_PATH

# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("📊 Customer Churn Predictor")
st.markdown("Predict whether a telecom customer will churn.")
st.divider()

# ─────────────────────────────────────────
# Load Pipeline
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return load_pipeline(PIPELINE_PATH)

pipeline = load_model()

# ─────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender",
                ["Male", "Female"])
    senior = st.selectbox("Senior Citizen",
                [0, 1])
    partner = st.selectbox("Partner",
                ["Yes", "No"])
    dependents = st.selectbox("Dependents",
                ["Yes", "No"])
    tenure = st.slider("Tenure (months)",
                0, 72, 12)
    phone = st.selectbox("Phone Service",
                ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines",
                ["Yes", "No",
                 "No phone service"])
    internet = st.selectbox("Internet Service",
                ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security",
                ["Yes", "No",
                 "No internet service"])
    backup = st.selectbox("Online Backup",
                ["Yes", "No",
                 "No internet service"])

with col2:
    protection = st.selectbox("Device Protection",
                    ["Yes", "No",
                     "No internet service"])
    tech = st.selectbox("Tech Support",
                    ["Yes", "No",
                     "No internet service"])
    tv = st.selectbox("Streaming TV",
                    ["Yes", "No",
                     "No internet service"])
    movies = st.selectbox("Streaming Movies",
                    ["Yes", "No",
                     "No internet service"])
    contract = st.selectbox("Contract",
                    ["Month-to-month",
                     "One year",
                     "Two year"])
    paperless = st.selectbox("Paperless Billing",
                    ["Yes", "No"])
    payment = st.selectbox("Payment Method",
                    ["Electronic check",
                     "Mailed check",
                     "Bank transfer (automatic)",
                     "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges ($)",
                    min_value=0.0,
                    max_value=200.0,
                    value=65.0)
    total = st.number_input("Total Charges ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=500.0)

st.divider()

# ─────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────
if st.button("🔮 Predict Churn", use_container_width=True):

    # Build input dataframe
    customer = pd.DataFrame([{
        'gender'          : gender,
        'SeniorCitizen'   : senior,
        'Partner'         : partner,
        'Dependents'      : dependents,
        'tenure'          : tenure,
        'PhoneService'    : phone,
        'MultipleLines'   : multiple,
        'InternetService' : internet,
        'OnlineSecurity'  : security,
        'OnlineBackup'    : backup,
        'DeviceProtection': protection,
        'TechSupport'     : tech,
        'StreamingTV'     : tv,
        'StreamingMovies' : movies,
        'Contract'        : contract,
        'PaperlessBilling': paperless,
        'PaymentMethod'   : payment,
        'MonthlyCharges'  : monthly,
        'TotalCharges'    : str(total)
    }])

    # Get prediction
    result = predict_churn(customer, pipeline)
    label = result['Churn_Label'].values[0]
    prob  = result['Churn_Probability'].values[0]

    st.divider()
    st.subheader("📈 Prediction Result")

    if prob >= 0.5:
        st.error(f"### {label}")
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.warning(
            "⚠️ This customer is at high risk of churning. "
            "Consider a retention offer."
        )
    else:
        st.success(f"### {label}")
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.info("✅ This customer is likely to stay.")