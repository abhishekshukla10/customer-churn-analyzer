print ("hi")
import streamlit as st
import pandas as pd
import joblib
print ("import complete")


# -----------------------------
# Configuration
# -----------------------------
THRESHOLD = 0.3

# -----------------------------
# Load Model + Features
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model_pipeline.pkl")
    features = joblib.load("models/model_features.pkl")
    return model, features

model, model_features = load_artifacts()

st.title("📊 Customer Churn Risk Predictor")

st.write("Enter customer details below:")

# -----------------------------
# User Inputs
# -----------------------------

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])


# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Churn Risk"):

    # Create input dictionary
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "TechSupport": tech_support,
        "InternetService": internet_service,
        "Dependents": dependents
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Align columns with training features
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    # Predict probability
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader(f"Churn Probability: {probability:.2f}")

    # -----------------------------
    # Risk Segmentation
    # -----------------------------
    if probability >= 0.6:
        risk = "High"
    elif probability >= THRESHOLD:
        risk = "Medium"
    elif probability >= 0.1:
        risk = "Low"
    else:
        risk = "Stable"

    st.subheader(f"Risk Level: {risk}")

      # -----------------------------
    # Business Action Layer
    # -----------------------------
    if contract in ["One year", "Two year"]:
        action = "Contract protected - Monitor only"

    elif probability >= 0.6:
        action = "Immediate retention call + discount offer"

    elif probability >= THRESHOLD:
        action = "Targeted retention email"

    elif probability >= 0.1:
        action = "Monitor behavior"

    else:
        action = "No action required"

    st.subheader(f"Recommended Action: {action}")


