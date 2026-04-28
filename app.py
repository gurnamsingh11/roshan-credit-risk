import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load model and feature names
model = joblib.load("credit_risk_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("💳 Credit Risk Prediction")
st.write("Enter the applicant details below to predict loan approval status.")

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=100.0, value=5.0)
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

with col2:
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
    loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.2)
    cb_person_default_on_file = st.selectbox("Historical Default", ["N", "Y"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)

# --- Encode categorical variables (same order as training) ---
# Replicate the LabelEncoder logic used during training
cat_encodings = {
    "person_home_ownership": ["MORTGAGE", "OTHER", "OWN", "RENT"],
    "loan_intent": ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}


def encode_value(col_name, value):
    return cat_encodings[col_name].index(value)


if st.button("🔍 Predict", type="primary"):
    # Build feature array in the same order as training
    features = np.array([[
        person_age,
        person_income,
        encode_value("person_home_ownership", person_home_ownership),
        person_emp_length,
        encode_value("loan_intent", loan_intent),
        encode_value("loan_grade", loan_grade),
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        encode_value("cb_person_default_on_file", cb_person_default_on_file),
        cb_person_cred_hist_length,
    ]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    st.divider()
    if prediction == 0:
        st.success(f"✅ **Low Risk** — Loan likely to be repaid (Confidence: {proba[0]*100:.1f}%)")
    else:
        st.error(f"⚠️ **High Risk** — Loan may default (Confidence: {proba[1]*100:.1f}%)")
