import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("📊 Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 50.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

if st.button("Predict Churn"):
    sample_input = {
        "gender": gender,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract
    }
    sample_df = pd.DataFrame([sample_input])
    sample_df = pd.get_dummies(sample_df, drop_first=True)
    sample_df = sample_df.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(sample_df)[0]
    prob = model.predict_proba(sample_df)[0][1]

    st.success(f"Prediction: {'⚠️ Churn' if pred==1 else '✅ Not Churn'} (Probability: {prob:.2f})")
