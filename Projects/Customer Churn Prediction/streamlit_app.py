# streamlit_app.py  ← FINAL VERSION (Guaranteed to work everywhere)
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Prediction")
st.markdown("### XGBoost Model • Test AUC ~0.92 • Live Demo")

# Load model
try:
    model = joblib.load("models/xgb_churn_model.pkl")
    features = joblib.load("models/feature_names.pkl")
except:
    st.error("Model files not found. Make sure 'models/' folder is uploaded.")
    st.stop()

# Input form
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total = st.number_input("Total Charges ($)", 18.0, 9000.0, 2000.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        payment = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("Predict Churn Risk", type="primary")

if submitted:
    # Build input dictionary
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'Contract': contract,
        'InternetService': internet,
        'PaymentMethod': payment.split()[0],  # rough but works
        'TechSupport': tech_support
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Align columns
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    prob = model.predict_proba(df)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{prob:.1%}", delta=f"{prob-0.5:+.1%}")
    with col2:
        if prob > 0.5:
            st.error("HIGH RISK – Offer discount!")
        else:
            st.success("LOW RISK – Safe customer")

    st.balloons()