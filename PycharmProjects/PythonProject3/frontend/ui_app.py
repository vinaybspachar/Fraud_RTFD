# ui_app.py

import streamlit as st
import requests

st.set_page_config(page_title="RTP Fraud Detection", page_icon="⚠️")

st.title("🚨 Real-Time Fraud Detection (APP & ATO + RTP Drain)")
st.markdown("This tool detects potential **fraudulent transactions** using both Rule-Based and ML-Based models. SHAP explainability is included.")

# Input Form
with st.form("fraud_form"):
    customer_id = st.text_input("Customer ID", value="CUST8084")
    transaction_type = st.selectbox("Transaction Type", [
        "Member-to-Member Transfer", "Member-to-External Transfer",
        "Loan Repayment", "Loan Disbursement", "RTP"
    ])
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
    device_type = st.selectbox("Device Type", ["Mobile", "Web", "ATM", "POS"])

    submitted = st.form_submit_button("Detect Fraud")

if submitted:
    # Prepare request payload
    payload = {
        "Customer_ID": customer_id,
        "Transaction_Type": transaction_type,
        "Transaction_Amount": transaction_amount,
        "Device_Type": device_type
    }

    try:
        # API request to FastAPI backend
        response = requests.post("http://localhost:8000/predict", json=payload)
        result = response.json()

        if response.status_code != 200:
            st.error(f"❌ Error: {result['detail']}")
        else:
            # Display results
            st.success("✅ Prediction Successful!")

            col1, col2 = st.columns(2)
            col1.metric("Rule-Based Result", result["rule_based_result"])
            col2.metric("ML Prediction", result["ml_prediction"])

            st.subheader("🔍 SHAP Explanation (Top 3 Features)")
            st.json(result["top_features"])

            if result["rule_based_result"] in ["APP Fraud", "ATO + RTP Drain"] or "APP Fraud" in result["ml_prediction"] or "ATO + RTP Drain" in result["ml_prediction"]:
                st.warning("🚨 Email alert has been sent to staff.")
            else:
                st.info("✅ No fraud detected. Email alert not triggered.")

    except Exception as e:
        st.error(f"⚠️ Internal Error: {str(e)}")
