# ui_app.py
import streamlit as st
import requests
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="RTP Fraud Detection", layout="centered")
st.title("🔍 Real-Time Payment Fraud Detection")

# --- Transaction Input Form ---
with st.form("txn_form"):
    st.subheader("📝 Enter Transaction Details")
    customer_id = st.text_input("Customer ID")
    txn_type = st.selectbox("Transaction Type", [
        "P2P Transfer", "Loan Disbursement",
        "Member-to-Member Transfer", "Member-to-External Transfer"
    ])
    txn_amount = st.number_input("Transaction Amount (₹)", min_value=0.0, format="%.2f")
    device_type = st.selectbox("Device Type", ["Mobile", "Web", "ATM", "POS"])
    payment_method = st.selectbox("Payment Method", [
        "Instant Bank Transfer", "Digital Wallet",
        "Credit Card", "Debit Card"
    ])
    submitted = st.form_submit_button("🚀 Predict Fraud")

# --- Predict Fraud ---
if submitted:
    if not customer_id:
        st.warning("Please enter a valid Customer ID.")
    else:
        # Prepare request payload
        payload = {
            "Customer_ID": customer_id.strip(),
            "Transaction_Type": txn_type,
            "Transaction_Amount": txn_amount,
            "Device_Type": device_type,
            "Payment_Method": payment_method
        }

        st.info("Sending data to fraud detection API...")

        try:
            # 🔁 Use local API or replace with deployed endpoint
            api_url = "http://127.0.0.1:8000/predict"

            response = requests.post(api_url, json=payload)

            if response.status_code == 200:
                result = response.json()

                # ✅ Prediction Summary
                st.success(f"🧠 Prediction: **{result['fraud_type']}** (Label: {result['prediction']})")
                st.info(f"📊 Actual (from DB): **{result['actual_fraud_type']}** (Label: {result['actual_label']})")

                # 📝 Save log to Excel
                log_entry = {
                    "Customer_ID": customer_id,
                    "Transaction_Type": txn_type,
                    "Transaction_Amount": txn_amount,
                    "Prediction": result['fraud_type'],
                    "Actual": result['actual_fraud_type']
                }

                try:
                    logs_df = pd.read_excel("fraud_logs.xlsx")
                    logs_df = pd.concat([logs_df, pd.DataFrame([log_entry])], ignore_index=True)
                except FileNotFoundError:
                    logs_df = pd.DataFrame([log_entry])

                logs_df.to_excel("fraud_logs.xlsx", index=False)
                st.success("📁 Prediction logged successfully.")

            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("🔌 Could not connect to FastAPI backend. Please ensure it is running.")
        except Exception as e:
            st.error(f"⚠️ Unexpected Error: {str(e)}")
