# backend/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from database import load_data_from_snowflake

# Load saved assets
model = joblib.load("xgb_model_multiclass.pkl")
encoders = joblib.load("encoders.pkl")
fraud_type_map = joblib.load("fraud_type_map.pkl")
fraud_type_map_inv = {v: k for k, v in fraud_type_map.items()}

# Load and preprocess dataset
query = "SELECT * FROM CREDITUNION_RTP_FRAUD_DATASET_W"
dataset = load_data_from_snowflake(query)
dataset.columns = [col.lower() for col in dataset.columns]

if "transaction_datetime" in dataset.columns:
    dataset["transaction_datetime"] = pd.to_datetime(dataset["transaction_datetime"], errors='coerce')
    dataset["hour"] = dataset["transaction_datetime"].dt.hour
    dataset["day"] = dataset["transaction_datetime"].dt.day
    dataset["weekday"] = dataset["transaction_datetime"].dt.weekday
else:
    raise ValueError("Missing 'transaction_datetime' column")

# API
app = FastAPI()

class Transaction(BaseModel):
    Customer_ID: str
    Transaction_Type: str
    Transaction_Amount: float
    Device_Type: str
    Payment_Method: str

@app.post("/predict")
def predict(txn: Transaction):
    try:
        data = txn.dict()
        cust_id = data["Customer_ID"].strip()

        cust_data = dataset[dataset["customer_id"] == cust_id]
        if cust_data.empty:
            raise HTTPException(status_code=404, detail="Customer ID not found")

        latest = cust_data.sort_values(by="transaction_datetime", ascending=False).iloc[0]

        input_data = {
            "Transaction_Type": data["Transaction_Type"],
            "Transaction_Amount": data["Transaction_Amount"],
            "Location": latest["location"],
            "Device_Type": data["Device_Type"],
            "Payment_Method": data["Payment_Method"],
            "Failed_Login_Attempts": latest["failed_login_attempts"],
            "New_Beneficiary_Added": latest["new_beneficiary_added"],
            "Unusual_Location": latest["unusual_location"],
            "Time_Gap_Between_Transactions": latest["time_gap_between_transactions"],
            "Transaction_Frequency_Per_Day": latest["transaction_frequency_per_day"],
            "Hour": latest["hour"],
            "Day": latest["day"],
            "Weekday": latest["weekday"],
            "is_rtp": 1 if data["Transaction_Type"].lower() == "rtp" else 0
        }

        actual_label = int(latest["fraud_label"])

        # Rule-based fraud detection
        if input_data["New_Beneficiary_Added"] == 1 and input_data["Transaction_Amount"] > 50000:
            return {
                "prediction": 1,
                "fraud_type": "APP (Rule-Based)",
                "actual_label": actual_label,
                "actual_fraud_type": fraud_type_map_inv.get(actual_label, "Unknown")
            }

        if input_data["Failed_Login_Attempts"] > 3 and input_data["Unusual_Location"] == 1:
            return {
                "prediction": 2,
                "fraud_type": "ATO (Rule-Based)",
                "actual_label": actual_label,
                "actual_fraud_type": fraud_type_map_inv.get(actual_label, "Unknown")
            }

        # Encode categorical features
        for field in ["Transaction_Type", "Location", "Device_Type", "Payment_Method"]:
            encoder = encoders.get(field.lower())
            if encoder:
                input_data[field] = int(encoder.transform([input_data[field]])[0])
            else:
                raise HTTPException(status_code=500, detail=f"Missing encoder for field: {field}")

        input_array = np.array([input_data[col] for col in input_data]).reshape(1, -1)

        prediction = int(model.predict(input_array)[0])

        return {
            "prediction": prediction,
            "fraud_type": fraud_type_map_inv.get(prediction, "Unknown") + " (ML-Based)",
            "actual_label": actual_label,
            "actual_fraud_type": fraud_type_map_inv.get(actual_label, "Unknown")
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
