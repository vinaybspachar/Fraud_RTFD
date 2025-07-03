# backend/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

from database import load_data_from_snowflake

# Load and prepare data
query = "SELECT * FROM CREDITUNION_RTP_FRAUD_DATASET_W"
df = load_data_from_snowflake(query)
df.columns = [col.lower() for col in df.columns]

# Encode categorical columns
categorical_columns = ['transaction_type', 'location', 'device_type', 'payment_method']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Time-based features
df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'], errors='coerce')
df['hour'] = df['transaction_datetime'].dt.hour
df['day'] = df['transaction_datetime'].dt.day
df['weekday'] = df['transaction_datetime'].dt.weekday

# Add is_rtp feature
if "rtp" in encoders["transaction_type"].classes_:
    rtp_index = list(encoders["transaction_type"].classes_).index("rtp")
    df["is_rtp"] = (df["transaction_type"] == rtp_index).astype(int)
else:
    df["is_rtp"] = 0

# Assign fraud types
def assign_fraud_type(row):
    if row["fraud_label"] == 0:
        return "None"
    elif row["new_beneficiary_added"] == 1 and row["transaction_amount"] > 5000:
        return "APP"
    elif row["failed_login_attempts"] > 3 and row["unusual_location"] == 1:
        return "ATO"
    else:
        return "None"

df["fraud_type"] = df.apply(assign_fraud_type, axis=1)
fraud_type_map = {"None": 0, "APP": 1, "ATO": 2}
df["fraud_type_code"] = df["fraud_type"].map(fraud_type_map)

# Feature and target
features = [
    "transaction_type", "transaction_amount", "location", "device_type", "payment_method",
    "failed_login_attempts", "new_beneficiary_added", "unusual_location",
    "time_gap_between_transactions", "transaction_frequency_per_day",
    "hour", "day", "weekday", "is_rtp"
]

X = df[features]
y = df["fraud_type_code"]

# SMOTE + model training
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    objective="multi:softmax",
    num_class=3,
    random_state=42
)
model.fit(X_resampled, y_resampled)

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model Accuracy:", model.score(X_test, y_test))

# Save artifacts
joblib.dump(model, "xgb_model_multiclass.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(fraud_type_map, "fraud_type_map.pkl")
print("✅ Model and encoders saved.")
