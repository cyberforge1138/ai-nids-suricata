import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# Load models and encoders
rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgb_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Example input alert
sample_alert = {
    "src_ip": "192.168.1.2",
    "dest_ip": "10.0.0.1",
    "src_port": 443,
    "dest_port": 51515,
    "proto": "TCP",
    "app_proto": "http",
    "http_host": "example.com",
    "http_url": "/index.html",
    "timestamp": "2023-06-20T10:15:00.000000Z"
}

# Convert to DataFrame
df = pd.DataFrame([sample_alert])

# Encode IPs and timestamp as numeric hashes
for col in ["src_ip", "dest_ip", "timestamp"]:
    df[col] = df[col].apply(lambda x: hash(x) % (10**8))

# Fill missing values
df = df.fillna("")

# One-hot encode categorical fields
df = pd.get_dummies(df)

# Align with feature columns
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0
df = df[feature_columns]

# Run inference
rf_pred = label_encoder.inverse_transform(rf_model.predict(df))[0]
xgb_pred = label_encoder.inverse_transform(xgb_model.predict(df))[0]

print(f"🟩 Random Forest Prediction: {rf_pred}")
print(f"🟦 XGBoost Prediction: {xgb_pred}")
