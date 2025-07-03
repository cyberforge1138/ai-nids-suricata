import json
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 📄 Load JSON data
filepath = "/home/fulgrim/alerts-only.json"
print(f"📄 Loading JSON: {filepath}")
with open(filepath, "r") as f:
    data = [json.loads(line) for line in f]
print(f"✅ Loaded {len(data)} rows")

# 📦 Convert to DataFrame
df = pd.DataFrame(data)

# 🎯 Filter for alert events
df = df[df["event_type"] == "alert"]
print(f"🎯 Filtered size: {len(df)}")

# 🧹 Extract and flatten fields
def extract_field(field, subfield, new_name):
    df[new_name] = df[field].apply(lambda x: x.get(subfield) if isinstance(x, dict) else np.nan)

fields = [
    ("flow", "pkts_toserver", "flow.pkts_toserver"),
    ("flow", "pkts_toclient", "flow.pkts_toclient"),
    ("flow", "bytes_toserver", "flow.bytes_toserver"),
    ("flow", "bytes_toclient", "flow.bytes_toclient"),
    ("alert", "signature", "alert_signature"),
    ("alert", "category", "alert_category"),
    ("alert", "severity", "alert_severity"),
    ("http", "hostname", "http_host"),
    ("http", "url", "http_url"),
]

for field, sub, name in fields:
    extract_field(field, sub, name)

df["src_ip"] = df["src_ip"]
df["dest_ip"] = df["dest_ip"]
df["src_port"] = df["src_port"]
df["dest_port"] = df["dest_port"]
df["proto"] = df["proto"]
df["app_proto"] = df["app_proto"]

# 🔍 Drop rows with missing critical fields
keep_cols = [
    "src_ip", "dest_ip", "src_port", "dest_port", "proto", "app_proto",
    "flow.pkts_toserver", "flow.pkts_toclient", "flow.bytes_toserver", "flow.bytes_toclient",
    "alert_signature", "alert_category", "alert_severity"
]
df = df[keep_cols].dropna()
print(f"🧹 Cleaned size: {df.shape}")

# ⚠️ Downsample for memory
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)
    print("⚠️  Downsampled to 10000 rows")

# 🎯 Encode target labels
original_encoder = LabelEncoder()
df["original_label"] = original_encoder.fit_transform(df["alert_signature"])

# Filter classes with at least 2 samples
counts = df["original_label"].value_counts()
valid_labels = counts[counts >= 2].index
df = df[df["original_label"].isin(valid_labels)]
print(f"🧪 Retained {len(df)} rows after filtering by class count")

# ✅ Reindex labels to ensure consecutive integers
unique_labels = sorted(df["original_label"].unique())
label_remap = {old: new for new, old in enumerate(unique_labels)}
df["label"] = df["original_label"].map(label_remap)

# 🔧 Prepare features
feature_cols = [
    "src_ip", "dest_ip", "src_port", "dest_port", "proto", "app_proto",
    "flow.pkts_toserver", "flow.pkts_toclient", "flow.bytes_toserver", "flow.bytes_toclient",
    "alert_category", "alert_severity"
]
X = df[feature_cols].copy()
y = df["label"]

# Encode categorical columns
for col in ["src_ip", "dest_ip", "proto", "app_proto", "alert_category"]:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 🚂 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 🌲 Train Random Forest
print("🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("✅ Random Forest model trained.")

# ⚡ Train XGBoost
print("⚡ Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
print("✅ XGBoost model trained.")

# 💾 Save everything
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb, "models/xgb_classifier.pkl")
joblib.dump(original_encoder, "models/label_encoder.pkl")

with open("models/feature_columns.json", "w") as f:
    json.dump(feature_cols, f)

print("✅ All models and encoders saved.")
