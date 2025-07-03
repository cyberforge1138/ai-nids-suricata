import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# File paths
DATA_FILE = "/home/fulgrim/alerts-only.json"
MODEL_DIR = "models"
FEATURE_COLUMNS_FILE = f"{MODEL_DIR}/feature_columns.json"
LABEL_ENCODER_FILE = f"{MODEL_DIR}/label_encoder.pkl"
RF_MODEL_FILE = f"{MODEL_DIR}/random_forest.pkl"
XGB_MODEL_FILE = f"{MODEL_DIR}/xgb_classifier.pkl"

# Load trained components
print("📄 Loading models and encoders...")
rf_model = joblib.load(RF_MODEL_FILE)
xgb_model = joblib.load(XGB_MODEL_FILE)
with open(FEATURE_COLUMNS_FILE, "r") as f:
    feature_columns = json.load(f)
label_encoder = joblib.load(LABEL_ENCODER_FILE)

# Load data
print(f"📄 Loading {DATA_FILE}...")
with open(DATA_FILE, "r") as f:
    raw_data = [json.loads(line) for line in f]

df = pd.DataFrame(raw_data)
print(f"✅ Loaded {len(df):,} rows")

# Keep only alert events
df = df[df["event_type"] == "alert"]
print(f"🎯 Filtered size: {len(df):,}")

# Extract fields from nested objects
flow_fields = ["pkts_toserver", "pkts_toclient", "bytes_toserver", "bytes_toclient"]
for field in flow_fields:
    df[f"flow.{field}"] = df["flow"].apply(lambda x: x.get(field) if isinstance(x, dict) else np.nan)

http_fields = ["hostname", "url"]
for field in http_fields:
    df[f"http_{field}"] = df["http"].apply(lambda x: x.get(field) if isinstance(x, dict) else "")

# Alert fields that were missing
df["alert_signature"] = df["alert"].apply(lambda x: x.get("signature") if isinstance(x, dict) else "")
df["alert_category"] = df["alert"].apply(lambda x: x.get("category") if isinstance(x, dict) else "")
df["alert_severity"] = df["alert"].apply(lambda x: x.get("severity") if isinstance(x, dict) else -1)

# Other top-level fields
df["src_ip"] = df["src_ip"]
df["dest_ip"] = df["dest_ip"]
df["src_port"] = df["src_port"]
df["dest_port"] = df["dest_port"]
df["proto"] = df["proto"]
df["app_proto"] = df["app_proto"]

# Drop rows without labels
df = df[df["alert_signature"] != ""]
print(f"🧹 Cleaned size: {df.shape}")

# Filter only known classes
df = df[df["alert_signature"].isin(label_encoder.classes_)]
print(f"🧪 Retained {len(df):,} rows after filtering unknown classes")

# Encode target
df["label"] = label_encoder.transform(df["alert_signature"])

# Encode categorical variables
for col in ["src_ip", "dest_ip", "proto", "app_proto", "http_hostname", "http_url", "alert_category"]:
    df[col] = df[col].astype("category").cat.codes

# Ensure correct feature order and presence
X = df[feature_columns]
y = df["label"]

# Evaluation: Random Forest
print("\n🌲 Random Forest Classifier Evaluation:")
rf_pred = rf_model.predict(X)
print(confusion_matrix(y, rf_pred))
print(classification_report(y, rf_pred, target_names=label_encoder.classes_))

# Evaluation: XGBoost
print("⚡ XGBoost Classifier Evaluation:")
xgb_pred = xgb_model.predict(X)
print(confusion_matrix(y, xgb_pred))
print(classification_report(y, xgb_pred, target_names=label_encoder.classes_))
