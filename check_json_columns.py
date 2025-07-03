import pandas as pd
import json

# Load JSON file
json_file = "small_alerts.json"
try:
    print(f"Loading {json_file}...")
    df = pd.read_json(json_file, lines=True)
except ValueError:
    print("Failed with line-delimited JSON. Trying as standard JSON list...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

# Show dataset shape and columns
print(f"\nLoaded {len(df)} records.")
print("\nAvailable columns:")
print(df.columns.tolist())

# Show the first few records as dictionaries
print("\nFirst 3 records (as dicts):")
print(df.head(3).to_dict(orient='records'))
