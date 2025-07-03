#script for parsing suricata data

import json
import pandas as pd

log_path = '/home/fulgrim/lab-logs/eve.json'

alerts = []

with open(log_path, 'r') as f:
    for line in f:
        try:
                entry = json.loads(line)
                if entry.get("event_type") == "alert":
                    alerts.append(entry)
        except json.JSONDecodeError:
                continue

df = pd.DataFrame(alerts)

print(df.head())

df.to_csv('alerts.csv', index=False)
