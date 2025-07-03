import pandas as pd

df = pd.read_csv('alerts.csv', nrows=5)
print(df.columns.tolist())

print(df['alert'].head(10))

print([col for col in df.columns if 'flag' in col.lower()])

