import pandas as pd
import numpy as np

df = pd.read_csv("gesture_data.csv")

print(f"Total samples: {len(df)}")
print(f"\nSamples per gesture:")
print(df["label"].value_counts().sort_index())
print(f"\nAny missing values: {df.isnull().sum().sum()}")
print(f"\nShape: {df.shape}")