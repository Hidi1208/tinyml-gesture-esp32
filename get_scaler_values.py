import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("gesture_data.csv")
X = df.drop("label", axis=1).values.astype(np.float32)

# Reshape to (samples*timesteps, 6) — treat every timestep as a row
X_reshaped = X.reshape(-1, 6)  # (30000, 6)

means = X_reshaped.mean(axis=0)
stds  = X_reshaped.std(axis=0)

print("MEANS:", means.tolist())
print("STDS: ", stds.tolist())