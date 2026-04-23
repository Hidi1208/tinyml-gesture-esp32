import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("gesture_data.csv")
X = df.drop("label", axis=1).values.astype(np.float32)  # (600, 300)
y = df["label"].values

# ── 2. Normalize ──────────────────────────────────────────────────────────────
# Scale each feature to zero mean, unit variance.
# IMPORTANT: fit only on training data, apply to all.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Save scaler — you'll need these values to normalize on the ESP32 too
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Reshape for 1D CNN: (samples, timesteps, features)
X_train = X_train.reshape(-1, 50, 6)
X_test  = X_test.reshape(-1, 50, 6)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 3. Build model ────────────────────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(50, 6)),

    layers.Conv1D(16, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPooling1D(pool_size=2),

    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")
])

model.summary()

# ── 4. Train ──────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.1f}%")

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["idle","shake_x","flick_up","twist"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"], label="train")
ax1.plot(history.history["val_accuracy"], label="val")
ax1.set_title("Accuracy"); ax1.legend()
ax2.plot(history.history["loss"], label="train")
ax2.plot(history.history["val_loss"], label="val")
ax2.set_title("Loss"); ax2.legend()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ── 6. Save model ─────────────────────────────────────────────────────────────
model.save("gesture_model.keras")
print("Model saved.")