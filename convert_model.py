import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# ── 1. Load scaler and data (needed for quantization calibration) ──────────────
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv("gesture_data.csv")
X = df.drop("label", axis=1).values.astype(np.float32)
X = scaler.transform(X)
X = X.reshape(-1, 50, 6)

# Representative dataset — TFLite uses this to calibrate int8 ranges
def representative_dataset():
    for i in range(len(X)):
        yield [X[i:i+1]]

# ── 2. Load trained model ──────────────────────────────────────────────────────
model = tf.keras.models.load_model("gesture_model.keras")

# ── 3. Convert to TFLite with full int8 quantization ──────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"TFLite model size: {len(tflite_model) / 1024:.1f} KB")

# ── 4. Verify quantized model accuracy ────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]["quantization"]

correct = 0
y = df["label"].values

for i in range(len(X)):
    sample = X[i:i+1].astype(np.float32)
    # Quantize input to int8
    sample_int8 = (sample / input_scale + input_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]["index"], sample_int8)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    pred = np.argmax(output)
    if pred == y[i]:
        correct += 1

print(f"Quantized model accuracy: {correct/len(X)*100:.1f}%")

# ── 5. Print scaler values — you'll hardcode these into the ESP32 sketch ───────
print(f"\nScaler means:  {scaler.mean_.tolist()}")
print(f"Scaler stds:   {scaler.scale_.tolist()}")