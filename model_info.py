import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("gesture_model.keras")

for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        print(f"Layer {i}: {layer.name}, weights shapes: {[w.shape for w in weights]}")