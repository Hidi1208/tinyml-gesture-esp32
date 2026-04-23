Good. Here's your README — save this as `README.md` in your project folder, then commit and push it.

```markdown
# TinyML Gesture Recognition on ESP32

A real-time gesture classification system using a 1D CNN trained on IMU time-series data, deployed on an ESP32 microcontroller with a custom C++ inference engine and zero external Arduino libraries.

## Results

| Metric | Value |
|---|---|
| Gestures classified | 4 (idle, shake X, flick up, twist) |
| Training samples | 600 (150 per class) |
| Test accuracy | 99.2% |
| Real-world accuracy | 100% |
| Model size (TFLite int8) | 23.6 KB |
| Inference engine | Custom C++ — no TFLite Micro library |

## System Architecture

```
┌─────────────────────────────────────────────┐
│                PC (Python)                  │
│                                             │
│  MPU-6050 → Serial → collect_data.py        │
│                           │                 │
│                    gesture_data.csv         │
│                           │                 │
│                    train_model.py           │
│                    (1D CNN, TensorFlow)     │
│                           │                 │
│                    convert_model.py         │
│                    (TFLite + int8 quant)    │
│                           │                 │
│                    export_engine.py         │
│                           │                 │
│                   inference_engine.h        │
└───────────────────────────┬─────────────────┘
                            │
                    flash to ESP32
                            │
┌───────────────────────────▼─────────────────┐
│              ESP32 + MPU-6050               │
│                                             │
│  IMU → normalize → inference_engine.h       │
│                           │                 │
│                    gesture label            │
│                    + LED feedback           │
└─────────────────────────────────────────────┘
```

## Hardware

| Component | Purpose |
|---|---|
| ESP32 DevKit v1 | Main MCU |
| MPU-6050 | 6-axis IMU (accelerometer + gyroscope) |

### Wiring

| MPU-6050 | ESP32 |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO 21 |
| SCL | GPIO 22 |
| AD0 | GND |

## Software Stack

- Python 3.11, TensorFlow 2.x, scikit-learn, NumPy, pyserial
- Arduino IDE, ESP32 Arduino Core

## Model Architecture

```
Input: (50 timesteps × 6 axes) = 300 features
→ Conv1D(16 filters, kernel=3, ReLU) + MaxPool
→ Conv1D(32 filters, kernel=3, ReLU) + MaxPool
→ Dense(32, ReLU) + Dropout(0.3)
→ Dense(4, Softmax)
```

Trained with Adam optimizer, sparse categorical crossentropy loss, early stopping.
Post-training int8 quantization applied via TFLite converter with representative dataset calibration.

## Inference Engine

Rather than using TFLite Micro, a custom C++ inference engine (`inference_engine.h`) was generated directly from the trained Keras model weights. It implements Conv1D, MaxPool1D, Dense, ReLU, and Softmax from scratch — resulting in a fully self-contained sketch with no external Arduino library dependencies.

## Reproduction

### 1. Install dependencies

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib pyserial
```

### 2. Collect gesture data

```bash
python collect_data.py
```

Flash `data_collection/data_collection.ino` to ESP32 first. Follow prompts to record 150 samples per gesture.

### 3. Train the model

```bash
python train_model.py
```

### 4. Convert to TFLite + quantize

```bash
python convert_model.py
```

### 5. Get scaler values

```bash
python get_scaler_values.py
```

### 6. Export inference engine

```bash
python export_engine.py
```

### 7. Flash to ESP32

Copy `inference_engine.h` into the `inference/` sketch folder. Update `MEANS` and `STDS` in `inference.ino` with values from step 5. Upload via Arduino IDE.

## Gestures

| Label | Gesture | Primary signal |
|---|---|---|
| idle | No movement | All axes near baseline |
| shake_x | Rapid back-and-forth | ax oscillates |
| flick_up | Quick upward snap | az spike |
| twist | Rotation around Z | gz spike |

## Project Context

Built as part of a TinyML portfolio project targeting embedded systems and robotics master's applications. Demonstrates the full pipeline from raw sensor data to on-device inference — data collection, model training, quantization, and bare-metal MCU deployment.
```



