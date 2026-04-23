import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("gesture_model.keras")

# Extract weights
conv1_w, conv1_b = model.layers[0].get_weights()  # (3,6,16), (16,)
conv2_w, conv2_b = model.layers[2].get_weights()  # (3,16,32), (32,)
dense1_w, dense1_b = model.layers[5].get_weights()  # (384,32), (32,)
dense2_w, dense2_b = model.layers[7].get_weights()  # (32,4), (4,)

def array_to_c(name, arr, dtype="float"):
    flat = arr.flatten().tolist()
    vals = ", ".join(f"{v:.6f}f" for v in flat)
    return f"const {dtype} {name}[{len(flat)}] = {{{vals}}};\n"

header = """#pragma once
#include <math.h>

// Auto-generated inference engine — no libraries needed

"""

header += array_to_c("CONV1_W", conv1_w)  # [3*6*16]
header += array_to_c("CONV1_B", conv1_b)  # [16]
header += array_to_c("CONV2_W", conv2_w)  # [3*16*32]
header += array_to_c("CONV2_B", conv2_b)  # [32]
header += array_to_c("DENSE1_W", dense1_w) # [384*32]
header += array_to_c("DENSE1_B", dense1_b) # [32]
header += array_to_c("DENSE2_W", dense2_w) # [32*4]
header += array_to_c("DENSE2_B", dense2_b) # [4]

header += """
// ReLU
inline float relu(float x) { return x > 0 ? x : 0; }

// 1D Conv: input (T, C_in), kernel (K, C_in, C_out), output (T, C_out), padding=same
void conv1d(const float* input, int T, int C_in,
            const float* kernel, int K, int C_out,
            const float* bias, float* output) {
  int pad = K / 2;
  for (int t = 0; t < T; t++) {
    for (int co = 0; co < C_out; co++) {
      float sum = bias[co];
      for (int k = 0; k < K; k++) {
        int ti = t - pad + k;
        if (ti < 0 || ti >= T) continue;
        for (int ci = 0; ci < C_in; ci++) {
          sum += input[ti * C_in + ci] * kernel[k * C_in * C_out + ci * C_out + co];
        }
      }
      output[t * C_out + co] = relu(sum);
    }
  }
}

// MaxPool1D: pool_size=2, stride=2
void maxpool1d(const float* input, int T, int C, float* output, int& T_out) {
  T_out = T / 2;
  for (int t = 0; t < T_out; t++) {
    for (int c = 0; c < C; c++) {
      float a = input[(t*2)   * C + c];
      float b = input[(t*2+1) * C + c];
      output[t * C + c] = a > b ? a : b;
    }
  }
}

// Dense layer
void dense(const float* input, int in_size,
           const float* weights, const float* bias,
           float* output, int out_size, bool apply_relu) {
  for (int o = 0; o < out_size; o++) {
    float sum = bias[o];
    for (int i = 0; i < in_size; i++) {
      sum += input[i] * weights[i * out_size + o];
    }
    output[o] = apply_relu ? relu(sum) : sum;
  }
}

// Softmax
void softmax(float* x, int n) {
  float max_val = x[0];
  for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
  for (int i = 0; i < n; i++) x[i] /= sum;
}

// Full inference: input is float[300] (50 timesteps x 6 axes, normalized)
void predict(const float* input, float* scores) {
  // Conv1 + Pool1: (50,6) -> (50,16) -> (25,16)
  static float c1[50 * 16];
  static float p1[25 * 16];
  int t1;
  conv1d(input, 50, 6, CONV1_W, 3, 16, CONV1_B, c1);
  maxpool1d(c1, 50, 16, p1, t1); // t1=25

  // Conv2 + Pool2: (25,16) -> (25,32) -> (12,32)
  static float c2[25 * 32];
  static float p2[12 * 32];
  int t2;
  conv1d(p1, 25, 16, CONV2_W, 3, 32, CONV2_B, c2);
  maxpool1d(c2, 25, 32, p2, t2); // t2=12, 12*32=384

  // Dense1: 384 -> 32
  static float d1[32];
  dense(p2, 384, DENSE1_W, DENSE1_B, d1, 32, true);

  // Dense2: 32 -> 4
  dense(d1, 32, DENSE2_W, DENSE2_B, scores, 4, false);

  softmax(scores, 4);
}
"""

with open("inference_engine.h", "w") as f:
    f.write(header)

print("Done. inference_engine.h generated.")
print(f"File size: {len(header)/1024:.1f} KB")