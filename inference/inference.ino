#include <Wire.h>
#include "inference_engine.h"

const float MEANS[6] = {-227.591659f, 1016.954224f, 9489.657227f, 8.386434f, -29.274366f, -283.500031f};
const float STDS[6]  = {12463.923828f, 6947.102051f, 12368.375977f, 11182.551758f, 14551.875000f, 12506.858594f};

const char* LABELS[] = {"idle", "shake_x", "flick_up", "twist"};
const int MPU_ADDR = 0x68;
const int SAMPLES  = 50;
const int LED      = 2; // built-in LED on most ESP32 DevKits

void readIMU(float* ax, float* ay, float* az,
             float* gx, float* gy, float* gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  *ax = (int16_t)(Wire.read() << 8 | Wire.read());
  *ay = (int16_t)(Wire.read() << 8 | Wire.read());
  *az = (int16_t)(Wire.read() << 8 | Wire.read());
  Wire.read(); Wire.read();
  *gx = (int16_t)(Wire.read() << 8 | Wire.read());
  *gy = (int16_t)(Wire.read() << 8 | Wire.read());
  *gz = (int16_t)(Wire.read() << 8 | Wire.read());
}

void ledIdle()   { digitalWrite(LED, LOW); }

void ledShakeX() {
  for (int i = 0; i < 6; i++) {
    digitalWrite(LED, HIGH); delay(60);
    digitalWrite(LED, LOW);  delay(60);
  }
}

void ledFlickUp() {
  digitalWrite(LED, HIGH); delay(400);
  digitalWrite(LED, LOW);
}

void ledTwist() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED, HIGH); delay(200);
    digitalWrite(LED, LOW);  delay(200);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  pinMode(LED, OUTPUT);
  digitalWrite(LED, LOW);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Serial.println("Running — move the sensor!");
}

void loop() {
  float input[300];
  float raw[6];
  int idx = 0;

  // Collect one window
  for (int i = 0; i < SAMPLES; i++) {
    readIMU(&raw[0], &raw[1], &raw[2], &raw[3], &raw[4], &raw[5]);
    for (int j = 0; j < 6; j++) {
      input[idx++] = (raw[j] - MEANS[j]) / STDS[j];
    }
    delay(50);
  }

  // Run inference
  float scores[4];
  predict(input, scores);

  int best = 0;
  for (int i = 1; i < 4; i++)
    if (scores[i] > scores[best]) best = i;

  // Only act if confidence is high enough
  if (scores[best] > 0.7f) {
    Serial.print(LABELS[best]);
    Serial.print(" (");
    Serial.print(scores[best] * 100, 1);
    Serial.println("%)");

    switch (best) {
      case 0: ledIdle();   break;
      case 1: ledShakeX(); break;
      case 2: ledFlickUp(); break;
      case 3: ledTwist();  break;
    }
  } else {
    Serial.println("uncertain");
  }
}