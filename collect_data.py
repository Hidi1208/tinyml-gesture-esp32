import serial
import csv
import time
import os

PORT = "COM8"        # Change this to your ESP32's port
BAUD = 115200
SAMPLES = 50
GESTURE_LABELS = {0: "idle", 1: "shake_x", 2: "flick_up", 3: "twist"}

def collect_gesture(ser, label, gesture_name, count):
    input(f"\n>>> Gesture: {gesture_name} | Sample {count+1} | Get ready, then press Enter...")
    
    # Send trigger to ESP32
    ser.write(b'g')
    
    # Wait for START
    while True:
        line = ser.readline().decode().strip()
        if line == "START":
            break
    
    print("    Recording... do the gesture now!")
    
    rows = []
    for _ in range(SAMPLES):
        line = ser.readline().decode().strip()
        if line == "END":
            break
        values = list(map(int, line.split(",")))
        rows.append(values)
    
    # Wait for END if not already received
    while True:
        line = ser.readline().decode().strip()
        if line == "END":
            break
    
    if len(rows) == SAMPLES:
        flat = [val for sample in rows for val in sample]
        flat.append(label)
        return flat
    else:
        print("    Incomplete sample, skipping.")
        return None

def main():
    output_file = "gesture_data.csv"
    file_exists = os.path.exists(output_file)
    
    ser = serial.Serial(PORT, BAUD, timeout=3)
    time.sleep(2)  # Let ESP32 boot
    ser.flushInput()
    
    print("=== Gesture Data Collector ===")
    print(f"Saving to: {output_file}")
    
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            # Header: ax0,ay0,...,gz49,label
            header = []
            for i in range(SAMPLES):
                for axis in ["ax","ay","az","gx","gy","gz"]:
                    header.append(f"{axis}{i}")
            header.append("label")
            writer.writerow(header)
        
        for label, gesture_name in GESTURE_LABELS.items():
            samples_needed = 150
            collected = 0
            
            print(f"\n{'='*40}")
            print(f"Collecting: {gesture_name.upper()} (need {samples_needed} samples)")
            print(f"{'='*40}")
            
            while collected < samples_needed:
                row = collect_gesture(ser, label, gesture_name, collected)
                if row:
                    writer.writerow(row)
                    f.flush()
                    collected += 1
                    print(f"    Saved. ({collected}/{samples_needed})")
    
    print("\nDone! gesture_data.csv is ready.")
    ser.close()

if __name__ == "__main__":
    main()