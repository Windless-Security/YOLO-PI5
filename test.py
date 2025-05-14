import subprocess
import csv
import datetime
import os

LOG_FILE = "hailo_detections_log.csv"

# Zorg dat logbestand bestaat met headers
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])

# Start de pipeline en vang stdout op
process = subprocess.Popen(
    [
        "rpicam-hello",
        "-t", "0",
        "--post-process-file", "/usr/share/rpi-camera-assets/hailo_yolov8_inference.json"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print("⏺️ Pipeline gestart — loggen naar hailo_detections_log.csv")

for line in process.stdout:
    # Zoek naar detectieregels (voorbeeldformaat afhankelijk van JSON)
    if "Detected object" in line and "class" in line:
        # Voorbeeldregel: Detected object: class=person conf=0.91 x1=0.25 y1=0.30 x2=0.45 y2=0.65
        parts = line.strip().split()
        data = {
            "class": parts[2].split("=")[1],
            "confidence": float(parts[3].split("=")[1]),
            "x1": float(parts[4].split("=")[1]),
            "y1": float(parts[5].split("=")[1]),
            "x2": float(parts[6].split("=")[1]),
            "y2": float(parts[7].split("=")[1]),
        }

        timestamp = datetime.datetime.now().isoformat()

        # Log naar CSV
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                data["class"],
                data["confidence"],
                data["x1"],
                data["y1"],
                data["x2"],
                data["y2"],
            ])

        print(f"➡️ {data['class']} ({data['confidence']:.2f}) logged.")
