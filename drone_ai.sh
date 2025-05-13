#!/bin/bash

# 📍 Modelpaden
HEF_PATH="$HOME/models/yolov5_vehicle_person.hef"
JSON_PATH="$HOME/models/hailo_yolov5_vehicle_person.json"

# 🧠 Output logbestand
LOG_PATH="$HOME/models/detections.csv"

# ✅ Check bestanden
if [[ ! -f "$HEF_PATH" || ! -f "$JSON_PATH" ]]; then
  echo "❌ Modelbestanden niet gevonden!"
  exit 1
fi

# ❇️ Start Hailo AI detectie met rpicam
echo "🚁 Drone AI Detectie gestart..."

rpicam-hello \
  --ai \
  --hef "$HEF_PATH" \
  --post-process-file "$JSON_PATH" \
  --info-text "Drone AI Detection" \
  --width 1280 \
  --height 720 \
  --framerate 30 \
  --annotate 1 \
  --annotate-file "$LOG_PATH" \
  --fullscreen \
  -t 0
