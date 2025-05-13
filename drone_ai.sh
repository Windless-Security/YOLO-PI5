#!/bin/bash

HEF_PATH="/usr/share/rpi-camera-assets/yolov5_personface.hef"
JSON_PATH="/usr/share/rpi-camera-assets/hailo_yolov5_personface.json"

if [[ ! -f "$HEF_PATH" || ! -f "$JSON_PATH" ]]; then
  echo "❌ Modelbestanden niet gevonden!"
  exit 1
fi

echo "🚁 Drone AI Detectie gestart..."

rpicam-hello \
  --ai \
  --hef "$HEF_PATH" \
  --post-process-file "$JSON_PATH" \
  --info-text "Drone AI Detection" \
  --width 1280 \
  --height 720 \
  --framerate 30 \
  --fullscreen \
  -t 0
