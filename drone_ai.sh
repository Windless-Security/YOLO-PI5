#!/bin/bash

HEF_PATH="/opt/hailo/hailo_app_zoo/models/yolov5_personface/yolov5_personface.hef"
JSON_PATH="/usr/share/rpi-camera-assets/hailo_yolov5_personface.json"

if [[ ! -f "$HEF_PATH" || ! -f "$JSON_PATH" ]]; then
  echo "❌ Vereiste bestanden niet gevonden"
  exit 1
fi

echo "🚁 Drone AI Detectie gestart..."

rpicam-hello \
  --ai \
  --hef "$HEF_PATH" \
  --post-process-file "$JSON_PATH" \
  --info-text "Drone AI" \
  --width 1280 \
  --height 720 \
  --framerate 30 \
  -t 0
