#!/bin/bash

# ✅ Pad naar je model en postprocess json
MODEL_PATH="/usr/share/hailo-models/yolov5s_personface_h8l.hef"
POSTPROCESS_JSON="/usr/share/rpi-camera-assets/hailo_yolov5_personface.json"

# ✅ Alleen 'person' label tonen (optioneel aanpassen in json)
# Zorg dat de json deze mapping bevat:
# { "label_map": { "0": "person" } }

# ✅ Start de live AI-demo
echo "🚀 Start YOLOv5s + Hailo detectie (alleen 'person')"
rpicam-hello --ai \
  --ai-model "$MODEL_PATH" \
  --post-process-file "$POSTPROCESS_JSON" \
  -t 0
