import cv2
import numpy as np
from hailo import HailoDevice, HEF, InferModel, VStreamsParams

# Pad naar het voorgecompileerde YOLOv8-model
HEF_PATH = "/usr/share/rpi-camera-assets/hailo_yolov8_inference.hef"

# Alleen deze klassen tonen (controleer class IDs in jouw model!)
TARGET_CLASSES = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}

# Initialiseer camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Start Hailo device
with HailoDevice() as device:
    hef = HEF(HEF_PATH)
    vstreams_params = VStreamsParams.from_hef(hef, key="yolov8")  # Let op: sleutel moet kloppen met je model
    with InferModel(device, hef, vstreams_params) as model:
        print("🚀 Detectie gestart - druk op Q om te stoppen.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (640, 640))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Run inference
            results = model.infer(rgb_frame)

            # Resultaten verwerken — afhankelijk van jouw postprocessing output
            for detection in results[0]:
                class_id, conf, x1, y1, x2, y2 = detection
                if int(class_id) in TARGET_CLASSES and conf > 0.4:
                    h, w, _ = frame.shape
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)
                    label = f"{TARGET_CLASSES[int(class_id)]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Drone AI Detectie", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
