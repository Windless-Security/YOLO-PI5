import cv2
import numpy as np
from hailo_platform import HEF, HailoStreamInterface, HailoDevice, HailoNetRunner
import time

# Pad naar het gecompileerde YOLOv8-model voor Hailo
HEF_PATH = "/usr/share/rpi-camera-assets/hailo_yolov8_inference.hef"

# Alleen deze objectklassen weergeven (YOLO index afhankelijk van model)
TARGET_CLASSES = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}

# Start Pi Camera 3
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Laad Hailo-model
hef = HEF(HEF_PATH)
with HailoDevice() as device:
    runner = HailoNetRunner(hef, device)
    input_info = runner.get_input_vstream_infos()[0]
    output_info = runner.get_output_vstream_infos()[0]

    with HailoStreamInterface(input_info, output_info, device) as interface:
        print("Detectie gestart - druk op Q om te stoppen")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Verkleinen naar modelinput
            resized = cv2.resize(frame, (640, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = rgb.astype(np.uint8)

            # Inference uitvoeren
            results = interface.infer(input_data)

            # Resultaten verwerken
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

            cv2.imshow("Drone Camera Detectie", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
