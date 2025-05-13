import cv2
import numpy as np
import time
from picamera2 import Picamera2
from hailo_platform import (
    HEF, VDevice, ConfigureParams, FormatType,
    InputVStreamParams, OutputVStreamParams,
    InferVStreams, HailoStreamInterface
)

# 📦 Pad naar je Hailo YOLOv8 model (.hef)
HEF_PATH = "/usr/share/hailo-models/yolov8_personface_h8l.hef"  # pas aan naar jouw modelnaam

# 📷 Start Pi-camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()
time.sleep(0.5)

# 🧠 Start AI-model
hef = HEF(HEF_PATH)

with VDevice() as device:
    configure_params = ConfigureParams.create_from_hef(hef, HailoStreamInterface.PCIe)
    network_group = device.configure(hef, configure_params)[0]
    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()

    input_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    input_name = input_infos[0].name
    model_shape = input_infos[0].shape  # [1, 3, 640, 640]
    _, _, in_h, in_w = model_shape

    with network_group.activate():
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            print("🚀 YOLOv8 AI demo gestart. Druk op 'q' om te stoppen.")

            while True:
                frame = picam2.capture_array()

                # 📐 Preprocess
                resized = cv2.resize(frame, (in_w, in_h))
                normalized = resized.astype(np.float32) / 255.0
                chw = np.transpose(normalized, (2, 0, 1))  # HWC → CHW
                chw = np.expand_dims(chw, axis=0)
                chw = np.ascontiguousarray(chw, dtype=np.float32)

                input_data = {input_name: chw}
                results = infer_pipeline.infer(input_data)
                output_data = results[output_infos[0].name]

                # 🧾 Verwacht formaat: [N, 6] → x1, y1, x2, y2, confidence, class_id
                for det in output_data:
                    x1, y1, x2, y2, conf, cls = det

                    if conf < 0.5 or int(cls) != 0:  # Alleen class_id 0 = "person"
                        continue

                    # Schaal bounding box naar originele frame grootte
                    h_ratio = frame.shape[0] / in_h
                    w_ratio = frame.shape[1] / in_w
                    x1 = int(x1 * w_ratio)
                    y1 = int(y1 * h_ratio)
                    x2 = int(x2 * w_ratio)
                    y2 = int(y2 * h_ratio)

                    # Teken bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 🎥 Toon frame
                cv2.imshow("YOLOv8 Person Detectie", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# 🧹 Opruimen
picam2.stop()
cv2.destroyAllWindows()
