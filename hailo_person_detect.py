import cv2
import numpy as np
import time
import csv
from datetime import datetime
from picamera2 import Picamera2
from hailo_platform import HEF, VDevice, ConfigureParams, FormatType, InputVStreamParams, OutputVStreamParams, InferVStreams, HailoStreamInterface

# 📁 Output CSV-bestand
CSV_PATH = "person_detections.csv"

# 📦 Pad naar je Hailo model
HEF_PATH = "/usr/share/hailo-models/yolov5s_personface_h8l.hef"

# 📷 Start de Pi-camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()
time.sleep(0.5)

# 📑 Laad het model
hef = HEF(HEF_PATH)

# 📄 Start logging-bestand
with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2'])

# 🧠 Start AI-inferentie
with VDevice() as device:
    configure_params = ConfigureParams.create_from_hef(hef, HailoStreamInterface.PCIe)
    network_group = device.configure(hef, configure_params)[0]
    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()

    input_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    with network_group.activate():
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            print("🚀 AI-inferentie gestart. Druk op 'q' om te stoppen.")

            while True:
                try:
                    frame = picam2.capture_array()
                except Exception as e:
                    print(f"❌ Fout bij lezen van camera: {e}")
                    break

                input_shape = input_infos[0].shape
                resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
                normalized = resized.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)
                input_data = {input_infos[0].name: input_tensor}

                # 🔍 Inferentie uitvoeren
                results = infer_pipeline.infer(input_data)
                output_data = results[output_infos[0].name]

                # 🔧 Parse detections (voor nu: fictieve data als voorbeeld)
                # LET OP: je moet dit vervangen met jouw echte outputstructuur!
                # Simulatie: [label, confidence, x1, y1, x2, y2]
                detecties = fake_parse_detections(output_data)

                # ✏️ Log & visualiseer alleen 'person'
                for det in detecties:
                    if det['label'] == 'person':
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        conf = det['confidence']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{det['label']} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        with open(CSV_PATH, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                datetime.now().isoformat(),
                                det['label'], conf, x1, y1, x2, y2
                            ])

                cv2.imshow("Personen Detectie", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# 🧹 Opruimen
picam2.stop()
cv2.destroyAllWindows()
