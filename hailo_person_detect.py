import cv2
import numpy as np
import time
from picamera2 import Picamera2
from hailo_platform import HEF, VDevice, ConfigureParams, FormatType, InputVStreamParams, OutputVStreamParams, InferVStreams, HailoStreamInterface

# 📦 Modelpad
HEF_PATH = "/usr/share/hailo-models/yolov5s_personface_h8l.hef"

# 📷 Start Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()
time.sleep(0.5)

# 📑 Laad het Hailo model
hef = HEF(HEF_PATH)

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

            # 📐 Resolutie bepalen op basis van modelinput
            input_shape = input_infos[0].shape
            if len(input_shape) == 4:
                in_h, in_w = input_shape[2], input_shape[3]
            elif len(input_shape) == 3:
                in_h, in_w = input_shape[1], input_shape[2]
            else:
                print("⚠️ Onbekende input shape:", input_shape)
                in_h, in_w = 320, 320

            while True:
                try:
                    frame = picam2.capture_array()
                except Exception as e:
                    print(f"❌ Fout bij lezen van camera: {e}")
                    break

                # 📊 Preprocessing
                resized = cv2.resize(frame, (in_w, in_h))
                normalized = resized.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)
                input_data = {input_infos[0].name: input_tensor}

                # 🔄 Inference uitvoeren
                _ = infer_pipeline.infer(input_data)

                # 🎥 Toon alleen live video
                cv2.imshow("Live Video (zonder verwerking)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# 🧹 Opruimen
picam2.stop()
cv2.destroyAllWindows()
