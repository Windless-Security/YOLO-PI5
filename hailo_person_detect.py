import cv2
import numpy as np
import time
from picamera2 import Picamera2
from hailo_platform import (
    HEF, VDevice, ConfigureParams, FormatType,
    InputVStreamParams, OutputVStreamParams,
    InferVStreams, HailoStreamInterface
)

HEF_PATH = "/usr/share/hailo-models/yolov5s_personface_h8l.hef"

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()
time.sleep(0.5)

hef = HEF(HEF_PATH)

with VDevice() as device:
    configure_params = ConfigureParams.create_from_hef(hef, HailoStreamInterface.PCIe)
    network_group = device.configure(hef, configure_params)[0]
    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()

    input_params = InputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=FormatType.FLOAT32
    )
    output_params = OutputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=FormatType.FLOAT32
    )

    # 📐 Model input shape ophalen
    input_name = input_infos[0].name
    model_shape = input_infos[0].shape  # [1, 3, 640, 640]
    print("📐 Model verwacht shape:", model_shape)

    if len(model_shape) != 4:
        raise RuntimeError("❌ Modelinputvorm is niet 4D!")

    batch, channels, in_h, in_w = model_shape
    expected_size = batch * channels * in_h * in_w
    print(f"✅ Modelinput verwacht {expected_size} elementen.")

    with network_group.activate():
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            print("🚀 AI-inferentie gestart. Druk op 'q' om te stoppen.")

            while True:
                try:
                    frame = picam2.capture_array()
                except Exception as e:
                    print(f"❌ Fout bij lezen van camera: {e}")
                    break

                # 🧼 Preprocessing exact zoals model verwacht
                resized = cv2.resize(frame, (in_w, in_h))
                normalized = resized.astype(np.float32) / 255.0
                chw = np.transpose(normalized, (2, 0, 1))         # HWC → CHW
                chw = np.expand_dims(chw, axis=0)                 # [1, 3, H, W]
                chw = np.ascontiguousarray(chw, dtype=np.float32)

                print("🧪 Inputvorm:", chw.shape)
                print("🔢 Totale elementen:", chw.size)

                if chw.size != expected_size:
                    print(f"❌ Mismatch! Verwacht {expected_size}, maar kreeg {chw.size}")
                    break

                input_data = {input_name: chw}

                try:
                    _ = infer_pipeline.infer(input_data)
                except Exception as e:
                    print(f"❌ Fout bij inferentie: {e}")
                    break

                cv2.imshow("Live beeld (AI actief)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

picam2.stop()
cv2.destroyAllWindows()
