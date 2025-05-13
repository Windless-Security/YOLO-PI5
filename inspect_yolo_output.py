import cv2
import numpy as np
import time
from picamera2 import Picamera2
from hailo_platform import (
    HEF, VDevice, ConfigureParams, FormatType,
    InputVStreamParams, OutputVStreamParams,
    InferVStreams, HailoStreamInterface
)

HEF_PATH = "/usr/share/hailo-models/yolov8_personface.hef"  # Pas aan indien nodig

# Start Pi Camera
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

    input_name = input_infos[0].name
    model_shape = input_infos[0].get_shape()
    print("📐 Verwachte modelinput shape:", model_shape)

    _, c, in_h, in_w = model_shape
    expected_dtype = np.float32

    with network_group.activate():
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            print("🚀 Start AI-inferentie")

            frame = picam2.capture_array()
            resized = cv2.resize(frame, (in_w, in_h))
            normalized = resized.astype(np.float32) / 255.0
            chw = np.transpose(normalized, (2, 0, 1))
            chw = np.expand_dims(chw, axis=0)
            chw = np.ascontiguousarray(chw, dtype=expected_dtype)

            input_data = {input_name: chw}
            results = infer_pipeline.infer(input_data)

            output_data = results[output_infos[0].name]

            print("▶️ Output shape:", output_data.shape)
            print("▶️ Eerste rij output:", output_data[0])

picam2.stop()
cv2.destroyAllWindows()
