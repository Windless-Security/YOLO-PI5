import cv2
import numpy as np
from hailo_platform import HEF, VDevice, ConfigureParams, FormatType, InputVStreamParams, OutputVStreamParams, InferVStreams, HailoStreamInterface

# Pad naar je HEF-bestand
HEF_PATH = "/usr/share/hailo-models/yolov5s_personface_h8l.hef"

# Open de webcam
cap = cv2.VideoCapture("libcamerasrc ! videoconvert ! appsink". cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Kan de camera niet openen.")
    exit()

# Laad het HEF-model
hef = HEF(HEF_PATH)

# Configureer het apparaat
with VDevice() as device:
    configure_params = ConfigureParams.create_from_hef(hef, HailoStreamInterface.PCIe)
    network_group = device.configure(hef, configure_params)[0]
    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()

    # Stel de parameters voor input en output in
    input_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    # Activeer het netwerk
    with network_group.activate():
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            print("🚀 AI-inferentie gestart. Druk op 'q' om te stoppen.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Fout bij het lezen van frame.")
                    break

                # Preprocessing: resize en normaliseer
                input_shape = input_infos[0].shape
                resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
                normalized = resized.astype(np.float32) / 255.0
                input_data = {input_infos[0].name: np.expand_dims(normalized, axis=0)}

                # Voer inferentie uit
                results = infer_pipeline.infer(input_data)
                output_data = results[output_infos[0].name]

                # Post-processing: hier zou je de output_data moeten verwerken
                # om bounding boxes te extraheren. Dit vereist kennis van het
                # specifieke model en outputformaat.

                # Voorbeeld: stel dat je een functie 'parse_detections' hebt
                # die de output_data verwerkt en een lijst van detecties retourneert.
                # detecties = parse_detections(output_data)

                # Voor elke detectie, teken een bounding box als het label 'person' is
                # for det in detecties:
                #     if det['label'] == 'person':
                #         x1, y1, x2, y2 = det['bbox']
                #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #         cv2.putText(frame, 'person', (x1, y1 - 10),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Toon het frame
                cv2.imshow("Personen Detectie", frame)

                # Stoppen met 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# Opruimen
cap.release()
cv2.destroyAllWindows()
