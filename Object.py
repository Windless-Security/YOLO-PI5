import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import time

# Laad YOLOv8 model
model = YOLO("best.pt")

# Start PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Video-opname instellen
timestamp = time.strftime("%Y%m%d-%H%M%S")
out = cv2.VideoWriter(f"opname_{timestamp}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      20, (1280, 720))

print("Druk op 'q' om te stoppen")

# Voor FPS-berekening
prev_time = time.time()

while True:
    frame = picam2.capture_array()

    # YOLO verwacht RGB input
    results = model(frame, conf=0.4, imgsz=640)

    # Teken bounding boxes
    result_frame = results[0].plot()

    # Bereken FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Voeg FPS-tekst toe
    cv2.putText(result_frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)


    # Toon beeld
    cv2.imshow("Live YOLOv8 Detectie", result_frame)

    # Schrijf frame weg naar video
    out.write(result_frame)

    # Stoppen met 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Opruimen
cv2.destroyAllWindows()
out.release()
picam2.close()
