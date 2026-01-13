heatmap.py
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model (person detection)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or bigger models for higher accuracy

# Parameters
VIDEO_SOURCE = 0  # 0 = webcam, or path to CCTV/drone video file
DENSITY_THRESHOLD = 50  # People count threshold to trigger alert
ALERT_COOLDOWN = 5  # Seconds before next alert

last_alert_time = 0

def generate_heatmap(frame, people_boxes):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for (x1, y1, x2, y2) in people_boxes:
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(heatmap, (center_x, center_y), 50, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (91, 91), 0)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    combined = cv2.addWeighted(frame, 0.7, heatmap_color, 0.5, 0)
    return combined

# Open video stream
cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)
    people_boxes = []

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # Class 0 = Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                people_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Count people
    people_count = len(people_boxes)
    cv2.putText(frame, f"People Count: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Generate heatmap overlay
    frame = generate_heatmap(frame, people_boxes)

    # Check for overcrowding
    if people_count > DENSITY_THRESHOLD:
        current_time = time.time()
        if current_time - last_alert_time > ALERT_COOLDOWN:
            print("⚠ ALERT: Overcrowding detected!")
            last_alert_time = current_time
        cv2.putText(frame, "⚠ OVERCROWDING ALERT!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display output
    cv2.imshow("AI Crowd Management", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
