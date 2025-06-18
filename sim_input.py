import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pyfirmata import Arduino, util
import time

# === Camera Setup ===
frame_width, frame_height = 1080, 720
CAM_FOV_H = 86   # Horizontal FOV of your camera
CAM_FOV_V = 55   # Vertical FOV of your camera
DEG_PER_PX_H = CAM_FOV_H / frame_width
DEG_PER_PX_V = CAM_FOV_V / frame_height

# === Arduino Setup ===
board = Arduino('COM9')  # Change to your COM port
pan_servo = board.get_pin('d:9:s')   # Servo on digital pin 9
tilt_servo = board.get_pin('d:10:s') # Servo on digital pin 10
time.sleep(2)  # Give time to initialize

# === Model and Tracker ===
model = YOLO("yolo11l.pt")
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
cap = cv2.VideoCapture(0)  # Or use 0 for webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h

        # Filter only drones (assume class 0 is drone)
        if cls_id != 0 or conf <= 0.7 or area < 500:
            continue

        detections.append(([x1, y1, w, h], conf, {"class_id": cls_id}))

    # Track drones
    tracks = tracker.update_tracks(detections, frame=frame)
    drone_detected = False

    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = map(int, track.to_ltrb())
        cx, cy = (l + r) // 2, (t + b) // 2

        # Draw tracking marker
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{track.track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # === Servo Angle Calculation ===
        dx_px = cx - frame_width // 2
        dy_px = cy - frame_height // 2
        pan_angle = 90 - dx_px * DEG_PER_PX_H
        tilt_angle = 90 + dy_px * DEG_PER_PX_V

        # Clamp to servo range
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))

        # Send to Arduino
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)

        print(f"[Track {track.track_id}] Pan: {pan_angle:.2f}°, Tilt: {tilt_angle:.2f}°")

        drone_detected = True

    if drone_detected:
        cvzone.putTextRect(frame, "Drone Detected", (20, 580), scale=2, thickness=3,
                           colorT=(255, 255, 255), colorR=(0, 0, 255), offset=10)

    # Show final output
    cv2.imshow("Drone Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
