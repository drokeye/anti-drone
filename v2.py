import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pyfirmata import Arduino, util
import time

# === Camera Setup ===
frame_width, frame_height = 1080, 720
CAM_FOV_H = 86  # Horizontal FOV in degrees
CAM_FOV_V = 55
DEG_PER_PX_H = CAM_FOV_H / frame_width
DEG_PER_PX_V = CAM_FOV_V / frame_height

# === Arduino Setup ===
board = Arduino('COM9')  # Change as needed
pan_servo = board.get_pin('d:9:s')
tilt_servo = board.get_pin('d:10:s')
time.sleep(2)  # Arduino initialization

# === Object Detection and Tracking ===
model = YOLO("best.pt")
tracker = DeepSort(max_age=15, n_init=2, nn_budget=50, max_cosine_distance=0.3)
cap = cv2.VideoCapture(0)

# === Tracking Cache for EMA
track_smoothing = {}
EMA_ALPHA = 0.3
ANGLE_EPSILON = 0.5  # Minimum change to update servo

# === Main Loop ===
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    results = model(frame, verbose=False)[0]
    detections = []

    # === Detection Filtering ===
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h

        if cls_id != 0 or conf < 0.7 or area < 1000:
            continue

        detections.append(([x1, y1, w, h], conf, {"class_id": cls_id}))

    tracks = tracker.update_tracks(detections, frame=frame)
    drone_detected = False

    active_tracks = []

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        l, t, r, b = map(int, track.to_ltrb())
        cx, cy = (l + r) // 2, (t + b) // 2

        track_id = track.track_id
        active_tracks.append(track_id)

        # === Exponential Moving Average (smoothen center position)
        if track_id not in track_smoothing:
            track_smoothing[track_id] = {"cx": cx, "cy": cy}
        else:
            track_smoothing[track_id]["cx"] = int(
                EMA_ALPHA * cx + (1 - EMA_ALPHA) * track_smoothing[track_id]["cx"])
            track_smoothing[track_id]["cy"] = int(
                EMA_ALPHA * cy + (1 - EMA_ALPHA) * track_smoothing[track_id]["cy"])

        smoothed_cx = track_smoothing[track_id]["cx"]
        smoothed_cy = track_smoothing[track_id]["cy"]

        # Draw visual tracking marker
        cv2.circle(frame, (smoothed_cx, smoothed_cy), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # === Servo Angle Calculation
        dx_px = smoothed_cx - frame_width // 2
        dy_px = smoothed_cy - frame_height // 2
        pan_angle = 90 - dx_px * DEG_PER_PX_H
        tilt_angle = 90 + dy_px * DEG_PER_PX_V

        # Clamp angles
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))

        # === Reduce jitter: only update servo if change is significant
        if "last_pan" not in track_smoothing[track_id]:
            track_smoothing[track_id]["last_pan"] = pan_angle
            track_smoothing[track_id]["last_tilt"] = tilt_angle

        if abs(track_smoothing[track_id]["last_pan"] - pan_angle) > ANGLE_EPSILON:
            pan_servo.write(pan_angle)
            track_smoothing[track_id]["last_pan"] = pan_angle

        if abs(track_smoothing[track_id]["last_tilt"] - tilt_angle) > ANGLE_EPSILON:
            tilt_servo.write(tilt_angle)
            track_smoothing[track_id]["last_tilt"] = tilt_angle

        print(f"[Track {track_id}] Pan: {pan_angle:.2f}°, Tilt: {tilt_angle:.2f}°")
        drone_detected = True

    # === Clean up stale tracks from cache
    to_delete = [tid for tid in track_smoothing if tid not in active_tracks]
    for tid in to_delete:
        del track_smoothing[tid]

    if drone_detected:
        cvzone.putTextRect(frame, "Drone Detected", (20, 580), scale=2, thickness=3,
                           colorT=(255, 255, 255), colorR=(0, 0, 255), offset=10)

    cv2.imshow("Drone Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
