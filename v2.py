import cv2
import cvzone
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pyfirmata import Arduino, util
from filterpy.kalman import KalmanFilter
from collections import defaultdict

# === Camera and Servo Configuration ===
CAM_FOV_H = 90
CAM_FOV_V = 90
face_size = 800

DEG_PER_PX_H = CAM_FOV_H / face_size
DEG_PER_PX_V = CAM_FOV_V / face_size

# === Arduino Setup (Uncomment when using real hardware) ===
# board = Arduino('COM9')
# pan_servo = board.get_pin('d:9:s')
# tilt_servo = board.get_pin('d:10:s')
# time.sleep(2)

# === Model and Tracker ===
model = YOLO("drone.pt")
tracker = DeepSort(max_age=40, n_init=1, nn_budget=100)
cap = cv2.VideoCapture("drone2.mp4")

kalman_filters = defaultdict(lambda: init_kalman())

# === Kalman Init ===
def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000.
    kf.R *= 5
    kf.Q *= 0.01
    return kf

# === Target Priority State ===
target_schedule = []
target_index = 0
previous_ids = set()

def update_schedule(confirmed_tracks):
    global target_schedule, target_index, previous_ids
    weighted_list = []
    current_ids = set()

    for track_id, cx, cy, *_ in confirmed_tracks:
        dx = cx - face_size // 2
        dy = cy - face_size // 2
        dist = np.sqrt(dx ** 2 + dy ** 2)
        priority = max(1, int(3000 / (dist + 1)))
        weighted_list.extend([track_id] * priority)
        current_ids.add(track_id)

    if current_ids != previous_ids:
        target_index = 0
        previous_ids = current_ids

    if len(confirmed_tracks) > 1:
        target_schedule = weighted_list
    else:
        target_schedule = [confirmed_tracks[0][0]] if confirmed_tracks else [None]

# === Main Loop ===
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    drone_detected = False
    results = model(frame, verbose=False)[0]
    detections = []

    frame_area = frame.shape[0] * frame.shape[1]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = w / h if h != 0 else 0

        print(f"Detected Class: {cls_id}, Confidence: {conf:.2f}, Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")

        # === Smart Filtering ===
        if cls_id != 0:
            continue
        if conf < 0.2:
            continue
        if area < 400 and conf < 0.2:
            print("Skipping: too small and low confidence")
            continue
        if area > 0.6 * frame_area:
            print("Skipping: too large")
            continue

        print(f"Valid Drone Detection: {x1}, {y1}, {w}, {h}")
        detections.append(([x1, y1, w, h], conf, {"class_id": cls_id}))

    tracks = tracker.update_tracks(detections, frame=frame)
    confirmed_tracks = []

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        l, t, r, b = map(int, track.to_ltrb())
        raw_cx, raw_cy = (l + r) // 2, (t + b) // 2

        kf = kalman_filters[track.track_id]
        kf.predict()
        kf.update(np.array([raw_cx, raw_cy]))
        cx, cy = int(kf.x[0]), int(kf.x[1])

        confirmed_tracks.append((track.track_id, cx, cy, l, t, r, b))
        drone_detected = True

    update_schedule(confirmed_tracks)

    if target_schedule:
        target_index %= len(target_schedule)
        target_id = target_schedule[target_index]
        target_index += 1

        for tid, cx, cy, l, t, r, b in confirmed_tracks:
            if tid == target_id:
                dx_px = cx - face_size // 2
                dy_px = cy - face_size // 2
                pan_angle = 90 - dx_px * DEG_PER_PX_H
                tilt_angle = 90 + dy_px * DEG_PER_PX_V

                pan_angle = max(0, min(180, pan_angle))
                tilt_angle = max(0, min(180, tilt_angle))

                # pan_servo.write(pan_angle)
                # tilt_servo.write(tilt_angle)

                cv2.rectangle(frame, (l, t), (r, b), (255, 0, 255), 3)
                cv2.putText(frame, f"TARGET {tid}", (l, t - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                break

    for tid, cx, cy, l, t, r, b in confirmed_tracks:
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {tid}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if drone_detected:
        cvzone.putTextRect(frame, "Drone Detected", (20, 40), scale=2, thickness=3,
                           colorT=(255, 255, 255), colorR=(0, 0, 255), offset=10)

    cv2.imshow("Drone Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
