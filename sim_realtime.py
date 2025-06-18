import cv2
import cvzone
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolo11l.pt")

tracker = DeepSort(max_age=60, nms_max_overlap=1.0)

cap = cv2.VideoCapture("drone3.mp4")

frame_width, frame_height = 640, 640

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    blurred_frame = frame_resized.copy()
    results = model(frame_resized, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])       # Class ID
        conf = float(box.conf[0])      # Confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h

        if cls_id != 0 or conf < 0.5 or area < 500:
            continue

        detections.append(([x1, y1, w, h], conf, None))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame_resized)
    drone_detected = False

    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = map(int, track.to_ltrb())
        cx, cy = (l + r) // 2, (t + b) // 2
        print("CX, CY", (cx, cy))
        cv2.circle(frame_resized, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(blurred_frame, (cx, cy), 8, (0, 0, 255), -1)
        drone_detected = True
    if drone_detected:
        cvzone.putTextRect(frame_resized, "Drone Detected", (20, 440), scale=2, thickness=3,
                           colorT=(255, 255, 255), colorR=(0, 0, 255), offset=10)
        cvzone.putTextRect(blurred_frame, "Drone Detected", (20, 440), scale=2, thickness=3,
                           colorT=(255, 255, 255), colorR=(0, 0, 255), offset=10)

    # Display side-by-side
    stacked_output = cvzone.stackImages([frame_resized, blurred_frame], 2, 0.7)
    cv2.imshow("Drone Tracker", stacked_output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()