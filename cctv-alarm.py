import cv2
import time
from ultralytics import YOLO
from playsound import playsound

# ------------------------------
# 1. Load YOLOv8 Model
# ------------------------------
model = YOLO("yolov8s.pt")
model.conf = 0.1  # Confidence threshold (0-1)

# Classes we care about: 0=person, 15=cat, 16=dog
classes_to_detect = [0, 15, 16]

# ------------------------------
# 2. RTSP Stream URL
# ------------------------------
rtsp_url = "rtsp://admin:<PASSWORD>@<IP>:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Cannot open RTSP stream")
    exit()

# ------------------------------
# 3. Live Detection Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)
        continue

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 360))

    # Run YOLOv8 detection
    results = model(frame_resized)[0]  # YOLOv8 returns a list of Results

    # Loop through detections
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in classes_to_detect:
            continue  # skip classes we don't care about
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Assign label
        if cls == 0:
            label = 'Human'
        elif cls == 15:
            label = 'Cat'
        elif cls == 16:
            label = 'Dog'

        # Alarm
        playsound('sonar.mp3')
        
        # Take snapshot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'./snapshots/snapshot_{label}_{timestamp}.jpg'

        # Save the original full-resolution frame, not the resized one
        cv2.imwrite(filename, frame)
        print(f"[INFO] Snapshot saved: {filename}")
        
        # Draw bounding box and label
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_resized, f'{label} {conf:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show live detection
    cv2.imshow("RTSP Human, Cat & Dog Detection", frame_resized)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
