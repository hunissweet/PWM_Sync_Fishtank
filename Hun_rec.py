import cv2
import numpy as np
import time
import subprocess   
from datetime import datetime

# ---------------------------con
# Global Variables
# ---------------------------
roi_start = None
roi_end = None
drawing = False
use_feature = 0
std_threshold = 20   # Default threshold

# Trigger control variables
previous_detected = False
last_trigger_time = 0
trigger_delay = 2.0     # for rising edge cooldown

absence_start_time = None
absence_delay = 3.0     # 3 seconds absence trigger

# ---------------------------

# Toggle Callback
# ---------------------------
def toggle_callback(val):
    global use_feature
    use_feature = val

# ---------------------------
# Threshold Callback
# ---------------------------
def threshold_callback(val):
    global std_threshold
    std_threshold = val

# ---------------------------
# Mouse Callback for ROI
# ---------------------------
def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        drawing = False


# ---------------------------
# Open Camera
# ---------------------------
cap = cv2.VideoCapture(0)

cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_callback)

# Create Trackbars
cv2.createTrackbar("Rec", "Webcam", 0, 1, toggle_callback)
cv2.createTrackbar("STD OBJ", "Webcam", 20, 100, threshold_callback)

prev_time = time.time()
con_obj = "no obj"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------
    # ROI Handling
    # ---------------------------
    if roi_start is None or roi_end is None:
        x1, y1 = 0, 0
        x2, y2 = w, h
    else:
        x1 = min(roi_start[0], roi_end[0])
        y1 = min(roi_start[1], roi_end[1])
        x2 = max(roi_start[0], roi_end[0])
        y2 = max(roi_start[1], roi_end[1])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    roi = gray[y1:y2, x1:x2]

    # ---------------------------
    # Compute STD
    # ---------------------------
    if roi.size > 0:
        stv = np.std(roi)
    else:
        stv = 0

    # ---------------------------
    # Detection Logic
    # ---------------------------
    detected = False
    if use_feature == 1 and stv > std_threshold:
        detected = True

    current_time = time.time()

    # ---------------------------
    # 1️⃣ Rising Edge Trigger (Object Appears)
    # ---------------------------
    if detected and not previous_detected:
        if current_time - last_trigger_time > trigger_delay:

            subprocess.Popen([
                "python",
                r"C:\Users\liang\Desktop\HunJang\PWM\PWM_Sync_Fishtank\Arduino_prompt.py",
                "--port", "COM14",
                "--freq", "50",
                "--pulse_us", "10"
            ])
            last_trigger_time = current_time
            print(f">>> TRIGGER SENT {datetime.now().strftime('%H:%M:%S')}")
            

        absence_start_time = None   # reset absence timer


    # ---------------------------
    # 2️⃣ Absence Timer Logic
    # ---------------------------
    if not detected:
        if previous_detected:
            # object just disappeared → start timer
            absence_start_time = current_time

        if absence_start_time is not None:
            if current_time - absence_start_time >= absence_delay:

                subprocess.Popen([
                    "python",
                    r"C:\Users\liang\Desktop\HunJang\PWM\PWM_Sync_Fishtank\Arduino_prompt.py",
                    "--port", "COM14",
                    "--freq", "0",
                    "--pulse_us", "10"
                ])

                print(f">>> Record stop {datetime.now().strftime('%H:%M:%S')}")

                absence_start_time = None   # prevent repeated firing


    # Update previous state
    previous_detected = detected

    # ---------------------------
    # FPS
    # ---------------------------
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # ---------------------------
    # Display
    # ---------------------------
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)

    cv2.putText(frame, f"ROI STD: {stv:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)

    cv2.putText(frame, f"Threshold: {std_threshold}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)

    if detected:
        con_obj = "Object detected"
    else:
        con_obj = "no obj"

    cv2.putText(frame, con_obj,
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0,0,255),
                3)

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        roi_start = None
        roi_end = None

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
