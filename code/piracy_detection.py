# piracy_detection.py
from ultralytics import YOLO
import cv2
import math
import time
import os
import tkinter as tk
from tkinter import messagebox
from threading import Thread, Lock

# --- Config ---
MODEL_PATH = os.path.join("yolo-Weights/yolov8s.pt")
POPUP_COOLDOWN = 5.0  # seconds between popups
CONF_THRESHOLD = 0.5

# --- Ensure model exists (helpful message) ---
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. You can download yolov8n.pt into yolo-Weights/ or set MODEL_PATH to 'yolov8n.pt' to let ultralytics auto-download it.")
# Load model (this will also try to download if you pass a simple filename)
model = YOLO(MODEL_PATH)

# Common COCO class names (80)
classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
              "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
              "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
              "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
              "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
              "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
              "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
              "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
              "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
              "teddy bear","hair drier","toothbrush"]

# --- Popup control ---
_last_popup_time = 0.0
_popup_lock = Lock()

def show_popup():
    """Displays a blocking tkinter warning. Runs in a separate thread."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Alert", "Cell Phone Detected! Stop Recording.")
    root.destroy()

def trigger_popup():
    global _last_popup_time
    now = time.time()
    with _popup_lock:
        if now - _last_popup_time < POPUP_COOLDOWN:
            return
        _last_popup_time = now
    Thread(target=show_popup, daemon=True).start()

# --- Start webcam ---
cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

print("Starting webcam. Press 'q' to quit.")
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam. Exiting.")
        break

    # Run model inference (stream=True is fine for frame-by-frame in a loop)
    results = model(img, stream=True)

    cell_phone_detected = False

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            try:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            except Exception:
                # fallback if structure differs
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords[:4])

            # draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            confidence = float(box.conf[0])
            confidence = math.ceil(confidence * 100) / 100.0

            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else str(cls)
            print(f"Class: {class_name}, Confidence: {confidence:.2f}")

            # if cell phone found
            if class_name.lower() == "cell phone" and confidence > CONF_THRESHOLD:
                cell_phone_detected = True
                trigger_popup()

            cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if cell_phone_detected:
        cv2.putText(img, "CELL PHONE DETECTED. STOP RECORDING", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
