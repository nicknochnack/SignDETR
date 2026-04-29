# VOXDEX/tools/collect_data.py

import cv2
import os
from datetime import datetime

BASE_DIR = "data/raw"
CLASSES = ["hello", "i_am_fine", "background"]

# create folders
for cls in CLASSES:
    os.makedirs(os.path.join(BASE_DIR, cls), exist_ok=True)

cap = cv2.VideoCapture(0)

print("Press h=hello, f=i_am_fine, b=background, q=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera failed")
        break

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    label = None
    if key == ord('h'):
        label = "hello"
    elif key == ord('f'):
        label = "i_am_fine"
    elif key == ord('b'):
        label = "background"
    elif key == ord('q'):
        break

    if label:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        path = os.path.join(BASE_DIR, label, filename)
        cv2.imwrite(path, frame)
        print(f"[SAVED] {label} → {path}")

cap.release()
cv2.destroyAllWindows()