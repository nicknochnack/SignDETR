# VOXDEX/tools/auto_label.py

import cv2
import mediapipe as mp
import os
import glob

mp_hands = mp.solutions.hands.Hands(static_image_mode=True)

INPUT_DIR = "data/raw"
OUT_IMG = "data/train/images"
OUT_LBL = "data/train/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

CLASSES = {
    "hello": 0,
    "i_am_fine": 1,
    "background": None
}

def save_label(path, lines):
    with open(path, "w") as f:
        for l in lines:
            f.write(l + "\n")

import uuid

for cls_name, cls_id in CLASSES.items():
    img_paths = glob.glob(os.path.join(INPUT_DIR, cls_name, "*.jpg"))

    for img_path in img_paths:
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(img_rgb)

        filename = f"{cls_name}_{uuid.uuid4().hex}.jpg"
        out_img_path = os.path.join(OUT_IMG, filename)
        out_lbl_path = os.path.join(OUT_LBL, filename.replace(".jpg", ".txt"))

        print(f"Processing {cls_name} → class_id={cls_id}")

        labels = []

        # ✅ CORRECTLY INDENTED LOGIC
        if cls_name == "background":
            labels = []
        else:
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    xs = [lm.x for lm in hand.landmark]
                    ys = [lm.y for lm in hand.landmark]

                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = (x_max - x_min)
                    height = (y_max - y_min)

                    labels.append(f"{cls_id} {x_center} {y_center} {width} {height}")
            else:
                # fallback
                labels.append(f"{cls_id} 0.5 0.5 1.0 1.0")

        # ✅ SAVE INSIDE LOOP
        cv2.imwrite(out_img_path, img)

        with open(out_lbl_path, "w") as f:
            for line in labels:
                f.write(line + "\n")