"""Real-time sign language detection: CameraHandler + GesturePredictor."""

from __future__ import annotations

import time
from collections import deque

import cv2

from camera_handler import CameraHandler
from gesture_predictor import GesturePredictor
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler

# --- Tunables (no magic numbers in call sites) ---
CHECKPOINT = "checkpoints/49_model.pt"
BOX_THRESHOLD = 0.35
DISPLAY_THRESHOLD = 0.20
TEMPORAL_WINDOW = 10
FPS_SMOOTH_FRAMES = 30
WINDOW_NAME = "VoxDex - Live"


class RollingFPS:
    """FPS from elapsed wall time over the last N frame timestamps."""

    def __init__(self, window: int = FPS_SMOOTH_FRAMES) -> None:
        self._window = max(2, window)
        self._t: deque[float] = deque(maxlen=self._window)

    def tick(self) -> None:
        self._t.append(time.perf_counter())

    def fps(self) -> float:
        if len(self._t) < 2:
            return 0.0
        dt = self._t[-1] - self._t[0]
        if dt <= 0:
            return 0.0
        return (len(self._t) - 1) / dt


def draw_overlay(
    frame,
    detections,
    predictor: GesturePredictor,
    stable_label: str,
    stable_conf: float,
    fps_value: float,
    instant: tuple[str, float] | None,
    display_threshold: float,
) -> None:
    """Single HUD bar + optional instant line; class-colored boxes (BGR)."""
    _h, w = frame.shape[:2]
    bar_h = 72
    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)

    instant_part = ""
    if instant is not None:
        instant_part = f"  |  now: {instant[0]} {instant[1]:.2f}"
    else:
        _lab, last_c = predictor.get_last_query()
        instant_part = f"  |  now: {_lab} {last_c:.2f} (<= {display_threshold:.2f})"

    line1 = (
        f"VoxDex  |  stable: {stable_label}  {stable_conf:.2f}"
        f"{instant_part}  |  FPS {fps_value:.1f}"
    )
    cv2.putText(
        frame,
        line1,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"boxes > {BOX_THRESHOLD:.2f}  |  display > {DISPLAY_THRESHOLD:.2f}",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    for det in detections:
        x1, y1, x2, y2 = (int(det.bbox_xyxy[0]), int(det.bbox_xyxy[1]), int(det.bbox_xyxy[2]), int(det.bbox_xyxy[3]))
        color = predictor.color_bgr(det.class_index)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        y0 = max(bar_h + 4, y1 - 6)
        cv2.rectangle(frame, (x1, y0 - th - 6), (x1 + tw + 8, y0 + 2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    logger = get_logger("realtime")
    detection_handler = DetectionHandler()
    logger.print_banner()
    logger.realtime("Initializing VoxDex real-time detection...")

    predictor = GesturePredictor(
        CHECKPOINT,
        box_threshold=BOX_THRESHOLD,
        display_threshold=DISPLAY_THRESHOLD,
        temporal_window=TEMPORAL_WINDOW,
    )
    camera = CameraHandler(0)
    fps_counter = RollingFPS(FPS_SMOOTH_FRAMES)

    frame_count = 0
    log_fps_start = time.time()

    logger.realtime("Starting camera — press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                logger.error("Failed to read frame from camera")
                break

            t0 = time.perf_counter()
            detections = predictor.predict(frame)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            stable_label, stable_conf = predictor.get_stable_prediction()
            instant = predictor.get_instant_hud_best()

            fps_counter.tick()
            fps_val = fps_counter.fps()

            draw_overlay(
                frame,
                detections,
                predictor,
                stable_label,
                stable_conf,
                fps_val,
                instant,
                DISPLAY_THRESHOLD,
            )

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - log_fps_start
                batch_fps = 30 / elapsed if elapsed > 0 else 0.0
                if detections:
                    detection_handler.log_detections(
                        [
                            {"class": d.label, "confidence": d.confidence, "bbox": list(d.bbox_xyxy)}
                            for d in detections
                        ],
                        frame_id=frame_count,
                    )
                detection_handler.log_inference_time(infer_ms, batch_fps)
                log_fps_start = time.time()

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.realtime("Stopping.")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
