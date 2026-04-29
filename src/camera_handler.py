"""Webcam capture for real-time inference."""

from __future__ import annotations

import cv2


class CameraHandler:
    """Thin wrapper around OpenCV VideoCapture."""

    def __init__(self, camera_id: int = 0, width: int | None = None, height: int | None = None):
        self._camera_id = camera_id
        self._cap = cv2.VideoCapture(camera_id)
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def capture(self):
        return self._cap

    def read(self):
        """Returns (ok, frame_bgr) as OpenCV convention."""
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()
