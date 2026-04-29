"""DETR-based gesture detection: preprocess → forward → labels + boxes."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch

from model import DETR
from utils.boxes import rescale_bboxes
from utils.logger import get_logger
from utils.setup import get_classes, get_colors


def _num_classes_from_state_dict(state_dict: dict[str, Any]) -> int:
    w = state_dict.get("linear_class.weight")
    if w is None:
        raise KeyError("Checkpoint missing linear_class.weight; cannot infer num_classes.")
    return int(w.shape[0]) - 1


def _resolve_class_names(config_names: list[str], checkpoint_num_classes: int, log: Any) -> list[str]:
    """Align config class names with checkpoint cardinality; log mismatches."""
    if checkpoint_num_classes < 0:
        raise ValueError("checkpoint_num_classes must be non-negative")

    if len(config_names) == checkpoint_num_classes:
        return list(config_names)

    if len(config_names) < checkpoint_num_classes:
        log.warning(
            f"[gesture_predictor] Config defines {len(config_names)} class name(s) but the checkpoint "
            f"has {checkpoint_num_classes} foreground class(es). "
            f"Extra outputs map to unknown_<index> (not silent)."
        )
        out = list(config_names)
        for i in range(len(config_names), checkpoint_num_classes):
            out.append(f"unknown_{i}")
        return out

    log.warning(
        f"[gesture_predictor] Config lists {len(config_names)} class name(s) but the checkpoint "
        f"has only {checkpoint_num_classes}. Extra config entries are ignored: "
        f"{config_names[checkpoint_num_classes:]!r}"
    )
    return list(config_names[:checkpoint_num_classes])


def _bgr_palette(num_classes: int, colors_rgb: list[list[int]]) -> list[tuple[int, int, int]]:
    bgr: list[tuple[int, int, int]] = []
    for i in range(num_classes):
        if i < len(colors_rgb):
            row = colors_rgb[i]
            bgr.append((int(row[2]), int(row[1]), int(row[0])))
        else:
            bgr.append((60 + (i * 37) % 180, 200, 80))
    return bgr


@dataclass
class Detection:
    class_index: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class GesturePredictor:
    """
    DETR inference with training-aligned preprocessing (RGB 224, ImageNet norm),
    dual confidence thresholds, checkpoint-driven num_classes, and temporal smoothing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        box_threshold: float,
        display_threshold: float,
        temporal_window: int = 10,
        device: str | torch.device | None = None,
    ):
        self._log = get_logger("gesture_predictor")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.box_threshold = box_threshold
        self.display_threshold = display_threshold

        raw = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        ckpt_num_classes = _num_classes_from_state_dict(raw)
        self._checkpoint_num_classes = ckpt_num_classes

        cfg = get_classes()
        if isinstance(cfg, str):
            raise RuntimeError(cfg)
        config_names = list(cfg)

        self.class_names = _resolve_class_names(config_names, ckpt_num_classes, self._log)

        colors = get_colors()
        if isinstance(colors, str):
            raise RuntimeError(colors)
        self._colors_bgr = _bgr_palette(ckpt_num_classes, colors)

        self.model = DETR(num_classes=ckpt_num_classes)
        self.model.load_state_dict(raw)
        self.model.to(self.device)
        self.model.eval()

        self._transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ]
        )

        self._temporal: deque[tuple[str, float]] = deque(maxlen=temporal_window)
        self._last_best_label: str = "—"
        self._last_best_conf: float = 0.0

    @property
    def checkpoint_num_classes(self) -> int:
        return self._checkpoint_num_classes

    def _tensor_from_frame_bgr(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = self._transform(image=rgb)
        return t["image"].unsqueeze(0).to(self.device)

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Run one forward pass, update temporal buffer with the best-query (label, confidence),
        return box detections above ``box_threshold`` only.
        """
        h, w = frame_bgr.shape[:2]
        batch = self._tensor_from_frame_bgr(frame_bgr)

        with torch.no_grad():
            out = self.model(batch)

        logits = out["pred_logits"]
        boxes = out["pred_boxes"]
        prob = logits.softmax(-1)[:, :, :-1]
        max_probs, max_classes = prob.max(-1)

        mp = max_probs[0]
        mc = max_classes[0]
        pb = boxes[0]

        q_best = int(mp.argmax().item())
        best_conf = float(mp[q_best].item())
        best_cls = int(mc[q_best].item())
        self._last_best_label = self.class_names[best_cls]
        self._last_best_conf = best_conf

        # Temporal buffer: one sample per frame (best query), always recorded
        self._temporal.append((self._last_best_label, self._last_best_conf))

        detections: list[Detection] = []
        for qi in range(mp.shape[0]):
            conf = float(mp[qi].item())
            if conf < self.box_threshold:
                continue
            ci = int(mc[qi].item())
            if ci >= len(self.class_names):
                continue
            box_norm = pb[qi : qi + 1]
            xyxy = rescale_bboxes(box_norm, (w, h))[0].detach().cpu().numpy()
            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            detections.append(
                Detection(
                    class_index=ci,
                    label=self.class_names[ci],
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )

        return detections

    def get_stable_prediction(self) -> tuple[str, float]:
        """Majority label over the temporal window; confidence = mean conf for the winning label."""
        if not self._temporal:
            return ("—", 0.0)

        buf = list(self._temporal)
        labels = [t[0] for t in buf]
        cnt = Counter(labels)
        max_votes = max(cnt.values())
        candidates = [lab for lab, c in cnt.items() if c == max_votes]

        def mean_conf_for(lab: str) -> float:
            vals = [t[1] for t in buf if t[0] == lab]
            return sum(vals) / len(vals) if vals else 0.0

        winner = max(candidates, key=lambda lab: (cnt[lab], mean_conf_for(lab)))
        return (winner, mean_conf_for(winner))

    def get_instant_hud_best(self) -> tuple[str, float] | None:
        """Best-query label/conf only if above ``display_threshold`` (for observability)."""
        if self._last_best_conf > self.display_threshold:
            return (self._last_best_label, self._last_best_conf)
        return None

    def get_last_query(self) -> tuple[str, float]:
        """Best object-query label and confidence from the most recent ``predict()`` call."""
        return (self._last_best_label, self._last_best_conf)

    def color_bgr(self, class_index: int) -> tuple[int, int, int]:
        return self._colors_bgr[class_index % len(self._colors_bgr)]

    @property
    def class_names_public(self) -> list[str]:
        return list(self.class_names)
