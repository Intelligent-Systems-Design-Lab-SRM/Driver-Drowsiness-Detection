"""OpenCV face detection and ONNX Runtime 68-point landmark detection."""

from dataclasses import dataclass
import os
import time
from importlib import import_module

import cv2
import numpy as np

try:
    ort = import_module("onnxruntime")
except ImportError:  # Keep configuration and unit tests importable without the model runtime.
    ort = None

LEFT_EYE = (36, 37, 38, 39, 40, 41)
RIGHT_EYE = (42, 43, 44, 45, 46, 47)
MOUTH = (62, 66, 48, 54)


@dataclass
class DetectionResult:
    face_detected: bool
    left_ear: float = 0.0
    right_ear: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    eye_closed_seconds: float = 0.0
    mouth_open_seconds: float = 0.0
    status: str = "NO FACE DETECTED"
    critical: bool = False
    yawn: bool = False


def calculate_ear(points: np.ndarray) -> float:
    if points.shape != (6, 2) or not np.isfinite(points).all():
        return 0.0
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    return 0.0 if horizontal <= 1e-9 else float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


class UnifiedDetector:
    def __init__(
        self, model_path, ear_threshold, eye_closed_seconds, mar_threshold,
        yawn_seconds, frame_scale=1.0, cascade_path="",
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required; install requirements.txt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Landmark model not found: {model_path}. "
                "Follow docs/MODEL_SETUP.md."
            )
        self.ear_threshold = ear_threshold
        self.eye_closed_seconds = eye_closed_seconds
        self.mar_threshold = mar_threshold
        self.yawn_seconds = yawn_seconds
        self.frame_scale = frame_scale
        self._eye_start = None
        self._mouth_start = None
        cascade = cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_detector = cv2.CascadeClassifier(cascade)
        if self._face_detector.empty():
            raise RuntimeError(f"Unable to load face detector: {cascade}")
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0]
        self._input_size = self._input.shape[-1]
        if not isinstance(self._input_size, int):
            self._input_size = 128

    @staticmethod
    def _points_from_output(output, box):
        values = np.asarray(output).reshape(-1)
        if values.size != 136:
            raise RuntimeError(f"Expected 68 landmark points, got {values.size // 2}")
        points = values.reshape(68, 2).astype(np.float32)
        if points.min() >= 0.0 and points.max() <= 1.0:
            points[:, 0] *= box[2] - box[0]
            points[:, 1] *= box[3] - box[1]
        else:
            points[:, 0] *= (box[2] - box[0]) / 128.0
            points[:, 1] *= (box[3] - box[1]) / 128.0
        points[:, 0] += box[0]
        points[:, 1] += box[1]
        return points

    def _landmarks(self, frame, box):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (self._input_size, self._input_size))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        output = self._session.run(
            None, {self._input.name: [crop]}
        )[0]
        return self._points_from_output(output, box)

    @staticmethod
    def _ear(landmarks, indices):
        return calculate_ear(landmarks[list(indices)])

    @staticmethod
    def _mar(landmarks):
        upper, lower, left, right = landmarks[list(MOUTH)]
        width = np.linalg.norm(right - left)
        return 0.0 if width <= 1e-9 else float(np.linalg.norm(upper - lower) / width)

    def process(self, frame):
        if self.frame_scale <= 0 or self.frame_scale > 1:
            raise ValueError("frame_scale must be greater than 0 and at most 1")
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a color image")
        if self.frame_scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.frame_scale, fy=self.frame_scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            self.reset()
            return DetectionResult(False)
        x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
        box = (int(x), int(y), int(x + width), int(y + height))
        landmarks = self._landmarks(frame, box)
        if landmarks is None:
            self.reset()
            return DetectionResult(False)
        now = time.monotonic()
        left_ear = self._ear(landmarks, LEFT_EYE)
        right_ear = self._ear(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        mar = self._mar(landmarks)
        if ear < self.ear_threshold:
            self._eye_start = now if self._eye_start is None else self._eye_start
        else:
            self._eye_start = None
        if mar > self.mar_threshold:
            self._mouth_start = now if self._mouth_start is None else self._mouth_start
        else:
            self._mouth_start = None
        eye_seconds = 0.0 if self._eye_start is None else now - self._eye_start
        mouth_seconds = 0.0 if self._mouth_start is None else now - self._mouth_start
        critical = eye_seconds >= self.eye_closed_seconds
        yawn = mouth_seconds >= self.yawn_seconds
        if critical and yawn:
            status = "CRITICAL: EYES CLOSED + YAWN"
        elif critical:
            status = "CRITICAL: PROLONGED EYE CLOSURE"
        elif yawn:
            status = "WARNING: POSSIBLE YAWN"
        elif self._eye_start is not None or self._mouth_start is not None:
            status = "CHECKING FOR DROWSINESS"
        else:
            status = "ALERT"
        return DetectionResult(
            True, left_ear, right_ear, ear, mar, eye_seconds, mouth_seconds,
            status, critical, yawn,
        )

    def reset(self):
        self._eye_start = None
        self._mouth_start = None

    def close(self):
        self._session = None
