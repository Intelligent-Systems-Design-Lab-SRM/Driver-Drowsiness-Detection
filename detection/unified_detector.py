"""Reusable MediaPipe Face Mesh based drowsiness detector."""

from dataclasses import dataclass
import time

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
MOUTH = (13, 14, 78, 308)


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
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    return 0.0 if horizontal == 0 else float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


class UnifiedDetector:
    def __init__(
        self,
        ear_threshold: float,
        eye_closed_seconds: float,
        mar_threshold: float,
        yawn_seconds: float,
        frame_scale: float = 1.0,
    ):
        self.ear_threshold = ear_threshold
        self.eye_closed_seconds = eye_closed_seconds
        self.mar_threshold = mar_threshold
        self.yawn_seconds = yawn_seconds
        self.frame_scale = frame_scale
        self._eye_start = None
        self._mouth_start = None
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _points(face, indices, width, height):
        return np.array(
            [[face.landmark[i].x * width, face.landmark[i].y * height] for i in indices],
            dtype=np.float64,
        )

    @classmethod
    def _mar(cls, face, width, height):
        points = cls._points(face, MOUTH, width, height)
        mouth_width = np.linalg.norm(points[2] - points[3])
        return 0.0 if mouth_width == 0 else float(
            np.linalg.norm(points[0] - points[1]) / mouth_width
        )

    def process(self, frame: np.ndarray) -> DetectionResult:
        if self.frame_scale <= 0 or self.frame_scale > 1:
            raise ValueError("frame_scale must be greater than 0 and at most 1")
        original_height, original_width = frame.shape[:2]
        if self.frame_scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(original_width * self.frame_scale), int(original_height * self.frame_scale)),
            )
        height, width = frame.shape[:2]
        results = self._mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.monotonic()

        if not results.multi_face_landmarks:
            self.reset()
            return DetectionResult(face_detected=False)

        face = results.multi_face_landmarks[0]
        left = self._points(face, LEFT_EYE, width, height)
        right = self._points(face, RIGHT_EYE, width, height)
        left_ear = calculate_ear(left)
        right_ear = calculate_ear(right)
        ear = (left_ear + right_ear) / 2.0
        mar = self._mar(face, width, height)

        if ear < self.ear_threshold:
            if self._eye_start is None:
                self._eye_start = now
        else:
            self._eye_start = None
        if mar > self.mar_threshold:
            if self._mouth_start is None:
                self._mouth_start = now
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

    def reset(self) -> None:
        self._eye_start = None
        self._mouth_start = None

    def close(self) -> None:
        self._mesh.close()
