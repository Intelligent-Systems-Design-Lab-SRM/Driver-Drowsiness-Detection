"""Raspberry Pi and Windows entry point for driver monitoring."""

import logging
import time

import cv2

from config import load_settings
from detection.unified_detector import UnifiedDetector
from hardware.buzzer_controller import BuzzerController
from hardware.camera_controller import CameraController


def _draw_overlay(frame, result, alarm_active, fps, face_missing):
    color = (0, 255, 0)
    if result.critical:
        color = (0, 0, 255)
    elif result.yawn or face_missing:
        color = (0, 165, 255)
    status = "WARNING: FACE NOT DETECTED" if face_missing else result.status
    lines = [
        f"STATUS: {status}",
        f"FACE: {'DETECTED' if result.face_detected else 'MISSING'}",
        f"EAR L/R: {result.left_ear:.3f} / {result.right_ear:.3f}",
        f"EYE CLOSED: {result.eye_closed_seconds:.1f}s",
        f"MAR: {result.mar:.3f}  YAWN: {'YES' if result.yawn else 'NO'}",
        f"ALARM: {'ON' if alarm_active else 'OFF'}  FPS: {fps:.1f}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame, line, (15, 30 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )


def main() -> int:
    settings = load_settings()
    logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO)
    camera = CameraController(
        settings.camera_index, settings.camera_width, settings.camera_height
    )
    if not camera.is_opened:
        camera.release()
        logging.error("Unable to open camera %s", settings.camera_index)
        return 1

    try:
        detector = UnifiedDetector(
            settings.landmark_model,
            settings.ear_threshold,
            settings.eye_closed_seconds,
            settings.mar_threshold,
            settings.yawn_seconds,
            settings.frame_scale,
            settings.face_cascade,
        )
    except (FileNotFoundError, RuntimeError) as error:
        camera.release()
        logging.error("Detector initialization failed: %s", error)
        return 1
    buzzer = BuzzerController(
        settings.buzzer_gpio,
        settings.buzzer_active_high,
        settings.mock_buzzer,
        settings.alarm_duration_seconds,
        settings.alarm_pause_seconds,
    )
    logging.info("Monitoring started; buzzer mock mode: %s", buzzer.is_mock)
    last_frame_time = time.monotonic()
    face_missing_start = None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                logging.error("Camera frame unavailable")
                break
            now = time.monotonic()
            result = detector.process(frame)
            if result.face_detected:
                face_missing_start = None
            elif face_missing_start is None:
                face_missing_start = now
            face_missing = (
                face_missing_start is not None
                and now - face_missing_start >= settings.face_missing_seconds
            )
            critical = result.critical
            buzzer.update(critical)
            fps = 1.0 / max(now - last_frame_time, 1e-6)
            last_frame_time = now

            if not settings.headless:
                _draw_overlay(frame, result, buzzer.is_active, fps, face_missing)
                cv2.imshow("Driver Drowsiness Detection System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    buzzer.test()
                if key == ord("r"):
                    detector.reset()
                    buzzer.off()
                    face_missing_start = None
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        buzzer.close()
        detector.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
