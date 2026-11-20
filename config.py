"""Runtime configuration for the driver drowsiness monitor."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    frame_scale: float = 1.0
    ear_threshold: float = 0.20
    eye_closed_seconds: float = 1.0
    mar_threshold: float = 0.20
    yawn_seconds: float = 1.2
    face_missing_seconds: float = 2.0
    buzzer_gpio: int = 18
    buzzer_active_high: bool = True
    alarm_duration_seconds: float = 0.5
    alarm_pause_seconds: float = 0.5
    debug: bool = False
    mock_buzzer: bool = False
    headless: bool = False


def load_settings() -> Settings:
    """Load supported overrides from environment variables."""
    values = {}
    converters = {
        "camera_index": int,
        "camera_width": int,
        "camera_height": int,
        "frame_scale": float,
        "ear_threshold": float,
        "eye_closed_seconds": float,
        "mar_threshold": float,
        "yawn_seconds": float,
        "face_missing_seconds": float,
        "buzzer_gpio": int,
        "alarm_duration_seconds": float,
        "alarm_pause_seconds": float,
        "debug": lambda value: value.lower() in {"1", "true", "yes", "on"},
        "mock_buzzer": lambda value: value.lower() in {"1", "true", "yes", "on"},
        "headless": lambda value: value.lower() in {"1", "true", "yes", "on"},
    }
    for name, converter in converters.items():
        environment_name = "DROWSINESS_" + name.upper()
        if environment_name in os.environ:
            values[name] = converter(os.environ[environment_name])
    return Settings(**values)
