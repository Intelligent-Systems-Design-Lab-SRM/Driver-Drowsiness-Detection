"""Safe buzzer adapter with GPIO Zero and development mock support."""

import logging
import time
from importlib import import_module
from typing import Optional

LOGGER = logging.getLogger(__name__)


class BuzzerController:
    def __init__(
        self,
        pin: int,
        active_high: bool = True,
        mock: bool = False,
        alarm_duration: float = 0.5,
        alarm_pause: float = 0.5,
    ):
        self.pin_number = pin
        self.alarm_duration = alarm_duration
        self.alarm_pause = alarm_pause
        self._active = False
        self._mock = mock
        self._output = None

        if not mock:
            try:
                buzzer_class = import_module("gpiozero").Buzzer
            except ImportError as error:
                LOGGER.warning("gpiozero is unavailable; using mock buzzer: %s", error)
                self._mock = True
            else:
                self._output = buzzer_class(pin, active_high=active_high)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_mock(self) -> bool:
        return self._mock

    def on(self) -> None:
        if self._output is not None:
            self._output.on()
        self._active = True

    def off(self) -> None:
        if self._output is not None:
            self._output.off()
        self._active = False

    def alarm(self) -> None:
        """Sound one alarm pulse without blocking the monitoring loop."""
        self.on()
        if self._mock:
            LOGGER.info("Mock buzzer alarm")

    def update(self, critical: bool, now: Optional[float] = None) -> None:
        """Run a repeatable alarm pattern from the main loop."""
        current = time.monotonic() if now is None else now
        if not hasattr(self, "_next_change"):
            self._next_change = current
        if critical and current >= self._next_change:
            if self.is_active:
                self.off()
                self._next_change = current + self.alarm_pause
            else:
                self.on()
                self._next_change = current + self.alarm_duration
        elif not critical and self.is_active:
            self.off()

    def test(self) -> None:
        self.alarm()
        time.sleep(self.alarm_duration)
        self.off()

    def close(self) -> None:
        self.off()
        if self._output is not None:
            self._output.close()
            self._output = None
