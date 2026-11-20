import unittest

from hardware.buzzer_controller import BuzzerController


class BuzzerTests(unittest.TestCase):
    def test_mock_alarm_can_start_and_stop(self):
        buzzer = BuzzerController(18, mock=True)
        buzzer.update(True, now=0)
        self.assertTrue(buzzer.is_active)
        buzzer.update(False, now=1)
        self.assertFalse(buzzer.is_active)
        buzzer.close()


if __name__ == "__main__":
    unittest.main()
