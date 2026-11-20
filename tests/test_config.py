import unittest

from config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_pi_friendly(self):
        settings = Settings()
        self.assertEqual((settings.camera_width, settings.camera_height), (640, 480))
        self.assertEqual(settings.buzzer_gpio, 18)


if __name__ == "__main__":
    unittest.main()
