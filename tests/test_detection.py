import unittest

import numpy as np

from detection.unified_detector import calculate_ear


class DetectionTests(unittest.TestCase):
    def test_zero_width_eye_is_safe(self):
        points = np.zeros((6, 2), dtype=float)
        self.assertEqual(calculate_ear(points), 0.0)


if __name__ == "__main__":
    unittest.main()
