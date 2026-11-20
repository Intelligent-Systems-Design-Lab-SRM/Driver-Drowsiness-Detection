"""OpenCV camera lifecycle wrapper."""

import cv2


class CameraController:
    def __init__(self, index: int, width: int, height: int):
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()
