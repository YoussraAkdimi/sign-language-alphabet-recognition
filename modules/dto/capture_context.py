import cv2.typing as cv2Typing
from mediapipe.tasks.python.vision.holistic_landmarker import (
    HolisticLandmarkerResult,
)

class CaptureContext:
    """
    A class that encapsulates the context of a capture session, including the camera number,
    the current frame, and the timestamp in milliseconds.
    """

    def __init__(self, camera_name: str):
        self._camera_name = camera_name
        self._timestamp_ms = 0
        self._detection_result = None

    @property
    def camera_name(self) -> str:
        """
        Get the name of the camera being used for capture.

        Returns:
            The name of the camera.
        """
        return self._camera_name

    @camera_name.setter
    def camera_name(self, value: str) -> None:
        """
        Set the name of the camera being used for capture.

        Args:
            value: The new name of the camera.
        """
        self._camera_name = value
        
    @property
    def detection_result(self) -> HolisticLandmarkerResult:
        """
        Get the current detection result.

        Returns:
            The current detection result.
        """
        return self._detection_result
    
    @detection_result.setter
    def detection_result(self, value: HolisticLandmarkerResult) -> None:
        """
        Set the current detection result.

        Args:
            value: The new detection result.
        """
        self._detection_result = value

    @property
    def timestamp_ms(self) -> int:
        """
        Get the current timestamp in milliseconds.

        Returns:
            The current timestamp in milliseconds.
        """
        return self._timestamp_ms

    @timestamp_ms.setter
    def timestamp_ms(self, value: int) -> None:
        """
        Set the current timestamp in milliseconds.

        Args:
            value: The new timestamp in milliseconds.
        """
        self._timestamp_ms = value
