from typing import Protocol

import cv2.typing as cv2Typing
from mediapipe.tasks.python.vision import HolisticLandmarkerResult

class ImageAnnotator(Protocol):
    """Writes contextual information onto an existing image, 
    e.g. camera number or detected landmarks.
    """
    def annotate_image(self, current_image: cv2Typing.MatLike, detection_result: HolisticLandmarkerResult) -> cv2Typing.MatLike:
        """Writes contextual information onto an existing image, 
        e.g. camera number or detected landmarks.
        
        Args:
            current_image: The image to annotate.
            detection_result: The result of the detection, which may contain landmarks or other information to annotate.
        Returns:
            The annotated image.
        """
        ...

