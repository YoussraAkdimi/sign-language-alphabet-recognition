from typing import Protocol, TYPE_CHECKING
import cv2.typing as cv2Typing

# if TYPE_CHECKING:
from ..dto.capture_context import CaptureContext

class ImageAnnotator(Protocol):
    """Writes contextual information onto an existing image, 
    e.g. camera number or detected landmarks.
    """
    def annotate_image(self, current_image: cv2Typing.MatLike, capture_context: CaptureContext) -> cv2Typing.MatLike:
        """Writes contextual information onto an existing image, 
        e.g. camera number or detected landmarks.
        
        Args:
            current_image: The image to annotate.
            detection_result: The result of the detection, which may contain landmarks or other information to annotate.
        Returns:
            The annotated image.
        """
        ...

class DetectionResultToDataConverter(Protocol):
    """Converts a detection result into a data structure suitable for further processing or analysis."""
    def convert(self, detection_result) -> dict:
        """Converts a detection result into a data structure suitable for further processing or analysis.
        
        Args:
            detection_result: The result of the detection, which may contain landmarks or other information to convert.
        Returns:
            A dictionary containing the converted data.
        """
        ...
