import cv2

from modules.context.capture_context import CaptureContext
from modules.utilities.drawing import draw_holistic_landmarks_on_image
from mediapipe.tasks.python.vision.holistic_landmarker import ( # otherwise type annotation fails
    HolisticLandmarkerResult,
)

class HolisticLandmarkAnnotator:
    def __init__(self):
        pass

    def annotate_image(self, image: cv2.typing.MatLike, capture_context: CaptureContext) -> cv2.typing.MatLike:

        # Draw pose landmarks if there are any detected
        if capture_context.detection_result and capture_context.detection_result.pose_landmarks:
            image = draw_holistic_landmarks_on_image(
                image, capture_context.detection_result
            )

        return image
