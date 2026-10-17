import cv2

from modules.dto.capture_context import CaptureContext
from modules.utilities.drawing import draw_holistic_landmarks_on_image
from mediapipe.tasks.python.vision.holistic_landmarker import ( # otherwise type annotation fails
    HolisticLandmarkerResult,
)

class HolisticLandmarkAnnotator:
    def __init__(self):
        pass

    def annotate_image(self, image: cv2.typing.MatLike, capture_context: CaptureContext) -> cv2.typing.MatLike:

        # cv2.putText(image,                      # Image (NumPy array)
        #     f"Camera {self.camera_number}\nPress 'q' to quit",                 # Text
        #     (20, 40),                   # Position (x, y)
        #     cv2.FONT_HERSHEY_SIMPLEX,   # Font
        #     1,                          # Font scale
        #     (0, 255, 0),                # Color (B, G, R): green
        #     2,                          # Thickness
        #     cv2.LINE_AA                 # Anti-aliasing
        # )

        # Draw pose landmarks if there are any detected
        if capture_context.detection_result and capture_context.detection_result.pose_landmarks:
            image = draw_holistic_landmarks_on_image(
                image, capture_context.detection_result
            )

        return image
