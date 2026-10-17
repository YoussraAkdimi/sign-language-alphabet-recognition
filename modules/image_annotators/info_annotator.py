import cv2

from modules.utilities.drawing import draw_holistic_landmarks_on_image
from mediapipe.tasks.python.vision.holistic_landmarker import ( # otherwise type annotation fails
    HolisticLandmarkerResult,
)

class InfoAnnotator:
    def __init__(self):
        pass

    def annotate_image(self, image: cv2.typing.MatLike, detection_result: HolisticLandmarkerResult) -> cv2.typing.MatLike:

        cv2.putText(image,                      # Image (NumPy array)
            f"Press 'q' to quit\nPress 'l' to toggle landmarks",                 # Text
            (20, 40),                   # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,   # Font
            1,                          # Font scale
            (0, 255, 0),                # Color (B, G, R): green
            2,                          # Thickness
            cv2.LINE_AA                 # Anti-aliasing
        )

        return image
