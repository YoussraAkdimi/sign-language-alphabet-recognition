import cv2

from modules.dto import CaptureContext
from modules.utilities.drawing import draw_holistic_landmarks_on_image
# from mediapipe.tasks.python.vision.holistic_landmarker import ( # otherwise type annotation fails
#     HolisticLandmarkerResult,
# )

class CameraAnnotator:
    def __init__(self):
        pass

    def annotate_image(self, image: cv2.typing.MatLike, capture_context: CaptureContext) -> cv2.typing.MatLike:

        height, width = image.shape[:2]

        cv2.putText(
            image,
            f"{capture_context.camera_name}",
            (20, height - 20),  # Bottom-left, 20 px from the edges
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        return image