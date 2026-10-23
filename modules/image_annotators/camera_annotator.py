import cv2

from modules.dto import CaptureContext
from modules.utilities.drawing import HorizontalPosition, VerticalPosition, draw_holistic_landmarks_on_image, draw_text_on_image

class CameraAnnotator:
    def __init__(self):
        pass

    def annotate_image(self, image: cv2.typing.MatLike, capture_context: CaptureContext) -> cv2.typing.MatLike:

        draw_text_on_image(text=capture_context.camera_name,
                           image=image,
                           x=HorizontalPosition.LEFT,
                           y=VerticalPosition.BOTTOM)
        
        return image