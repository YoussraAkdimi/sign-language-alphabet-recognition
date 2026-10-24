import cv2

from modules.context import CaptureContext, TrainingContext
from modules.utilities.drawing import HorizontalPosition, VerticalPosition, draw_holistic_landmarks_on_image, draw_text_on_image

class TrainingAnnotator:
    def __init__(self):
        pass

    def annotate_image(self, image: cv2.typing.MatLike, capture_context: CaptureContext) -> cv2.typing.MatLike:

        if isinstance(capture_context, TrainingContext):
            height, width = image.shape[:2]

            text = f"{capture_context.action}/{capture_context.sequence}/{capture_context.frame}"
            draw_text_on_image(text=text, 
                               image=image, 
                               x=HorizontalPosition.MIDDLE, 
                               y=VerticalPosition.TOP)

            if capture_context.countdown_value > 0:
                text = [capture_context.action, str(capture_context.countdown_value)]
                draw_text_on_image(text=text, 
                                   image=image, 
                                   x=HorizontalPosition.MIDDLE, 
                                   y=VerticalPosition.MIDDLE,
                                   margin_y=-20,
                                   font_scale=10)

        return image