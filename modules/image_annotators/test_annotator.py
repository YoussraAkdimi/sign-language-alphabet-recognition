import cv2

from modules.dto import CaptureContext, TestContext
from modules.utilities.drawing import HorizontalPosition, VerticalPosition, draw_text_on_image

class TestAnnotator:
    def __init__(self):
        pass

    def annotate_image(self, image: cv2.typing.MatLike, capture_context: CaptureContext) -> cv2.typing.MatLike:

        if isinstance(capture_context, TestContext):
            height, width = image.shape[:2]

            text = f"{capture_context.result}"
            draw_text_on_image(text=text, 
                               image=image, 
                               x=HorizontalPosition.MIDDLE, 
                               y=VerticalPosition.TOP)

        return image