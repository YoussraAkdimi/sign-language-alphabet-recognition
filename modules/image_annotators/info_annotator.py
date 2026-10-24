import cv2

from modules.context import CaptureContext
from modules.utilities.drawing import HorizontalPosition, VerticalPosition, draw_text_on_image

class InfoAnnotator:
    def annotate_image(
        self,
        image: cv2.typing.MatLike,
        capture_context: CaptureContext
    ) -> cv2.typing.MatLike:

        lines = [
            "'q': quit",
            "'l': toggle landmarks",
            "'f': toggle face blendshapes",
            "'c': change camera",
        ]

        draw_text_on_image(text=lines, 
                           image=image,
                           x=HorizontalPosition.LEFT,
                           y=VerticalPosition.TOP)

        return image
    