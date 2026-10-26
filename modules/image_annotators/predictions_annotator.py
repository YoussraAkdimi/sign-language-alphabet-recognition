import cv2
import numpy as np

from modules.context import CaptureContext, PredictionContext
from modules.utilities.drawing import HorizontalPosition, VerticalPosition, draw_text_on_image

class PredictionsAnnotator:
    def __init__(self, panel_width: int = 350):
        self.panel_width = panel_width

    def annotate_image(self, image: cv2.typing.MatLike, capture_context: CaptureContext) -> cv2.typing.MatLike:

        if not isinstance(capture_context, PredictionContext):
            return image

        
        line_spacing = 3

        height, _ = image.shape[:2]

        # Create a panel to the right of the image.
        panel = np.full(
            (height, self.panel_width, 3),
            (35, 35, 35),
            dtype=np.uint8,
        )

        y = 10

        for index, action in enumerate(capture_context.actions):
            y = y + line_spacing  # next line y position (add margin)

            if y >= height:
                break

            prediction_percentage = capture_context.model_prediction[index] if capture_context.model_prediction is not None else 0.0

            text = (
                f"{action}: "
                f"{prediction_percentage:.3f}"
            )

            _, y = draw_text_on_image(text, 
                                panel,
                                x=HorizontalPosition.LEFT,
                                y=y,
                                font_scale=1.5)

        # Combine the original image and the panel.
        return np.hstack((image, panel))