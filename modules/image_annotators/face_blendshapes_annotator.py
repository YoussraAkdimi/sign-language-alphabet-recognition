import cv2
import numpy as np

from mediapipe.tasks.python.vision.holistic_landmarker import (
    HolisticLandmarkerResult,
)

from modules.dto import CaptureContext
from modules.utilities.drawing import HorizontalPosition, VerticalPosition, draw_text_on_image


class FaceBlendshapesAnnotator:
    def __init__(self, panel_width: int = 350):
        self.panel_width = panel_width

    def annotate_image(
        self,
        image: cv2.typing.MatLike,
        capture_context: CaptureContext,
    ) -> cv2.typing.MatLike:
        line_spacing = 3

        height, _ = image.shape[:2]

        # Create a panel to the right of the image.
        panel = np.full(
            (height, self.panel_width, 3),
            (35, 35, 35),
            dtype=np.uint8,
        )

        _, y = draw_text_on_image("Face Blendshapes",
                           panel,
                           x=HorizontalPosition.LEFT,
                           y=VerticalPosition.TOP,
                           font_scale=0.7)

        y = y + line_spacing*3

        if capture_context.detection_result and capture_context.detection_result.face_blendshapes:
            categories = capture_context.detection_result.face_blendshapes

            categories = sorted(
                categories,
                key=lambda category: category.category_name,
                reverse=True,
            )

            for category in categories:
                y = y + line_spacing  # next line y position (add margin)

                if y >= height:
                    break

                text = (
                    f"{category.category_name}: "
                    f"{category.score:.3f}"
                )

                _, y = draw_text_on_image(text, 
                                   panel,
                                   x=HorizontalPosition.LEFT,
                                   y=y,
                                   font_scale=0.5)

        # Combine the original image and the panel.
        return np.hstack((image, panel))