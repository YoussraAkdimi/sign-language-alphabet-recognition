import cv2
import numpy as np

from mediapipe.tasks.python.vision.holistic_landmarker import (
    HolisticLandmarkerResult,
)

from modules.dto import CaptureContext


class FaceBlendshapesAnnotator:
    def __init__(self, panel_width: int = 350):
        self.panel_width = panel_width

    def annotate_image(
        self,
        image: cv2.typing.MatLike,
        capture_context: CaptureContext,
    ) -> cv2.typing.MatLike:

        height, width = image.shape[:2]

        # Create a panel to the right of the image.
        panel = np.full(
            (height, self.panel_width, 3),
            (35, 35, 35),
            dtype=np.uint8,
        )

        # Add a heading.
        cv2.putText(
            panel,
            "Face Blendshapes",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if capture_context.detection_result and capture_context.detection_result.face_blendshapes:
            categories = capture_context.detection_result.face_blendshapes

            categories = sorted(
                categories,
                key=lambda category: category.category_name,
                reverse=True,
            )

            for index, category in enumerate(categories):
                y = 65 + index * 28

                if y >= height:
                    break

                text = (
                    f"{category.category_name}: "
                    f"{category.score:.3f}"
                )

                cv2.putText(
                    panel,
                    text,
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Combine the original image and the panel.
        return np.hstack((image, panel))