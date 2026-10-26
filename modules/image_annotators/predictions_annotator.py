
import cv2
import numpy as np

from modules.context import CaptureContext, PredictionContext
from modules.utilities.drawing import (
    HorizontalPosition,
    VerticalPosition,
    draw_text_on_image,
)


class PredictionsAnnotator:
    def __init__(
        self,
        panel_width: int = 350,
        bar_height: int = 14,
        row_height: int = 55,
    ):
        self.panel_width = panel_width
        self.bar_height = bar_height
        self.row_height = row_height

    def annotate_image(
        self,
        image: cv2.typing.MatLike,
        capture_context: CaptureContext,
    ) -> cv2.typing.MatLike:

        if not isinstance(capture_context, PredictionContext):
            return image

        height, _ = image.shape[:2]

        # Create a dark panel to the right of the image.
        panel = np.full(
            (height, self.panel_width, 3),
            (35, 35, 35),
            dtype=np.uint8,
        )

        padding = 15
        bar_width = self.panel_width - 2 * padding
        y = 10

        predictions = capture_context.model_prediction

        for index, action in enumerate(capture_context.actions):
            if y + self.row_height > height:
                break

            # Get the prediction probability.
            probability = (
                float(predictions[index])
                if predictions is not None and index < len(predictions)
                else 0.0
            )

            # Keep the probability within the expected range.
            probability = float(np.clip(probability, 0.0, 1.0))

            # Draw the action label and probability.
            text = f"{action}: {probability:.3f}"

            _, text_y = draw_text_on_image(
                text,
                panel,
                x=HorizontalPosition.LEFT,
                y=y,
                font_scale=1,
            )

            # Position the bar below the text.
            bar_y = min(
                text_y + 5,
                height - self.bar_height - 1,
            )

            # Draw the full background bar.
            cv2.rectangle(
                panel,
                (padding, bar_y),
                (padding + bar_width, bar_y + self.bar_height),
                (65, 65, 65),
                thickness=-1,
            )

            # Draw the foreground bar proportional to the probability.
            filled_width = int(bar_width * probability)

            if filled_width > 0:
                cv2.rectangle(
                    panel,
                    (padding, bar_y),
                    (padding + filled_width, bar_y + self.bar_height),
                    (70, 210, 100),
                    thickness=-1,
                )

            # Add vertical spacing before the next action.
            y += self.row_height

        # Combine the original image and the prediction panel.
        return np.hstack((image, panel))
