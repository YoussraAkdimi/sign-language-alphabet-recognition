import cv2

from modules.dto import CaptureContext, TrainingContext
from modules.utilities.drawing import draw_holistic_landmarks_on_image

class TrainingAnnotator:
    def __init__(self):
        pass

    def annotate_image(self, image: cv2.typing.MatLike, capture_context: CaptureContext) -> cv2.typing.MatLike:

        if isinstance(capture_context, TrainingContext):
            height, width = image.shape[:2]

            text = f"{capture_context.action}/{capture_context.sequence}/{capture_context.frame}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # locate the center of the screen
            x = (width - text_width) // 2
            y = 40

            cv2.putText(
                image,
                text,
                (x, y),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )

            if capture_context.countdown_value > 0:
                text = str(capture_context.countdown_value)
                font_scale = 10

                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )

                # locate the center of the screen
                x = (width - text_width) // 2
                y = (height + text_height) // 2

                cv2.putText(
                    image,
                    text,
                    (x, y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                    cv2.LINE_AA,
                )

        return image