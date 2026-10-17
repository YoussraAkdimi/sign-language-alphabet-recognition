import cv2

from modules.dto import CaptureContext

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

        x, y = 20, 50
        line_height = 30
        padding = 10

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        # Calculate the background rectangle dimensions
        text_sizes = [
            cv2.getTextSize(line, font, font_scale, thickness)[0]
            for line in lines
        ]

        max_width = max(width for width, height in text_sizes)
        text_height = text_sizes[0][1]

        rect_x1 = x - padding
        rect_y1 = y - text_height - padding
        rect_x2 = x + max_width + padding
        rect_y2 = y + (len(lines) - 1) * line_height + padding

        # Draw a filled background rectangle
        cv2.rectangle(
            image,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            (40, 40, 40),  # Dark gray background
            cv2.FILLED
        )

        # Draw the text over the rectangle
        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (x, y + i * line_height),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA
            )

        return image