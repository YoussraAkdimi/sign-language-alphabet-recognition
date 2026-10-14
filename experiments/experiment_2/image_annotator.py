import cv2

from modules.utilities.drawing import draw_landmarks_on_image

class image_annotator:
    def __init__(self, camera_number):
        self.camera_number = camera_number

    def annotate(self, image, detection_result):

        cv2.putText(image,                      # Image (NumPy array)
            f"Camera {self.camera_number}\nPress 'q' to quit",                 # Text
            (20, 40),                   # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,   # Font
            1,                          # Font scale
            (0, 255, 0),                # Color (B, G, R): green
            2,                          # Thickness
            cv2.LINE_AA                 # Anti-aliasing
        )

        if detection_result is not None:
        # Draw landmarks on the original frame
            image = draw_landmarks_on_image(image, detection_result)

        return image
