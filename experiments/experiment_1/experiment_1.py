import sys
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import matplotlib.pyplot as plt

# add three levels of folders to the system path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.utilities.drawing import draw_landmarks_on_image

MODEL_PATH: str = '././models/hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("./test_images/woman_hands.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# Convert RGB to BGR for your helper function
image_bgr = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(
    image_bgr,
    detection_result
)

# Convert BGR back to RGB for Matplotlib
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# STEP 6: Display the annotated image
plt.imshow(annotated_image_rgb)
plt.axis("off")
plt.show()

detector.close()
