import sys
from pathlib import Path

# add three levels of folders to the system path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib import image
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cv2_enumerate_cameras import enumerate_cameras
import image_annotator

from modules.utilities.drawing import draw_hand_landmarks_on_image

# get a list of the available cameras and check if there are any
cameras = enumerate_cameras(cv2.CAP_AVFOUNDATION)

if len(cameras) > 0:
    print(f"Found {len(cameras)} camera(s).")

    for camera in cameras:
        print(f"Index: {camera.index}, Name: {camera.name}")
else:
    print("No cameras found.")
    exit(1)

# choose the first camera in the list (index 0) for video capture
camera_index = cameras[0].index

MODEL_PATH: str = '././models/hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

print("Camera opened:", cap.isOpened())

image_annotator = image_annotator.image_annotator(camera_number=camera_index)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe image from the RGB frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect hand landmarks
    detection_result = detector.detect(mp_image)

    # annotate the frame with camera number and landmarks
    annotated_frame = image_annotator.annotate(frame, detection_result=detection_result)  # Annotate the frame with camera number

    # Display the annotated frame
    cv2.imshow('Video capture', annotated_frame)

    # wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# some cleanup code to release the camera and close the window
cap.release()
cv2.destroyAllWindows()