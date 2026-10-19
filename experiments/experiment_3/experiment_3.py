import sys
from pathlib import Path

# add three levels of folders to the system path so that
# we can resolve modules from other paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib import image
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cv2_enumerate_cameras import enumerate_cameras
from modules.utilities import CameraManager
from modules.dto import CaptureContext
from modules.image_annotators import CompositeAnnotator
from modules.detection_result_converters import HolisticLandmarksFeatureExtracter

MODEL_PATH: str = '././models/holistic_landmarker.task'

print(f"MediaPipe version: {mp.__version__}")

camera_manager = CameraManager()

if not camera_manager.has_cameras:
    print("No cameras found.")
    exit(1)

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5)


image_annotator = CompositeAnnotator.default_annotator()

cap, _, camera_name = camera_manager.start_camera()

capture_context = CaptureContext(camera_name)
feature_extracter = HolisticLandmarksFeatureExtracter()

with HolisticLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
        else:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a MediaPipe image from the RGB frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Detect hand landmarks
            detection_result = landmarker.detect_for_video(
                mp_image,
                capture_context.timestamp_ms
            )

            capture_context.detection_result = detection_result

            # annotate the frame with camera number and landmarks
            annotated_frame = image_annotator.annotate_image(frame, capture_context=capture_context)  # Annotate the frame with camera number

            features: np.array = feature_extracter.extract(capture_context.detection_result)

            # Check that the returned number of features is the required length
            assert(len(features) == feature_extracter.feature_length)

            print(features.shape)
            # if (capture_context.timestamp_ms > 4000):
            #     print(converted_data)
            #     exit(0)

            # Display the annotated frame
            cv2.imshow('Video capture', annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('c'):
            cap, _, camera_name = camera_manager.start_camera(camera_manager.next_camera_index)
            capture_context.camera_name = camera_name
        elif not image_annotator.handle_key_press(key):
            pass

# some cleanup code to release the camera and close the window
camera_manager.release()
cv2.destroyAllWindows()