import sys
from pathlib import Path
import time

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
from modules.logic.video_loop import capture_video_data
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore

MEDIAPIPE_MODEL_PATH: str = '././models/holistic_landmarker.task'
DATA_PATH: Path = Path('./mp_data')
TENSORFLOW_MODEL_PATH: Path = Path('./output_models/action.h5')

actions = sorted(
    path.name for path in DATA_PATH.iterdir()
    if path.is_dir()
)

no_sequences = 30    # 30 videos worth of data
sequence_length = 30 # use 30 frames for an action

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
    base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5)

image_annotator = CompositeAnnotator.default_annotator()

cap, _, camera_name = camera_manager.start_camera()

capture_context = CaptureContext(camera_name)
feature_extracter = HolisticLandmarksFeatureExtracter()

sequence = []
sentence = []
threshold = 0.9

continue_processing = True

model = load_model(TENSORFLOW_MODEL_PATH)

# Optional: inspect the expected input shape
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

with HolisticLandmarker.create_from_options(options) as landmarker:

    while continue_processing:
        capture_video_data(cap, landmarker, image_annotator, capture_context)

        features: np.array = feature_extracter.extract(capture_context.detection_result)

        # Check that the returned number of features is the required length
        assert(len(features) == feature_extracter.feature_length)

        sequence.append(features)
        sequence = sequence[-30:]    # keep last thirty frames

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            #print(actions[np.argmax(res)])
            #print(res)

            if res[np.argmax(res)] > threshold:
                action = actions[np.argmax(res)]

                if len(sentence) > 0:
                    if sentence[-1] != action:
                        sentence.append(action)
                        print(sentence)
                else:
                    sentence.append(action)
                    print(sentence)
                                        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            continue_processing = False
            break
        if key == ord('c'):
            cap, _, camera_name = camera_manager.start_camera(camera_manager.next_camera_index)
            capture_context.camera_name = camera_name
        elif not image_annotator.handle_key_press(key):
            pass

# some cleanup code to release the camera and close the window
camera_manager.release()
cv2.destroyAllWindows()