import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib import image
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cv2_enumerate_cameras import enumerate_cameras
from modules.utilities import CameraManager
from modules.context import PredictionContext
from modules.image_annotators import CompositeAnnotator
from modules.detection_result_converters import HolisticLandmarksFeatureExtracter
from modules.logic.video_loop import capture_video_data
from modules.protocols import GesturePredictionCollector
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
from tensorflow.keras.models import Model # type: ignore

MEDIAPIPE_MODEL_PATH: str = str(
    Path(__file__).resolve().parents[2] / "task_models" / "holistic_landmarker.task"
)

DATA_PATH: Path = Path('./mp_data')
TENSORFLOW_MODEL_PATH: Path = Path('./output_models/simple_model.h5')

def real_life_test(trained_model: Model, gesture_prediction_collector: GesturePredictionCollector, sequence_length: int= 30) -> None:
    actions = sorted(
        path.name for path in DATA_PATH.iterdir()
        if path.is_dir() and not path.name.startswith('_')
    )

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

    prediction_context = PredictionContext(camera_name, actions)

    print(f"Read actions: {prediction_context._actions}")

    feature_extracter = HolisticLandmarksFeatureExtracter()

    sequence = []
    sentence = []
    threshold = 0.4

    continue_processing = True

    # Optional: inspect the expected input shape
    print("Model input shape:", trained_model.input_shape)
    print("Model output shape:", trained_model.output_shape)

    try:
        with HolisticLandmarker.create_from_options(options) as landmarker:

            while continue_processing:
                capture_video_data(cap, landmarker, image_annotator, prediction_context)

                features: np.array = feature_extracter.extract(prediction_context.detection_result)

                # Check that the returned number of features is the required length
                assert(len(features) == feature_extracter.feature_length)

                sequence.append(features)
                sequence = sequence[-sequence_length:]    # keep last thirty frames

                if len(sequence) == sequence_length:
                    model_prediction_for_recent_sequence = trained_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                    prediction_context.result = gesture_prediction_collector.add_model_prediction(model_prediction_for_recent_sequence)

                    if prediction_context.result:
                        action = actions[np.argmax(prediction_context.result)]
                        print(action)
                        if action != "idle":
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
                    prediction_context.camera_name = camera_name
                elif not image_annotator.handle_key_press(key):
                    pass

    finally:
        # some cleanup code to release the camera and close the window
        camera_manager.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()