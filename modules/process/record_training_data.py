import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matplotlib import image
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cv2_enumerate_cameras import enumerate_cameras
from modules.utilities import CameraManager
from modules.context import TrainingContext
from modules.image_annotators import CompositeAnnotator
from modules.features import HolisticLandmarksFeatureExtracter
from modules.logic.video_loop import capture_video_data
from modules.protocols import ExtractFeaturesFromDetectionResult

TASK_MODEL_PATH: str = './task_models/holistic_landmarker.task'

actions = ["idle"] #['slapen', 'vogel', 'poes' ]

def record_training_data(output_data_path: Path, 
                         actions: list[str], 
                         feature_extracter: ExtractFeaturesFromDetectionResult,
                         no_sequences: int = 30, 
                         no_of_frames: int = 30) -> Path:

    for action in actions:
        for sequence in range(no_sequences):
            sequence_path = output_data_path / action.zfill(3) / str(sequence).zfill(3)
            sequence_path.mkdir(parents=True, exist_ok=True)

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
        base_options=BaseOptions(model_asset_path=TASK_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5)

    image_annotator = CompositeAnnotator.default_annotator()

    cap, _, camera_name = camera_manager.start_camera()

    capture_context = TrainingContext(camera_name)
    #feature_extracter = HolisticLandmarksFeatureExtracter()

    continue_processing = True

    try:
        with HolisticLandmarker.create_from_options(options) as landmarker:
            for capture_context.action in actions:
                if not continue_processing:
                    break

                for capture_context.sequence in range(no_sequences):
                    if not continue_processing:
                        break

                    # Countdown before capturing the sequence
                    countdown_seconds = 3
                    for remaining in range(countdown_seconds, 0, -1):
                        capture_context.countdown_value = remaining
                        start_time = time.time()

                        while time.time() - start_time < 1:
                            capture_video_data(cap, landmarker, image_annotator, capture_context)
                            key = cv2.waitKey(1) & 0xFF

                            if key == ord('q'):
                                continue_processing = False
                                break

                    capture_context.countdown_value = 0
                    
                    # start capturing
                    for capture_context.frame in range(no_of_frames):

                            capture_video_data(cap, landmarker, image_annotator, capture_context)

                            feature = feature_extracter.extract(capture_context.detection_result)

                            # save the features in a file
                            file_path = output_data_path / capture_context.action / str(capture_context.sequence).zfill(3) / str(capture_context.frame).zfill(3)
                            
                            feature.save(file_path)

                            key = cv2.waitKey(1) & 0xFF

                            if key == ord('q'):
                                continue_processing = False
                                break
                            if key == ord('c'):
                                cap, _, camera_name = camera_manager.start_camera(camera_manager.next_camera_index)
                                capture_context.camera_name = camera_name
                            elif not image_annotator.handle_key_press(key):
                                pass

    finally:
        camera_manager.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()