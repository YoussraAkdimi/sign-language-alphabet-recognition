from dataclasses import dataclass
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
from modules.utilities.drawing import HorizontalPosition, VerticalPosition, draw_holistic_landmarks_on_image, draw_text_on_image
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from modules.features import FeatureTransformer
from modules.protocols import ExtractFeaturesFromDetectionResult

@dataclass
class ReconstructedHolisticLandmarks:
    left_hand_landmarks: list[NormalizedLandmark] # type: ignore
    right_hand_landmarks: list[NormalizedLandmark] # type: ignore
    pose_landmarks: list[NormalizedLandmark] # type: ignore
    face_landmarks: list[NormalizedLandmark] # type: ignore
    
def _reconstruct_landmarks(data: np.ndarray,) -> ReconstructedHolisticLandmarks:

    def convert(points: np.ndarray) -> list[NormalizedLandmark]: # type: ignore
        if np.all(points == 0):
            return []

        return [
            NormalizedLandmark(
                x=float(x),
                y=float(y),
                z=float(z),
            )
            for x, y, z in points
        ]

    return ReconstructedHolisticLandmarks(
        left_hand_landmarks=convert(data[0:63].reshape(21, 3)),
        right_hand_landmarks=convert(data[63:126].reshape(21, 3)),
        pose_landmarks=convert(data[126:225].reshape(33, 3)),
        face_landmarks=convert(data[225:1659].reshape(478, 3)),
    )

MODEL_PATH: str = '././models/holistic_landmarker.task'

def show_trained_data(data_path: Path, 
                      feature_extracter: ExtractFeaturesFromDetectionResult,
                      feature_transformer: FeatureTransformer) -> None:
    
    continue_playing = True

    try:
        for action_path in sorted(data_path.iterdir()):
            if not action_path.is_dir() or not continue_playing:
                continue

            for sequence_path in sorted(action_path.iterdir()):
                if (
                    not action_path.is_dir() or 
                    action_path.name.startswith('_') or
                    not continue_playing
                ):
                    continue

                # start playing
                for frame_path in sorted(sequence_path.iterdir()):
                    if not frame_path.is_file or not continue_playing:
                        continue

                    # load the feature data from file
                    feature = feature_extracter.load_feature(frame_path)

                    # transform (if any defined) the feature
                    feature = feature_transformer.transform(feature)
                    
                    data = feature.get_data()

                    landmarks = _reconstruct_landmarks(data)

                    # Use an RGB image, as expected by your drawing function
                    rgb_image = np.zeros((720, 1280, 3), dtype=np.uint8)

                    annotated_image = draw_holistic_landmarks_on_image(
                        rgb_image,
                        landmarks,
                    )

                    text = f"{action_path.name}/{sequence_path.name}/{frame_path.name}"
                    draw_text_on_image(text=text, 
                                    image=annotated_image, 
                                    x=HorizontalPosition.MIDDLE, 
                                    y=VerticalPosition.TOP)

                    cv2.imshow("Reconstructed Holistic Landmarks", annotated_image)

                    key = cv2.waitKey(100) & 0xFF

                    if key == ord('q'):
                        continue_playing = False
                        break

    # some cleanup code to release the camera and close the window
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()