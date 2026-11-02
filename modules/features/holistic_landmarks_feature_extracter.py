from pathlib import Path
from typing import ClassVar

import numpy as np
from .holistic_landmarks_feature import HolisticLandmarksFeature
from enum import Enum
from mediapipe.tasks.python.vision.holistic_landmarker import ( # otherwise type annotation fails
    HolisticLandmarkerResult,
)

class HolisticLandmarksFeatureExtracter:
    
    def __init__(self):
        pass

    """
    A class that converts a HolisticLandmarkerResult into a dictionary containing the relevant data.
    """

    def extract(self, detection_result: HolisticLandmarkerResult) -> HolisticLandmarksFeature:
        """
        Converts a HolisticLandmarkerResult into a dictionary containing the relevant data.

        Args:
            detection_result: The result of the detection, which may contain landmarks or other information to convert.
        Returns:
            A dictionary containing the converted data.
        """
        return HolisticLandmarksFeature.from_detection_result(detection_result)

    def load_feature(self, path: Path) -> HolisticLandmarksFeature:
        return HolisticLandmarksFeature.load(path)