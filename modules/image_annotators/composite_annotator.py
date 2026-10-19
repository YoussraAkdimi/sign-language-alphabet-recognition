from __future__ import annotations

from enum import Enum
from modules.dto import CaptureContext
from modules.image_annotators.camera_annotator import CameraAnnotator
from modules.image_annotators.info_annotator import InfoAnnotator
from modules.image_annotators.training_annotator import TrainingAnnotator
from modules.image_annotators.holistic_landmark_annotator import HolisticLandmarkAnnotator
from modules.image_annotators.face_blendshapes_annotator import FaceBlendshapesAnnotator

from modules.protocols import ImageAnnotator

import cv2.typing as cv2Typing
 # import like this otherwise type annotation fails
from mediapipe.tasks.python.vision.holistic_landmarker import (
    HolisticLandmarkerResult,
)

class AnnotatorTypes(Enum):
    CAMERA_ANNOTATOR = CameraAnnotator
    HOLISTIC_LANDMARK_ANNOTATOR = HolisticLandmarkAnnotator
    FACE_BLENDSHAPES_ANNOTATOR = FaceBlendshapesAnnotator
    INFO_ANNOTATOR = InfoAnnotator
    TRAINING_ANNOTATOR = TrainingAnnotator
    
class CompositeAnnotator:
    def __init__(self):
        self.__annotators = {}

    def add_annotator(self, annotator: ImageAnnotator, enabled: bool = True) -> None:
        self.__annotators[annotator] = enabled

    def annotate_image(self, current_image: cv2Typing.MatLike, capture_context: CaptureContext) -> cv2Typing.MatLike:
        for annotator, enabled in self.__annotators.items():
            if enabled:
                current_image = annotator.annotate_image(current_image, capture_context)
        
        return current_image

    def handle_key_press(self, key: int) -> bool:

        # Handle key press events to toggle annotators
        if key == ord('l'):
            self.toggle_annotator(AnnotatorTypes.HOLISTIC_LANDMARK_ANNOTATOR)
            return True
        elif key == ord('f'):
            self.toggle_annotator(AnnotatorTypes.FACE_BLENDSHAPES_ANNOTATOR)
            return True
        
        # Key not handled
        return False

    def toggle_annotator(self, annotator_type: AnnotatorTypes) -> None:
        for annotator, enabled in self.__annotators.items():
            print(f"Checking annotator: {annotator} of type {type(annotator)} against {annotator_type.value}")
            if isinstance(annotator, annotator_type.value):
                self._toggle_annotator(annotator)

    def _enable_annotator(self, annotator: ImageAnnotator, enabled: bool = True) -> None:
        if annotator in self.__annotators:
            self.__annotators[annotator] = enabled

    def _toggle_annotator(self, annotator: ImageAnnotator) -> None:
        if annotator in self.__annotators:
            self.__annotators[annotator] = not self.__annotators[annotator]

    @classmethod
    def default_annotator(cls) -> CompositeAnnotator:
        composite_annotator = cls()

        for annotator_type in AnnotatorTypes:
            composite_annotator.add_annotator(
                annotator_type.value(),
                enabled=True,
            )

        return composite_annotator
