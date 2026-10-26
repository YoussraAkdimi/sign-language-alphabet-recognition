from modules.image_annotators.composite_annotator import CompositeAnnotator

from modules.image_annotators import (CameraAnnotator, 
                                      InfoAnnotator, 
                                      HolisticLandmarkAnnotator,
                                      PredictionsAnnotator,
                                      AnnotatorTypes)
from modules.context import CaptureContext
import cv2.typing as cv2Typing

class RealLifeTestAnnotator:
    def __init__(self):
        self._composite_annotator = CompositeAnnotator.from_annotator_types(CameraAnnotator,
                                                                            InfoAnnotator,
                                                                            HolisticLandmarkAnnotator,
                                                                            PredictionsAnnotator)

    def annotate_image(self, current_image: cv2Typing.MatLike, capture_context: CaptureContext) -> cv2Typing.MatLike:
        return self._composite_annotator.annotate_image(current_image=current_image, capture_context=capture_context)
    
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
