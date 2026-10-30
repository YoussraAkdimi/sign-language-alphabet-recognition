from pathlib import Path
from typing import ClassVar, Protocol, TYPE_CHECKING
import cv2.typing as cv2Typing
from tensorflow.keras.models import Model # type: ignore
import numpy as np

# if TYPE_CHECKING:
from ..context.capture_context import CaptureContext

class ImageAnnotator(Protocol):
    """Writes contextual information onto an existing image, 
    e.g. camera number or detected landmarks.
    """
    def annotate_image(self, current_image: cv2Typing.MatLike, capture_context: CaptureContext) -> cv2Typing.MatLike:
        """Writes contextual information onto an existing image, 
        e.g. camera number or detected landmarks.
        
        Args:
            current_image: The image to annotate.
            detection_result: The result of the detection, which may contain landmarks or other information to annotate.
        Returns:
            The annotated image.
        """
        ...

class Feature(Protocol):
    def save(path: Path) -> None:
        ...

    def get_data(self) -> np.ndarray:
        ...

    def apply_weights(self, left_hand: float, right_hand: float, pose: float, face: float) -> "Feature":
        ...

class ExtractFeaturesFromDetectionResult(Protocol):
    feature_length: ClassVar[int]
    
    """Converts a detection result into a data structure suitable for further processing or analysis."""
    def extract(self, detection_result) -> Feature:
        """Converts a detection result into a data structure suitable for further processing or analysis.
        
        Args:
            detection_result: The result of the detection, which may contain landmarks or other information to convert.
        Returns:
            A dictionary containing the converted data.
        """
        ...

    def load_feature(self, path: Path) -> Feature:
        ...

class ModelFactory(Protocol):
    def create(self, input_shape: tuple, action_length: int) -> Model:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def nr_of_epochs(self) -> int:
        ...

class GesturePredictionCollector(Protocol):
    def add_model_prediction(self, prediction: np.ndarray) -> int | None:
        ...

class FeatureTransform(Protocol):
    def transform(self, feature: Feature) -> Feature:
        ...