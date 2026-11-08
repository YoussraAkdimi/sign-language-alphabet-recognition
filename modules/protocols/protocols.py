from pathlib import Path
from typing import ClassVar, Protocol, TYPE_CHECKING
import cv2.typing as cv2Typing # type: ignore
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
    """
    Host all data and operations on a single feature, which is a collection of landmarks or other data points.
    """
    def save(path: Path) -> None:
        ...

    def get_data(self) -> np.ndarray:
        ...

    def apply_weights(self, left_hand: float, right_hand: float, pose: float, face: float) -> "Feature":
        ...

    def center_at_pose(self) -> "Feature":
        ...

    def normalize(self) -> "Feature":
        ...

    def remove_leg_data(self) -> "Feature":
        ...

    def remove_z_coordinates(self) -> " Feature":
        ...

class ExtractFeaturesFromDetectionResult(Protocol):
    """
    Converts a detection result into a data structure suitable for further processing or analysis.
    """

    feature_length: ClassVar[int]
    
    def extract(self, detection_result) -> Feature:
        """
        Converts a detection result into a data structure suitable for further processing or analysis.

        Args:
            detection_result: The result of the detection, which may contain landmarks or other information to convert.
        Returns:
            A dictionary containing the converted data.
        """
        ...

    def load_feature(self, path: Path) -> Feature:
        """
        Loads a feature from a file.

        Args:
            path: The path to the file containing the feature data.

        Returns:
            The loaded feature.
        """
        ...

class ModelFactory(Protocol):
    """
    A factory for creating machine learning models.
    """
    def create(self, input_shape: tuple, action_length: int) -> Model:
        ...

    @property
    def name(self) -> str:
        """
        Gets the name of the model.
        """
        ...

    @property
    def nr_of_epochs(self) -> int:
        """
        Gets the number of epochs for training the model.
        """
        ...

class GesturePredictionCollector(Protocol):
    """
    Collects predictions from a gesture recognition model and provides methods to access the collected predictions.
    """
    def add_model_prediction(self, prediction: np.ndarray) -> int | None:
        ...

    def reset(self) -> None:
        ...

class FeatureTransform(Protocol):
    def transform(self, feature: Feature) -> Feature:
        ...