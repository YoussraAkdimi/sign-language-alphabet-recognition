from pathlib import Path
from typing import ClassVar

import numpy as np
from enum import Enum
from mediapipe.tasks.python.vision.holistic_landmarker import ( # otherwise type annotation fails
    HolisticLandmarkerResult,
)
from mediapipe.tasks.python.components.containers.landmark import (
    NormalizedLandmark,
)

class HolisticLandmarksFeature:
  
    # Left hand   21 * x * y * z =   63 values
    # Right hand  21 * x * y * z =   63 values
    # Pose        33 * x * y * z =   99 values
    # Face       478 * x * y * z = 1434 values
    #                              ----------- +
    #                              1659 values

    feature_length: ClassVar[int] = 1659
    
    class DataDimensions(Enum):
        LEFT_HAND = 21
        RIGHT_HAND = 21
        POSE=33
        FACE=478

    _NO_LEFT_HAND_DATA: ClassVar[np.ndarray] = np.zeros(DataDimensions.LEFT_HAND.value * 3)
    _NO_RIGHT_HAND_DATA: ClassVar[np.ndarray] = np.zeros(DataDimensions.RIGHT_HAND.value * 3)
    _NO_POSE_DATA: ClassVar[np.ndarray] = np.zeros(DataDimensions.POSE.value * 3)
    _NO_FACE_DATA: ClassVar[np.ndarray] = np.zeros(DataDimensions.FACE.value * 3)

    def __init__(self, data: np.ndarray | None = None):
        self._data = data

    def save(self, path: Path) -> None:
        np.save(path, self._data)

    @staticmethod
    def load(path: Path) -> "HolisticLandmarksFeature":
        data = np.load(path)
        return HolisticLandmarksFeature(data)
    
    def __len__(self) -> int:
        return self._data.shape[0] if self._data is not None else 0

    def get_data(self) -> np.ndarray:
        return self._data

    def apply_weights(self,
                    left_hand: float = 1.0,
                    right_hand: float = 1.0,
                    pose: float = 1.0,
                    face: float = 1.0) -> "HolisticLandmarksFeature":

        if self._data is None:
                return self

        weighted_data = self._data.copy()

        weights = [
            (self.DataDimensions.LEFT_HAND, left_hand),
            (self.DataDimensions.RIGHT_HAND, right_hand),
            (self.DataDimensions.POSE, pose),
            (self.DataDimensions.FACE, face),
        ]

        start = 0

        for dimension, weight in weights:
            end = start + dimension.value * 3
            weighted_data[start:end] *= weight
            start = end

        return HolisticLandmarksFeature(weighted_data)

    def center_at_pose(self) -> "HolisticLandmarksFeature":
        centered_data = self._data.copy()

        # Left hand
        lh_start = 0
        lh_end = lh_start + self.DataDimensions.LEFT_HAND.value * 3
        lh = centered_data[lh_start:lh_end].reshape(-1, 3)

        # Right hand
        rh_start = lh_end
        rh_end = rh_start + self.DataDimensions.RIGHT_HAND.value * 3
        rh = centered_data[rh_start:rh_end].reshape(-1, 3)

        # Pose
        pose_start = rh_end
        pose_end = pose_start + self.DataDimensions.POSE.value * 3
        pose = centered_data[pose_start:pose_end].reshape(-1, 3)

        # Face
        face_start = pose_end
        face_end = face_start + self.DataDimensions.FACE.value * 3
        face = centered_data[face_start:face_end].reshape(-1, 3)

        # Calculate global center using shoulder landmarks
        center = (pose[11] + pose[12]) / 2

        # Center all landmarks relative to the pose
        lh -= center
        rh -= center
        pose -= center
        face -= center

        return HolisticLandmarksFeature(centered_data)

    def normalize(self) -> "HolisticLandmarksFeature":
        if self._data is None:
            return self

        normalized_data = self._data.copy()

        # Extract pose landmarks
        pose_start = (
            self.DataDimensions.LEFT_HAND.value * 3
            + self.DataDimensions.RIGHT_HAND.value * 3
        )
        pose_end = pose_start + self.DataDimensions.POSE.value * 3

        pose = normalized_data[pose_start:pose_end].reshape(-1, 3)

        # Calculate shoulder width
        shoulder_width = np.linalg.norm(pose[11] - pose[12])

        # Avoid division by zero
        if shoulder_width < 1e-8:
            return self

        # Normalize all landmarks
        normalized_data /= shoulder_width

        return self.__class__(normalized_data)

    # z-normalization
    def standardize(self,
                    mean: np.ndarray,
                    std: np.ndarray) -> "HolisticLandmarksFeature":

        data = self._data.copy()

        data = (data - mean) / np.maximum(std, 1e-8)

        return self.__class__(data)
    
    @classmethod
    def from_detection_result(cls, detection_result: HolisticLandmarkerResult) -> "HolisticLandmarksFeature":
        """
        Converts a HolisticLandmarkerResult into a dictionary containing the relevant data.

        Args:
            detection_result: The result of the detection, which may contain landmarks or other information to convert.
        Returns:
            A dictionary containing the converted data.
        """
        result = cls()

        if detection_result is not None:
            # print(f"Length of pose landmarks {len(detection_result.pose_landmarks)}")
            # print(f"Length of face landmarks {len(detection_result.face_landmarks)}")
            # print(f"Length of left hand landmarks {len(detection_result.left_hand_landmarks)}")
            # print(f"Length of right hand landmarks {len(detection_result.right_hand_landmarks)}")

            lh_data = result._get_flattened_landmark_data(detection_result.left_hand_landmarks, cls._NO_LEFT_HAND_DATA)
            rh_data = result._get_flattened_landmark_data(detection_result.right_hand_landmarks, cls._NO_RIGHT_HAND_DATA)
            pose_data = result._get_flattened_landmark_data(detection_result.pose_landmarks, cls._NO_POSE_DATA)
            face_data = result._get_flattened_landmark_data(detection_result.face_landmarks, cls._NO_FACE_DATA)

            result._data = np.concat([lh_data, rh_data, pose_data, face_data])

        return result

    def _get_flattened_landmark_data(self, landmarks: list[NormalizedLandmark], default_data: list[float]) -> np.ndarray:
        flattened_data = (
            np.array([
                [landmark.x, landmark.y, landmark.z]
                for landmark in landmarks
            ]).flatten()
            if landmarks
            else default_data
        )

        return flattened_data
