import numpy as np
from enum import Enum
from mediapipe.tasks.python.vision.holistic_landmarker import ( # otherwise type annotation fails
    HolisticLandmarkerResult,
)
from mediapipe.tasks.python.components.containers.landmark import (
    NormalizedLandmark,
)

class HolisticLandmarksFeatureExtracter:

    class DataDimensions(Enum):
        LEFT_HAND = 21
        RIGHT_HAND = 21
        POSE=33
        FACE=478

    def __init__(self):
        self._no_left_hand_data = np.zeros(self.DataDimensions.LEFT_HAND.value*3)
        self._no_right_hand_data = np.zeros(self.DataDimensions.RIGHT_HAND.value*3)
        self._no_pose_data = np.zeros(self.DataDimensions.POSE.value*3)
        self._no_face_data = np.zeros(self.DataDimensions.FACE.value*3)

    """
    A class that converts a HolisticLandmarkerResult into a dictionary containing the relevant data.
    """

    def convert(self, detection_result: HolisticLandmarkerResult) -> dict:
        """
        Converts a HolisticLandmarkerResult into a dictionary containing the relevant data.

        Args:
            detection_result: The result of the detection, which may contain landmarks or other information to convert.
        Returns:
            A dictionary containing the converted data.
        """
        if detection_result is None:
            return {}

        # print(f"Length of pose landmarks {len(detection_result.pose_landmarks)}")
        # print(f"Length of face landmarks {len(detection_result.face_landmarks)}")
        # print(f"Length of left hand landmarks {len(detection_result.left_hand_landmarks)}")
        # print(f"Length of right hand landmarks {len(detection_result.right_hand_landmarks)}")

        lh_data = self._get_flattened_landmark_data(detection_result.left_hand_landmarks, self._no_left_hand_data)
        rh_data = self._get_flattened_landmark_data(detection_result.right_hand_landmarks, self._no_right_hand_data)
        pose_data = self._get_flattened_landmark_data(detection_result.pose_landmarks, self._no_pose_data)
        face_data = self._get_flattened_landmark_data(detection_result.face_landmarks, self._no_face_data)

        return np.concat((lh_data, rh_data, pose_data, face_data))

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
