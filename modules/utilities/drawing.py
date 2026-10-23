#@markdown We implemented some functions to visualize the holistic landmark detection results.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum, auto

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

class VerticalPosition(Enum):
    TOP = auto()
    MIDDLE = auto()
    BOTTOM = auto()

class HorizontalPosition(Enum):
    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()

def draw_holistic_landmarks_on_image(rgb_image, detection_result):
    # pose_landmarks = detection_result["pose_landmarks"]
    # face_landmarks = detection_result["face_landmarks"]
    # left_hand_landmarks = detection_result["left_hand_landmarks"]
    # right_hand_landmarks = detection_result["right_hand_landmarks"]    
    pose_landmarks = detection_result.pose_landmarks
    face_landmarks = detection_result.face_landmarks
    left_hand_landmarks = detection_result.left_hand_landmarks
    right_hand_landmarks = detection_result.right_hand_landmarks

    annotated_image = np.copy(
        rgb_image[:, :, :3] if rgb_image.shape[2] == 4 else rgb_image
    )

    # Draw pose landmarks
    if pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2
            )
        )

    # Draw face landmarks
    if face_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
        )

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
        )

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    # Draw hand landmarks and handedness labels
    height, width, _ = annotated_image.shape

    hands = [
        (left_hand_landmarks, "Left"),
        (right_hand_landmarks, "Right")
    ]

    for hand_landmarks, hand_name in hands:
        if hand_landmarks:
            # Draw hand landmarks
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
            )

            # Calculate the top-left corner of the hand's bounding box
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]

            text_x = int(min(x_coordinates) * width)
            text_y = max(0, int(min(y_coordinates) * height) - MARGIN)

            # Draw the hand name
            cv2.putText(
                annotated_image,
                hand_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA
            )

    return annotated_image
  
# def plot_face_blendshapes_bar_graph(face_blendshapes):
#   # Extract the face blendshapes category names and scores.
#   face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
#   face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
#   # The blendshapes are ordered in decreasing score value.
#   face_blendshapes_ranks = range(len(face_blendshapes_names))

#   fig, ax = plt.subplots(figsize=(12, 12))
#   bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
#   ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
#   ax.invert_yaxis()

#   # Label each bar with values
#   for score, patch in zip(face_blendshapes_scores, bar.patches):
#     plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

#   ax.set_xlabel('Score')
#   ax.set_title("Face Blendshapes")
#   plt.tight_layout()
#   plt.show()

def draw_hand_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks,
      vision.HandLandmarksConnections.HAND_CONNECTIONS,
      drawing_styles.get_default_hand_landmarks_style(),
      drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def draw_text_on_image(text: str | list[str],
                       image: cv2.typing.MatLike,
                       x: int | HorizontalPosition = HorizontalPosition.MIDDLE,
                       y: int | VerticalPosition = VerticalPosition.MIDDLE,
                       margin_x: int = 20,
                       margin_y: int = 20,
                       font_scale: float = 1.0,
                       thickness: int = 2,
                       line_spacing: int = 5,
                      ) -> list[tuple[int, int]]:

    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Support both a single string and multiple lines.
    lines = text.splitlines() if isinstance(text, str) else text

    if not lines:
        return []

    # Measure every line.
    sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)
        for line in lines
    ]

    text_widths = [size[0][0] for size in sizes]
    text_heights = [size[0][1] for size in sizes]
    baselines = [size[1] for size in sizes]

    # Calculate the total height of the text block.
    line_step = max(text_heights) + max(baselines) + line_spacing
    block_height = (
        max(text_heights)
        + (len(lines) - 1) * line_step
        + max(baselines)
    )

    # Determine the top of the text block.
    if isinstance(y, VerticalPosition):
        match y:
            case VerticalPosition.TOP:
                top = margin_y
            case VerticalPosition.MIDDLE:
                top = (height - block_height) // 2
            case VerticalPosition.BOTTOM:
                top = height - margin_y - block_height
    else:
        top = y

    positions = []

    for i, (line, text_width, text_height) in enumerate(
        zip(lines, text_widths, text_heights)
    ):
        # Determine the horizontal position of this line.
        if isinstance(x, HorizontalPosition):
            match x:
                case HorizontalPosition.LEFT:
                    line_x = margin_x
                case HorizontalPosition.MIDDLE:
                    line_x = (width - text_width) // 2
                case HorizontalPosition.RIGHT:
                    line_x = width - text_width - margin_x
        else:
            line_x = x

        # OpenCV expects the baseline, not the top-left corner.
        line_y = top + text_height + i * line_step

        cv2.putText(
            image,
            line,
            (line_x, line_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

        positions.append((line_x, line_y))

    return line_x, line_y
