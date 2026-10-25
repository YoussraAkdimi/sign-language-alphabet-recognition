
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Zorg dat de modules vanuit de projectmap gevonden worden.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.utilities import CameraManager
from modules.context import PredictionContext
from modules.image_annotators import CompositeAnnotator
from modules.detection_result_converters import HolisticLandmarksFeatureExtracter
from modules.logic.video_loop import capture_video_data

MEDIAPIPE_MODEL_PATH = (
    PROJECT_ROOT / "task_models" / "holistic_landmarker.task"
)

DATA_PATH = PROJECT_ROOT / "mp_data"

TENSORFLOW_MODEL_PATH = (
    PROJECT_ROOT / "output_models" / "simple_model.h5"
)

# Het model is getraind op 30 frames per gebaar.
SEQUENCE_LENGTH = 30

# Minimale lengte van een handmatige opname.
MIN_RECORDING_FRAMES = 5

# Voorkom dat een opname onbeperkt doorgaat.
MAX_RECORDING_FRAMES = 300

def resample_sequence(sequence, target_length):
    """
    Zet een opname met een variabel aantal frames om
    naar target_length frames met lineaire interpolatie.

    Input:  (aantal_frames, aantal_features)
    Output: (target_length, aantal_features)
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(
            f"Verwachte 2D sequentie, kreeg {sequence.shape}"
        )

    num_frames, num_features = sequence.shape

    if num_frames == 0:
        raise ValueError("De opname bevat geen frames.")

    if num_frames == 1:
        return np.repeat(sequence, target_length, axis=0)

    old_positions = np.linspace(0.0, 1.0, num_frames)
    new_positions = np.linspace(0.0, 1.0, target_length)

    result = np.empty(
        (target_length, num_features),
        dtype=np.float32
    )

    for feature_idx in range(num_features):
        result[:, feature_idx] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_idx]
        )

    return result


def classify_recording(sequence, model, actions):
    """
    Verwerk een complete opname en voorspel precies één keer.
    """
    num_frames = len(sequence)

    if num_frames < MIN_RECORDING_FRAMES:
        print(
            f"\nOpname te kort: {num_frames} frames. "
            f"Minimaal {MIN_RECORDING_FRAMES} nodig."
        )
        return None

    processed = resample_sequence(
        sequence,
        SEQUENCE_LENGTH
    )

    # Keras verwacht:
    # (batch_size, sequence_length, num_features)
    model_input = np.expand_dims(processed, axis=0)

    probabilities = model.predict(
        model_input,
        verbose=0
    )[0]

    predicted_idx = int(np.argmax(probabilities))
    predicted_action = actions[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    print("\n" + "=" * 45)
    print("CLASSIFICATIERESULTAAT")
    print("=" * 45)
    print(f"Opgenomen frames: {num_frames}")
    print(f"Verwerkte frames: {SEQUENCE_LENGTH}")
    print()

    print("Kansen per klasse:")

    ranked = sorted(
        zip(actions, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    for action, probability in ranked:
        print(f"  {action:<12} {float(probability) * 100:6.2f}%")

    print()
    print(f"VOORSPELLING: {predicted_action}")
    print(f"CONFIDENCE:   {confidence * 100:.2f}%")
    print("=" * 45 + "\n")

    return probabilities


def validate_files():
    for path in [
        MEDIAPIPE_MODEL_PATH,
        TENSORFLOW_MODEL_PATH
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Bestand niet gevonden: {path}"
            )

    if not DATA_PATH.is_dir():
        raise FileNotFoundError(
            f"Datasetmap niet gevonden: {DATA_PATH}"
        )

def main():
    validate_files()

    actions = sorted(
        path.name
        for path in DATA_PATH.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    if not actions:
        raise RuntimeError(
            "Geen klassen gevonden in mp_data."
        )

    print(f"MediaPipe version: {mp.__version__}")
    print("Klassen:", actions)

    # Laad jullie bestaande LSTM-model.
    model = load_model(TENSORFLOW_MODEL_PATH)

    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    # Controleer of de huidige klassen overeenkomen
    # met het aantal uitgangen van het model.
    if model.output_shape[-1] != len(actions):
        raise ValueError(
            f"Model heeft {model.output_shape[-1]} outputs, "
            f"maar er zijn {len(actions)} klassen: {actions}. "
            "Controleer of mp_data overeenkomt met "
            "de dataset waarop het model getraind is."
        )

    if model.input_shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Model verwacht {model.input_shape[1]} frames, "
            f"maar SEQUENCE_LENGTH is {SEQUENCE_LENGTH}."
        )

    feature_extractor = HolisticLandmarksFeatureExtracter()

    if model.input_shape[-1] != feature_extractor.feature_length:
        raise ValueError(
            f"Model verwacht {model.input_shape[-1]} features, "
            f"maar de extractor levert "
            f"{feature_extractor.feature_length}."
        )

    camera_manager = CameraManager()

    if not camera_manager.has_cameras:
        print("Geen camera gevonden.")
        return

    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = (
        mp.tasks.vision.HolisticLandmarkerOptions
    )
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MEDIAPIPE_MODEL_PATH)
        ),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5
    )

    image_annotator = CompositeAnnotator.default_annotator()

    cap, _, camera_name = camera_manager.start_camera()

    prediction_context = PredictionContext(
        camera_name,
        actions
    )

    recording = False
    recorded_sequence = []
    last_result = None

    print("\n" + "=" * 45)
    print("HANDMATIGE GEBAARHERKENNING")
    print("=" * 45)
    print("B = Begin opname")
    print("S = Stop opname + classificatie")
    print("C = Wissel camera")
    print("Q = Afsluiten")
    print("=" * 45)
    print("\nDruk op B om een gebaar op te nemen.\n")

    try:
        with HolisticLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                # Deze bestaande functie verzorgt de webcam,
                # MediaPipe-detectie en annotaties.
                capture_video_data(
                    cap,
                    landmarker,
                    image_annotator,
                    prediction_context
                )

                # Lees de toetsen na het verwerken van het frame.
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Programma afgesloten.")
                    break

                elif key == ord("c"):
                    # Wissel camera en annuleer een eventuele opname.
                    recording = False
                    recorded_sequence = []

                    cap, _, camera_name = (
                        camera_manager.start_camera(
                            camera_manager.next_camera_index
                        )
                    )

                    prediction_context.camera_name = camera_name
                    print(f"Camera gewijzigd naar: {camera_name}")
                    continue

                elif key == ord("b"):
                    # Begin een nieuwe opname.
                    recorded_sequence = []
                    recording = True
                    last_result = None

                    print("\n[START] Opname begonnen.")
                    print("Voer nu je gebaar uit.")
                    print("Druk op S om te stoppen.\n")

                    continue

                elif key == ord("s"):
                    if not recording:
                        print(
                            "Er loopt geen opname. "
                            "Druk eerst op B."
                        )
                        continue

                    # Stop met opnemen.
                    recording = False

                    print(
                        f"\n[STOP] Opname gestopt: "
                        f"{len(recorded_sequence)} frames."
                    )

                    last_result = classify_recording(
                        recorded_sequence,
                        model,
                        actions
                    )

                    if last_result is not None:
                        prediction_context.result = last_result

                    print(
                        "Klaar voor een nieuw gebaar. "
                        "Druk op B.\n"
                    )

                    continue

                # Verzamel alleen landmarks tijdens opname.
                if recording:
                    features = feature_extractor.extract(
                        prediction_context.detection_result
                    )

                    features = np.asarray(
                        features,
                        dtype=np.float32
                    )

                    if features.size != feature_extractor.feature_length:
                        raise ValueError(
                            f"Onverwachte featurelengte: "
                            f"{features.size}"
                        )

                    recorded_sequence.append(features.copy())

                    if len(recorded_sequence) % 10 == 0:
                        print(
                            f"Opgenomen: "
                            f"{len(recorded_sequence)} frames"
                        )

                    if len(recorded_sequence) >= MAX_RECORDING_FRAMES:
                        recording = False

                        print(
                            "\nMaximale opnamelengte bereikt."
                        )

                        last_result = classify_recording(
                            recorded_sequence,
                            model,
                            actions
                        )

                        if last_result is not None:
                            prediction_context.result = last_result

                # Andere bestaande annotator-toetsen blijven werken.
                if key not in (255,):
                    image_annotator.handle_key_press(key)

    finally:
        camera_manager.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
