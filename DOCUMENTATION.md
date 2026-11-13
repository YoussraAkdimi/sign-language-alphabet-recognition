# Technical Documentation
## Sign Language Alphabet Recognition

## 1. Overview

This project implements a video-based gesture-recognition pipeline using MediaPipe holistic landmark detection and TensorFlow/Keras sequence classification. It was developed for Assignment 3 of the *Key Topics in AI* course.

A webcam frame is processed by MediaPipe to detect hand, pose and face landmarks. These coordinates are converted into a numerical feature vector. A sequence of 30 consecutive frame vectors is then classified by a neural network.

The current experiment configuration contains four classes: `idle`, `poes`, `slapen` and `vogel`. These are the labels used by the checked-in experiment setup; the repository should not be described as a complete 26-letter alphabet classifier.

## 2. High-level pipeline

1. **Capture:** OpenCV reads frames from a camera.
2. **Detection:** MediaPipe Holistic Landmarker detects face, pose and hand landmarks.
3. **Feature extraction:** landmarks are flattened into a 1,659-value vector per frame.
4. **Preprocessing:** optional transformations normalize, center, weight or zero selected coordinate groups.
5. **Sequence assembly:** 30 consecutive frame vectors form one model input sample.
6. **Classification:** a TensorFlow/Keras model estimates class probabilities.
7. **Temporal stabilization:** an exponential moving average and a candidate-frame threshold reduce flicker in live predictions.
8. **Visualization:** annotator classes overlay landmarks, status information and predictions on video.

## 3. Repository layout

| Location | Purpose |
|---|---|
| `environment.yml` | Conda environment specification; environment name `ktai_3`, Python 3.12 |
| `main_files/test_different_transformations_simple_model.ipynb` | Main feature-transformation experiment and independent test evaluation |
| `main_files/test_different_transformations_multi_seed.ipynb` | Additional multi-seed experiment |
| `main_files/experiments/experiment.py` | Webcam-oriented data capture, training and live inference workflow |
| `modules/context/` | Context/state objects for capture and prediction |
| `modules/features/` | Landmark feature representation, feature extraction and transformation orchestration |
| `modules/features/transformations/` | Individual feature transformations |
| `modules/image_annotators/` | Image/video overlay implementations |
| `modules/logic/` | Shared video-capture processing loop |
| `modules/model_factory/` | Keras model factory implementations |
| `modules/prediction_collectors/` | Temporal prediction smoothing and commitment |
| `modules/process/` | Data recording, visualization, model training and live testing |
| `modules/protocols/` | Protocol interfaces used by the modules |
| `modules/utilities/` | Camera management, drawing, file loading/saving and dataset utilities |
| `task_models/` | MediaPipe task model assets |
| `mp_data/` | Training landmark sequences |
| `mp_data_validation/` | Validation landmark sequences |
| `mp_data_new_test/` | Independent test sequences referenced by the main notebook |
| `figures/` | Exported evaluation plots |
| `test_images/` | Example image asset |

The main notebook expects to be run with its working directory set to `main_files/`; it derives the repository root from the parent of the current working directory.

## 4. Environment and dependencies

The checked-in `environment.yml` defines a Conda environment named `ktai_3` with Python 3.12. It includes:

- NumPy and pandas
- Matplotlib
- scikit-learn
- Jupyter and ipykernel
- OpenCV camera enumeration helper (`cv2-enumerate-cameras`)
- `opencv-contrib-python==4.11.0.86`
- MediaPipe
- TensorFlow
- TensorBoard
- certifi

Use the commands in [README.md](README.md) to clone the repository, create the Conda environment and launch the notebook.

## 5. Landmark feature representation

The feature implementation is in `modules/features/holistic_landmarks_feature.py`. Each landmark contributes three values: X, Y and Z.

| Group | Landmarks | Scalar values |
|---|---:|---:|
| Left hand | 21 | 63 |
| Right hand | 21 | 63 |
| Pose | 33 | 99 |
| Face | 478 | 1,434 |
| **Total** | **553** | **1,659** |

The flat vector is concatenated in the order left hand, right hand, pose, face. Missing groups are replaced with zero arrays of the corresponding size. Each frame feature can be saved to or loaded from a NumPy `.npy` file.

The extractor, `HolisticLandmarksFeatureExtracter`, converts a MediaPipe holistic detection result into this representation. The resulting model input for one sequence has shape `(30, 1659)`; a full dataset has shape `(number_of_sequences, 30, 1659)`.

## 6. Feature preprocessing

`FeatureTransformer` is a sequential pipeline: it applies the configured transformation objects in the order in which they are provided. The notebook compares each transformation individually and a combined pipeline.

### 6.1 Normalization

Implemented by `FeatureNormalizer` and `HolisticLandmarksFeature.normalize()`.

The implementation identifies non-zero landmarks, computes the minimum and maximum X/Y coordinates and scales the X/Y values using the largest bounding-box dimension. A common scale is used to preserve the aspect ratio. Z values are not normalized by this operation.

### 6.2 Pose centering

Implemented by `FeatureCentering` and `center_at_pose()`.

The midpoint between pose landmarks 11 and 12 (the two shoulder landmarks) is calculated and subtracted from all landmark groups. This expresses positions relative to the shoulder midpoint.

### 6.3 Landmark-group weighting

Implemented by `FeatureWeights` and `apply_weights()`.

Separate multipliers can be applied to left hand, right hand, pose and face coordinates. The main combined experiment uses:

| Group | Weight |
|---|---:|
| Left hand | 10.0 |
| Right hand | 10.0 |
| Pose | 2.0 |
| Face | 0.5 |

### 6.4 Removing Z coordinates

Implemented by `FeatureRemoveZCoordinates`.

Every third value in the flattened feature vector (the Z coordinate) is set to zero. The overall vector length remains 1,659.

### 6.5 Removing lower-body pose landmarks

Implemented by `FeatureRemoveLegData`.

Pose landmarks 25–32 are set to zero. This suppresses lower-body information while keeping the original vector dimensions.

### 6.6 Combined preprocessing pipeline

The combined pipeline used in the main notebook and the live experiment is configured in this order:

1. Remove lower-body pose landmarks.
2. Set Z coordinates to zero.
3. Normalize X/Y landmark positions.
4. Center all landmark groups at the shoulder midpoint.
5. Apply group weights (left hand 10.0, right hand 10.0, pose 2.0, face 0.5).

The order is meaningful because the transformations are applied sequentially.

## 7. Dataset organization and loader

The data-loading logic is in `modules/utilities/file.py`. The expected directory format is:

```text
mp_data/
  <class-name>/
    <sequence-id>/
      000.npy
      001.npy
      ...
      029.npy
```

The same convention is used for validation and test folders. The loader:

- iterates through class directories in sorted order;
- ignores directory names starting with an underscore;
- iterates through each sequence directory in sorted order;
- reads all `.npy` frame features;
- checks that each sequence has exactly 30 frames;
- applies the supplied feature transformer to every frame;
- builds integer labels from the sorted class-directory order;
- returns float32 data, int16 labels and the action-name list.

Every split must use the same class-directory names and class ordering. The main notebook explicitly checks that the independent test class list matches the training class list.

The data-recording routine defaults to 30 recordings (sequences) per class and 30 frames per sequence. It stores frame-level feature vectors in the same class/sequence layout.

## 8. Model architectures

Model definitions are provided through factories in `modules/model_factory/`. The main notebook selects `SimpleModel`.

### 8.1 SimpleModel

The main model is a Keras Sequential network:

| Layer | Configuration |
|---|---|
| Input | Sequence shape, normally `(30, 1659)` |
| GRU | 64 units, `return_sequences=True`, dropout 0.3, recurrent dropout 0.1 |
| GRU | 32 units, dropout 0.3 |
| Dense | 32 units, ReLU activation, L2 regularization of 1e-4 |
| Dropout | 0.3 |
| Output Dense | Number of action classes, softmax activation |

The output layer produces one probability per action class.

### 8.2 Other available model factories

The repository also contains LSTM-based and noise-augmented alternatives, including `SimpleLSTMModel`, `SimpleModelWithNoise` and `SimpleLSTMModelWithNoise`. The LSTM model has two recurrent layers (32 units with sequence output, then 16 units), followed by a 16-unit ReLU layer and a softmax output. Inspect each factory before substituting it into an experiment; the main feature-comparison notebook specifically instantiates `SimpleModel`.

## 9. Training and validation

Training is implemented in `modules/process/train_and_validate_model.py`.

The training function:

1. Sets Python, NumPy and TensorFlow random seeds (default seed 42).
2. Converts integer labels to one-hot categorical arrays.
3. Prints class names, sequence count, tensor shape, data type and NaN/infinite-value counts.
4. Creates a model through the selected model factory.
5. Compiles it with Adam, categorical cross-entropy and categorical accuracy.
6. Fits the model for the supplied epoch count using validation data.
7. Writes TensorBoard logs to `logs/`.
8. Evaluates the validation loss and accuracy.
9. Prints accuracy, a confusion matrix and a classification report.

In the main notebook, `NR_OF_EPOCHS` is set to 50. The training function contains examples of EarlyStopping and ModelCheckpoint callbacks, but they are commented out in the current source and are not active in the standard training run.

## 10. Experiment design and evaluation

The main notebook is `main_files/test_different_transformations_simple_model.ipynb`.

It loads the training and validation datasets, creates transformed copies of those data, and trains a separate model for each preprocessing configuration. The main comparisons are:

- Baseline (no transformation)
- Normalized landmarks
- Weighted landmarks
- Centered landmarks
- Z coordinates removed
- Lower-body pose landmarks removed
- All transformations combined

The training curves plot loss and categorical accuracy for each experiment.

For independent evaluation, the notebook loads `mp_data_new_test/`, checks the class ordering against the training actions, applies the corresponding transformation to the test data and computes predictions for each trained model. It visualizes confusion matrices and compares the metrics below.

### Metrics

- **Accuracy:** proportion of test sequences classified correctly.
- **Balanced accuracy:** average recall across classes, useful when class counts differ.
- **Macro F1:** F1-score calculated for each class and averaged equally across classes.
- **Confusion matrix:** counts of true versus predicted labels, useful for seeing which classes are confused.
- **Per-class F1:** makes class-specific strengths and weaknesses visible.

Exported plots are stored in `figures/`:

- `training_curves.png`
- `confusion_matrix.png`
- `class-wise-performace.png` (repository filename spelling preserved)

The notebook is the source for recalculating experiment results. Committed images are snapshots and may not exactly match results from a fresh run if datasets, seeds, package versions or hardware differ.

The second notebook, `test_different_transformations_multi_seed.ipynb`, is an additional experiment intended for multiple-seed analysis. It may require substantially more training time because the model comparisons are repeated.

## 11. Live prediction

The live workflow is assembled in `main_files/experiments/experiment.py` and implemented mainly in `modules/process/real_life_test.py`.

The inference flow:

1. Opens a camera through the camera manager.
2. Runs MediaPipe Holistic Landmarker in video mode.
3. Extracts the holistic feature for each frame.
4. Applies the configured feature transformer.
5. Maintains a rolling sequence of the latest 30 features.
6. Runs the trained model on the sequence.
7. Sends class probabilities to `EMAPredictionCollector`.
8. Prints/annotates a gesture when the collector commits a prediction.

The live script uses the same class list and combined feature transformations as its training setup. It saves the model to `output_models/simple_model.keras` through the project save helper.

### Prediction smoothing and event commitment

`modules/prediction_collectors/ema_prediction_collector.py` smooths model probabilities using an exponential moving average:

`smoothed_t = alpha * prediction_t + (1 - alpha) * smoothed_(t-1)`

The collector then:

- finds the highest-probability class;
- checks whether its smoothed confidence exceeds a threshold;
- ignores idle and low-confidence predictions;
- requires a candidate class to persist for a configured number of frames;
- emits a class index only when armed;
- re-arms after a configured number of idle predictions.

In `experiment.py`, the collector is configured with threshold 0.7, alpha 0.6, 10 commit frames, 5 idle frames to re-arm and idle class index 0.

## 12. Data capture and visualization

### Recording

`modules/process/record_training_data.py` captures webcam data. It creates the class/sequence directories, performs a three-second countdown and captures 30 frames per sequence by default. Each captured frame is converted to a landmark feature and saved as a NumPy file when `whatIf` is false.

The capture workflow uses `q` to stop and `c` to switch to the next available camera where supported. Camera and OpenCV resources are released in a `finally` block.

### Inspecting recorded sequences

`modules/process/show_trained_data.py` loads saved features, applies the chosen transformations, reconstructs the landmark groups and draws them on a blank canvas. Press `q` to stop playback.

### Annotators

The implementations under `modules/image_annotators/` provide composable overlays for landmarks, camera information, training state and predictions.

## 13. Artifacts and output directories

| Path | Description |
|---|---|
| `figures/` | Committed training/evaluation figures |
| `logs/` | TensorBoard logs; ignored by Git |
| `output_models/` | Saved Keras models; ignored by Git |
| Dataset folders | Frame-level NumPy landmark features |

The experiment script saves a model at `output_models/simple_model.keras`. The save helper creates the parent directory when needed. The live test implementation contains a separate default path constant; the experiment script passes its already-loaded model into the live test function.

## 14. Reproduction notes and implementation caveats

- **Notebook working directory:** the simple-model notebook sets `PROJECT_ROOT = Path.cwd().parents[0]`. Launch it with the working directory set to `main_files/` so its project-root calculation points at the repository root.
- **Independent test data:** the notebook expects `mp_data_new_test/`. Ensure this folder contains the expected class/sequence structure before running the independent test cells.
- **Sequence length:** `load_dataset()` raises an error unless each sequence contains exactly 30 frame files.
- **Class order:** class directories are sorted alphabetically. Keep names identical across training, validation and test sets.
- **Capture mode:** the current `experiment.py` sets `whatIfMode = True`. Therefore, the validation capture invocation does not write new validation features. Change this only when intentionally recording data.
- **Callbacks:** Early stopping and model checkpoint code in the training module is commented out.
- **Live inference:** the live workflow requires camera access and the MediaPipe asset `task_models/holistic_landmarker.task`.
- **Environment compatibility:** TensorFlow, MediaPipe and OpenCV binaries can have platform-specific constraints. Use the provided environment as the intended baseline and resolve installation issues for the target OS if needed.

## 15. References

- MediaPipe Holistic Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker
- TensorFlow: https://www.tensorflow.org/
- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html

---

For cloning, environment setup, notebook execution and troubleshooting, see [README.md](README.md).
