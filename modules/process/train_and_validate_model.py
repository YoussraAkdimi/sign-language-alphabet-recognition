import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, set_random_seed # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping # type: ignore
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, classification_report, accuracy_score
from modules.protocols import ExtractFeaturesFromDetectionResult
from modules.features.feature_transformer import FeatureTransformer

from modules.protocols import ModelFactory
from modules.model_factory import SimpleModel
from modules.utilities import load_dataset

DATA_PATH: Path = Path('./mp_data')
VALIDATE_PATH: Path = Path('./mp_data')
LOG_PATH: Path = Path('./logs')
MODEL_PATH: Path = Path('./output_models')

# Choose the model here
model_factory: ModelFactory = SimpleModel()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    
def train_and_validate_model(model_factory: ModelFactory, 
                             actions: list[str],
                             train_dataset: np.ndarray,
                             train_labels: np.ndarray,
                             validation_dataset: np.ndarray,
                             validation_labels: np.ndarray,
                             nr_of_epochs: int = 400,
                             seed: int = 42) -> Model:

    set_seed(seed)

    x_train = train_dataset
    y_train = to_categorical(train_labels).astype(int)

    x_validation = validation_dataset
    y_validation = to_categorical(validation_labels).astype(int)

    print("Actions:", actions)
    print("Number of sequences:", len(x_train))
    print("Number of classes:", len(actions))
    print("Input shape:", x_train.shape)
    print("Input dtype:", x_train.dtype)
    print("NaN values:", np.isnan(x_train).sum())
    print("Infinite values:", np.isinf(x_train).sum())

    # for action_num, action in enumerate(actions):
    #     count = np.count_nonzero(labels == action_num)
    #     print(f"{action}: {count} sequences")

    tb_callback = TensorBoard(log_dir=LOG_PATH)

    model = model_factory.create(input_shape=x_train.shape[1:], action_length=len(actions))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    # stop training if the validation loss does notimprove for 20 epochs, and restore the best weights
    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     patience=20,
    #     restore_best_weights=True
    # )

    # This can be added the callback to save the best model
    # checkpoint = ModelCheckpoint(filepath=MODEL_PATH / 'best_model.keras',
    #                              monitor='val_loss',
    #                              mode='min',
    #                              save_best_only=True,
    #                              verbose=1)

    # verbose=1 means show progress bar, verbose=2 means show one line per epoch, verbose=0 means silent
    history = model.fit(x_train, 
                        y_train, 
                        epochs = nr_of_epochs, 
                        callbacks=[tb_callback], 
                        validation_data=(x_validation, y_validation),
                        verbose=1)

    # ----- VALIDATION STEP -----

    val_loss, val_acc = model.evaluate(
        x_validation,
        y_validation,
        verbose=0
    )

    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    # Per-sequence predictions
    probs = model(x_validation, training=False).numpy()

    y_pred = np.argmax(probs, axis=1)
    y_true = np.asarray(validation_labels)

    print("Accuracy:", accuracy_score(y_true, y_pred))

    print("Confusion matrix:")
    print("-----------------")
    print(confusion_matrix(y_true, y_pred))

    print("Classification report")
    print("---------------------")
    print(classification_report(
        y_true,
        y_pred,
        target_names=actions,
        zero_division=0
    ))
    #----------------

    # Use this if you want to reload the best model after training
    # best_model = load_model(MODEL_PATH / 'best_model.keras')
    # best_model.summary()  

    return model, history
