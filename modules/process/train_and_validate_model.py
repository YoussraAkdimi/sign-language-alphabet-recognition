import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
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

def train_and_validate_model(model_factory: ModelFactory, 
                             feature_extracter: ExtractFeaturesFromDetectionResult, 
                             feature_transformer: FeatureTransformer,
                             train_dataset_path: Path = DATA_PATH,
                             validation_dataset_path: Path = VALIDATE_PATH,
                             nr_of_epochs: int = 400) -> Model:

    # load the trining dataset
    dataset, labels, actions = load_dataset(train_dataset_path, 
                                            feature_extracter=feature_extracter,
                                            feature_transformer=feature_transformer)

    # load the validation dataset
    dataset_validation, labels_validation, actions_validation = load_dataset(validation_dataset_path,
                                                                             feature_extracter=feature_extracter,
                                                                             feature_transformer=feature_transformer)
 
    x = dataset
    y = to_categorical(labels).astype(int)

    x_validation = dataset_validation
    y_validation = to_categorical(labels_validation).astype(int)
    

    print("Actions:", actions)
    print("Number of sequences:", len(x))
    print("Number of classes:", len(actions))
    print("Input shape:", x.shape)
    print("Input dtype:", x.dtype)
    print("NaN values:", np.isnan(x).sum())
    print("Infinite values:", np.isinf(x).sum())

    for action_num, action in enumerate(actions):
        count = np.count_nonzero(labels == action_num)
        print(f"{action}: {count} sequences")

    print(np.array(x).shape)

    # create a training set and a test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, stratify=labels, random_state=42)

    tb_callback = TensorBoard(log_dir=LOG_PATH)

    model = model_factory.create(input_shape=x.shape[1:], action_length=len(actions))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    model.fit(x_train, y_train, epochs = nr_of_epochs, callbacks=[tb_callback])

    #- VALIDATION STEP -----
    # Loss and accuracy on the validation set
    val_loss, val_acc = model.evaluate(x_validation, y_validation)
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    # Per-sequence predictions
    probs = model.predict(x_validation)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.asarray(labels_validation)

    print("Accuracy:", accuracy_score(y_true, y_pred))

    print("Confusion matrix:")
    print("-----------------")
    
    print(confusion_matrix(y_true, y_pred))

    print("Classification report")
    print("---------------------")

    print(classification_report(y_true, y_pred, target_names=actions))
    #----------------

    model.summary()  

    # # test a prediction
    # prediction = model.predict(x_test)

    # predicted_action = actions[np.argmax(prediction[0])]
    # expected_action = actions[np.argmax(y_test[0])]

    # print(prediction)
    # print(predicted_action)
    # print(expected_action)

    # # validate model

    # yhat = model.predict(x_test)

    # ytrue = np.argmax(y_test, axis=1).tolist()
    # yhat = np.argmax(yhat, axis=1).tolist()

    # multilabel_confusion_matrix(ytrue, yhat)

    # score = accuracy_score(ytrue, yhat)
    # print(f"Accuracy score: {score}")

    return model
