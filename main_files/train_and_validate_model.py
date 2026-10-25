import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dropout, Input, LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from modules.protocols import ModelFactory
from modules.model_factory import SimpleModel

DATA_PATH: Path = Path('./mp_data')
LOG_PATH: Path = Path('./logs')
MODEL_PATH: Path = Path('./output_models')

# Choose the model here
model_factory: ModelFactory = SimpleModel()

label_map = {} #{label: num for num, label in enumerate(actions)}

sequences, labels, actions = [], [], []  # sequence=input, labels=output

action_num = -1

for action_path in sorted(DATA_PATH.iterdir()):

    if not action_path.is_dir() or action_path.name.startswith('_'):
        continue

    action_num += 1
    action = action_path.name
    actions.append(action)

    print(f"Num={action_num}, action={action}")
          
    for sequence_path in sorted(action_path.iterdir()):

        window = []
        for frame_path in sorted(sequence_path.iterdir()):
            data = np.load(frame_path)

            window.append(data)

        sequences.append(window)

        labels.append(action_num)

x = np.asarray(sequences, dtype=np.float32)
y = to_categorical(labels).astype(int)

print("Actions:", actions)
print("Number of sequences:", len(x))
print("Number of classes:", len(actions))
print("Input shape:", x.shape)
print("Input dtype:", x.dtype)
print("NaN values:", np.isnan(x).sum())
print("Infinite values:", np.isinf(x).sum())

for action_num, action in enumerate(actions):
    print(f"{action}: {labels.count(action_num)} sequences")

print(np.array(sequences).shape)

# create a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

tb_callback = TensorBoard(log_dir=LOG_PATH)

model = model_factory.create(input_shape=x.shape[1:], action_length=len(actions))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"]
)

model.fit(x_train, y_train, epochs = model_factory.nr_of_epochs, callbacks=[tb_callback])

model.summary()  

model.save(MODEL_PATH / f'{model_factory.name}.h5')

# test a prediction
prediction = model.predict(x_test)

predicted_action = actions[np.argmax(prediction[0])]
expected_action = actions[np.argmax(y_test[0])]

print(prediction)
print(predicted_action)
print(expected_action)

# validate model

yhat = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

multilabel_confusion_matrix(ytrue, yhat)

score = accuracy_score(ytrue, yhat)
print(f"Accuracy score: {score}")
