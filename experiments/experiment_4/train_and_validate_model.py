import sys
from pathlib import Path

# add three levels of folders to the system path so that
# we can resolve modules from other paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

DATA_PATH: Path = Path('./mp_data')
LOG_PATH: Path = Path('./logs')
MODEL_PATH: Path = Path('./output_models')

actions = ['hello', 'thanks', 'iloveyou']
no_sequences = 5    # 30 videos worth of data
sequence_length = 30 # use 30 frames for an action

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []  # sequence=input, labels=output
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame in range(sequence_length):
            npy_path = file_path = DATA_PATH / action / str(sequence) / f"{frame}.npy"
            res = np.load(npy_path)

            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print(np.array(sequences).shape)

x = np.array(sequences)
y = to_categorical(labels).astype(int)

# create a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

tb_callback = TensorBoard(log_dir=LOG_PATH)

model = Sequential([
    Input(shape=x.shape[1:]),
    LSTM(64, return_sequences=True, activation='relu'),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=200, callbacks=[tb_callback])

model.summary()  

model.save(MODEL_PATH / 'action.h5')

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
