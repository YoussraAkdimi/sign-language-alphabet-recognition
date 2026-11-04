from modules.protocols import ModelFactory
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dropout, Input, LSTM, Dense, GRU, GaussianNoise # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore

class SimpleLSTMModelWithNoise(ModelFactory):
    @property
    def name(self) -> str:
        return "simple_lstm_model"

    @property
    def nr_of_epochs(self) -> int:
        return 1000
    
    def create(self, input_shape: tuple, action_length: int):
        model = Sequential([
            Input(shape=input_shape),
            GaussianNoise(0.01),
            LSTM(32, return_sequences=True),
            LSTM(16),
            Dense(16, activation="relu"),
            Dense(action_length, activation="softmax")
        ])
        return model