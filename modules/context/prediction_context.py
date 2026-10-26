import numpy as np

from modules.context.capture_context import CaptureContext

class PredictionContext(CaptureContext):
    def __init__(self, camera_name: str, actions: list[str]):
        super().__init__(camera_name)

        self._actions = actions
        self._action = None
        self._model_prediction = None

    @property
    def actions(self) -> str:
        """
        Get the name of the action being guessed.

        Returns:
            The name of the action.
        """
        return self._actions

    @property
    def action(self) -> str:
        """
        Get the name of the action being guessed.

        Returns:
            The name of the action.
        """
        return self._action

    @action.setter
    def action(self, value: str) -> None:
        """
        Set the name of the action being guessed.

        Args:
            value: The new name of the action.
        """
        self._action = value

    @property
    def model_prediction(self) -> np.ndarray | None:
        return self._model_prediction

    @model_prediction.setter
    def model_prediction(self, value: np.ndarray) -> None:
        self._model_prediction = value