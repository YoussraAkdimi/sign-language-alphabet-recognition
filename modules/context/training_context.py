from modules.context.capture_context import CaptureContext

class TrainingContext(CaptureContext):
    """
    A class that encapsulates the context of a training session, including the camera number,
    the current frame, the timestamp in milliseconds, and the action being captured.
    """

    def __init__(self, camera_name: str):
        """
        Initialize the TrainingContext with the specified camera name.
        """

        super().__init__(camera_name)

        self._action = None
        self._sequence = 0
        self._frame = 0
        self._countdown_value = 0

    @property
    def action(self) -> str:
        """
        Get the name of the action being captured.

        Returns:
            The name of the action.
        """
        return self._action

    @action.setter
    def action(self, value: str) -> None:
        """
        Set the name of the action being captured.

        Args:
            value: The new name of the action.
        """
        self._action = value

    @property
    def sequence(self) -> str:
        """
        Get the name of the sequence being captured.

        Returns:
            The name of the sequence.
        """
        return self._sequence

    @sequence.setter
    def sequence(self, value: str) -> None:
        """
        Set the name of the sequence being captured.

        Args:
            value: The new sequence of the action.
        """
        self._sequence = value

    @property
    def frame(self) -> str:
        """
        Get the name of the frame being captured.

        Returns:
            The value of the frame.
        """
        return self._frame

    @frame.setter
    def frame(self, value: str) -> None:
        """
        Set the frame of the sequence being captured.

        Args:
            value: The new frame of the action.
        """
        self._frame = value     

    @property
    def countdown_value(self) -> int | None:
        """
        Current countdown value or None if no value set
        """
        return self._countdown_value

    @countdown_value.setter
    def countdown_value(self, value: int) -> None:
        """
        Set the countdown value.
        """
        self._countdown_value = value
