import numpy as np

class EMAPredictionCollector:
    """
    Exponential Moving Average class

    todo: class receives a model prediction and outputs an action index when
          deemed appropriate, or None if it does not select an action index
    """
    def __init__(self, threshold=0.4, alpha=0.4, commit_frames=4,
                 idle_frames_to_rearm=3, idle_index: int | None = None):
        self.threshold = threshold
        self.alpha = alpha
        self.commit_frames = commit_frames
        self.idle_frames_to_rearm = idle_frames_to_rearm
        self.idle_index = idle_index
        self.reset()

    @property
    def smoothed(self):
        return self._smoothed

    def reset(self):
        self._smoothed = None
        self._candidate = None
        self._candidate_count = 0
        self._armed = True
        self._idle_count = 0

    def add_model_prediction(self, prediction: np.ndarray):
        if self._smoothed is None:
            self._smoothed = prediction.copy()
        else:
            self._smoothed = (
                self.alpha * prediction
                + (1 - self.alpha) * self._smoothed
            )

        best = int(np.argmax(self._smoothed))
        confident = self._smoothed[best] > self.threshold

        # Track idle using the raw model prediction.
        raw_best = int(np.argmax(prediction))

        if raw_best == self.idle_index:
            self._idle_count += 1
        else:
            self._idle_count = 0

        if self._idle_count >= self.idle_frames_to_rearm:
            self._armed = True
            self._candidate = None
            self._candidate_count = 0

        # Do not emit idle or low-confidence predictions.
        if not confident or best == self.idle_index:
            self._candidate = None
            self._candidate_count = 0
            return None

        if best == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = best
            self._candidate_count = 1

        if self._armed and self._candidate_count >= self.commit_frames:
            self._armed = False
            return best

        return None

