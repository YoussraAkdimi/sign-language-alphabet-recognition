import cv2
from cv2_enumerate_cameras import enumerate_cameras

class CameraManager:
    def __init__(self):
        self._load_cameras()
        self._current_camera_index = 0
        self._cap = None

    def list_cameras(self):
        if len(self._cameras) > 0:
            print(f"Found {len(self._cameras)} camera(s).")
            for camera in self._cameras:
                print(f"Index: {camera.index}, Name: {camera.name}")
        else:
            print("No cameras found.")
            exit(1)

    @property
    def has_cameras(self) -> bool:
        return len(self._cameras) > 0

    def start_camera(self, camera_index: int=0):
        if not self.has_cameras:
            print("No cameras found.")
            exit(1)

        # Release the old camera first
        if self._cap is not None:
            self._cap.release()

        # do some checking here if the index is allowed

        self._cap  = cv2.VideoCapture(self._cameras[camera_index].index, cv2.CAP_AVFOUNDATION)

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS: {fps}")

        self._current_camera_index = camera_index

        return self._cap, self._current_camera_index, self._cameras[camera_index].name
    
    def _load_cameras(self):
        self._cameras = enumerate_cameras(cv2.CAP_AVFOUNDATION)

    @property
    def next_camera_index(self) -> int:
        return (self._current_camera_index + 1) % len(self._cameras)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
