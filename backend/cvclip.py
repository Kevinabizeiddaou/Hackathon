# cvclip.py
import cv2
from typing import Optional

class CVClip:
    """
    Minimal MoviePy-like wrapper using OpenCV.

    Exposed attributes/methods:
      - .fps (float)
      - .duration (float, seconds)
      - .get_frame(t_seconds) -> HxWx3 uint8 (RGB)
      - .subclip(start, end) -> CVClip (view on [start,end))
      - .close() (no-op)
    """
    def __init__(self, path: str, start: float = 0.0, end: Optional[float] = None):
        self.path = path

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        raw_duration = (frame_count / fps) if fps > 0 else 0.0
        cap.release()

        self.fps = float(fps)
        self._start = max(0.0, float(start))
        self._end = float(end) if end is not None else float(raw_duration)
        self.duration = max(0.0, self._end - self._start)

    def get_frame(self, t: float):
        """
        Return an RGB frame (numpy uint8 array) at time t (seconds) within THIS slice.
        """
        t_local = max(0.0, min(self.duration, float(t)))
        t_global = self._start + t_local

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

        cap.set(cv2.CAP_PROP_POS_MSEC, t_global * 1000.0)
        ok, bgr = cap.read()
        cap.release()

        if not ok or bgr is None:
            # Fallback one frame earlier if we're too close to the tail.
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_global - (1.0 / max(1.0, self.fps))) * 1000.0)
            ok, bgr = cap.read()
            cap.release()
            if not ok or bgr is None:
                raise ValueError(f"Could not read frame at {t_global:.3f}s")

        # BGR -> RGB
        return bgr[:, :, ::-1]

    def subclip(self, start: float, end: float):
        start = float(start)
        end = float(end)
        start = max(0.0, min(self.duration, start))
        end = max(start, min(self.duration, end))
        return CVClip(self.path, self._start + start, self._start + end)

    def close(self):
        # We don't keep persistent handles; nothing to close.
        pass
