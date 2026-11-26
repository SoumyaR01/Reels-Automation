"""Face detection and tracking for smart reframing (No identity recognition)."""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import cv2


@dataclass
class FaceBBox:
    timestamp: float
    x: float  # normalized [0, 1]
    y: float
    w: float
    h: float
    confidence: float


def detect_faces(proxy_path: Path, sample_interval_sec: float = 1.0) -> List[FaceBBox]:
    """Detect faces across video timeline to guide virtual camera reframing.

    Uses OpenCV's Haar/DNN face detector as lightweight default, maintaining policy compliance:
    Detection and localization only, absolutely no identity recognition.
    """
    cap = cv2.VideoCapture(str(proxy_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    step_frames = max(1, int(fps * sample_interval_sec))
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    faces: List[FaceBBox] = []
    face_cascade = None
    try:
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        face_cascade = None

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step_frames == 0:
            t = float(frame_idx / fps)
            if face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
                    for (x, y, w, h) in detected:
                        faces.append(
                            FaceBBox(
                                timestamp=t,
                                x=float(x / width),
                                y=float(y / height),
                                w=float(w / width),
                                h=float(h / height),
                                confidence=0.85,
                            )
                        )
                except Exception:
                    pass

        frame_idx += 1

    cap.release()
    return faces
