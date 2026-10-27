"""Shared face pipeline: decode -> detect largest face (MediaPipe) -> crop with margin.

Model-specific preprocessing (resize, normalise) lives with each model in
modalities/face.py. Everything here works on in-memory arrays; nothing is
written to disk.
"""
import io
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import FACE_MARGIN, FACE_MIN_CONFIDENCE
from ..errors import AppError
from ..schemas import FaceBox


@dataclass
class FaceCrop:
    crop: Image.Image  # RGB crop of the face, with margin, squared
    box: FaceBox  # detected face box (before margin), normalised to [0, 1]
    faces_found: int


def decode_image(data: bytes, allowed_formats=("JPEG", "PNG", "WEBP")) -> np.ndarray:
    """Decode image bytes to an RGB uint8 array. Raises AppError on bad input."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img = ImageOps.exif_transpose(img)  # respect phone camera orientation
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise AppError(400, "invalid_image", "The file could not be read as an image. It may be corrupt.")
    if allowed_formats and fmt not in allowed_formats:
        raise AppError(415, "unsupported_file_type",
                       f"Unsupported image format {fmt}. Allowed: {', '.join(allowed_formats)}.")
    return np.asarray(img)


# One MediaPipe detector per (model, threshold) for the whole process. Creating a second detector while an
# old one is being garbage-collected deadlocks MediaPipe's serial dispatcher (seen when the app is created more
# than once, e.g. in tests), so detectors are created once and never released.
_SHARED: dict = {}


class FaceDetector:
    """Thin wrapper over the MediaPipe Tasks face detector."""

    def __init__(self, model_path: Path, min_confidence: float = FACE_MIN_CONFIDENCE):
        # Imported lazily so the rest of the app can import without mediapipe.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        key = (str(Path(model_path).resolve()), float(min_confidence))
        if key not in _SHARED:
            options = vision.FaceDetectorOptions(
                # Pass the bytes rather than the path: avoids Windows path issues.
                base_options=mp_python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=min_confidence,
            )
            # the detector is not safe for concurrent calls, so it shares one lock with every wrapper
            _SHARED[key] = (vision.FaceDetector.create_from_options(options), threading.Lock())
        self._detector, self._lock = _SHARED[key]

    def detect(self, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return face boxes as (x, y, w, h) in pixels, largest first."""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        with self._lock:
            result = self._detector.detect(image)
        boxes = []
        for d in result.detections:
            b = d.bounding_box
            if b.width > 0 and b.height > 0:
                boxes.append((b.origin_x, b.origin_y, b.width, b.height))
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_face(rgb: np.ndarray, box: tuple[int, int, int, int], margin: float = FACE_MARGIN) -> Image.Image:
    """Crop a square region around the box, padded by `margin` x box size on each side.

    The models were trained on square 48x48 crops, so the crop is made square
    (using the longer side) to avoid distorting the face when resized.
    """
    H, W = rgb.shape[:2]
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)
    x0 = int(round(max(0, cx - side / 2)))
    y0 = int(round(max(0, cy - side / 2)))
    x1 = int(round(min(W, cx + side / 2)))
    y1 = int(round(min(H, cy + side / 2)))
    return Image.fromarray(rgb[y0:y1, x0:x1])


def extract_largest_face(detector: FaceDetector, rgb: np.ndarray) -> FaceCrop | None:
    boxes = detector.detect(rgb)
    if not boxes:
        return None
    H, W = rgb.shape[:2]
    x, y, w, h = boxes[0]
    norm = FaceBox(x=max(0.0, x / W), y=max(0.0, y / H), w=min(1.0, w / W), h=min(1.0, h / H))
    return FaceCrop(crop=crop_face(rgb, boxes[0]), box=norm, faces_found=len(boxes))
