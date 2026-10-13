"""Face emotion analyzer: loads the trained CNN and the face detector once, then serves predictions.

Reuses the training-time code in project/code/infer.py (architecture, crop, preprocessing,
MediaPipe + Haar detector), so the web app and the offline tool cannot drift apart.
"""
import io
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
PROJECT_CODE = ROOT / "project" / "code"

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_CODE))
import infer  # noqa: E402  (training code: SmallCNN, FaceFinder, crop_face, to_model_tensor)

CLASSES = infer.CLASSES
DEFAULT_MODEL = infer.DEFAULT_MODEL


class Analyzer:
    def __init__(self, model_path: Path | None = None, detect_every: int = 1):
        model_path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL))
        classes = json.loads((ROOT / "project" / "models" / "classes.json").read_text(encoding="utf-8"))
        if classes != CLASSES:
            raise RuntimeError("classes.json does not match the model class order")
        self.device = torch.device("cpu")
        self.model = infer.load_model(model_path, self.device)
        self.model_name = model_path.name
        self.finder = infer.FaceFinder()
        self.detector_name = "mediapipe" if self.finder.detector is not None else "haar"
        self.detect_every = max(1, detect_every)
        self._frame_no = 0
        self._last_box = None
        self._lock = threading.Lock()  # tracking state is shared, so one predict at a time

    # ---- helpers -------------------------------------------------------
    def _probs(self, face_rgb_224: np.ndarray) -> np.ndarray:
        x = infer.to_model_tensor(face_rgb_224).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()

    def _find_box(self, rgb: np.ndarray, allow_track: bool):
        """Largest face box (x, y, w, h) in pixels. Re-detects every `detect_every` frames.
        On the frames in between it reuses the last box (tracking), so detection costs less."""
        self._frame_no += 1
        if allow_track and self._last_box is not None and self._frame_no % self.detect_every != 0:
            return self._last_box, True
        box = self.finder.find(rgb)
        self._last_box = box
        return box, False

    def _result(self, probs, box, rgb_shape, ms, tracked=False):
        h, w = rgb_shape[:2]
        x, y, bw, bh = box
        return {
            "face_found": True,
            "box": {"x": x / w, "y": y / h, "w": bw / w, "h": bh / h},
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
            "top_label": CLASSES[int(np.argmax(probs))],
            "confidence": round(float(np.max(probs)), 4),
            "tracked": tracked,
            "ms": round(ms, 1),
        }

    @staticmethod
    def _no_face(ms):
        return {"face_found": False, "box": None, "probabilities": None, "top_label": None,
                "confidence": None, "tracked": False, "ms": round(ms, 1)}

    # ---- public API ----------------------------------------------------
    def predict_rgb(self, rgb: np.ndarray, allow_track: bool = False) -> dict:
        """One RGB frame -> result dict. Predicts the largest face only; no face -> face_found False."""
        t0 = time.perf_counter()
        with self._lock:
            box, tracked = self._find_box(rgb, allow_track)
            if box is None:
                return self._no_face((time.perf_counter() - t0) * 1000)
            crop = infer.crop_face(rgb, box)
            probs = self._probs(crop)
        return self._result(probs, box, rgb.shape, (time.perf_counter() - t0) * 1000, tracked)

    def predict_image_bytes(self, data: bytes) -> dict:
        """Full-resolution still image (EXIF orientation applied)."""
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception as exc:  # PIL raises several types for bad input
            raise ValueError(f"invalid image: {exc}") from exc
        result = self.predict_rgb(np.asarray(img), allow_track=False)
        result["model"] = self.model_name
        return result

    def predict_jpeg_bytes(self, data: bytes) -> dict:
        """Live webcam frame (JPEG) -> result dict, with tracking enabled."""
        buf = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("invalid jpeg frame")
        return self.predict_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), allow_track=True)

    def predict_frames(self, images: list[bytes]) -> dict:
        """Several stills: average the probabilities over the frames that have a face."""
        t0 = time.perf_counter()
        probs_list = []
        for data in images:
            r = self.predict_image_bytes(data) if isinstance(data, bytes) else None
            if r and r["face_found"]:
                probs_list.append([r["probabilities"][c] for c in CLASSES])
        out = {"frames_total": len(images), "frames_with_face": len(probs_list), "model": self.model_name,
               "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if not probs_list:
            out.update({"face_found": False, "probabilities": None, "top_label": None, "confidence": None})
            return out
        mean = np.mean(np.array(probs_list), axis=0)
        out.update({
            "face_found": True,
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, mean)},
            "top_label": CLASSES[int(np.argmax(mean))],
            "confidence": round(float(np.max(mean)), 4),
        })
        return out

    def health(self) -> dict:
        return {"status": "ok", "model": self.model_name, "classes": CLASSES, "device": str(self.device),
                "detector": self.detector_name, "detect_every": self.detect_every}
