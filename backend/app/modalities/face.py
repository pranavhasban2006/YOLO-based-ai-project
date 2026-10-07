"""Facial emotion modality: model definitions, checkpoint loading and preprocessing."""
import logging
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .. import config
from ..errors import AppError, no_face
from ..face.pipeline import FaceDetector, extract_largest_face
from ..schemas import FaceBox, Prediction
from .base import Predictor

log = logging.getLogger(__name__)


# ---- Model definitions. These must match training exactly for the weights to load.

def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        nn.MaxPool2d(2))


class EmotionCNN(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        self.features = nn.Sequential(block(1, 64), block(64, 128),
                                      block(128, 256), block(256, 512))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(512, n))

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(n=7) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, n)
    return m


# ---- Preprocessing, one per model, applied to an RGB face crop.

CNN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

RESNET_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MODEL_SPECS: dict[str, tuple[Callable[[], nn.Module], Callable]] = {
    "resnet18": (build_resnet18, RESNET_TRANSFORM),
    "cnn": (EmotionCNN, CNN_TRANSFORM),
}


def load_checkpoint(name: str, path: Path) -> nn.Module:
    """Build the named architecture and load a state_dict saved with torch.save(model.state_dict())."""
    builder, _ = MODEL_SPECS[name]
    model = builder()
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Tolerate checkpoints saved from nn.DataParallel ("module." prefix); weights are unchanged.
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class FacePredictor(Predictor):
    modality = "face"
    is_mock = False

    def __init__(self):
        self.detector: FaceDetector | None = None
        self.model: nn.Module | None = None
        self.model_name: str | None = None
        self.transform = None
        self.loaded: list[str] = []
        self.load_errors: dict[str, str] = {}
        self._lock = threading.Lock()

    # ---- startup

    def load(self) -> None:
        self.detector = FaceDetector(config.FACE_DETECTOR_PATH)

        preferred = config.ACTIVE_FACE_MODEL
        if preferred not in MODEL_SPECS:
            raise ValueError(f"ACTIVE_FACE_MODEL must be one of {list(MODEL_SPECS)}, got {preferred!r}")
        order = [preferred] + [n for n in MODEL_SPECS if n != preferred]

        # Try the configured model first, then fall back to the other one.
        for name in order:
            path = config.FACE_CHECKPOINTS[name]
            if not path.exists():
                self.load_errors[name] = f"file not found: {path}"
                log.warning("Face checkpoint %s missing at %s", name, path)
                continue
            try:
                model = load_checkpoint(name, path)
            except Exception as e:  # a bad checkpoint should not take the server down
                self.load_errors[name] = f"failed to load: {e}"
                log.exception("Failed to load face checkpoint %s", name)
                continue
            self.loaded.append(name)
            if self.model is None:
                self.use_model(name, model)
            else:
                del model  # only the active model is kept in memory
        if self.model is None:
            log.error("No face checkpoint could be loaded; face endpoints will return 503.")
        elif self.model_name != preferred:
            log.warning("Requested face model %r unavailable; falling back to %r", preferred, self.model_name)

    def use_model(self, name: str, model: nn.Module) -> None:
        self.model = model.eval()
        self.model_name = name
        self.transform = MODEL_SPECS[name][1]

    @property
    def ready(self) -> bool:
        return self.model is not None and self.detector is not None

    @property
    def info(self) -> dict:
        return {
            "modality": self.modality,
            "mock": False,
            "ready": self.ready,
            "active_model": self.model_name,
            "requested_model": config.ACTIVE_FACE_MODEL,
            "loaded_checkpoints": self.loaded,
            "unavailable_checkpoints": self.load_errors,
            "detector_loaded": self.detector is not None,
        }

    # ---- inference

    def _require_ready(self):
        if not self.ready:
            raise AppError(503, "model_unavailable",
                           "No facial emotion model is loaded. Put resnet18_fer.pt or cnn_baseline.pt "
                           "in the models/ folder and restart the backend.")

    @torch.inference_mode()
    def classify_crop(self, crop: Image.Image) -> np.ndarray:
        x = self.transform(crop).unsqueeze(0)
        with self._lock:
            logits = self.model(x)
        return torch.softmax(logits, dim=1)[0].numpy()

    def analyse(self, rgb: np.ndarray) -> tuple[np.ndarray, FaceBox, int] | None:
        """Detect the largest face and classify it. Returns None if no face is found."""
        self._require_ready()
        face = extract_largest_face(self.detector, rgb)
        if face is None:
            return None
        return self.classify_crop(face.crop), face.box, face.faces_found

    def predict(self, rgb: np.ndarray) -> Prediction:
        result = self.analyse(rgb)
        if result is None:
            raise no_face()
        probs, _, _ = result
        return Prediction.from_probs(self.modality, probs)
