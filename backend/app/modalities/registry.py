"""The single place where modality predictors are registered.

Replacing a mock with a real model is a one-line change here, e.g.
    from .audio import AudioPredictor
    register(AudioPredictor())
"""
from .audio import AudioPredictor
from .base import Predictor
from .eeg import MockEEGPredictor
from .face import FacePredictor
from .text import TextPredictor

_registry: dict[str, Predictor] = {}


def register(predictor: Predictor) -> None:
    _registry[predictor.modality] = predictor


def get(modality: str) -> Predictor:
    return _registry[modality]


def all_predictors() -> dict[str, Predictor]:
    return dict(_registry)


def is_mock(modality: str) -> bool:
    p = _registry.get(modality)
    return p is None or p.is_mock


def register_defaults() -> None:
    register(FacePredictor())
    # Audio and text fall back to mock behaviour (mock=True) if their checkpoint or dependencies are missing.
    register(AudioPredictor())
    register(TextPredictor())
    register(MockEEGPredictor())
