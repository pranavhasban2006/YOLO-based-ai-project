"""The interface every modality predictor implements.

To add a real model for a modality: write a class that subclasses Predictor,
set `modality`, return Prediction objects from `predict`, and register it in
modalities/registry.py. Routes look predictors up through the registry, so no
route code needs to change.
"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..config import CLASSES
from ..schemas import Prediction


class Predictor(ABC):
    modality: str = ""
    is_mock: bool = False

    def load(self) -> None:
        """Load weights. Called once at startup."""

    @property
    def info(self) -> dict:
        return {"modality": self.modality, "mock": self.is_mock}

    @abstractmethod
    def predict(self, data: Any) -> Prediction:
        """Return a Prediction for one input (bytes, text, signal window, ...)."""


class MockPredictor(Predictor):
    """Placeholder that returns random but well-formed probabilities with mock=True."""
    is_mock = True

    def __init__(self, modality: str, seed: int | None = None):
        self.modality = modality
        self._rng = np.random.default_rng(seed)

    def predict(self, data: Any = None) -> Prediction:
        probs = self._rng.dirichlet(np.ones(len(CLASSES)))
        return Prediction.from_probs(self.modality, probs, mock=True)
