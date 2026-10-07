"""EEG modality. Placeholder until the EEG model and IoT sensor stream are ready.

To plug in the real model: implement a Predictor subclass whose `predict` takes a
window of samples (list of floats, or a dict of channels) and returns
Prediction.from_probs("eeg", probs), then register it in registry.py.
"""
from .base import MockPredictor


class MockEEGPredictor(MockPredictor):
    def __init__(self):
        super().__init__("eeg")
