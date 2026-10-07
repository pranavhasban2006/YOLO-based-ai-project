"""Text emotion modality. Placeholder until the text model is ready.

To plug in the real model: implement a Predictor subclass whose `predict` takes a
string and returns Prediction.from_probs("text", probs), then register it in
registry.py. Keep the class order in config.CLASSES.
"""
from .base import MockPredictor


class MockTextPredictor(MockPredictor):
    def __init__(self):
        super().__init__("text")
