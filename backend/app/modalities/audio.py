"""Speech emotion modality. Placeholder until the audio model is ready.

To plug in the real model: implement a Predictor subclass here (load weights in
`load`, take raw audio bytes in `predict`, return Prediction.from_probs("audio", probs)),
then register it in registry.py in place of MockAudioPredictor. Keep the class
order in config.CLASSES.
"""
from .base import MockPredictor


class MockAudioPredictor(MockPredictor):
    def __init__(self):
        super().__init__("audio")
