import pytest
import torch

from app import config
from app.modalities.face import (CNN_TRANSFORM, RESNET_TRANSFORM, EmotionCNN, build_resnet18,
                                 load_checkpoint)
from app.schemas import Prediction

EXPECTED_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


@pytest.mark.parametrize("name,shape", [("resnet18", (1, 3, 224, 224)), ("cnn", (1, 1, 48, 48))])
def test_checkpoint_loads(name, shape):
    path = config.FACE_CHECKPOINTS[name]
    if not path.exists():
        pytest.skip(f"checkpoint not present: {path}")
    model = load_checkpoint(name, path)  # strict=True: fails on any architecture mismatch
    assert not model.training
    with torch.no_grad():
        out = model(torch.zeros(shape))
    assert out.shape == (1, 7)


def test_architectures_have_seven_outputs():
    assert EmotionCNN().head[-1].out_features == 7
    assert build_resnet18().fc.out_features == 7


def test_preprocessing_shapes():
    from PIL import Image
    crop = Image.new("RGB", (100, 120), (128, 128, 128))
    assert CNN_TRANSFORM(crop).shape == (1, 48, 48)
    assert RESNET_TRANSFORM(crop).shape == (3, 224, 224)


def test_class_order():
    assert config.CLASSES == EXPECTED_CLASSES
    for i, name in enumerate(EXPECTED_CLASSES):
        probs = [0.0] * 7
        probs[i] = 1.0
        assert Prediction.from_probs("face", probs).label == name


def test_health_reports_class_order_and_mocks(client):
    body = client.get("/health").json()
    assert body["classes"] == EXPECTED_CLASSES
    assert body["modalities"]["face"]["mock"] is False
    assert set(body["mocked"]) == {"audio", "text", "eeg"}
