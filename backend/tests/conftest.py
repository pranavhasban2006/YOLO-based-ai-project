import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.modalities import registry
from app.modalities.face import EmotionCNN


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture(scope="session")
def face_rgb() -> np.ndarray:
    """A real photo of a face: skimage's 'astronaut' (Eileen Collins, NASA, public domain)."""
    from skimage import data
    return data.astronaut()


@pytest.fixture(scope="session")
def face_jpeg(face_rgb) -> bytes:
    return jpeg_bytes(face_rgb)


@pytest.fixture(scope="session")
def no_face_jpeg() -> bytes:
    rng = np.random.default_rng(0)
    # Smooth gradient plus mild noise: a valid photo-like image with no face in it.
    g = np.linspace(40, 200, 320, dtype=np.float32)
    img = np.stack([np.tile(g, (240, 1))] * 3, axis=-1) + rng.normal(0, 5, (240, 320, 3))
    return jpeg_bytes(np.clip(img, 0, 255).astype(np.uint8))


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=tmp_path / "test.db")
    with TestClient(app) as c:
        face = registry.get("face")
        c.used_real_checkpoint = face.model is not None
        if face.model is None:
            # No checkpoint in models/: use a randomly initialised CNN so the
            # pipeline and response shape can still be tested. Accuracy is not tested.
            torch.manual_seed(0)
            face.use_model("cnn", EmotionCNN())
        yield c
