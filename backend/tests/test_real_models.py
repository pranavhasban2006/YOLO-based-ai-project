"""The real audio and text models (skipped when their checkpoints are not present)."""
import io
import wave

import numpy as np
import pytest

from app import config
from app.config import CLASSES
from app.errors import AppError
from app.modalities import audio, text

TEXT_PT = config.MODELS_DIR / "text_roberta_bilstm_attn_goemotions.pt"
AUDIO_PT = config.MODELS_DIR / "audio_hubert_transformer.pt"


def wav_bytes(seconds=1.0, sr=22050) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_decode_resamples_and_pads_to_four_seconds():
    w = audio.decode(wav_bytes(1.0, 22050))
    assert w.dtype == np.float32 and w.shape == (audio.LEN,)
    assert np.abs(w[audio.SR + 200:]).max() == 0          # zero padding after the 1 s of audio


def test_decode_rejects_garbage():
    with pytest.raises(AppError):
        audio.decode(b"RIFF....WAVE not really audio")


@pytest.mark.skipif(not TEXT_PT.exists(), reason="text checkpoint not present")
def test_text_model_predicts_obvious_emotions(monkeypatch):
    monkeypatch.setattr(text, "CHECKPOINT", TEXT_PT)
    p = text.TextPredictor()
    p.load()
    assert not p.is_mock
    for sentence, label in [("Thank you so much, this made my day!", "happy"),
                            ("I miss her every single day", "sad")]:
        pred = p.predict(sentence)
        assert pred.mock is False and len(pred.probs) == len(CLASSES)
        assert pred.label == label


@pytest.mark.skipif(not AUDIO_PT.exists(), reason="audio checkpoint not present")
def test_audio_model_returns_real_prediction(monkeypatch):
    monkeypatch.setattr(audio, "CHECKPOINT", AUDIO_PT)
    p = audio.AudioPredictor()
    p.load()
    assert not p.is_mock
    pred = p.predict(wav_bytes(2.0, 16000))
    assert pred.mock is False and abs(sum(pred.probs) - 1) < 1e-4
