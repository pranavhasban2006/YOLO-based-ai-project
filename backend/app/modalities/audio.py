"""Speech emotion modality: frozen HuBERT encoder + 2-layer Transformer head (RAVDESS + TESS + SAVEE).

The head must stay identical to SeqTransformer in project/code/train_audio.py. Input audio (wav, webm, ogg,
mp3, m4a ...) is decoded with PyAV, mixed to mono, resampled to 16 kHz and centre-cropped / zero-padded to
4 s, exactly like training. Until the model loads the predictor behaves like the old mock (mock=True).
"""
import io
import logging
import os
import threading

import numpy as np
import torch
from torch import nn

from .. import config
from ..errors import AppError
from ..schemas import Prediction
from .base import MockPredictor

log = logging.getLogger(__name__)

ENCODER = "facebook/hubert-base-ls960"
SR, LEN = 16000, 4 * 16000
CHECKPOINT = config.MODELS_DIR / os.getenv("AUDIO_CHECKPOINT", "audio_hubert_transformer.pt")


class SeqTransformer(nn.Module):
    """HuBERT frame features -> 2-layer Transformer encoder -> mean pool."""
    def __init__(self, n=7, d=768, k=256):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, k))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(k, 4, k * 4, 0.1, batch_first=True,
                                                                   norm_first=True), 2)
        self.head = nn.Sequential(nn.LayerNorm(k), nn.Dropout(0.3), nn.Linear(k, n))

    def forward(self, x):
        return self.head(self.tf(self.proj(x)).mean(1))


def decode(data: bytes) -> np.ndarray:
    """Any container/codec PyAV understands -> mono float32 at 16 kHz, 4 s centre crop / zero pad."""
    import av
    try:
        with av.open(io.BytesIO(data)) as c:
            stream = next(s for s in c.streams if s.type == "audio")
            rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
            chunks = []
            for frame in c.decode(stream):
                chunks += [f.to_ndarray().reshape(-1) for f in rs.resample(frame)]
            chunks += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    except (StopIteration, av.error.FFmpegError) as e:
        raise AppError(422, "bad_audio", f"Could not decode the audio file: {e}") from e
    w = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    if len(w) < SR // 4:
        raise AppError(422, "bad_audio", "Audio is shorter than 0.25 s.")
    if len(w) >= LEN:
        s = (len(w) - LEN) // 2
        return w[s:s + LEN]
    return np.pad(w, (0, LEN - len(w)))


class AudioPredictor(MockPredictor):
    """Real model once `load` succeeds; until then identical to the mock (mock=True)."""

    def __init__(self):
        super().__init__("audio")
        self.encoder = None
        self.extractor = None
        self.head = None
        self._lock = threading.Lock()

    @property
    def info(self) -> dict:
        return {"modality": "audio", "mock": self.is_mock,
                "model": None if self.is_mock else "HuBERT + Transformer"}

    def load(self) -> None:
        if not CHECKPOINT.exists():
            log.warning("Audio checkpoint missing at %s; audio stays mocked", CHECKPOINT)
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModel
            # Prefer the local Hugging Face cache: no network at startup, and no background conversion thread.
            try:
                self.extractor = AutoFeatureExtractor.from_pretrained(ENCODER, local_files_only=True)
                self.encoder = AutoModel.from_pretrained(ENCODER, local_files_only=True).eval()
            except OSError:
                self.extractor = AutoFeatureExtractor.from_pretrained(ENCODER)
                self.encoder = AutoModel.from_pretrained(ENCODER).eval()
            head = SeqTransformer(len(config.CLASSES))
            head.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
            self.head = head.eval()
            self.is_mock = False
            log.info("Audio model loaded from %s", CHECKPOINT.name)
        except Exception:  # a missing dependency should not take the server down
            log.exception("Could not load the audio model; audio stays mocked")

    def predict(self, data) -> Prediction:
        if self.head is None:
            return super().predict(data)
        w = decode(data)
        x = self.extractor([w], sampling_rate=SR, return_tensors="pt")
        with self._lock, torch.no_grad():
            h = self.encoder(**x).last_hidden_state
            probs = torch.softmax(self.head(h), 1)[0]
        return Prediction.from_probs("audio", probs.tolist())


MockAudioPredictor = AudioPredictor  # old name, kept for imports
