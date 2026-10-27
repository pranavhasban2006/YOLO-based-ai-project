"""Speech A1 -> A2 cascade for the live app: CNN (A1) -> GRU (A2), trained by project/code/cascade.py.

Stage 1: log-mel -> 2D CNN -> frame sequence -> pooled linear classifier = A1's scores.
Stage 2: the GRU reads A1's frame sequence; final scores = A1 scores + GRU(A1 output), exactly as in training.
Live audio has no true label, so A2 re-checks every clip (the "A2 on all samples" setting of the report).

The modules must stay identical to MelCNN / RNNHead in cascade.py. torchaudio is not needed: the mel filterbank
and the STFT window are stored in the A1 checkpoint and the spectrogram is computed with torch.stft.
"""
import logging
import os
import threading

import torch
from torch import nn

from .. import config
from ..schemas import Prediction
from .audio import decode

log = logging.getLogger(__name__)

CASCADE_DIR = config.MODELS_DIR / os.getenv("AUDIO_CASCADE_DIR", "cascade_live/models")
A1_CKPT, A2_CKPT = CASCADE_DIR / "audio_cnn_A1.pt", CASCADE_DIR / "audio_cnn_gru_stage2.pt"
N_FFT, HOP = 1024, 320


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class CNNA1(nn.Module):
    """A1 alone. `features` returns the frame sequence B,T',256 that A2 receives."""
    def __init__(self, fb, window, n=7):
        super().__init__()
        self.register_buffer("fb", fb)            # 513 x 64 mel filterbank (torchaudio MelSpectrogram)
        self.register_buffer("window", window)    # Hann window, 1024
        self.cnn = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, n))

    def features(self, w):
        spec = torch.stft(w, N_FFT, HOP, N_FFT, self.window, center=True, pad_mode="reflect",
                          return_complex=True).abs().pow(2)                      # B,513,T
        m = torch.log(torch.matmul(spec.transpose(1, 2), self.fb).transpose(1, 2) + 1e-6)
        m = ((m - m.mean((1, 2), keepdim=True)) / (m.std((1, 2), keepdim=True) + 1e-6)).unsqueeze(1)
        return self.cnn(m).mean(2).transpose(1, 2)

    def scores(self, f):
        return self.head(f.mean(1))


class GRUA2(nn.Module):
    def __init__(self, n=7, d=256, h=128):
        super().__init__()
        self.rnn = nn.GRU(d, h, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, n))

    def forward(self, f):
        return self.head(self.rnn(f)[1][-1])


class AudioCascade:
    model_name = "CNN (A1) -> GRU (A2)"

    def __init__(self):
        self.a1 = self.a2 = None
        self._lock = threading.Lock()

    def _load(self):
        sd1 = torch.load(A1_CKPT, map_location="cpu", weights_only=True)
        a1 = CNNA1(sd1["backbone.mel.mel.mel_scale.fb"], sd1["backbone.mel.mel.spectrogram.window"])
        a1.load_state_dict({"fb": a1.fb, "window": a1.window,
                            **{k.removeprefix("backbone."): v for k, v in sd1.items() if k.startswith("backbone.cnn.")},
                            **{k: v for k, v in sd1.items() if k.startswith("head.")}})
        a2 = GRUA2()
        a2.load_state_dict({k.removeprefix("a2."): v for k, v in
                            torch.load(A2_CKPT, map_location="cpu", weights_only=True).items()})
        self.a1, self.a2 = a1.eval(), a2.eval()
        log.info("Audio cascade loaded from %s", CASCADE_DIR)

    @property
    def available(self) -> bool:
        return A1_CKPT.exists() and A2_CKPT.exists()

    def predict(self, data: bytes) -> dict:
        w = torch.from_numpy(decode(data)).unsqueeze(0)
        with self._lock, torch.no_grad():
            if self.a1 is None:
                self._load()
            f = self.a1.features(w)
            s1 = self.a1.scores(f)
            s2 = s1 + self.a2(f)
        a1 = Prediction.from_probs("audio", torch.softmax(s1, 1)[0].tolist())
        final = Prediction.from_probs("audio", torch.softmax(s2, 1)[0].tolist())
        return {"a1": a1, "final": final}


cascade = AudioCascade()
