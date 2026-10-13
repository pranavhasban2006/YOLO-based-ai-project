"""Speech emotion inference.

Usage (project venv):
    python -I project/code/infer_audio.py                 # accuracy on 20 test-split clips
    python -I project/code/infer_audio.py clip.wav        # one file, prints JSON

Mirrors the Colab training notebook (project/colab_upload_audio_v2/audio_colab_train.ipynb):
  - 16 kHz mono float32, first channel, first 4.0 s, zero-padded to 4.0 s (64,000 samples)
  - MelSpectrogram(16000, n_fft=512, hop=160, n_mels=64), log(x + 1e-6),
    per-clip mean/std normalisation (done inside Net.forward, as in training)
  - 4 conv blocks (32/64/128/256), global average pool, dropout 0.4, linear to 7

For raw input (an upload or a recording) `prepare=True` also applies the dataset
preparation used before training: trim silence (librosa.effects.trim, top_db=30) and
centre-crop to 4 s. Clips already in data/audio_16k are used with prepare=False.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "audio_best.pt"
CLASSES_PATH = PROJECT_DIR / "models" / "audio_classes.json"
AUDIO_DIR = PROJECT_DIR / "data" / "audio_16k"

SR, SEC = 16000, 4.0
N = int(SR * SEC)
CLASSES = json.loads(CLASSES_PATH.read_text())


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, hop_length=160, n_mels=64)
        self.tm = torchaudio.transforms.TimeMasking(30)
        self.fm = torchaudio.transforms.FrequencyMasking(10)
        ch = [1, 32, 64, 128, 256]
        layers = []
        for a, b in zip(ch, ch[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                       nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.cnn = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(256, 7)

    def forward(self, w):
        x = torch.log(self.mel(w.float()) + 1e-6)
        x = (x - x.mean((1, 2), keepdim=True)) / (x.std((1, 2), keepdim=True) + 1e-5)
        x = x.unsqueeze(1)
        if self.training:
            x = self.fm(self.tm(x))
        x = self.cnn(x).mean((2, 3))
        return self.fc(self.drop(x))


_model = None
_device = None


def load_model(device=None):
    global _model, _device
    if _model is None:
        _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        m = Net()
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        _model = m.to(_device).eval()
    return _model


def to_clip(wave, sr, prepare=False):
    """Any waveform -> float32 array of exactly N samples at 16 kHz."""
    w = np.asarray(wave, dtype=np.float32)
    if w.ndim > 1:  # (samples, channels) from soundfile
        w = w[:, 0] if w.shape[1] <= w.shape[0] else w[0]
    if sr != SR:
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
    if prepare:
        import librosa
        w, _ = librosa.effects.trim(w, top_db=30)
        if len(w) > N:
            s = (len(w) - N) // 2
            w = w[s:s + N]
    w = w[:N]
    return np.pad(w, (0, N - len(w))).astype(np.float32)


def predict_audio(wav, sr=None, prepare=None):
    """wav: path to an audio file, or a numpy array (then sr is required).

    prepare defaults to False for paths inside data/audio_16k (already prepared) and
    True for everything else. Returns dict(label, confidence, probs{class: p}).
    """
    if isinstance(wav, (str, Path)):
        p = Path(wav)
        if prepare is None:
            prepare = AUDIO_DIR not in p.resolve().parents
        wave, sr = sf.read(str(p), dtype="float32")
    else:
        if sr is None:
            raise ValueError("sr is required when passing an array")
        wave = wav
        prepare = True if prepare is None else prepare
    clip = to_clip(wave, sr, prepare)
    model = load_model()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(clip)[None].to(_device)), 1)[0].cpu().numpy()
    k = int(probs.argmax())
    return {"label": CLASSES[k], "confidence": float(probs[k]),
            "probs": {c: float(p) for c, p in zip(CLASSES, probs)}}


def _check(n_per_clip=20, seed=42):
    import pandas as pd
    df = pd.read_csv(AUDIO_DIR / "index.csv")
    test = df[df.split == "test"].sample(n_per_clip, random_state=seed)
    ok = 0
    for r in test.itertuples():
        out = predict_audio(AUDIO_DIR / r.path)
        ok += out["label"] == r.label
        print(f"{r.dataset:8s} true={r.label:9s} pred={out['label']:9s} conf={out['confidence']:.2f}")
    print(f"accuracy on {n_per_clip} test clips: {ok / n_per_clip:.3f}")

    # Larger check against the Colab figure: the whole test split.
    full = df[df.split == "test"]
    ok = sum(predict_audio(AUDIO_DIR / r.path)["label"] == r.label for r in full.itertuples())
    print(f"accuracy on all {len(full)} test clips: {ok / len(full):.4f} (Colab E2 ALL: 0.5795)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(predict_audio(sys.argv[1]), indent=1))
    else:
        _check()
