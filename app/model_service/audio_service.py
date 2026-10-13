"""Speech emotion service: decodes an uploaded or recorded clip and runs project/code/infer_audio.py.

WAV is read directly. Anything else (browser recordings are WebM/Opus) goes through ffmpeg
when it is installed; without ffmpeg only WAV is accepted.
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project" / "code"))
import infer_audio  # noqa: E402  (training-time architecture and preprocessing)

MAX_SECONDS = 60
MIN_SECONDS = 0.5
LOW_CONFIDENCE = 0.5


class AudioError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _ffmpeg_decode(data: bytes) -> tuple[np.ndarray, int]:
    exe = ffmpeg_path()
    if exe is None:
        raise AudioError("ffmpeg_missing",
                         "This file is not a readable WAV and ffmpeg is not installed. Upload a .wav file.", 415)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.bin")
        Path(src).write_bytes(data)
        proc = subprocess.run(
            [exe, "-nostdin", "-loglevel", "error", "-t", str(MAX_SECONDS), "-i", src,
             "-f", "wav", "-ac", "1", "-ar", str(infer_audio.SR), "pipe:1"],
            capture_output=True, timeout=30,
        )
    if proc.returncode != 0 or not proc.stdout:
        raise AudioError("invalid_audio", "Could not decode this audio file.", 400)
    wave, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return wave, sr


def decode(data: bytes) -> tuple[np.ndarray, int]:
    try:
        wave, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        return _ffmpeg_decode(data)
    return wave, sr


class AudioAnalyzer:
    def __init__(self):
        infer_audio.load_model("cpu")
        self.model_name = infer_audio.MODEL_PATH.name

    def health(self) -> dict:
        return {"model": self.model_name, "ffmpeg": ffmpeg_path() is not None,
                "classes": infer_audio.CLASSES}

    def predict_bytes(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        wave, sr = decode(data)
        if wave.ndim > 1:
            wave = wave[:, 0]
        seconds = len(wave) / sr
        if seconds < MIN_SECONDS:
            raise AudioError("audio_too_short", f"Clip is {seconds:.1f} s; send at least {MIN_SECONDS} s.")
        if seconds > MAX_SECONDS:
            raise AudioError("audio_too_long", f"Clip is longer than {MAX_SECONDS} s.", 413)
        if float(np.max(np.abs(wave))) < 1e-4:
            raise AudioError("silent_audio", "The clip is silent.")
        out = infer_audio.predict_audio(wave, sr)  # raw input: trim silence, centre-crop to 4 s
        return {
            "top_label": out["label"],
            "confidence": out["confidence"],
            "probabilities": out["probs"],
            "uncertain": out["confidence"] < LOW_CONFIDENCE,
            "duration_s": round(seconds, 2),
            "model": self.model_name,
            "ms": round((time.perf_counter() - t0) * 1000),
        }
