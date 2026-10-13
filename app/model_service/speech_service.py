"""Speech to text with faster-whisper (CPU, int8) plus the combined per-answer analysis.

The audio is decoded in memory and discarded after the request; nothing is written to disk by this module.
"""
import os
import threading
import time

import numpy as np
import torch
import torchaudio

import audio_service
import prosody

MODEL_NAME = os.environ.get("WHISPER_MODEL", "base.en")
_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8", cpu_threads=max(2, (os.cpu_count() or 4) // 2))
    return _model


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def to_16k(data: bytes) -> np.ndarray:
    wave, sr = audio_service.decode(data)
    if wave.ndim > 1:
        wave = wave[:, 0]
    if sr != prosody.SR:
        wave = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)), sr, prosody.SR).numpy()
    return wave.astype(np.float32)


def transcribe_wave(wave: np.ndarray) -> dict:
    t0 = time.perf_counter()
    if float(np.max(np.abs(wave))) < 1e-4 or len(wave) < prosody.SR * 0.3:
        return {"text": "", "words": [], "duration_s": round(len(wave) / prosody.SR, 3), "ms": 0, "model": MODEL_NAME}
    model = _get_model()
    segments, info = model.transcribe(wave, language="en", beam_size=1, word_timestamps=True, vad_filter=True,
                                      condition_on_previous_text=False)
    words, parts = [], []
    for seg in segments:
        parts.append(seg.text.strip())
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
    return {"text": " ".join(p for p in parts if p).strip(), "words": words, "duration_s": round(info.duration, 3),
            "ms": round((time.perf_counter() - t0) * 1000), "model": MODEL_NAME}


def analyse_answer(data: bytes, voice_emotion=None) -> dict:
    """One call for one answer: transcript, prosody and (when the audio model is loaded) voice emotion probabilities."""
    wave = to_16k(data)
    tr = transcribe_wave(wave)
    pro = prosody.extract(wave, tr["words"])
    emo = None
    if voice_emotion is not None:
        try:
            import infer_audio
            if float(np.max(np.abs(wave))) >= 1e-4 and len(wave) >= prosody.SR * 0.5:
                out = infer_audio.predict_audio(wave, prosody.SR)
                emo = {"top_label": out["label"], "confidence": out["confidence"], "probabilities": out["probs"]}
        except Exception:
            emo = None
    return {"transcript": tr, "prosody": pro, "voice_emotion": emo}
