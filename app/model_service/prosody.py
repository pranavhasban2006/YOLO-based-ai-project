"""Prosody features for one spoken answer (heuristic indicators, not a clinical measure).

Input: mono float32 waveform at 16 kHz, optionally word timings from the transcriber.
Output (all floats or None):
  duration_s        length of the clip
  latency_s         silence before the first speech (how long the person took to start)
  speech_s          time from first to last speech
  speech_rate_wps   words per second over the speech span (needs word timings)
  pause_ratio       share of the speech span that is silence between speech parts
  pause_mean_s      mean length of those pauses (pauses of 0.25 s or more)
  n_pauses
  pitch_mean_hz     mean fundamental frequency over voiced frames (pyin)
  pitch_std_st      pitch variability in semitones
  energy_mean_db    mean frame energy over speech frames, dB relative to full scale
  energy_std_db     variability of that energy
"""
import numpy as np

SR = 16000
FRAME = 400  # 25 ms
HOP = 160  # 10 ms
MIN_PAUSE_S = 0.25
MERGE_GAP_S = 0.15


def _frame_rms(w: np.ndarray) -> np.ndarray:
    if len(w) < FRAME:
        w = np.pad(w, (0, FRAME - len(w)))
    n = 1 + (len(w) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return np.sqrt(np.mean(w[idx] ** 2, axis=1) + 1e-12)


def speech_mask(w: np.ndarray) -> np.ndarray:
    """Energy-based voice activity per 10 ms frame."""
    rms = _frame_rms(w)
    peak = float(np.percentile(rms, 95))
    thr = max(0.008, 0.12 * peak)
    return rms > thr


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    # merge very short gaps (a stop consonant is not a pause)
    gap = int(MERGE_GAP_S * SR / HOP)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return merged


def extract(wave: np.ndarray, words: list[dict] | None = None, sr: int = SR) -> dict:
    w = np.asarray(wave, dtype=np.float32)
    if w.ndim > 1:
        w = w[:, 0]
    if sr != SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    out = {k: None for k in ("duration_s", "latency_s", "speech_s", "speech_rate_wps", "pause_ratio", "pause_mean_s",
                             "n_pauses", "pitch_mean_hz", "pitch_std_st", "energy_mean_db", "energy_std_db")}
    out["duration_s"] = round(len(w) / SR, 3)
    if len(w) < SR * 0.3 or float(np.max(np.abs(w))) < 1e-4:
        return out

    mask = speech_mask(w)
    segs = _segments(mask)
    if not segs:
        return out
    fps = SR / HOP
    first, last = segs[0][0] / fps, segs[-1][1] / fps
    out["latency_s"] = round(first, 3)
    out["speech_s"] = round(last - first, 3)

    gaps = [(b[0] - a[1]) / fps for a, b in zip(segs, segs[1:])]
    pauses = [g for g in gaps if g >= MIN_PAUSE_S]
    span = max(last - first, 1e-6)
    out["n_pauses"] = len(pauses)
    out["pause_ratio"] = round(sum(pauses) / span, 4)
    out["pause_mean_s"] = round(float(np.mean(pauses)), 3) if pauses else 0.0

    if words:
        ws = [x for x in words if x.get("end") is not None and x.get("start") is not None]
        if len(ws) >= 2:
            dur = ws[-1]["end"] - ws[0]["start"]
            if dur > 0.2:
                out["speech_rate_wps"] = round(float(len(ws) / dur), 3)

    rms = _frame_rms(w)
    voiced_rms = rms[mask[: len(rms)]] if len(mask) >= len(rms) else rms[mask]
    if len(voiced_rms) > 3:
        db = 20 * np.log10(voiced_rms + 1e-9)
        out["energy_mean_db"] = round(float(np.mean(db)), 2)
        out["energy_std_db"] = round(float(np.std(db)), 2)

    try:
        import librosa
        seg = w[int(first * SR): int(last * SR) + 1]
        if len(seg) > SR * 0.4:
            f0, voiced, _ = librosa.pyin(seg, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=320)
            f = f0[voiced & ~np.isnan(f0)]
            if len(f) >= 5:
                out["pitch_mean_hz"] = round(float(np.mean(f)), 1)
                st = 12 * np.log2(f / np.mean(f))
                out["pitch_std_st"] = round(float(np.std(st)), 3)
    except Exception:  # pitch is optional; never fail the whole analysis
        pass
    return out
