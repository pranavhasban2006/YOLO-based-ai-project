"""Clip-level facial emotion: sample at a fixed rate, classify, smooth, aggregate.

Video is decoded with PyAV directly from the in-memory upload.
"""
import io
import logging

import av
import numpy as np

from .config import CLASSES, VIDEO_MAX_SAMPLES, VIDEO_SAMPLE_FPS, VIDEO_SMOOTH_WINDOW
from .errors import AppError
from .modalities.face import FacePredictor
from .schemas import TimelinePoint, VideoPrediction

log = logging.getLogger(__name__)


def corrupt(detail: str = "") -> AppError:
    msg = "The video could not be decoded. It may be corrupt or use an unsupported codec."
    return AppError(422, "corrupt_video", f"{msg} {detail}".strip())


def sample_frames(data: bytes, fps: float = VIDEO_SAMPLE_FPS, max_samples: int = VIDEO_MAX_SAMPLES):
    """Yield (t_seconds, rgb_array) at roughly `fps`, from in-memory video bytes."""
    try:
        container = av.open(io.BytesIO(data), mode="r")
    except (av.error.FFmpegError, ValueError, OSError) as e:
        raise corrupt() from e
    try:
        if not container.streams.video:
            raise corrupt("No video stream was found in the file.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        step = 1.0 / fps
        next_t = 0.0
        n = 0
        decoded_any = False
        try:
            for frame in container.decode(stream):
                decoded_any = True
                t = frame.time if frame.time is not None else n * step
                if t + 1e-6 < next_t:
                    continue
                yield float(t), frame.to_ndarray(format="rgb24")
                n += 1
                if n >= max_samples:
                    log.info("Video truncated at %d samples", max_samples)
                    break
                # Catch up if the source has gaps, so samples stay ~1/fps apart.
                next_t = max(next_t + step, t + step / 2)
        except av.error.FFmpegError as e:
            if not decoded_any:
                raise corrupt() from e
            log.warning("Video decode stopped early: %s", e)  # keep what decoded cleanly
        if not decoded_any:
            raise corrupt("No frames could be decoded.")
    finally:
        container.close()


def moving_average(rows: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average over rows (each row is a probability vector)."""
    out = np.empty_like(rows)
    for i in range(len(rows)):
        out[i] = rows[max(0, i - window + 1): i + 1].mean(axis=0)
    return out


def predict_video(predictor: FacePredictor, data: bytes) -> VideoPrediction:
    predictor._require_ready()
    points: list[TimelinePoint] = []
    raw: list[np.ndarray] = []
    face_idx: list[int] = []
    duration = 0.0

    for t, rgb in sample_frames(data):
        duration = t
        result = predictor.analyse(rgb)
        if result is None:
            points.append(TimelinePoint(t=round(t, 3), face=False))
            continue
        probs, box, _ = result
        face_idx.append(len(points))
        raw.append(probs)
        points.append(TimelinePoint(t=round(t, 3), face=True, face_box=box))

    frames_without_face = len(points) - len(raw)
    if not raw:
        raise AppError(422, "no_face_detected",
                       f"No face was detected in any of the {len(points)} sampled frames.")

    smoothed = moving_average(np.stack(raw), VIDEO_SMOOTH_WINDOW)
    for i, p in zip(face_idx, smoothed):
        k = int(p.argmax())
        points[i].probs = [round(float(v), 5) for v in p]
        points[i].label = CLASSES[k]
        points[i].confidence = round(float(p[k]), 5)

    clip = smoothed.mean(axis=0)
    k = int(clip.argmax())
    return VideoPrediction(
        modality="face",
        probs=[float(v) for v in clip],
        label=CLASSES[k],
        confidence=float(clip[k]),
        mock=False,
        faces_found=len(raw),
        frames_sampled=len(points),
        frames_without_face=frames_without_face,
        duration=round(duration, 3),
        timeline=points,
        model=predictor.model_name,
    )
