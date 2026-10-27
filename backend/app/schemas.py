from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .config import CLASSES


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FaceBox(BaseModel):
    """Face box in normalised [0, 1] image coordinates."""
    x: float
    y: float
    w: float
    h: float


class Prediction(BaseModel):
    """The common prediction shape returned by every modality."""
    modality: str
    probs: list[float]
    label: str
    confidence: float
    timestamp: str = Field(default_factory=now_iso)
    mock: bool = False

    @field_validator("probs")
    @classmethod
    def _seven_probs(cls, v: list[float]) -> list[float]:
        if len(v) != len(CLASSES):
            raise ValueError(f"probs must have {len(CLASSES)} values in order {CLASSES}")
        return v

    @classmethod
    def from_probs(cls, modality: str, probs, mock: bool = False, **extra) -> "Prediction":
        probs = [float(p) for p in probs]
        idx = max(range(len(probs)), key=probs.__getitem__)
        return cls(modality=modality, probs=probs, label=CLASSES[idx],
                   confidence=probs[idx], mock=mock, **extra)


class ImagePrediction(Prediction):
    faces_found: int
    face_box: Optional[FaceBox] = None
    model: str
    session_id: Optional[int] = None


class TimelinePoint(BaseModel):
    t: float  # seconds into the clip
    face: bool
    probs: Optional[list[float]] = None  # smoothed; None when no face in this frame
    label: Optional[str] = None
    confidence: Optional[float] = None
    face_box: Optional[FaceBox] = None


class VideoPrediction(Prediction):
    faces_found: int  # sampled frames that contained a face
    frames_sampled: int
    frames_without_face: int
    duration: float
    timeline: list[TimelinePoint]
    model: str
    session_id: Optional[int] = None


class CascadePrediction(Prediction):
    """A1 -> A2 cascade: the top-level fields are the final (A1 + A2) answer, `a1` is A1 alone."""
    a1: Prediction
    changed: bool  # A2 changed A1's label
    model: str
    session_id: Optional[int] = None


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class FusedRequest(BaseModel):
    predictions: list[Prediction]


class FusedPrediction(Prediction):
    contributing: list[str]
    skipped: list[dict]
    weights: dict[str, float]
    session_id: Optional[int] = None
