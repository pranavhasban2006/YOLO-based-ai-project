"""Late fusion: confidence-weighted average of per-modality probability vectors."""
import numpy as np

from .config import CLASSES
from .errors import AppError
from .modalities import registry
from .schemas import FusedPrediction, Prediction


def fuse(predictions: list[Prediction]) -> FusedPrediction:
    """Fuse the supplied predictions.

    A prediction is skipped if it is flagged mock, or if the backend's predictor
    for that modality is still a mock (so a client cannot slip placeholder data
    in by setting mock=false), or if its modality is unknown, or if an earlier
    entry already supplied that modality. Registered modalities that were not
    supplied are reported as skipped with reason "not_supplied".

    Each remaining modality is weighted by its own confidence (max probability):
        fused = sum(c_m * p_m) / sum(c_m)
    """
    known = registry.all_predictors()
    used: dict[str, Prediction] = {}
    skipped: list[dict] = []

    for p in predictions:
        if p.modality not in known:
            skipped.append({"modality": p.modality, "reason": "unknown_modality"})
        elif p.mock or registry.is_mock(p.modality):
            skipped.append({"modality": p.modality, "reason": "mock"})
        elif p.modality in used:
            skipped.append({"modality": p.modality, "reason": "duplicate"})
        else:
            used[p.modality] = p

    supplied = {p.modality for p in predictions}
    for m in known:
        if m not in supplied:
            skipped.append({"modality": m, "reason": "not_supplied"})

    if not used:
        raise AppError(422, "no_real_modalities",
                       "Nothing to fuse: every supplied prediction was missing or placeholder (mock) data.")

    weights: dict[str, float] = {}
    total = np.zeros(len(CLASSES))
    for m, p in used.items():
        probs = np.clip(np.asarray(p.probs, dtype=float), 0, None)
        s = probs.sum()
        probs = probs / s if s > 0 else np.full(len(CLASSES), 1 / len(CLASSES))
        c = float(probs.max())
        weights[m] = c
        total += c * probs
    fused = total / sum(weights.values())
    norm_weights = {m: round(w / sum(weights.values()), 5) for m, w in weights.items()}

    k = int(fused.argmax())
    return FusedPrediction(
        modality="fused",
        probs=[float(v) for v in fused],
        label=CLASSES[k],
        confidence=float(fused[k]),
        mock=False,
        contributing=list(used),
        skipped=skipped,
        weights=norm_weights,
    )
