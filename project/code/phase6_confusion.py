"""Phase 6 helper: confusion matrices of the E3 held-out runs.

The Colab run saved each confusion matrix only as an annotated PNG (confusion_<ds>.png), not as numbers.
The counts below were read from those PNGs. They are checked here against the per-class support and
recall that Colab did save in metrics.json, so a misread digit fails the check.

Rows = true class, columns = predicted class, both in CLASSES order.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Transcribed from colab_results/E3_cnn_holdout_<ds>/confusion_<ds>.png
MATRICES = {
    "fer2013": [
        [392, 2291, 203, 35, 16, 56, 32],
        [18, 194, 6, 1, 1, 4, 4],
        [111, 442, 182, 32, 3, 11, 32],
        [1540, 5264, 609, 829, 223, 473, 261],
        [2125, 7113, 1097, 169, 148, 900, 221],
        [404, 2927, 368, 52, 20, 425, 26],
        [681, 1994, 1065, 92, 69, 91, 248],
    ],
    "jaffe": [
        [1, 0, 1, 0, 22, 3, 3],
        [0, 6, 1, 0, 3, 17, 2],
        [0, 0, 13, 0, 3, 14, 2],
        [0, 0, 0, 25, 5, 0, 1],
        [0, 0, 7, 0, 17, 3, 3],
        [0, 1, 3, 1, 5, 19, 2],
        [0, 0, 3, 2, 3, 0, 22],
    ],
    "ravdess": [
        [540, 378, 21, 35, 733, 187, 26],
        [8, 1748, 4, 26, 32, 99, 3],
        [123, 438, 368, 43, 464, 458, 26],
        [54, 128, 0, 1601, 82, 53, 2],
        [79, 81, 2, 632, 1633, 451, 2],
        [24, 635, 0, 72, 397, 792, 0],
        [232, 309, 174, 215, 610, 329, 51],
    ],
    "crema_d": [
        [1148, 944, 769, 636, 1856, 342, 660],
        [445, 3887, 279, 527, 569, 469, 179],
        [488, 678, 2057, 268, 1589, 493, 782],
        [131, 407, 161, 4559, 576, 157, 364],
        [440, 495, 596, 199, 2987, 384, 334],
        [383, 1074, 1007, 266, 2258, 1001, 361],
        [0, 0, 0, 0, 0, 0, 0],
    ],
}


def check(ds, cm, metrics):
    """Row sums must equal the saved per-class support; diagonal / row sum must equal the saved recall."""
    problems = []
    per_class = metrics["tests"][ds]["per_class"]
    for i, c in enumerate(CLASSES):
        row = sum(cm[i])
        sup = per_class[c]["support"]
        if row != sup:
            problems.append(f"{ds} {c}: row sum {row} != support {sup}")
        if sup > 0:
            rec = cm[i][i] / row
            if abs(rec - per_class[c]["recall"]) > 0.0005:
                problems.append(f"{ds} {c}: recall {rec:.4f} != saved {per_class[c]['recall']:.4f}")
    return problems


def load_all():
    """Returns {ds: matrix as np.array} after the checks pass. Raises if any check fails."""
    out, problems = {}, []
    for ds, cm in MATRICES.items():
        metrics = json.loads((ROOT / "colab_results" / f"E3_cnn_holdout_{ds}" / "metrics.json").read_text(encoding="utf-8"))
        problems += check(ds, cm, metrics)
        out[ds] = np.array(cm)
    if problems:
        raise ValueError("confusion transcription failed checks:\n" + "\n".join(problems))
    return out


def top_pairs(cm, k=3):
    """Largest off-diagonal cells, true -> predicted, as (count, row_rate, true, pred)."""
    cells = []
    for i in range(len(CLASSES)):
        row = cm[i].sum()
        for j in range(len(CLASSES)):
            if i != j and cm[i][j] > 0:
                cells.append((int(cm[i][j]), float(cm[i][j]) / row if row else 0.0, CLASSES[i], CLASSES[j]))
    cells.sort(reverse=True)
    return cells[:k]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mats = load_all()
    print("all matrices pass row-sum and recall checks")
    for ds, cm in mats.items():
        print(ds)
        for count, rate, t, p in top_pairs(cm, 6):
            print(f"  {t:>8} -> {p:<8} {count:>6}  ({rate:.1%} of true {t})")
