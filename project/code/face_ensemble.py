r"""Accuracy boosts that reuse the trained face models (no retraining):

  1. Test-time augmentation (TTA): average the softmax of each image and its horizontal mirror.
  2. Ensemble: average TTA softmax over a subset of models. Members are chosen greedily on the VALIDATION
     split (same 90/10 split as train_face.py), never on test, so the test score stays an honest estimate.
  3. Stacked features: concatenates the chosen members' feature CSVs into project/features/face_stack.csv,
     which then goes through protocol.py (80:20 / 70:30 +/- SMOTE, 10-fold CV) like any single model.

Usage (repo root, after all face models are trained and their feature CSVs exported):
    project\.venv-train\Scripts\python project\code\face_ensemble.py
Writes project/reports/face_tta_ensemble_metrics.json (+ per-model probability cache in project/features/).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
import train_face as tf  # noqa: E402
from torchvision import datasets  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "project" / "features"
MODELS = ["cnn", "resnet18", "mobilenet_attn", "cnn_bilstm", "densenet_se", "effnet_cbam", "resnet50_vit"]


def probs(name, device):
    """Softmax on val and test, plain and with horizontal-flip TTA. Cached per model."""
    cache = FEAT / f"face_{name}_probs.npz"
    if cache.exists():
        return dict(np.load(cache))
    model = tf.build(name)
    model.load_state_dict(torch.load(ROOT / "models" / tf.CHECKPOINTS[name], map_location="cpu", weights_only=True))
    model.to(device).eval()
    tfm = tf.transforms_for(name, False)
    train_full = datasets.ImageFolder(tf.DATA / "train", tfm)
    idx = np.random.default_rng(0).permutation(len(train_full))     # identical to train_face.py's split
    sets = {"val": Subset(train_full, idx[:len(idx) // 10]), "test": datasets.ImageFolder(tf.DATA / "test", tfm)}
    out = {}
    for split, ds in sets.items():
        plain, flip, ys = [], [], []
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            for x, y in DataLoader(ds, batch_size=64, num_workers=2):
                x = x.to(device)
                plain.append(torch.softmax(model(x).float(), 1).cpu())
                flip.append(torch.softmax(model(torch.flip(x, dims=[-1])).float(), 1).cpu())
                ys.append(y)
        out[f"{split}_plain"], out[f"{split}_flip"] = torch.cat(plain).numpy(), torch.cat(flip).numpy()
        out[f"{split}_y"] = torch.cat(ys).numpy()
    FEAT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def scores(y, p):
    pr, rc, f, s = precision_recall_fscore_support(y, p, labels=range(7), zero_division=0)
    return {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4),
            "per_class": {c: {"precision": round(pr[i], 4), "recall": round(rc[i], 4), "f1": round(f[i], 4),
                              "support": int(s[i])} for i, c in enumerate(tf.CLASSES)},
            "confusion_matrix": confusion_matrix(y, p, labels=range(7)).tolist()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [m for m in MODELS if (ROOT / "models" / tf.CHECKPOINTS[m]).exists()]
    P = {}
    for m in names:
        P[m] = probs(m, device)
        print(f"{m}: probabilities ready", flush=True)
    tta = lambda m, s: (P[m][f"{s}_plain"] + P[m][f"{s}_flip"]) / 2
    yv, yt = P[names[0]]["val_y"], P[names[0]]["test_y"]
    val_f1 = lambda ms: f1_score(yv, sum(tta(m, "val") for m in ms).argmax(1), average="macro")

    res = {"tta": {}}
    for m in names:
        res["tta"][m] = {"test_plain": scores(yt, P[m]["test_plain"].argmax(1)),
                         "test_tta": scores(yt, tta(m, "test").argmax(1))}
        print(f"{m}: test macroF1 {res['tta'][m]['test_plain']['macro_f1']} -> TTA "
              f"{res['tta'][m]['test_tta']['macro_f1']}", flush=True)

    # Greedy forward selection on validation macro-F1 (stop when adding a model no longer helps).
    chosen, best = [], -1.0
    while True:
        cand = [(val_f1(chosen + [m]), m) for m in names if m not in chosen]
        if not cand or max(cand)[0] <= best:
            break
        best, pick = max(cand)
        chosen.append(pick)
    ens = sum(tta(m, "test") for m in chosen).argmax(1)
    res["ensemble"] = {"members": chosen, "selected_on": "validation macro-F1 (greedy forward selection)",
                       "val_macro_f1": round(best, 4), "test": scores(yt, ens)}
    print(f"ensemble {chosen}: val macroF1 {best:.4f}, test acc {res['ensemble']['test']['accuracy']} "
          f"macroF1 {res['ensemble']['test']['macro_f1']}", flush=True)
    (ROOT / "project" / "reports" / "face_tta_ensemble_metrics.json").write_text(json.dumps(res, indent=2))

    # Stacked feature CSV for the protocol (rows are the same test images in the same order in every CSV).
    parts = [pd.read_csv(FEAT / f"face_{m}.csv") for m in chosen]
    for p in parts[1:]:
        assert (p["label"].to_numpy() == parts[0]["label"].to_numpy()).all()
    stack = pd.concat([p.drop(columns="label").add_prefix(f"{m}_") for m, p in zip(chosen, parts)], axis=1)
    stack["label"] = parts[0]["label"]
    stack.to_csv(FEAT / "face_stack.csv", index=False)
    print(f"face_stack.csv: {stack.shape[1] - 1} features from {chosen}")


if __name__ == "__main__":
    main()
