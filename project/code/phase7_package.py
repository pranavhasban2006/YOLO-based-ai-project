"""Phase 7 step 1: package the selected face-emotion model.

Winner rule (from Phase 6): the architecture with the higher mean E2 macro-F1 in
results/comparison_all.csv. The winning checkpoint is copied as a plain state_dict to
models/face_emotion_best.pt, next to classes.json. Run with the project venv under python -I.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import SmallCNN  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    comp = pd.read_csv(PROJECT / "results" / "comparison_all.csv", keep_default_na=False)
    e2 = comp[(comp["experiment"] == "E2") & (comp["dataset"] == "mean")]
    means = dict(zip(e2["model"], e2["macro_f1"].astype(float)))
    print("E2 mean macro-F1:", means)
    winner = max(means, key=means.get)
    assert winner == "cnn", f"Phase 7 expects the CNN to win on E2 mean macro-F1, got {winner}"

    src = PROJECT / "models" / "E2_cnn" / "best.pt"
    dst = PROJECT / "models" / "face_emotion_best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=True)
    assert isinstance(ck, dict) and all(torch.is_tensor(v) for v in ck.values()), "best.pt is not a plain state_dict"
    model = SmallCNN(n_classes=len(CLASSES))
    model.load_state_dict(ck, strict=True)  # fails loudly if the architecture does not match
    model.eval()
    torch.save(ck, dst)  # state_dict only, no optimiser or epoch info

    same = (PROJECT / "models" / "best_model.pt").exists() and sha256(PROJECT / "models" / "best_model.pt") == sha256(src)
    print("copied", src, "->", dst, "| same bytes as models/best_model.pt:", same)

    classes = json.loads((PROJECT / "results" / "classes.json").read_text(encoding="utf-8"))
    assert classes == CLASSES, f"classes.json differs: {classes}"
    (PROJECT / "models" / "classes.json").write_text(json.dumps(CLASSES, indent=2) + "\n", encoding="utf-8")
    print("wrote models/classes.json and verified state_dict loads into SmallCNN(7)")


if __name__ == "__main__":
    main()
