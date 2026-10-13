"""Phase 7 step 2: run infer.py on 3 test images and 1 test video from the held-out test split.

Samples: seed 42, drawn from data/splits.csv where split == test.
  image 1: fer2013 original 48x48 frame (no detector crop, the whole image is used if no face is found)
  image 2: jaffe face crop
  image 3: crema_d face crop
  video 1: RAVDESS full clip (5 frames sampled, probabilities averaged)
The true label is shown next to each output. It is not passed to infer.py.
Writes logs/phase7_infer_test.json. Run with the project venv under python -I.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
SEED = 42
PLAN = [  # (dataset, kind in splits.csv, what to feed infer.py)
    ("fer2013", "image", "crop"),  # fer2013 rows point at the original 48x48 image
    ("jaffe", "image", "crop"),  # jaffe rows point at a 224x224 face crop
    ("crema_d", "video", "crop"),  # crema_d rows are face crops of video frames
    ("ravdess", "video", "clip"),  # the full RAVDESS clip, 5 frames sampled inside infer.py
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    splits = pd.read_csv(PROJECT / "data" / "splits.csv", keep_default_na=False)
    test = splits[splits["split"] == "test"]

    samples = []
    for dataset, kind, feed in PLAN:
        pool = test[(test["dataset"] == dataset) & (test["kind"] == kind)]
        row = pool.sample(n=1, random_state=SEED).iloc[0]
        if feed == "clip":
            src = REPO / "datasets" / row["path"]  # original clip
        else:
            src = REPO / row["face_path"]  # fer2013 original or a 224x224 face crop
        assert src.is_file(), f"missing sample file: {src}"
        out_kind = "video" if feed == "clip" else "image"
        samples.append({"dataset": dataset, "kind": out_kind, "label": row["label"], "path": str(src)})

    results = []
    for s in samples:
        cmd = [sys.executable, "-I", str(PROJECT / "code" / "infer.py"), s["path"]]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(f"infer.py failed on {s['path']}:\n{proc.stderr}")
        out = json.loads(proc.stdout)
        results.append({"sample": s, "output": out})
        print(f"--- {s['dataset']} {s['kind']}  true label: {s['label']}")
        print(json.dumps(out, indent=2, ensure_ascii=False))

    log = PROJECT / "logs" / "phase7_infer_test.json"
    log.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", log)


if __name__ == "__main__":
    main()
