r"""Packs the WHOLE project (code, app, trained models, feature CSVs, every dataset) into 5 zips for Google Drive.

Inside every zip the paths keep the repo layout (YOLO-based-ai-project/...), so after unzipping in Colab the
training scripts run unchanged.

  FULL_1_code_results.zip   all code, backend, frontend, reports, logs, faculty CSV/images, test samples
  FULL_2_models.zip         the 27 trained checkpoints (.pt)
  FULL_3_features.zip       the deep-feature CSV of every model
  FULL_4_data.zip           FER-2013 images, text datasets, speech audio + caches, processed WESAD windows
  FULL_5_wesad_raw.zip      the raw WESAD recordings (17 GB)

Usage (repo root): project\.venv-train\Scripts\python project\code\make_full_package.py <part 1-5>
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Colab_FULL_upload"
TOP = Path(ROOT.name)
SKIP_DIRS = {"node_modules", "__pycache__", "Colab_FULL_upload", "models", "data", "features"}


def part1():
    for f in ROOT.rglob("*"):
        rel = f.relative_to(ROOT)
        if not f.is_file() or any(p.startswith(".") for p in rel.parts):
            continue
        if any(p in SKIP_DIRS for p in rel.parts[:-1]) and rel.parts[:2] != ("backend", "data"):
            continue
        if f.suffix in (".zip", ".pt") or f.name.endswith(".ipynb"):
            continue
        yield f


PARTS = {
    1: ("FULL_1_code_results.zip", part1),
    2: ("FULL_2_models.zip", lambda: (ROOT / "models").rglob("*")),
    3: ("FULL_3_features.zip", lambda: (ROOT / "project" / "features").rglob("*")),
    4: ("FULL_4_data.zip", lambda: [*(ROOT / "project" / "data").glob("*.npz"),
                                    *[f for d in ("fer2013", "text", "audio")
                                      for f in (ROOT / "project" / "data" / d).rglob("*")]]),
    5: ("FULL_5_wesad_raw.zip", lambda: (ROOT / "project" / "data" / "wesad").rglob("*")),
}

if __name__ == "__main__":
    n = int(sys.argv[1])
    name, files = PARTS[n]
    OUT.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(OUT / name, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in files():
            if f.is_file():
                z.write(f, TOP / f.relative_to(ROOT))
                count += 1
    print(f"{name}: {count} files, {(OUT / name).stat().st_size / 1e9:.2f} GB")
